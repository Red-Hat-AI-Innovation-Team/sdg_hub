# SPDX-License-Identifier: Apache-2.0

SDG_IMPORT_REF:=e5368f0b0a2d019f6aa56837f9a3c6ee6ba72197

#
# If you want to see the full commands, run:
#   NOISY_BUILD=y make
#
ifeq ($(NOISY_BUILD),)
    ECHO_PREFIX=@
    CMD_PREFIX=@
    PIPE_DEV_NULL=> /dev/null 2> /dev/null
else
    ECHO_PREFIX=@\#
    CMD_PREFIX=
    PIPE_DEV_NULL=
endif

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: action-lint actionlint
action-lint: actionlint
actionlint: ## Lint GitHub Action workflows
	$(ECHO_PREFIX) printf "  %-12s .github/...\n" "[ACTION LINT]"
	$(CMD_PREFIX) if ! command -v actionlint $(PIPE_DEV_NULL) ; then \
		echo "Please install actionlint." ; \
		echo "go install github.com/rhysd/actionlint/cmd/actionlint@latest" ; \
		exit 1 ; \
	fi
	$(CMD_PREFIX) if ! command -v shellcheck $(PIPE_DEV_NULL) ; then \
		echo "Please install shellcheck." ; \
		echo "https://github.com/koalaman/shellcheck#user-content-installing" ; \
		exit 1 ; \
	fi
	$(CMD_PREFIX) actionlint -color

.PHONY:check
check: ## Check git diff between this repo and the CLI generator directory
	@(git remote | grep -q "^instructlab_repo") || git remote add instructlab_repo https://github.com/instructlab/instructlab
	@git fetch instructlab_repo
	@echo "==="
	@echo "=== CHANGES SINCE LAST IMPORT FROM instructlab/instructlab repo:"
	@echo "==="
	@git diff $(SDG_IMPORT_REF)..instructlab_repo/main -- src/instructlab/generator/ | cat

.PHONY: check-tox
check-tox:
	@command -v tox &> /dev/null || (echo "'tox' is not installed" && exit 1)

.PHONY: md-lint
md-lint: ## Lint markdown files
	$(ECHO_PREFIX) printf "  %-12s ./...\n" "[MD LINT]"
	$(CMD_PREFIX) podman run --rm -v $(CURDIR):/workdir --security-opt label=disable docker.io/davidanson/markdownlint-cli2:latest > /dev/null

.PHONY: spellcheck
spellcheck: ## Spellcheck markdown files
	tox p -e spellcheck

.PHONY: spellcheck-sort
spellcheck-sort: .spellcheck-en-custom.txt ## Sort spellcheck directory
	sort -d -f -o $< $<

.PHONY: verify
verify: check-tox ## Run linting, typing, and formatting checks via tox
	tox p -e fastlint,mypy,ruff
