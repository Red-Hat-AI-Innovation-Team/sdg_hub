# CLAUDE.md

## Project Overview

**Requirements:** Python 3.10+

SDG Hub is a Python framework for synthetic data generation using composable blocks and flows. Blocks are processing units that transform datasets; flows chain blocks into pipelines defined in YAML.

Core concept: `dataset -> Block1 -> Block2 -> Block3 -> enriched_dataset`

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Agent Knowledge Base

| Task | Read this first |
|------|----------------|
| Adding a block | docs/agent-knowledge/block-invariants.md |
| Adding a flow | docs/agent-knowledge/flow-invariants.md |
| Adding a connector | docs/agent-knowledge/connector-invariants.md |
| Writing tests | docs/agent-knowledge/testing-standards.md |
| Reviewing code | docs/agent-knowledge/grading-criteria.md |
| Deciding to fix vs escalate | docs/agent-knowledge/decision-rubric.md |
| Checking quality status | docs/agent-knowledge/QUALITY.md |
| All principles | docs/agent-knowledge/core-principles.md |

Full index: docs/agent-knowledge/index.md

## Development Commands

**Use `uv` for all Python commands and package management.**

```bash
# Install with dev dependencies
uv pip install .[dev]

# IMPORTANT: Always install pre-commit hooks after cloning
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# Other install targets
uv pip install .           # Core only
uv pip install .[vllm]     # With vLLM support
uv pip install .[examples] # With examples dependencies
```

### Testing

```bash
# Unit tests (excludes slow/integration)
uv run pytest tests/blocks tests/connectors tests/flow tests/utils -m "not (examples or slow)"

# Structural tests (architecture enforcement)
uv run pytest tests/structural/

# With coverage
uv run pytest --cov=sdg_hub --cov-report=term tests/blocks tests/connectors tests/flow tests/utils

# Integration tests (requires API keys)
uv run pytest tests/integration -v -s
```

### Linting and Formatting

```bash
uv run ruff check --fix src/ tests/    # Lint with auto-fix
uv run ruff format src/ tests/         # Format
uv run mypy src/                       # Type check
```

### Pre-commit Hooks

Hooks run automatically on commit:
- **uv-lock**: Keeps `uv.lock` in sync with `pyproject.toml`
- **ruff**: Linter with auto-fix
- **ruff-format**: Code formatter
- **mypy**: Type checker (runs `mypy src/sdg_hub`)
- **conventional-pre-commit**: Enforces [Conventional Commits](https://www.conventionalcommits.org/)

Commit prefixes: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

## Common Pitfalls

- `flow.set_model_config(model="...", api_key="...")` must be called before `generate()` for any flow containing LLM blocks
- Use `flow.dry_run(dataset)` to validate a pipeline end-to-end without making LLM calls
- `runtime_params` can be passed to `flow.generate(dataset, runtime_params={...})` to override block config at execution time
- LiteLLM reads standard env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) automatically -- you can skip passing `api_key` if the env var is set
- The `FlowCheckpointer` enables resumable execution; pass `checkpoint_dir` to `generate()` to persist intermediate results

## CI Requirements

All PRs must pass:

| Check | Command | Workflow |
|-------|---------|----------|
| Conventional Commits | `commitlint` | commitlint.yml |
| Ruff formatting | `ruff format --check src/ tests/` | lint.yml |
| Ruff linting | `ruff check src/ tests/` | lint.yml |
| Type checking | `mypy src/sdg_hub` | lint.yml |
| Unit tests | `pytest tests/blocks tests/connectors tests/flow tests/utils` | test.yml |
| Structural tests | `pytest tests/structural/` | test.yml |
| Lock file sync | `uv lock --check` | lock.yml |
| Markdown linting | `markdownlint-cli2` | docs.yml |
| GitHub Actions lint | `actionlint` | actionlint.yml |
