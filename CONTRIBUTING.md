# Contributing to SDG Hub

Thank you for your interest in contributing to SDG Hub. Whether you're
fixing a bug, adding a new block, improving documentation, or suggesting
a feature, your contributions are welcome.

## Ways to Contribute

- **Report bugs** -- open an issue with a minimal reproducible example
- **Suggest features** -- open an issue describing the use case and proposed API
- **Submit code** -- fix bugs, add blocks/flows/connectors, improve internals
- **Improve documentation** -- fix errors, add examples, clarify explanations
- **Review pull requests** -- provide constructive feedback on open PRs

## Reporting Issues

### Bug Reports

Before opening a new issue, search [existing issues](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/issues)
to avoid duplicates. When filing a bug report, include:

- Python version and OS
- SDG Hub version (`pip show sdg-hub`)
- Minimal code to reproduce the issue
- Full traceback or error message
- Expected vs. actual behavior

### Feature Requests

Describe the motivation and use case. Include a code example showing your
desired API if possible. For new blocks or flows, explain what data
transformation you need and why existing components don't cover it.

## Development Setup

### Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/sdg_hub.git
cd sdg_hub

# 2. Install dev dependencies
uv sync --extra dev

# 3. Install pre-commit hooks (required)
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

### Running Tests

```bash
# Unit tests (excludes slow/integration tests)
uv run pytest tests/blocks tests/connectors tests/flow tests/utils \
  -m "not (examples or slow)"

# With coverage
uv run pytest --cov=sdg_hub --cov-report=term \
  tests/blocks tests/connectors tests/flow tests/utils

# Single test file or pattern
uv run pytest tests/blocks/test_specific.py
uv run pytest -k "test_pattern"
```

### Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting,
and [mypy](https://mypy-lang.org/) for type checking. Pre-commit hooks
run these automatically on commit, but you can run them manually:

```bash
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

## Pull Request Process

### Workflow

1. **Fork** the repository and create a branch from `main`
2. **Implement** your changes with tests
3. **Run tests and linting** locally to verify everything passes
4. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/)
   format (enforced by pre-commit hooks)
5. **Push** your branch and open a pull request
6. **Address** review feedback
7. A maintainer will merge when approved

### Commit Messages

Format: `<type>(<scope>): <description>`

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`,
`test`, `build`, `ci`, `chore`, `revert`.

Examples:

```
feat(blocks): add TextSummarizerBlock for document summarization
fix(flows): correct parameter validation in QA generation flow
docs(blocks): update LLM block examples with new model config
```

### PR Checklist

Before submitting, verify:

- [ ] Code follows the project's style (Ruff passes, no mypy errors)
- [ ] Tests cover the new or changed behavior
- [ ] All existing tests still pass
- [ ] Documentation is updated if the public API changed
- [ ] Commit messages follow Conventional Commits format

### CI Checks

All PRs must pass these automated checks before merging:

| Check | Command |
|-------|---------|
| Ruff linting | `ruff check src/ tests/` |
| Ruff formatting | `ruff format --check src/ tests/` |
| Type checking | `mypy src/sdg_hub` |
| Unit tests | `pytest tests/blocks tests/connectors tests/flow tests/utils` |
| Lock file sync | `uv lock --check` |
| Conventional Commits | `commitlint` |
| Markdown linting | `markdownlint-cli2` |
| GitHub Actions lint | `actionlint` |

## Contributing Blocks, Flows, and Connectors

For detailed contribution workflows -- including how to structure new
blocks, create flow YAML files, and register connectors -- see the
[Development Guide](docs/development.md).

Quick reference:

- **New blocks** go in `src/sdg_hub/core/blocks/<category>/`, registered
  with `@BlockRegistry.register()`, tested in `tests/blocks/<category>/`
- **New flows** go in `src/sdg_hub/flows/<category>/<use_case>/` with a
  `flow.yaml` and prompt templates
- **New connectors** go in `src/sdg_hub/core/connectors/agent/`, registered
  with `@ConnectorRegistry.register()`

## Documentation

Docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).
To preview locally:

```bash
uv pip install .[docs]
uv run mkdocs serve
```

The site renders at `http://localhost:8000`. API reference is auto-generated
from docstrings via [mkdocstrings](https://mkdocstrings.github.io/).

## Community Guidelines

- Be respectful and inclusive in all interactions
- Provide constructive, specific feedback
- Follow existing patterns in the codebase
- Report security issues responsibly via private disclosure
