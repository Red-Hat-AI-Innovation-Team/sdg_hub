# Installation

SDG Hub requires **Python 3.10+** and can be installed from PyPI or from source for development.

## Basic Installation

**pip:**

```bash
pip install sdg-hub
```

**uv (recommended):**

```bash
uv pip install sdg-hub
```

Or add it to an existing uv project:

```bash
uv init my-sdg-project
cd my-sdg-project
uv add sdg-hub
```

## Optional Dependencies

SDG Hub defines several optional dependency groups for different use cases. Install them by appending the group name in brackets.

### examples

Heavy dependencies for running example notebooks (document parsing, knowledge mixing, embeddings). Not required for core functionality.

Includes: tabulate, transformers, langchain-text-splitters, docling, scikit-learn, polars, matplotlib, spacy, nltk, sentence-transformers, instructor, fastapi, ipykernel.

**pip:**

```bash
pip install sdg-hub[examples]
```

**uv:**

```bash
uv pip install sdg-hub[examples]
```

### integration

Minimal dependencies for integration testing.

Includes: nest-asyncio.

**pip:**

```bash
pip install sdg-hub[integration]
```

**uv:**

```bash
uv pip install sdg-hub[integration]
```

### docs

Dependencies for building the documentation site.

Includes: mkdocs-material, mkdocstrings, griffe-pydantic, mkdocs-llmstxt.

**pip:**

```bash
pip install sdg-hub[docs]
```

**uv:**

```bash
uv pip install sdg-hub[docs]
```

### dev

Development and testing tools. See [Development Setup](#development-setup) below.

## Development Setup

To contribute to SDG Hub or run tests locally:

### 1. Clone and install

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
cd sdg_hub

# Install with development dependencies
uv pip install .[dev]

# Alternative: use uv sync with the lock file
uv sync --extra dev
```

### 2. Install pre-commit hooks

This step is required. The hooks enforce formatting, linting, and conventional commit messages on every commit.

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

The pre-commit configuration runs:

- **uv-lock** -- keeps `uv.lock` in sync with `pyproject.toml`
- **ruff** -- linter with auto-fix
- **ruff-format** -- code formatter
- **conventional-pre-commit** -- enforces [Conventional Commits](https://www.conventionalcommits.org/) message format (prefixes: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`)

### 3. Run tests

```bash
# Unit tests (excludes slow and integration tests)
uv run pytest tests/blocks tests/connectors tests/flow tests/utils -m "not (examples or slow)"

# With coverage
uv run pytest --cov=sdg_hub --cov-report=term tests/blocks tests/connectors tests/flow tests/utils
```

### 4. Lint and format

```bash
# Lint with auto-fix
uv run ruff check --fix src/ tests/

# Format
uv run ruff format src/ tests/

# Type checking
uv run mypy src/
```

### Dev extra contents

The `dev` group includes: coverage, mypy, nbconvert, pre-commit, pytest, pytest-asyncio, pytest-cov, pytest-env, pytest-html, ruff.

## Verify Installation

After installing, confirm everything works:

```python
from sdg_hub import FlowRegistry, BlockRegistry

# Discover available components
FlowRegistry.discover_flows()
BlockRegistry.discover_blocks()

# Check counts
print(f"Available flows: {len(FlowRegistry.list_flows())}")
print(f"Available blocks: {len(BlockRegistry.list_blocks())}")
```

## Next Steps

Once installed, head to the [Quick Start](quickstart.md) guide to build your first synthetic data pipeline.
