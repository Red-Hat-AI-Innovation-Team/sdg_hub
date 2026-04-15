# Development Guide

This guide covers everything you need to contribute to SDG Hub, from
initial setup through CI requirements and contribution workflows.

## Setup

### Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) package manager

### Clone and Install

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
cd sdg_hub
```

Install with development dependencies:

```bash
uv pip install .[dev]
# or equivalently:
uv sync --extra dev
```

### Pre-commit Hooks

Pre-commit hooks are required for all contributors. Install them immediately
after cloning:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

## Testing

All commands below are verified against `pyproject.toml` and the CI
workflow in `.github/workflows/test.yml`.

### Unit Tests

```bash
uv run pytest tests/blocks tests/connectors tests/flow tests/utils -m "not (examples or slow)"
```

### Unit Tests with Coverage

```bash
uv run pytest --cov=sdg_hub --cov-report=term tests/blocks tests/connectors tests/flow tests/utils
```

### Integration Tests

Integration tests require API keys and the `integration` extra:

```bash
uv sync --extra dev --extra integration
uv run pytest tests/integration -v -s
```

### Running Specific Tests

```bash
# Single file
uv run pytest tests/blocks/test_specific_file.py

# Pattern match
uv run pytest -k "test_pattern"
```

### Pytest Configuration

The following settings are defined in `pyproject.toml` under
`[tool.pytest.ini_options]`:

| Setting | Value |
|---------|-------|
| `asyncio_mode` | `auto` |
| `asyncio_default_fixture_loop_scope` | `function` |
| `LOG_LEVEL` | `WARNING` (via `pytest-env`) |

Custom markers:

- `integration` -- notebook-based end-to-end tests
- `slow` -- long-running tests (>60 s)

## Linting and Formatting

### Ruff

Ruff handles both linting and formatting. Configuration lives in
`pyproject.toml` under `[tool.ruff]`.

```bash
# Lint with auto-fix
uv run ruff check --fix src/ tests/

# Format
uv run ruff format src/ tests/

# Check only (no changes) -- same as CI
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Enabled rule sets: `E` (pycodestyle), `F` (Pyflakes), `I` (isort),
`N` (pep8-naming), `Q` (flake8-quotes), `TID` (flake8-tidy-imports).

Target version: Python 3.10. Line length: 88.

### Mypy

```bash
uv run mypy src/sdg_hub
```

The CI workflow runs `uv run mypy src/sdg_hub --show-error-codes`.
Configuration is in `pyproject.toml` under `[tool.mypy]` with
`disable_error_code = ["import-not-found", "import-untyped"]`.

### Make Targets

The `Makefile` provides two optional convenience targets that require
external tools:

```bash
make actionlint    # Lint GitHub Actions (requires actionlint + shellcheck)
make md-lint       # Lint markdown files (requires podman)
make verify        # Run ruff check + ruff format --check
```

## Pre-commit Hooks

The following hooks are configured in `.pre-commit-config.yaml` and run
automatically on each commit:

| Hook | Stage | Source | What It Does |
|------|-------|--------|--------------|
| `uv-lock` | `pre-commit` | `astral-sh/uv-pre-commit` | Keeps `uv.lock` in sync with `pyproject.toml` |
| `ruff` | `pre-commit` | `astral-sh/ruff-pre-commit` | Lints Python code with auto-fix (`--fix`) |
| `ruff-format` | `pre-commit` | `astral-sh/ruff-pre-commit` | Formats Python code |
| `conventional-pre-commit` | `commit-msg` | `compilerla/conventional-pre-commit` | Validates commit message format |

The `conventional-pre-commit` hook runs at the `commit-msg` stage, which
is why it requires the separate install command
(`uv run pre-commit install --hook-type commit-msg`).

## CI Requirements

All pull requests must pass these checks before merging. The table below
lists each check with its exact command and workflow file.

| Check | Command | Workflow |
|-------|---------|----------|
| Ruff formatting | `ruff format --check src/ tests/` | `lint.yml` |
| Ruff linting | `ruff check src/ tests/` | `lint.yml` |
| Mypy type checking | `mypy src/sdg_hub --show-error-codes` | `lint.yml` |
| Unit tests | `pytest --cov=sdg_hub tests/blocks tests/connectors tests/flow tests/utils -m "not (examples or slow)"` | `test.yml` |
| Conventional commits | `commitlint` | `commitlint.yml` |
| Lock file sync | `uv lock --check` | `lock.yml` |
| Markdown lint | `markdownlint-cli2` | `docs.yml` |
| GitHub Actions lint | `actionlint` | `actionlint.yml` |
| Integration tests | `pytest tests/integration -v -s` | `integration-test.yml` (gated) |

Unit tests run on Python 3.10 and 3.11 (Ubuntu), plus Python 3.11 (macOS).

Integration tests are gated: they run on push to `main`, on
`workflow_dispatch`, or on PRs with the `run-integration-tests` label
when relevant paths change.

## Contributing Blocks

Blocks live under `src/sdg_hub/core/blocks/` in category directories:

- `llm/` -- LLM-powered blocks (chat, prompt building, text parsing)
- `transform/` -- data transformation blocks (column operations, text manipulation)
- `filtering/` -- data filtering blocks with quality thresholds
- `agent/` -- agent framework integration blocks

### Creating a Block

1. Create a new file in the appropriate category directory.
2. Inherit from `BaseBlock` and implement the `generate()` method.
3. Register with the `@BlockRegistry.register()` decorator.
4. Add tests in `tests/blocks/<category>/`.

Minimal template:

```python
from typing import Any

import pandas as pd

from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry


@BlockRegistry.register("MyBlock", "transform", "Short description")
class MyBlock(BaseBlock):
    """What this block does.

    Parameters
    ----------
    custom_param : str
        Description of the parameter.
    """

    custom_param: str = "default"

    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        # Implementation here
        return samples
```

### Block Test Template

```python
import pandas as pd
import pytest

from sdg_hub.core.blocks.transform.my_block import MyBlock


class TestMyBlock:
    def test_basic(self):
        block = MyBlock(
            block_name="test",
            input_cols=["text"],
            output_cols=["result"],
        )
        df = pd.DataFrame({"text": ["hello", "world"]})
        result = block(df)
        assert "result" in result.columns

    def test_missing_column(self):
        block = MyBlock(
            block_name="test",
            input_cols=["missing"],
            output_cols=["result"],
        )
        df = pd.DataFrame({"other": ["data"]})
        with pytest.raises(Exception):
            block(df)
```

### Block Checklist

- Block placed in the correct category directory
- Inherits from `BaseBlock` and implements `generate()`
- Registered with `@BlockRegistry.register(name, category, description)`
- Pydantic field validation for configuration
- Tests cover success, error, and edge cases
- Docstring with parameter descriptions

## Contributing Flows

Flows are defined as YAML files under `src/sdg_hub/flows/`.

### Directory Structure

```
src/sdg_hub/flows/<category>/<use_case>/<variant>/
    flow.yaml
    prompts/
        prompt_template.yaml
```

### flow.yaml Requirements

Every flow must include:

```yaml
metadata:
  name: "flow_name"
  version: "1.0.0"
  author: "Author Name"
  description: "What this flow does"

parameters:
  param_name:
    type: "string"
    default: "value"
    description: "What this parameter controls"

blocks:
  - block_type: "BlockTypeName"
    block_config:
      block_name: "unique_name"
      # block-specific config
```

### Flow Checklist

- Directory structure follows the convention above
- `flow.yaml` includes complete metadata
- Required input columns documented
- Supporting prompt templates included
- Integration tests validate execution

## Contributing Connectors

Connectors handle communication with external agent frameworks and live
under `src/sdg_hub/core/connectors/agent/`.

### Creating a Connector

1. Create a new file in `src/sdg_hub/core/connectors/agent/`.
2. Inherit from `BaseAgentConnector`.
3. Register with `@ConnectorRegistry.register("name")`.
4. Implement required methods.

```python
from sdg_hub.core.connectors.agent.base import BaseAgentConnector
from sdg_hub.core.connectors.registry import ConnectorRegistry


@ConnectorRegistry.register("myframework")
class MyFrameworkConnector(BaseAgentConnector):
    def build_request(self, **kwargs):
        # Build the HTTP request for the agent framework
        ...

    def parse_response(self, response):
        # Parse the raw response
        return response

    @classmethod
    def extract_text(cls, response):
        # Extract text from agent response (used by AgentResponseExtractorBlock)
        return None

    @classmethod
    def extract_session_id(cls, response):
        return None

    @classmethod
    def extract_tool_trace(cls, response):
        return None
```

The `extract_*` class methods are used by `AgentResponseExtractorBlock`
to extract structured data from agent responses without changing block code.

## Git Workflow

### Conventional Commits

Commit messages must follow
[Conventional Commits](https://www.conventionalcommits.org/) format.
This is enforced by the `conventional-pre-commit` hook at the `commit-msg`
stage and validated in CI by `commitlint`.

Format: `<type>(<scope>): <description>`

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`,
`test`, `build`, `ci`, `chore`, `revert`.

Examples:

```
feat(blocks): add TextSummarizerBlock for document summarization
fix(flows): correct parameter validation in QA generation flow
docs(blocks): update LLM block examples with new model config
test(connectors): add integration tests for LangGraph connector
```

### Branch Naming

- `feature/<description>` -- new features (blocks, flows, connectors)
- `fix/<description>` -- bug fixes
- `docs/<description>` -- documentation changes
- `chore/<description>` -- maintenance tasks

### Pull Request Process

1. Create a feature branch from `main`.
2. Implement changes with tests.
3. Run tests and linting locally:

    ```bash
    uv run pytest tests/blocks tests/connectors tests/flow tests/utils -m "not (examples or slow)"
    uv run ruff check src/ tests/
    uv run ruff format --check src/ tests/
    uv run mypy src/sdg_hub
    ```

4. Open a PR with a clear description.
5. Address review feedback.
6. Squash and merge when approved.

## Docstring Guidelines

Docstrings are optional but recommended for public API functions, complex
functions, and core framework components. Use NumPy-style format:

```python
def my_function(param1: str, param2: int = 5) -> bool:
    """One-line summary.

    Longer description if needed.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int, optional
        Description of param2 (default: 5).

    Returns
    -------
    bool
        Description of return value.

    Raises
    ------
    ValueError
        When invalid parameters are provided.
    """
```
