# CLAUDE.md

## Project Overview

**Requirements:** Python 3.10+

SDG Hub is a Python framework for synthetic data generation using composable blocks and flows. Blocks are processing units that transform datasets; flows chain blocks into pipelines defined in YAML.

Core concept: `dataset → Block₁ → Block₂ → Block₃ → enriched_dataset`

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
- **conventional-pre-commit**: Enforces [Conventional Commits](https://www.conventionalcommits.org/)

Commit prefixes: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

## Codebase Orientation

### Block System (`src/sdg_hub/core/blocks/`)

Blocks are organized by category:

| Category | What's there |
|----------|-------------|
| `llm/` | LLMChatBlock, PromptBuilderBlock, LLMResponseExtractorBlock |
| `parsing/` | TagParserBlock, RegexParserBlock, JSONParserBlock |
| `transform/` | TextConcatBlock, RenameColumnsBlock, MeltColumnsBlock, RowMultiplierBlock, and others |
| `filtering/` | ColumnValueFilterBlock |
| `agent/` | AgentBlock, AgentResponseExtractorBlock |
| `mcp/` | MCPAgentBlock (agentic tool-use with remote MCP servers) |

Run `BlockRegistry.discover_blocks()` to see all registered blocks.

### Flow System (`src/sdg_hub/core/flow/`)

Flows are YAML-defined pipelines that chain blocks. Key entry points: `Flow.from_yaml()`, `flow.generate()`, `flow.dry_run()`, `flow.set_model_config()`.

Pre-built flows live in `src/sdg_hub/flows/` (knowledge infusion, red-teaming, RAG evaluation, MCP distillation, text analysis). Use `FlowRegistry.discover_flows()` to list them.

### Connectors (`src/sdg_hub/core/connectors/`)

Agent framework connectors for external integrations:
- **Langflow** (`langflow`) -- visual LLM app builder
- **LangGraph** (`langgraph`) -- stateful multi-actor agents

Configure via `flow.set_agent_config(agent_framework="...", agent_url="...", agent_api_key="...")`.

## Creating New Blocks

1. Inherit from `BaseBlock`, implement `generate()`
2. Use Pydantic fields for configuration
3. Use `input_cols`/`output_cols` for column handling
4. Register with `@BlockRegistry.register(name, category, description)`
5. Add tests under `tests/blocks/`

## Creating New Connectors

1. Create a file in `src/sdg_hub/core/connectors/agent/`
2. Inherit from `BaseAgentConnector`
3. Implement `build_request()` and `parse_response()`
4. Register with `@ConnectorRegistry.register("name")`

## Testing Guidelines

- Tests organized by category under `tests/blocks/`, `tests/connectors/`, `tests/flow/`, `tests/utils/`
- Test config files in `tests/blocks/testdata/`
- Mock LLM clients when testing LLM-powered blocks
- Test both success and error cases

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
| Lock file sync | `uv lock --check` | lock.yml |
| Markdown linting | `markdownlint-cli2` | docs.yml |
| GitHub Actions lint | `actionlint` | actionlint.yml |
