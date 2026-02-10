# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Requirements:** Python 3.9+

SDG Hub is a Python framework for synthetic data generation using composable blocks and flows. Transform datasets through **building-block composition** - mix and match LLM-powered and traditional processing blocks like Lego pieces to create sophisticated data generation workflows.

**Core Concepts:**
- **Blocks** are composable units that transform datasets - think data processing Lego pieces
- **Flows** orchestrate multiple blocks into complete pipelines defined in YAML
- Simple concept: `dataset → Block₁ → Block₂ → Block₃ → enriched_dataset`

## Development Commands

**Use `uv` for all Python commands and package management.**

### Setup and Installation
```bash
# Install core dependencies
uv pip install .

# Install with development dependencies
uv pip install .[dev]
# Alternative: uv sync --extra dev

# Install with optional vLLM support
uv pip install .[vllm]
# Alternative: uv sync --extra vllm

# Install with examples dependencies
uv pip install .[examples]
# Alternative: uv sync --extra examples
```

### Testing
```bash
# Run all unit tests
uv run pytest tests/blocks tests/connectors tests/flow tests/utils -m "not (examples or slow)"

# Run tests with coverage
uv run pytest --cov=sdg_hub --cov-report=term tests/blocks tests/connectors tests/flow tests/utils

# Run specific test file
uv run pytest tests/test_specific_file.py

# Run tests matching pattern
uv run pytest -k "test_pattern"

# Run integration tests (requires API keys)
uv run pytest tests/integration -v -s
```

### Linting and Formatting
```bash
# Run ruff linter with auto-fix
uv run ruff check --fix src/ tests/

# Run ruff formatter
uv run ruff format src/ tests/

# Check only (no fixes) - same as CI
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

### Other Make targets
```bash
make actionlint    # Lint GitHub Actions
make md-lint       # Lint markdown files
```

## Core Architecture

### Block System
The framework is built around a modular block system with **composability at its core** - mix and match blocks to build simple transformations or complex multi-stage pipelines:

- **BaseBlock** (`src/sdg_hub/core/blocks/base.py`): Abstract base class for all processing blocks with Pydantic validation
- **BlockRegistry** (`src/sdg_hub/core/blocks/registry.py`): Auto-discovery system for organizing blocks with zero setup
- Blocks are organized in categories:
  - `llm/`: LLM-powered blocks (chat, prompt building, text parsing) with async execution
  - `transform/`: Data transformation blocks (column operations, text manipulation)
  - `filtering/`: Data filtering blocks with quality thresholds
  - `evaluation/`: Quality evaluation blocks (faithfulness, relevancy assessment)
  - `agent/`: Agent framework integration blocks (Langflow, etc.)

**Key Benefits**: Type-safe composition, automatic validation, rich logging, and high-performance async processing.

### Flow System
Flows orchestrate multiple blocks into data processing pipelines:

- **Flow** (`src/sdg_hub/core/flow/base.py`): Main flow execution class with Pydantic validation
- **FlowRegistry** (`src/sdg_hub/core/flow/registry.py`): Registry for flow discovery
- **FlowMetadata** (`src/sdg_hub/core/flow/metadata.py`): Metadata and parameter definitions
- **FlowValidator** (`src/sdg_hub/core/flow/validation.py`): YAML structure validation

### Flow Configuration
Flows are defined in YAML files with this structure:
```yaml
metadata:
  name: "flow_name"
  version: "1.0.0"
  author: "Author Name"
  description: "Flow description"

parameters:
  param_name:
    type: "string"
    default: "default_value"
    description: "Parameter description"

blocks:
  - block_type: "BlockTypeName"
    block_config:
      block_name: "unique_block_name"
      # block-specific configuration
```

### Built-in Flow Discovery
The framework includes auto-discovery for flows in `src/sdg_hub/flows/`. Example flow structure:
```
flows/qa_generation/document_grounded_qa/multi_summary_qa/instructlab/
├── flow.yaml                    # Main flow definition
├── atomic_facts.yaml           # Sub-flow configurations
├── detailed_summary.yaml
└── generate_questions_responses.yaml
```

### Connector System
Connectors handle communication with external agent frameworks:

- **BaseConnector** (`src/sdg_hub/core/connectors/base.py`): Abstract base for all connectors
- **ConnectorRegistry** (`src/sdg_hub/core/connectors/registry.py`): Auto-discovery for connectors
- **BaseAgentConnector** (`src/sdg_hub/core/connectors/agent/base.py`): Base class for agent framework connectors

**Supported Agent Frameworks:**
- **Langflow** (`src/sdg_hub/core/connectors/agent/langflow.py`): Visual LLM app builder

**Using AgentBlock:**
```python
from sdg_hub.core.blocks.agent import AgentBlock

block = AgentBlock(
    block_name="my_agent",
    agent_framework="langflow",  # Connector name from registry
    agent_url="http://localhost:7860/api/v1/run/my-flow",
    agent_api_key="your-api-key",  # Optional
    input_cols=["question"],
    output_cols=["response"],
    extract_response=True,  # Extract text from response (Langflow-specific)
)

result = block.generate(dataset)
```

**Adding New Connectors:**
1. Create a new file in `src/sdg_hub/core/connectors/agent/`
2. Inherit from `BaseAgentConnector`
3. Implement `build_request()` and `parse_response()` methods
4. Register with `@ConnectorRegistry.register("name")`

## Key Patterns

### Block Development
When creating new blocks:
1. Inherit from `BaseBlock` and implement the `generate()` method
2. Use Pydantic field validation for configuration
3. Follow the standardized column handling patterns (`input_cols`, `output_cols`)
4. Register blocks in appropriate category directories
5. Include proper error handling and logging

### Dataset Processing
All blocks operate on HuggingFace `datasets.Dataset` objects:
- Input validation ensures required columns exist
- Output validation prevents column collisions
- Rich logging provides processing summaries
- Empty dataset handling with appropriate errors


## Testing Guidelines

- Tests are organized by block category under `tests/blocks/`
- Use `pytest` fixtures for common test data
- Test configuration files are in `tests/blocks/testdata/`
- Follow the existing pattern of testing both success and error cases
- Mock LLM clients when testing LLM-powered blocks

## Important Notes

- Always use `uv` for Python package management
- The framework uses Pydantic extensively for validation and configuration
- LLM clients are managed through the `client_manager.py` system
- Path resolution is handled centrally in `utils/path_resolution.py`
- Error handling follows custom exception patterns in `utils/error_handling.py`