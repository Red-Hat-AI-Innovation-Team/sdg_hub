# SDG Hub

A modular Python framework for building synthetic data generation pipelines using composable blocks and flows.

## Core Concept

SDG Hub transforms datasets through building-block composition. Chain blocks together to create multi-step data generation workflows:

```
dataset --> Block1 --> Block2 --> Block3 --> enriched_dataset
```

Each block performs one transformation -- an LLM call, a text parse, a column filter -- and passes the result to the next block. Flows define these chains in YAML so they are portable and reproducible.

## Key Capabilities

- **Composable blocks** -- LLM, parsing, transform, filtering, and agent blocks that snap together in any order
- **YAML-defined flow pipelines** -- declare multi-block workflows in configuration, not code
- **Auto-discovery** -- `FlowRegistry` and `BlockRegistry` find and catalog all available components automatically
- **Async LLM processing** -- 100+ LLM providers supported through LiteLLM (OpenAI, Anthropic, Ollama, vLLM, and more)
- **Pydantic validation** -- every block and flow config is validated at construction time
- **Rich monitoring and logging** -- formatted tables, progress bars, and structured logs throughout execution

## Installation

```bash
pip install sdg-hub
```

Or with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install sdg-hub
```

See the [Installation guide](installation.md) for optional dependencies and development setup.

## Quick Example

```python
from sdg_hub import FlowRegistry, Flow

# Discover all available flows
FlowRegistry.discover_flows()

# List flows programmatically
flows = FlowRegistry.list_flows()
# Returns: [{"id": "flow-id", "name": "Flow Name"}, ...]

# Load a flow from the registry
flow_path = FlowRegistry.get_flow_path("flow-id-or-name")
flow = Flow.from_yaml(flow_path)

# Check model requirements
print(flow.get_default_model())          # e.g., "openai/gpt-4o"
print(flow.get_model_recommendations())  # {"default": ..., "compatible": ..., "experimental": ...}

# Configure the LLM
flow.set_model_config(
    model="openai/gpt-4o",
    api_key="your-api-key",
)
```

## What to Read Next

- [Installation](installation.md) -- optional dependencies, development setup, and verification
- [Quick Start](quickstart.md) -- end-to-end walkthrough from loading a flow to generating data
- [Core Concepts](concepts.md) -- blocks, flows, registries, and dataset handling explained
- [Blocks](blocks/index.md) -- reference for all block types (LLM, parsing, transform, filtering, agent)
- [Flows](flows/index.md) -- YAML flow format, built-in flows, and creating custom flows
