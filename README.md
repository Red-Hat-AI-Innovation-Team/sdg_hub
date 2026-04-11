# `sdg_hub`: Synthetic Data Generation Toolkit

[![Build](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/pypi.yml/badge.svg?branch=main)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/pypi.yml)
[![Release](https://img.shields.io/github/v/release/Red-Hat-AI-Innovation-Team/sdg_hub)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/releases)
[![License](https://img.shields.io/github/license/Red-Hat-AI-Innovation-Team/sdg_hub)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/blob/main/LICENSE)
[![Tests](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml/badge.svg)](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub/graph/badge.svg?token=SP75BCXWO2)](https://codecov.io/gh/Red-Hat-AI-Innovation-Team/sdg_hub)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Red-Hat-AI-Innovation-Team/sdg_hub)

<p align="center">
  <img src="docs/assets/sdg-hub-cover.png" alt="SDG Hub Cover" width="400">
</p>

A modular Python framework for building synthetic data generation pipelines using composable blocks and flows. Transform datasets through **building-block composition** - mix and match LLM-powered and traditional processing blocks to create sophisticated data generation workflows.

Full documentation available in the [`docs/`](docs/) directory or at [DeepWiki](https://deepwiki.com/Red-Hat-AI-Innovation-Team/sdg_hub).

## Key Features

**Modular Composability** - Mix and match blocks like Lego pieces. Build simple transformations or complex multi-stage pipelines with YAML-configured flows.

**Async Performance** - High-throughput LLM processing with built-in error handling.

**Built-in Validation** - Pydantic-based type safety ensures your configurations and data are correct before execution.

**Auto-Discovery** - Automatic block and flow registration. No manual imports or complex setup.

**Rich Monitoring** - Detailed logging with progress bars and execution summaries.

**Dataset Schema Discovery** - Instantly discover required data formats. Get empty DataFrames with correct schema for easy validation and data preparation.

**Easily Extensible** - Create custom blocks with simple inheritance. Rich logging and monitoring built-in.


## Installation

Recommended: Install uv  — see https://docs.astral.sh/uv/getting-started/installation/

```bash
# Production
uv pip install sdg-hub

# Development
git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
cd sdg_hub
uv pip install .[dev]
# or: uv sync --extra dev
```

### Optional Dependencies
```bash
# For vLLM support
uv pip install sdg-hub[vllm]

# For examples
uv pip install sdg-hub[examples]
```

## Quick Start

### Core Concepts

**Blocks** are composable units that transform datasets. Each block performs a specific task: LLM chat, text parsing, agent tool-calling, filtering, or transformation.

**Flows** orchestrate multiple blocks into complete pipelines defined in YAML. Chain blocks together to create sophisticated data generation workflows.

```
dataset --> Block1 --> Block2 --> Block3 --> enriched_dataset
```

### Example: MCP Server Distillation

This example uses the built-in **MCP Server Distillation** flow to generate tool-use training data. The flow takes an MCP server's tool definitions, uses a frontier model to explore the server, synthesizes realistic questions, and captures expert tool-calling trajectories -- producing supervised fine-tuning data so a smaller model can learn to use the same tools.

```python
import pandas as pd
from sdg_hub import FlowRegistry, Flow

# Discover available flows
FlowRegistry.discover_flows()

# Load the MCP distillation flow by name or ID
flow_path = FlowRegistry.get_flow_path("MCP Server Distillation")
flow = Flow.from_yaml(flow_path)

# Check what the flow needs
requirements = flow.get_dataset_requirements()
print(f"Required columns: {requirements.required_columns}")
# -> ['tool_list', 'mcp_server_name', 'mcp_server_description']

# Prepare your dataset -- one row per MCP server
dataset = pd.DataFrame({
    "tool_list": [[
        {
            "name": "search_products",
            "description": "Search product catalog by query",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "get_inventory",
            "description": "Check inventory levels for a product",
            "inputSchema": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    ]],
    "mcp_server_name": ["ShopInsights Analytics"],
    "mcp_server_description": ["E-commerce analytics platform for product search and inventory"],
})

# Configure the frontier model and agent endpoint
flow.set_model_config(
    model="openai/gpt-4o",
    api_key="your-openai-key",
)
flow.set_agent_config(
    agent_framework="langflow",
    agent_url="http://localhost:7860/api/v1/run/default",
)

# Dry run to validate the pipeline
dry_result = flow.dry_run(dataset, sample_size=1)
print(f"Output columns: {dry_result['final_dataset']['columns']}")

# Generate training data
result = flow.generate(dataset)
```

The output contains question-trajectory pairs ready for supervised fine-tuning: each row has a realistic user question, the tools that should be called, and the full expert tool-calling trace with arguments and outputs.

### Flow Discovery

Every flow has a unique, human-readable ID for easy reference:

```python
from sdg_hub import FlowRegistry

# Search by tag
agentic_flows = FlowRegistry.search_flows(tag="agentic")
eval_flows = FlowRegistry.search_flows(tag="evaluation")

# Browse by category
categories = FlowRegistry.get_flows_by_category()
for category, flows in categories.items():
    print(f"{category}: {[f['name'] for f in flows]}")
```


## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---

Built by the Red Hat AI Innovation Team
