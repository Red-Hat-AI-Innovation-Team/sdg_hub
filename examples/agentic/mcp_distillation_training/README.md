# MCP Server Distillation — E-commerce Example

Automatically generate tool-use training data to teach a small language model how to use your MCP server's tools.

## What this example does

A small model (e.g., Qwen3-8B) connected to your MCP server struggles with tool selection, parameter extraction, and multi-step chaining. This pipeline uses a **frontier model** to generate expert-quality training data that you can use to fine-tune the small model.

```
  ┌───────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │  Expert   │   │ Question │   │ Quality  │   │ Expert   │   │ Format   │
  │Exploration│──▶│Synthesis │──▶│ Filter   │──▶│Trajectory│──▶│ Training │
  │           │   │          │   │          │   │          │   │   Data   │
  └───────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
   Frontier model  Teacher LLM   Teacher LLM    Frontier model  Structured
   calls MCP tools generates Qs  scores quality  solves via MCP  messages
```

**Key idea:** The frontier model *actively explores* your MCP server — calling every tool, discovering real data, mapping tool relationships — then generates training examples grounded in what it found. This is fundamentally different from approaches that only read tool schemas.

## Files

| File | Description |
|---|---|
| `server.py` | **ShopInsights Analytics Platform** — a 15-tool FastMCP e-commerce server with deliberate ambiguity clusters |
| `data.py` | Deterministic seed data: 51 products, 30 customers, 200 orders, inventory, carts, promotions |
| `demo.ipynb` | Step-by-step tutorial notebook walking through the full pipeline |

## The MCP Server

The ShopInsights server provides 15 tools organized into **ambiguity clusters** — groups of tools that look similar but serve different purposes. This makes it challenging for small models and is representative of real-world MCP servers.

| Cluster | Tools | Challenge |
|---|---|---|
| Product Discovery | `search_products`, `browse_catalog`, `get_trending_products`, `get_product_details` | Which search/browse tool for a given query? |
| Sales & Revenue | `get_sales_data`, `get_revenue_report`, `get_store_overview` | Per-product units vs. aggregate revenue vs. snapshot |
| Customer Analytics | `get_customer_segments`, `get_customer_profile`, `get_abandoned_carts` | Individual vs. aggregate vs. behavioral |
| Multi-step | `analyze_product_performance`, `compare_products`, `forecast_demand`, `get_inventory_status`, `create_promotion` | Require chaining 2-4 tools |

## Prerequisites

1. **Python 3.10+** with `uv`
2. **Langflow** running with a frontier model agent (e.g., GPT-5.2) connected to the MCP server
3. **OpenAI API key** (or compatible) for the teacher LLM

## Quick start

### 1. Install dependencies

```bash
uv pip install .[dev]
uv pip install fastmcp
```

### 2. Start the MCP server

```bash
uv run python examples/agentic/mcp_distillation_training/server.py
```

The server starts on port 8008 (configurable via `PORT` env var) using Streamable HTTP transport.

### 3. Connect Langflow

In Langflow, create an agent flow with:
- **Model:** A frontier model (e.g., GPT-5.2)
- **MCP Server:** Point to `http://localhost:8008`

### 4. Run the pipeline

```python
from sdg_hub import Flow

flow = Flow.from_yaml("src/sdg_hub/flows/agentic/mcp_distillation/flow.yaml")

# Teacher model for question synthesis + quality scoring
flow.set_model_config(model="openai/gpt-5.2", api_key="...")

# Frontier model agent in Langflow
flow.set_agent_config(
    agent_framework="langflow",
    agent_url="http://localhost:7860/api/v1/run/<your-flow-id>",
    agent_api_key="...",
)

result = flow.generate(dataset)
```

Or follow the step-by-step [tutorial notebook](demo.ipynb).

## Pipeline stages

The flow (`src/sdg_hub/flows/agentic/mcp_distillation/flow.yaml`) runs 6 stages:

### Stage 1: Expert Exploration

The frontier model connects to your MCP server and **actively calls every tool**. It discovers:
- Real data entities (product IDs, customer names, category hierarchies)
- Tool relationships (which outputs feed into which inputs)
- Ambiguity clusters (which tools overlap and when to use each)
- Edge cases (error conditions, empty results, parameter constraints)

The output is a structured "server understanding" document used to ground all subsequent question generation.

### Stage 2: Diversity

Multiplies each input row (×10 by default) and samples different tool subsets to ensure the training data covers diverse tool combinations.

### Stage 3: Question Synthesis

A teacher LLM generates realistic user questions using:
- The sampled tool subset
- The exploration findings (real entities, known tool relationships)

Questions reference real data discovered during exploration, not hypothetical scenarios.

### Stage 4: Question Quality Filter

Each question is scored on difficulty, realism, and uniqueness. Only "good" or "excellent" questions survive.

### Stage 5: Expert Trajectory Generation

The frontier model **solves each question** by actually calling MCP tools. The full tool call trace is captured: tool names, arguments, structured outputs, and the final synthesized answer.

### Stage 6: Response Quality Filter

Trajectories are scored on completeness. Incomplete responses are filtered out.

## Output format

Each training example is a structured function-calling conversation:

```json
[
  {"role": "system", "content": "<tool declarations for all 15 tools>"},
  {"role": "user", "content": "Find the top trending laptop and check inventory"},
  {"role": "assistant", "content": "", "function_call": {"name": "get_trending_products", "arguments": "{...}"}},
  {"role": "function", "content": "{\"trending\": [...]}", "name": "get_trending_products"},
  {"role": "assistant", "content": "", "function_call": {"name": "get_inventory_status", "arguments": "{...}"}},
  {"role": "function", "content": "{\"inventory\": [...]}", "name": "get_inventory_status"},
  {"role": "assistant", "content": "The AeroBook Gaming X17 is the top trending laptop..."}
]
```

Compatible with standard SFT frameworks (TRL, Axolotl, LLaMA-Factory).

## Scaling up

Increase volume and complexity via runtime parameter overrides:

```python
result = flow.generate(
    dataset,
    runtime_params={
        "multiply_tool_rows": {"num_samples": 50},  # 50 candidates instead of 10
        "sample_tools": {"num_samples": 3},          # 3-tool combos (harder questions)
    },
)
```

For multiple MCP servers, create a multi-row input DataFrame — one row per server. The pipeline explores each independently.

## Adapting for your own MCP server

1. Replace `server.py` and `data.py` with your own MCP server
2. Extract tool schemas into the input DataFrame format (`tool_list`, `mcp_server_name`, `mcp_server_description`)
3. Set up a Langflow agent with your frontier model + your MCP server
4. Run the pipeline — the exploration phase will automatically discover your server's data and tool relationships
