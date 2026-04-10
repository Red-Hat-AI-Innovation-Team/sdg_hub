# MCP Distillation for Agent Evaluation

Generate synthetic evaluation benchmarks and evaluate your agent's tool-use
performance using sdg_hub — validated against [Accenture's mcp-bench](https://github.com/Accenture/mcp-bench).

## sdg_hub features used

This example showcases two core sdg_hub capabilities:

1. **[MCP Server Distillation Flow](../../../src/sdg_hub/flows/agentic/mcp_distillation/)** —
   A 23-block pipeline that generates high-quality tool-use evaluation tasks through
   expert distillation. A frontier model explores your agent's MCP tools, generates
   grounded questions, and produces gold-standard trajectories.

2. **[LangGraph Connector](../../../src/sdg_hub/core/connectors/agent/langgraph.py)** —
   Connects sdg_hub's `AgentBlock` to any LangGraph-deployed agent. Supports runtime
   model swapping via `run_config.configurable`, enabling the same agent to be used
   for both data generation (with a frontier model) and evaluation (with target models).

## Overview

```
You have an agent (LangGraph + MCP servers)
  → Plug in a frontier model → sdg_hub generates evaluation data
  → Swap the model → evaluate through the same agent
  → Compare rankings across models
```

The key insight: **the same agent harness** is used for both generation and evaluation.
Only the underlying LLM changes. This means you're evaluating your full agent stack
(tools, guardrails, orchestration), not just the bare model.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `generate.ipynb` | **Synthetic task generation** — uses the distillation flow with a frontier model through your agent to produce `benchmark_tasks.jsonl` |
| `evaluate.ipynb` | **Agent evaluation** — runs target models through the same agent, scores with a 6-dimension LLM-as-judge, produces rankings |

## Files

| File | Description |
|------|-------------|
| `eval_utils.py` | Shared utilities: trace normalization, formatting, programmatic metrics |
| `start_servers.sh` | Start/stop/check MCP servers (native FastMCP for Python servers) |
| `start_agents.sh` | Start/stop/check LangGraph agents with configurable model support |
| `.env.example` | Template for API keys and agent URLs |

## Quick start

```bash
cd examples/agentic/mcp_distillation_evaluation

# 1. Start MCP servers
git clone https://github.com/Accenture/mcp-bench.git ../../mcp-bench
bash start_servers.sh

# 2. Start LangGraph agents (one per server, model swappable at runtime)
bash start_agents.sh

# 3. Configure
cp .env.example .env  # add your OPENAI_API_KEY

# 4. Generate evaluation tasks (generate.ipynb)
# 5. Evaluate models (evaluate.ipynb)
```

## How evaluation works

### Programmatic metrics (computed from traces)

| Metric | What it measures |
|--------|-----------------|
| Tool recall | Did the model call the same tools as the expert? |
| Tool precision | Were all the model's tool calls relevant? |
| Order match | Were tools called in the right sequence? (LCS-based) |
| Parameter match | Did the model use correct argument keys and values? |

### LLM-as-judge dimensions (1-10 scale, aligned with mcp-bench)

| Dimension | Group | What it measures |
|-----------|-------|-----------------|
| Task fulfillment | Task Completion | % of requirements correctly completed |
| Grounding | Task Completion | % of claims grounded in actual tool outputs |
| Tool appropriateness | Tool Selection | Were the right tools selected? |
| Parameter accuracy | Tool Selection | Were parameters correct and complete? |
| Dependency awareness | Planning | Were cross-tool dependencies handled? |
| Parallelism & efficiency | Planning | Were redundant calls avoided? |

The judge receives **full execution traces** (tool names, arguments, and outputs)
for both the expert and the model, enabling fine-grained comparison.

## Evaluating local models

To evaluate a locally served model (e.g., via sglang), add it to `MODEL_CONFIGS`
in `evaluate.ipynb`:

```python
MODEL_CONFIGS = {
    "gpt-4o": {},
    "Qwen3.5-27B": {"api_base": "http://localhost:30000/v1", "api_key": "dummy"},
}
```

The model name and connection details are passed to the LangGraph agent at runtime
via `configurable` — no agent redeployment needed.

**Important**: When serving models via sglang for tool-use, enable the tool-call
parser. Example for Qwen3.5:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-27B \
    --tp-size 4 --port 30000 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```
