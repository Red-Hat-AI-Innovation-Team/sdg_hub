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

## Results

7 models evaluated on 111 synthetic tasks across 6 MCP servers. Scores are
**per-server averages** (each server contributes equally) using a 6-dimension
LLM-as-judge with full trace comparison.

| Server | Tasks | GPT-5 | Claude Sonnet 4-6 | GPT-4o | Qwen3.5-27B | GPT-4o-mini | Llama-3.3-70B | Qwen3-32B |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical Calculator | 30 | 0.911 | 0.860 | 0.881 | 0.756 | 0.868 | 0.305 | 0.179 |
| Weather Data | 17 | 0.756 | 0.770 | 0.732 | 0.696 | 0.753 | 0.670 | 0.134 |
| DEX Paprika | 23 | 0.740 | 0.712 | 0.675 | 0.608 | 0.631 | 0.489 | 0.117 |
| Reddit | 9 | 0.710 | 0.703 | 0.721 | 0.594 | 0.692 | 0.486 | 0.133 |
| Wikipedia | 28 | 0.624 | 0.637 | 0.663 | 0.603 | 0.608 | 0.492 | 0.130 |
| Car Price Evaluator | 4 | 0.582 | 0.580 | 0.507 | 0.813 | 0.479 | 0.530 | 0.120 |
| **OVERALL** | **111** | **0.720** | **0.710** | **0.696** | **0.678** | **0.672** | **0.495** | **0.136** |

**Ranking: GPT-5 > Claude Sonnet 4-6 > GPT-4o > Qwen3.5-27B > GPT-4o-mini > Llama-3.3-70B > Qwen3-32B**

### Validation against mcp-bench

We validated by running all 7 models through [mcp-bench's](https://github.com/Accenture/mcp-bench)
own evaluation pipeline and comparing rankings:

```
Our synthetic benchmark: GPT-5 > Claude Sonnet 4-6 > GPT-4o > Qwen3.5-27B > GPT-4o-mini > Llama-3.3-70B > Qwen3-32B
mcp-bench evaluation:    GPT-5 > Claude Sonnet 4-6 > GPT-4o > Qwen3.5-27B > GPT-4o-mini > Llama-3.3-70B > Qwen3-32B

Kendall's tau:   1.000  (p=0.0004)
Spearman's rho:  1.000  (p=0.0000)
```

Perfect rank agreement across all 7 models (4 proprietary + 3 open-source),
statistically significant at p=0.0004.

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
