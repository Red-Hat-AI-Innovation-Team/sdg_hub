# MCP Evaluation Benchmark — Multi-Server Example

Generate synthetic evaluation benchmarks for MCP tool-use skills across multiple servers using the sdg_hub distillation flow.

## What this example does

Given any MCP server, the pipeline automatically generates evaluation tasks that test an AI agent's ability to select the right tools, extract correct parameters, chain tool calls, and synthesize results. The generated tasks come with expert-quality gold-standard trajectories for comparison.

```
  MCP Server ──▶ Explore ──▶ Generate Questions ──▶ Expert Solves ──▶ Evaluation Dataset
                  │              │                      │                    │
            Frontier model   Teacher LLM          Frontier model      question +
            calls all tools  creates questions     solves via MCP     target_tools +
            maps behavior    grounded in real      producing gold     expert_trajectory +
                             server data           trajectories       quality ratings
```

## Servers

This example uses 6 data-dependent MCP servers where models genuinely need the tools (can't answer from internal knowledge alone):

| Server | Tools | Data Type | Why tools are needed |
|--------|-------|-----------|---------------------|
| Weather Data | 4 | Live API | Dynamic weather — changes every hour |
| Medical Calculator | 22 | Specialized formulas | Clinical calculations (eGFR, MELD, CHA2DS2-VASc) |
| Wikipedia | 9 | Live API | Article content too vast to memorize |
| Car Price Evaluator | 3 | Database | Brazilian vehicle market pricing |
| Reddit | 2 | Live API | Live posts/comments — dynamic content |
| DEX Paprika | 11 | Live API | DeFi/crypto market data — changes every second |

The servers come from [Accenture's mcp-bench](https://github.com/Accenture/mcp-bench) repository. We selected servers where models genuinely need tool access — servers like Math MCP and Unit Converter were excluded because strong models (e.g., GPT-5) can solve those tasks internally without calling any tools.

## Files

| File | Description |
|---|---|
| `demo.ipynb` | End-to-end tutorial: setup, generate tasks, evaluate models, see results |
| `start_servers.sh` | Start/stop/check all 6 MCP servers via supergateway |
| `.env.example` | Template for API keys and Langflow agent URLs |
| `outputs/` | Pre-generated evaluation tasks (112 tasks across 6 servers) |
| `evaluation_results.jsonl` | Pre-computed evaluation scores (3 models × 112 tasks = 336 rows) |

## Quick start

### 1. Clone mcp-bench (provides the MCP servers)

All commands below assume you are in the example directory:

```bash
cd examples/agentic/mcp_evaluation
```

Clone mcp-bench as a sibling directory:

```bash
git clone https://github.com/Accenture/mcp-bench.git ../mcp-bench
```

### 2. Start MCP servers

```bash
bash start_servers.sh          # install deps + start all 6 servers
bash start_servers.sh --check  # verify they're running
```

`start_servers.sh` uses paths relative to its own location, so it works from any directory.

### 3. Set up Langflow agents

Start Langflow (`uvx langflow run`) and create one agent flow per server:

1. Create a flow with an **Agent** + **MCP Tools** component
2. Point MCP Tools at the server URL (e.g., `http://localhost:8001/mcp`)
3. Set Agent max iterations to **100**
4. Configure the Agent's LLM (e.g., GPT-5.2 via OpenAI key)
5. Note the flow URL and add it to `.env`

Repeat for each server. The exploration step may occasionally fail if the frontier model
probes edge cases — simply re-run the cell if a server fails. Pre-generated tasks are
provided in `outputs/` so you can skip task generation entirely if needed.

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and Langflow flow URLs
```

### 5. Run the notebook

Open `demo.ipynb` and follow the steps.

## Results

With 3 models (GPT-5, GPT-4o, GPT-4o-mini) evaluated on 112 synthetic tasks:

| Server | Tasks | GPT-5 | GPT-4o | GPT-4o-mini |
|:---|---:|---:|---:|---:|
| Medical Calculator | 38 | 0.974 | 0.944 | 0.927 |
| Weather Data | 16 | 0.815 | 0.811 | 0.782 |
| DEX Paprika | 20 | 0.760 | 0.640 | 0.612 |
| Wikipedia | 27 | 0.705 | 0.728 | 0.694 |
| Reddit | 6 | 0.686 | 0.646 | 0.678 |
| Car Price Evaluator | 5 | 0.556 | 0.388 | 0.453 |
| **OVERALL** | **112** | **0.814** | **0.778** | **0.759** |

**Ranking: GPT-5 > GPT-4o > GPT-4o-mini**

## Validation against mcp-bench

[MCP-Bench](https://github.com/Accenture/mcp-bench) is a comprehensive evaluation framework
by Accenture that assesses LLMs' tool-use capabilities through the Model Context Protocol. It
provides an end-to-end pipeline for evaluating how effectively different LLMs can discover,
select, and utilize tools to solve real-world tasks across 28 MCP servers.

We validated our synthetic benchmark by running mcp-bench's own evaluator (TaskEvaluator with
6-subdimension LLM judge) on the same 6 servers with the same 3 models, using mcp-bench's
pre-existing benchmark tasks (2 per server, 12 total):

| Server | Tasks | GPT-5 | GPT-4o | GPT-4o-mini |
|:---|---:|---:|---:|---:|
| Medical Calculator | 2 | 0.608 | 0.475 | 0.400 |
| Weather Data | 2 | 0.550 | 0.458 | 0.392 |
| DEX Paprika | 2 | 0.517 | 0.367 | 0.383 |
| Wikipedia | 2 | 0.475 | 0.400 | 0.367 |
| Reddit | 2 | 0.458 | 0.408 | 0.400 |
| Car Price Evaluator | 2 | 0.267 | 0.192 | 0.200 |
| **OVERALL** | **12** | **0.479** | **0.383** | **0.357** |

**mcp-bench ranking: GPT-5 > GPT-4o > GPT-4o-mini**

### Rank comparison

Both our synthetic benchmark and mcp-bench produce the same model ordering:

```
Kendall's tau:   1.000
Spearman's rho:  1.000
Pairwise agree:  3/3
Rank match:      YES
```

The absolute scores differ (our flow: 0.76-0.81, mcp-bench: 0.36-0.48) because the
evaluation methods are different, but the **ranking signal is preserved** — which validates
that our distillation flow generates evaluation-quality synthetic data.

## How evaluation works

Each model is scored against the expert gold-standard trajectory on 6 metrics:

| Metric | Type | What it measures |
|--------|------|-----------------|
| Tool recall | Programmatic | Did the model call the same tools as the expert? |
| Tool precision | Programmatic | Were all the model's tool calls expected? |
| Order match | Programmatic | Were tools called in the right sequence? |
| Task completion | LLM judge (0-10) | Did the model achieve the task objectives? |
| Tool usage | LLM judge (0-10) | Were tools used with correct parameters? |
| Answer quality | LLM judge (0-10) | Is the final answer as complete as the expert's? |

The overall score averages all 6 metrics (normalized to 0-1).

## Evaluating local models

To evaluate a locally served model (e.g., via vLLM or TGI), add it to `MODEL_CONFIGS` in the notebook:

```python
MODEL_CONFIGS = {
    "openai/gpt-4o": {},
    # vLLM-hosted model
    "hosted_vllm/my-finetuned-model": {
        "api_base": "http://localhost:8000/v1",
        "api_key": "dummy",
    },
    # Any OpenAI-compatible endpoint (TGI, Ollama, etc.)
    "openai/my-local-model": {
        "api_base": "http://localhost:11434/v1",
        "api_key": "dummy",
    },
}
```

The evaluation loop uses `MCPAgentBlock` which passes the model name and `api_base` to LiteLLM. Use the `hosted_vllm/` prefix for vLLM endpoints, or `openai/` for any other OpenAI-compatible API.
