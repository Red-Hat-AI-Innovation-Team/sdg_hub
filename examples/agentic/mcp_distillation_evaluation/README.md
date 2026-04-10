# MCP Evaluation Benchmark — Multi-Server Example

Generate synthetic evaluation benchmarks for MCP tool-use skills across multiple servers using the sdg_hub distillation flow, then evaluate models with a 6-dimension LLM-as-judge aligned with [mcp-bench](https://github.com/Accenture/mcp-bench).

## What this example does

Given any MCP server, the pipeline automatically generates evaluation tasks that test an AI agent's ability to select the right tools, extract correct parameters, chain tool calls, and synthesize results. The generated tasks come with expert-quality gold-standard trajectories for comparison.

```
  MCP Server ──▶ Explore ──▶ Generate Questions ──▶ Expert Solves ──▶ Evaluation Dataset
                  │              │                      │                    │
            Frontier model   Teacher LLM          Frontier model      question +
            calls all tools  creates questions     solves via MCP     expert_tools +
            maps behavior    grounded in real      producing gold     expert_trajectory +
                             server data           trajectories       quality ratings
```

## Servers

This example uses 6 data-dependent MCP servers where models genuinely need the tools (can't answer from internal knowledge alone):

| Server | Tools | Data Type | Why tools are needed |
|--------|-------|-----------|---------------------|
| Weather Data | 4 | Live API | Dynamic weather — changes every hour |
| Medical Calculator | 22 | Specialized formulas | Clinical calculations (eGFR, MELD, CHA₂DS₂-VASc) |
| Wikipedia | 9 | Live API | Article content too vast to memorize |
| Car Price Evaluator | 3 | Database | Brazilian vehicle market pricing |
| Reddit | 2 | Live API | Live posts/comments — dynamic content |
| DEX Paprika | 11 | Live API | DeFi/crypto market data — changes every second |

The servers come from [Accenture's mcp-bench](https://github.com/Accenture/mcp-bench) repository. We selected servers where models genuinely need tool access — servers like Math MCP and Unit Converter were excluded because strong models (e.g., GPT-5) can solve those tasks internally without calling any tools.

## Files

| File | Description |
|---|---|
| `demo.ipynb` | End-to-end tutorial: setup, generate tasks, evaluate models, see results |
| `eval_utils.py` | Evaluation utilities: trace extraction, formatting, programmatic metrics |
| `start_servers.sh` | Start/stop/check all 6 MCP servers (native FastMCP for Python, supergateway for Node.js) |
| `start_agents.sh` | Start/stop/check LangGraph agents for task generation |
| `.env.example` | Template for API keys and LangGraph agent URLs |
| `benchmark_tasks.jsonl` | Pre-generated evaluation tasks (111 tasks across 6 servers) |
| `evaluation_results.jsonl` | Pre-computed evaluation scores (5 models × 111 tasks = 555 rows) |

## Quick start

### 1. Clone mcp-bench (provides the MCP servers)

All commands below assume you are in the example directory:

```bash
cd examples/agentic/mcp_distillation_evaluation
```

Clone mcp-bench into the examples directory:

```bash
git clone https://github.com/Accenture/mcp-bench.git ../../mcp-bench
```

### 2. Start MCP servers

```bash
bash start_servers.sh          # install deps + start all 6 servers
bash start_servers.sh --check  # verify they're running
```

The 5 Python servers run natively via FastMCP (no supergateway needed). DEX Paprika (Node.js) uses supergateway as a fallback.

### 3. Set up LangGraph agents (for task generation only)

Each MCP server needs a LangGraph agent connected to it for the task generation step (Section 2 of the notebook). This is NOT needed for evaluation — evaluation uses MCPAgentBlock directly.

```bash
bash start_agents.sh           # start all 6 agents on ports 2024-2029
bash start_agents.sh --check   # verify they're running
```

The agents are thin ReAct wrappers that connect to MCP servers via `langchain_mcp_adapters`. The `start_agents.sh` script handles configuration and deployment.

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Run the notebook

Open `demo.ipynb` and follow the steps. Pre-generated tasks and evaluation results are provided — you can skip directly to Section 5 (Results) to see the output.

## Results

7 models evaluated on 111 synthetic tasks across 6 servers using a 6-dimension
LLM-as-judge with full trace comparison (tool names, arguments, outputs).
Scores are **per-server averages** — each server contributes equally.

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

Notable findings:
- **Qwen3.5-27B** (open-source, 27B) scores between GPT-4o and GPT-4o-mini — competitive with proprietary models
- **Llama-3.3-70B** (open-source, 70B) scores 0.495 despite being 2.5x larger than Qwen3.5 — tool-calling quality varies significantly across model families
- **Qwen3-32B** scores 0.136 due to missing `--tool-call-parser` in sglang — the model generates tool calls as text but sglang doesn't parse them into structured function calls

## Validation against mcp-bench

[MCP-Bench](https://github.com/Accenture/mcp-bench) is a comprehensive evaluation framework
by Accenture that assesses LLMs' tool-use capabilities through the Model Context Protocol.

We validated our synthetic benchmark by running all 7 models through mcp-bench's own
evaluation pipeline (using their `TaskEvaluator` judge on their tasks) and comparing
the resulting rankings:

```
Our synthetic benchmark: GPT-5 > Claude Sonnet 4-6 > GPT-4o > Qwen3.5-27B > GPT-4o-mini > Llama-3.3-70B > Qwen3-32B
mcp-bench evaluation:    GPT-5 > Claude Sonnet 4-6 > GPT-4o > Qwen3.5-27B > GPT-4o-mini > Llama-3.3-70B > Qwen3-32B

Kendall's tau:   1.000  (p=0.0004)
Spearman's rho:  1.000  (p=0.0000)
```

Perfect rank agreement across all 7 models, statistically significant at p=0.0004.
The synthetic benchmark produces identical model rankings to mcp-bench across
4 proprietary API models and 3 open-source locally-hosted models.

## How evaluation works

The evaluation uses two complementary scoring approaches:

### Programmatic metrics (computed from traces)

| Metric | What it measures |
|--------|-----------------|
| Tool recall | Did the model call the same tools as the expert? |
| Tool precision | Were all the model's tool calls relevant? |
| Order match | Were tools called in the right sequence? (LCS-based) |
| Parameter match | Did the model use correct argument keys and values? |

### LLM-as-judge dimensions (1-10 scale, aligned with mcp-bench)

| Dimension | Aggregate Group | What it measures |
|-----------|----------------|-----------------|
| Task fulfillment | Task Completion | % of task requirements correctly completed |
| Grounding | Task Completion | % of claims grounded in actual tool outputs |
| Tool appropriateness | Tool Selection | Were the right tools selected for each subtask? |
| Parameter accuracy | Tool Selection | Were tool parameters correct and complete? |
| Dependency awareness | Planning | Were cross-tool dependencies handled correctly? |
| Parallelism & efficiency | Planning | Were redundant calls avoided? Were independent calls parallelized? |

The judge receives **full execution traces** (tool names, arguments, and outputs) for
both the expert and the model, enabling fine-grained comparison of parameter choices
and execution strategy — not just tool names.

The overall score per server averages all 10 metrics (judge scores normalized to 0-1).
The aggregate overall is the mean of per-server scores — each server contributes
equally regardless of task count.

## Evaluating local models

To evaluate a locally served model (e.g., via vLLM or sglang), add it to `MODEL_CONFIGS` in the notebook:

```python
MODEL_CONFIGS = {
    "openai/gpt-4o": {},
    # sglang-hosted model (must have --tool-call-parser enabled)
    "hosted_vllm/Qwen3.5-27B": {
        "api_base": "http://localhost:30000/v1",
        "api_key": "dummy",
    },
    # Vertex AI (Application Default Credentials)
    "vertex_ai/claude-sonnet-4-6": {"api_key": None},
}
```

The evaluation uses `MCPAgentBlock` which passes the model name and `api_base` to LiteLLM. Use the `hosted_vllm/` prefix for vLLM/sglang endpoints.

**Important**: When serving models via sglang for tool-use evaluation, you must enable the tool-call parser. Without it, tool calls are returned as raw text instead of structured function calls. Example for Qwen3.5:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-27B \
    --tp-size 4 --port 30000 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
```

Refer to each model's HuggingFace model card for the correct `--tool-call-parser` value.

## Dataset schema (benchmark_tasks.jsonl)

| Column | Type | Description |
|--------|------|-------------|
| `server` | str | MCP server name |
| `question` | str | The evaluation prompt |
| `expert_answer` | str | Gold-standard answer text |
| `expert_tools` | list[str] | Ordered list of tool names the expert called |
| `expert_tool_trace` | list[dict] | Full tool trace: name, input args, output per call |
| `question_quality_rating` | str | "good" or "excellent" (from distillation quality filter) |
| `completeness_rating` | str | "mostly complete" or "fully complete" |
