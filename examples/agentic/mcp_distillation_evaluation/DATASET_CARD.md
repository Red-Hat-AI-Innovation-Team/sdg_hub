---
license: apache-2.0
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- mcp
- tool-use
- function-calling
- benchmark
- evaluation
- synthetic-data
- agentic
size_categories:
- n<1K
---

# MCP Tool-Use Evaluation Benchmark

A synthetic evaluation benchmark for assessing LLM tool-use capabilities against MCP (Model Context Protocol) servers. Generated using [sdg_hub](https://github.com/ibm/sdg_hub)'s MCP distillation flow.

## Dataset Description

111 evaluation tasks across 6 data-dependent MCP servers, each with expert-quality gold-standard tool traces produced by a frontier model (GPT-5.2). Tasks are designed to test tool selection, parameter accuracy, dependency chaining, and result synthesis — skills that models cannot fake with internal knowledge alone.

### Servers

| Server | Tasks | Tools | Data Type |
|--------|-------|-------|-----------|
| Medical Calculator | 30 | 22 | Clinical formulas (eGFR, MELD, CHA₂DS₂-VASc) |
| Wikipedia | 28 | 9 | Live article content and search |
| DEX Paprika | 23 | 11 | DeFi/crypto market data |
| Weather Data | 17 | 4 | Live weather API |
| Reddit | 9 | 2 | Live posts and comments |
| Car Price Evaluator | 4 | 3 | Brazilian vehicle pricing |

### Generation Method

Tasks were generated using the **MCP Server Distillation** flow from sdg_hub:

1. **Exploration**: A frontier model (GPT-5.2) actively explores each MCP server, calling tools and mapping their behavior
2. **Question generation**: Grounded in real server data and tool relationships, the model generates multi-tool questions at varying complexity levels (2, 4, 8 tools)
3. **Quality filtering**: An LLM judge scores question quality and filters low-quality items
4. **Expert trajectory**: The frontier model solves each question via the MCP server, producing gold-standard tool traces
5. **Response filtering**: Only tasks with "mostly complete" or "fully complete" expert solutions are kept

### Validation

Model rankings produced by this benchmark match [Accenture's mcp-bench](https://github.com/Accenture/mcp-bench) with perfect rank correlation:

```
Our benchmark:     GPT-5 > Claude Sonnet 4-6 > GPT-4o > GPT-4o-mini > Qwen3-32B
mcp-bench ranking: GPT-5 > Claude Sonnet 4-6 > GPT-4o > GPT-4o-mini > Qwen3-32B

Kendall's tau:  1.000 (p=0.017)
Spearman's rho: 1.000 (p=0.000)
```

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `server` | string | MCP server name |
| `question` | string | The evaluation prompt |
| `expert_answer` | string | Gold-standard answer text |
| `expert_tools` | list[string] | Ordered list of tool names the expert called |
| `expert_tool_trace` | list[object] | Full tool trace with `name`, `input` (args dict), `output` |
| `question_quality_rating` | string | "good" or "excellent" |
| `completeness_rating` | string | "mostly complete" or "fully complete" |

## Usage

### With sdg_hub

```python
from datasets import load_dataset

ds = load_dataset("your-org/mcp-eval-benchmark")

# Evaluate a model using MCPAgentBlock
from sdg_hub.core.blocks import MCPAgentBlock

block = MCPAgentBlock(
    block_name="eval",
    mcp_server_url="http://localhost:8001/mcp",
    model="openai/gpt-4o",
    input_cols=["question"],
    output_cols=["model_trace"],
)
results = block.generate(ds["train"])
```

### With the evaluation notebook

See the [example notebook](https://github.com/ibm/sdg_hub/tree/main/examples/agentic/mcp_distillation_evaluation) for the full evaluation pipeline including LLM-as-judge scoring.

## MCP Server Setup

The MCP servers come from [Accenture's mcp-bench](https://github.com/Accenture/mcp-bench). To run evaluation, you need the servers running locally. See the example README for setup instructions.

## Citation

If you use this benchmark, please cite:

```bibtex
@software{sdg_hub_mcp_eval,
  title={MCP Tool-Use Evaluation Benchmark},
  author={SDG Hub Contributors},
  year={2026},
  url={https://github.com/ibm/sdg_hub},
  note={Generated using sdg_hub MCP distillation flow}
}
```

## License

Apache 2.0
