---
name: synthetic-data-generation
description: Generate synthetic data using sdg_hub with composable blocks and YAML flows. Use when the user wants to create training datasets, generate QA pairs, run data generation pipelines, build custom flows, produce synthetic data from documents, use agent frameworks for data generation, or distill MCP tool-use traces. Supports pre-built flows, custom Python scripts, and YAML flow authoring with 20+ blocks, agent connectors (Langflow, LangGraph), MCP tool-use, and 100+ LLM providers via LiteLLM.
---

# Synthetic Data Generation with SDG Hub

Generate synthetic data using composable blocks and flows. Blocks are processing units that transform datasets; flows chain blocks into pipelines defined in YAML.

Core concept: `dataset -> Block_1 -> Block_2 -> Block_3 -> enriched_dataset`

## Choose Your Approach

| Approach | When to Use |
|----------|-------------|
| **Pre-built flow** | Standard pipeline exists for your task (QA generation, text analysis, red-teaming, RAG eval, MCP distillation) |
| **Custom Python** | Quick experiments, ad-hoc generation, custom logic |
| **Custom YAML flow** | Reusable pipeline, team sharing, complex multi-block workflows |
| **Agent-based** | Need external agent frameworks (Langflow, LangGraph) or MCP tool-use in your pipeline |

## Approach A: Pre-Built Flows

### Step 1: Discover flows

```python
# play.py
from sdg_hub import FlowRegistry

# List all flows
for f in FlowRegistry.list_flows():
    print(f"- {f['name']} (tags: {f.get('tags', [])})")

# Search by tag
FlowRegistry.search_flows(tag="qa-generation")
```

Consult `references/pre_built_flows.md` for the full catalog with descriptions and required inputs.

### Step 2: Load and inspect

```python
from sdg_hub import Flow, FlowRegistry

path = FlowRegistry.get_flow_path("Flow Name or ID")
flow = Flow.from_yaml(path)
flow.print_info()

# Check what dataset columns are needed
reqs = flow.get_dataset_requirements()
if reqs:
    print(f"Required columns: {reqs.required_columns}")
```

### Step 3: Configure model

```python
import os

flow.set_model_config(
    model="openai/gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# For local models (vLLM, Ollama)
flow.set_model_config(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY"
)
```

See `references/model_configs.md` for all supported providers (OpenAI, Anthropic, Azure, vLLM, Ollama, Together, Groq, Bedrock, etc.).

### Step 4: Prepare data and dry run

```python
import pandas as pd

df = pd.DataFrame({"document": ["Your text here..."]})

# Validate dataset against flow requirements
errors = flow.validate_dataset(df)
if errors:
    print(f"Fix these: {errors}")

# Dry run with 2 samples -- do this before every full run
dry = flow.dry_run(df, sample_size=2)
print(f"Success: {dry['execution_successful']}")
for block in dry['blocks_executed']:
    print(f"  {block['block_name']}: {block['execution_time_seconds']:.2f}s")
```

### Step 5: Generate and save

```python
# Full run with checkpointing for large datasets
result = flow.generate(
    df,
    checkpoint_dir="./checkpoints",
    save_freq=100,
    max_concurrency=5
)

result.to_parquet("output.parquet")
```

## Approach B: Custom Python Scripts

Use blocks directly for ad-hoc experiments.

### Basic: Single block

```python
# play.py
from sdg_hub.core.blocks import LLMChatBlock
import pandas as pd

block = LLMChatBlock(
    block_name="gen",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o-mini",
    api_key="sk-...",
    temperature=0.7
)

df = pd.DataFrame({
    "messages": [[
        {"role": "system", "content": "You generate QA pairs."},
        {"role": "user", "content": "Generate a fun fact about Python."}
    ]]
})

result = block(df)
print(result["response"].iloc[0])
```

### Chain: Multiple blocks

```python
from sdg_hub.core.blocks import LLMChatBlock, TagParserBlock
import pandas as pd

# Step 1: Generate
llm = LLMChatBlock(
    block_name="gen",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o-mini",
    api_key="sk-..."
)

# Step 2: Parse with tags
parser = TagParserBlock(
    block_name="parse",
    input_cols="response",
    output_cols=["question", "answer"],
    start_tags=["<question>", "<answer>"],
    end_tags=["</question>", "</answer>"]
)

df = pd.DataFrame({
    "messages": [[
        {"role": "user", "content": "Generate a QA pair. Use <question>...</question> and <answer>...</answer> tags."}
    ]]
})

result = parser(llm(df))
print(result[["question", "answer"]])
```

### Batch processing for large datasets

```python
from tqdm import tqdm

def process_in_batches(df, block, batch_size=50):
    results = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size].copy()
        results.append(block(batch))
    return pd.concat(results, ignore_index=True)
```

See `references/block_reference.md` for all 20+ available blocks and their configurations.

## Approach C: Authoring Custom Flow YAMLs

Build incrementally -- start with one block, test, add the next.

### Step 1: Define the data contract

```python
# play.py - Clarify inputs and outputs first
import pandas as pd

input_df = pd.DataFrame({
    "document": ["Climate change is accelerating..."],
    "domain": ["environment"]
})
print("Input columns:", list(input_df.columns))

expected_outputs = ["document", "domain", "question", "response"]
print("Expected output:", expected_outputs)
```

### Step 2: Minimal YAML

```yaml
# flow.yaml
metadata:
  name: "My QA Flow"
  version: "0.1.0"
  author: "Your Name"
  description: "Generate QA pairs from documents"
  dataset_requirements:
    required_columns: ["document"]

blocks:
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_prompt"
      input_cols: ["document"]
      output_cols: "messages"
      prompt_config_path: "prompts/qa.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate"
      input_cols: "messages"
      output_cols: "raw_response"
      temperature: 0.7
      async_mode: true

  - block_type: "TagParserBlock"
    block_config:
      block_name: "parse"
      input_cols: "raw_response"
      output_cols: ["question", "response"]
      start_tags: ["<question>", "<answer>"]
      end_tags: ["</question>", "</answer>"]
```

### Step 3: Create prompt template

```yaml
# prompts/qa.yaml (relative to flow.yaml)
- role: system
  content: |
    You generate question-answer pairs from documents.

- role: user
  content: |
    Generate one question and answer from this document.
    Use <question>...</question> and <answer>...</answer> tags.

    {document}
```

### Step 4: Test incrementally

```python
# play.py
from sdg_hub import Flow
import pandas as pd

flow = Flow.from_yaml("flow.yaml")
flow.set_model_config(model="openai/gpt-4o-mini", api_key="sk-...")

df = pd.DataFrame({"document": ["Python was created by Guido van Rossum in 1991."]})

# Dry run first
dry = flow.dry_run(df, sample_size=1)
print(f"Success: {dry['execution_successful']}")

# Full run
if dry['execution_successful']:
    result = flow.generate(df)
    print(result[["document", "question", "response"]])
```

See `references/yaml_schema.md` for the complete YAML structure and `references/flow_patterns.md` for common patterns (quality filtering, parallel paths, multi-step extraction).

## Approach D: Agent and MCP Pipelines

### Agent frameworks (Langflow, LangGraph)

Use `AgentBlock` to call external agent frameworks as pipeline steps:

```python
from sdg_hub.core.blocks.agent import AgentBlock

block = AgentBlock(
    block_name="my_agent",
    agent_framework="langflow",       # or "langgraph"
    agent_url="http://localhost:7860/api/v1/run/my-flow",
    agent_api_key="your-key",
    input_cols=["question"],
    output_cols=["agent_response"],
    extract_response=True
)

result = block.generate(dataset)
```

In YAML flows, configure agent blocks with `set_agent_config()`:

```python
flow = Flow.from_yaml("flow.yaml")
if flow.is_agent_config_required():
    flow.set_agent_config(
        agent_framework="langgraph",
        agent_url="http://localhost:8123",
        agent_api_key="your-key"
    )
```

### MCP tool-use distillation

`MCPAgentBlock` connects an LLM to a remote MCP server for agentic tool-use. The LLM calls tools in a loop, producing full traces for training data:

```yaml
- block_type: "MCPAgentBlock"
  block_config:
    block_name: "mcp_agent"
    input_cols: "messages"
    output_cols: "agent_trace"
    mcp_server_url: "http://localhost:3000/mcp"
    max_iterations: 10
```

See the pre-built `MCP Server Distillation` flow in `references/pre_built_flows.md` for a complete pipeline.

## Flow Methods Quick Reference

```python
flow = Flow.from_yaml("flow.yaml")

# Model configuration
flow.set_model_config(model="...", api_key="...", blocks=["specific_block"])
flow.is_model_config_required()
flow.get_default_model()
flow.get_model_recommendations()

# Agent configuration
flow.set_agent_config(agent_framework="...", agent_url="...", agent_api_key="...")
flow.is_agent_config_required()

# Dataset validation
flow.validate_dataset(df)
flow.get_dataset_requirements()

# Execution
flow.dry_run(df, sample_size=2)
flow.generate(df, checkpoint_dir="./ckpt", save_freq=100, max_concurrency=5)

# Inspection
flow.print_info()
flow.to_yaml("output_flow.yaml")
```

## Block Discovery

```python
from sdg_hub.core.blocks import BlockRegistry

BlockRegistry.discover_blocks()                    # Rich table of all blocks
BlockRegistry.list_blocks(category="llm")          # By category
BlockRegistry.list_blocks(grouped=True)            # Grouped by category
BlockRegistry.categories()                         # All categories
```

## Data I/O

```python
import pandas as pd

# Load
df = pd.read_csv("input.csv")
df = pd.read_parquet("input.parquet")
df = pd.read_json("input.jsonl", lines=True)

# From HuggingFace
from datasets import load_dataset
df = load_dataset("your_dataset", split="train").to_pandas()

# Save
result.to_parquet("output.parquet")
result.to_csv("output.csv", index=False)
result.to_json("output.jsonl", orient="records", lines=True)

# Push to HuggingFace Hub
from datasets import Dataset
Dataset.from_pandas(result).push_to_hub("username/dataset")
```

## Quality Checklist

Before using generated data:

- [ ] Dry run succeeded with `sample_size=2`?
- [ ] Output columns are correct?
- [ ] Sample outputs look reasonable (spot-check 5-10)?
- [ ] No excessive nulls or empty values?
- [ ] Data saved to durable storage?

## Common Issues

**"Column X not found"** -- Input data is missing a required column. Run `flow.get_dataset_requirements()` to see what the flow expects, then check your DataFrame columns.

**Empty or null outputs** -- The LLM response didn't match the parser pattern. Check the raw LLM output before parsing, and adjust your prompt template or parser config.

**Rate limit errors** -- Reduce `max_concurrency` in `flow.generate()` or add `timeout` and `num_retries` to `set_model_config()`.

**Slow generation** -- Use `async_mode: true` on LLMChatBlock, increase `max_concurrency`, or use checkpointing to resume interrupted runs.

**Model not responding** -- Verify your model config works with a single-sample test:
```python
from sdg_hub.core.blocks import LLMChatBlock
block = LLMChatBlock(block_name="test", input_cols="messages", output_cols="r", model="...", api_key="...")
block(pd.DataFrame({"messages": [[{"role": "user", "content": "hello"}]]}))
```

## Reference Files

Detailed documentation for specific topics:

- `references/block_reference.md` -- All 20+ blocks with YAML configs and usage examples
- `references/pre_built_flows.md` -- Catalog of pre-built flows with inputs, outputs, and usage
- `references/model_configs.md` -- LLM provider configurations (OpenAI, Anthropic, vLLM, Ollama, etc.)
- `references/yaml_schema.md` -- Complete flow YAML structure and validation rules
- `references/flow_patterns.md` -- Common composition patterns (LLM chain, quality filtering, parallel paths, agent integration)
