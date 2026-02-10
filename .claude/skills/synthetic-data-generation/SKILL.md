---
name: Synthetic Data Generation
description: Generate synthetic data, run a flow, create training data, produce datasets, or author custom flow YAMLs using sdg_hub
---

# Synthetic Data Generation Workflow

Generate synthetic data using SDG Hub - either with pre-built flows (YAML) or custom Python scripts.

## Workflow Philosophy

1. **Make it work** → **Make it right** → **Make it fast**
2. **Test with `play.py`** - verify before full runs
3. **Dry run first** - always test with 2 samples
4. **Checkpoint large runs** - resume if interrupted
5. **Validate output** - check quality before using

## Two Approaches

| Approach | When to Use |
|----------|-------------|
| **YAML Flows** | Standard pipelines, reusable workflows, team sharing |
| **Python Scripts** | Ad-hoc experiments, custom logic, quick tests |

## Quick Start: YAML Flow

```python
# play.py - Quick test
from sdg_hub import Flow
import pandas as pd

# Load
flow = Flow.from_yaml("path/to/flow.yaml")

# Configure model
flow.set_model_config(
    model="openai/gpt-4o-mini",
    api_key="sk-..."
)

# Test data
df = pd.DataFrame({"text": ["Test input"]})

# Dry run (always first!)
dry = flow.dry_run(df, sample_size=1)
print(f"Success: {dry['execution_successful']}")

# Generate
if dry['execution_successful']:
    result = flow.generate(df)
    print(result)
```

## Quick Start: Python Script

```python
# play.py - Direct block usage
from sdg_hub.core.blocks import LLMChatBlock
import pandas as pd

block = LLMChatBlock(
    block_name="gen",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o-mini",
    api_key="sk-..."
)

df = pd.DataFrame({
    "messages": [[
        {"role": "user", "content": "Generate a fun fact."}
    ]]
})

result = block(df)
print(result["response"].iloc[0])
```

## Step-by-Step: Using Pre-Built Flows

### Step 1: Discover Available Flows

```python
# play.py
from sdg_hub import FlowRegistry

# List all flows
flows = FlowRegistry.list_flows()
for f in flows:
    print(f"- {f['name']}")
    print(f"  Tags: {f.get('tags', [])}")

# Search by tag
qa_flows = FlowRegistry.search_flows(tag="qa-generation")
```

### Step 2: Explore a Flow

```python
# play.py
from sdg_hub import Flow, FlowRegistry

# Get path
path = FlowRegistry.get_flow_path("Structured Text Insights Extraction Flow")
print(f"Path: {path}")

# Load and inspect
flow = Flow.from_yaml(path)
flow.print_info()

# Check requirements
reqs = flow.get_dataset_requirements()
if reqs:
    print(f"Required columns: {reqs.required_columns}")

# Check if model config needed
if flow.is_model_config_required():
    print(f"Model required. Default: {flow.get_default_model()}")
```

### Step 3: Prepare Input Data

```python
# play.py
import pandas as pd

# Option A: Create manually
df = pd.DataFrame({
    "text": [
        "Climate change is accelerating...",
        "AI is transforming healthcare...",
    ]
})

# Option B: Load from file
df = pd.read_csv("input.csv")
df = pd.read_parquet("input.parquet")
df = pd.read_json("input.jsonl", lines=True)

# Option C: From HuggingFace
from datasets import load_dataset
hf = load_dataset("your_dataset", split="train")
df = hf.to_pandas()

# Validate against flow
errors = flow.validate_dataset(df)
if errors:
    print(f"Validation errors: {errors}")
else:
    print("✓ Dataset valid")
```

### Step 4: Configure Model

```python
# play.py
flow.set_model_config(
    model="openai/gpt-4o-mini",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=1024
)

# Or for local models
flow.set_model_config(
    model="meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY"
)
```

### Step 5: Dry Run

**Always do this before full generation!**

```python
# play.py
dry = flow.dry_run(df, sample_size=2)

print(f"Success: {dry['execution_successful']}")
print(f"Time: {dry['total_execution_time_seconds']:.2f}s")

for block in dry['blocks_executed']:
    print(f"  {block['block_name']}:")
    print(f"    Time: {block['execution_time_seconds']:.2f}s")
    print(f"    Rows: {block['input_rows']} -> {block['output_rows']}")

# Estimate full run
if dry['execution_successful']:
    per_sample = dry['total_execution_time_seconds'] / 2
    estimated = per_sample * len(df)
    print(f"\nEstimated full run: {estimated/60:.1f} minutes")
```

### Step 6: Generate

```python
# play.py - Small test first
result = flow.generate(df.head(10))
print(result.columns)
print(result.head())

# Full run with checkpointing
result = flow.generate(
    df,
    checkpoint_dir="./checkpoints",
    save_freq=100,              # Save every 100 samples
    log_dir="./logs",
    max_concurrency=5           # Rate limit
)
```

### Step 7: Save Output

```python
# play.py
# Parquet (recommended - preserves types)
result.to_parquet("output.parquet")

# CSV
result.to_csv("output.csv", index=False)

# JSON Lines
result.to_json("output.jsonl", orient="records", lines=True)

# HuggingFace Hub
from datasets import Dataset
Dataset.from_pandas(result).push_to_hub("username/dataset")
```

## Step-by-Step: Custom Python Script

### Step 1: Define I/O

```python
# play.py
import pandas as pd

# What do I have?
input_df = pd.DataFrame({
    "document": ["Doc 1 content", "Doc 2 content"]
})

# What do I want?
# -> question, answer columns for each document
```

### Step 2: Test Block Standalone

```python
# play.py
from sdg_hub.core.blocks import LLMChatBlock
import pandas as pd

block = LLMChatBlock(
    block_name="test",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o-mini",
    api_key="sk-..."
)

# Single test
df = pd.DataFrame({
    "messages": [[
        {"role": "system", "content": "You generate QA pairs."},
        {"role": "user", "content": "Generate a question about Python."}
    ]]
})

result = block(df)
print(result["response"].iloc[0])
```

### Step 3: Build Pipeline

```python
# play.py
from sdg_hub.core.blocks import LLMChatBlock, TextParserBlock
import pandas as pd

def generate_qa(documents: list[str], model: str, api_key: str) -> pd.DataFrame:
    """Generate QA pairs from documents."""

    # Prepare messages
    df = pd.DataFrame({
        "document": documents,
        "messages": [
            [
                {"role": "system", "content": "Generate Q&A pairs."},
                {"role": "user", "content": f"Document: {doc}\n\nGenerate Question: and Answer:"}
            ]
            for doc in documents
        ]
    })

    # Generate
    llm = LLMChatBlock(
        block_name="gen",
        input_cols="messages",
        output_cols="response",
        model=model,
        api_key=api_key,
        temperature=0.7
    )
    df = llm(df)

    # Parse
    parser = TextParserBlock(
        block_name="parse",
        input_cols="response",
        output_cols=["question", "answer"],
        pattern=r"Question:\s*(.+?)\s*Answer:\s*(.+)",
        flags="DOTALL"
    )
    df = parser(df)

    return df[["document", "question", "answer"]]

# Test
result = generate_qa(
    ["Python is a programming language."],
    model="openai/gpt-4o-mini",
    api_key="sk-..."
)
print(result)
```

### Step 4: Add Batching

```python
# play.py
from tqdm import tqdm
import pandas as pd

def generate_in_batches(
    df: pd.DataFrame,
    batch_size: int = 50
) -> pd.DataFrame:
    """Generate with progress bar."""
    results = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        result = process_batch(batch)  # Your processing function
        results.append(result)

    return pd.concat(results, ignore_index=True)
```

### Step 5: Add Error Handling (if needed)

```python
# play.py - Only if explicitly requested
import time

def generate_with_retry(df, max_retries=3):
    for attempt in range(max_retries):
        result = block(df)
        if not result["response"].isna().any():
            return result
        time.sleep(2 ** attempt)
    return result  # Return partial on failure
```

## Quality Checklist

Before using generated data:

- [ ] Dry run succeeded?
- [ ] Output columns correct?
- [ ] Sample outputs look reasonable?
- [ ] No excessive nulls/empty values?
- [ ] Data saved to durable storage?

## Reference Files

Detailed documentation in `references/`:
- `pre_built_flows.md` - Available flows and their purposes
- `custom_scripts.md` - Python script patterns and examples
- `model_configs.md` - Model provider configurations
- `yaml_schema.md` - Complete flow YAML structure reference
- `flow_patterns.md` - Common flow patterns (LLM chain, quality filtering, parallel paths, etc.)
- `block_reference.md` - Available blocks and their YAML configurations

## Authoring Custom Flows

When no pre-built flow fits your use case, author a custom flow YAML. Use an **incremental, test-driven** approach.

### Step 1: Define the Data Contract

Before any YAML, clarify I/O in `play.py`:

```python
# play.py - Define data contract
import pandas as pd

# INPUT: What dataset comes in?
input_df = pd.DataFrame({
    "document": [
        "Climate change is accelerating...",
        "AI is transforming healthcare..."
    ],
    "domain": ["environment", "technology"]
})
print("Input columns:", list(input_df.columns))

# OUTPUT: What should come out?
expected_output_columns = [
    "document",   # Original
    "domain",     # Original
    "question",   # Generated
    "response",   # Generated
]
print("Expected output columns:", expected_output_columns)
```

### Step 2: Start with Minimal YAML

```yaml
# flow.yaml
metadata:
  name: "My QA Generation Flow"
  version: "0.1.0"
  author: "Your Name"
  description: "Generate QA pairs from documents"
  dataset_requirements:
    required_columns:
      - "document"

blocks:
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_qa_prompt"
      input_cols: ["document"]
      output_cols: "qa_prompt"
      prompt_config_path: "qa_prompt.yaml"
```

### Step 3: Create Prompt Template

```yaml
# qa_prompt.yaml (same directory as flow.yaml)
- role: system
  content: |
    You generate question-answer pairs from documents.

- role: user
  content: |
    Generate one question and answer from this document:

    {document}

    Format:
    Question: <your question>
    Answer: <your answer>
```

### Step 4: Test the First Block

```python
# play.py
from sdg_hub import Flow
import pandas as pd

flow = Flow.from_yaml("flow.yaml")
df = pd.DataFrame({"document": ["Python was created by Guido van Rossum in 1991."]})

result = flow.dry_run(df, sample_size=1)
print(f"Success: {result['execution_successful']}")
```

### Step 5: Add LLM + Parsing Blocks

```yaml
# flow.yaml - Add generation and parsing
blocks:
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_qa_prompt"
      input_cols: ["document"]
      output_cols: "qa_prompt"
      prompt_config_path: "qa_prompt.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate_qa"
      input_cols: "qa_prompt"
      output_cols: "qa_response"
      temperature: 0.7
      max_tokens: 512

  - block_type: "TextParserBlock"
    block_config:
      block_name: "parse_qa"
      input_cols: "qa_response"
      output_cols:
        - "question"
        - "response"
      pattern: "Question:\\s*(.+?)\\s*Answer:\\s*(.+)"
      flags: "DOTALL"
```

### Step 6: Test Full Flow

```python
# play.py
from sdg_hub import Flow
import pandas as pd

flow = Flow.from_yaml("flow.yaml")
flow.set_model_config(model="openai/gpt-4o-mini", api_key="sk-...")

df = pd.DataFrame({
    "document": [
        "Python was created by Guido van Rossum in 1991.",
        "Machine learning uses algorithms to learn from data."
    ]
})

# Dry run first
dry = flow.dry_run(df, sample_size=2)
print(f"Dry run success: {dry['execution_successful']}")
for block in dry['blocks_executed']:
    print(f"  {block['block_name']}: {block['execution_time_seconds']:.2f}s")

# Full run
if dry['execution_successful']:
    result = flow.generate(df)
    print(result[["document", "question", "response"]])
    result.to_parquet("output.parquet")
```

### Flow Authoring Checklist

- [ ] Does metadata include name, version, author, description?
- [ ] Are dataset_requirements.required_columns specified?
- [ ] Is each block tested individually in `play.py`?
- [ ] Does dry_run succeed with sample_size=2?
- [ ] Is the output format correct?

See `references/yaml_schema.md` for full YAML structure, `references/block_reference.md` for available blocks, and `references/flow_patterns.md` for common patterns.

## Quick Reference

### Flow Methods

```python
flow = Flow.from_yaml("flow.yaml")
flow.set_model_config(model="...", api_key="...")
flow.validate_dataset(df)
flow.dry_run(df, sample_size=2)
flow.generate(df, checkpoint_dir="...", save_freq=100)
flow.print_info()
```

### Block Direct Usage

```python
from sdg_hub.core.blocks import LLMChatBlock

block = LLMChatBlock(
    block_name="gen",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o-mini",
    api_key="..."
)
result = block(df)
```

### Input/Output Formats

```python
# Load
df = pd.read_csv("input.csv")
df = pd.read_parquet("input.parquet")
df = pd.read_json("input.jsonl", lines=True)

# Save
df.to_parquet("output.parquet")
df.to_csv("output.csv", index=False)
df.to_json("output.jsonl", orient="records", lines=True)
```
