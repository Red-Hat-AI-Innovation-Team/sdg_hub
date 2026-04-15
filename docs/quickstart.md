# Quick Start

By the end of this guide, you will have loaded a built-in flow, validated it
with a dry run, and generated synthetic data.

This guide walks through the standard SDG Hub workflow: discover a flow,
inspect its requirements, configure a model, validate with a dry run, and
generate synthetic data.

## Prerequisites

Install SDG Hub:

```bash
uv pip install sdg-hub
```

## Step 1: Discover and Load a Flow

SDG Hub ships with built-in flows. Use `FlowRegistry` to browse them, then
load one by ID or name.

> **Note:** `discover_flows()` must be called before any registry lookups.
> It scans registered search paths and populates the registry.

```python
from sdg_hub import FlowRegistry, Flow
from datasets import Dataset

# Discover built-in flows (must be called before any registry lookups)
FlowRegistry.discover_flows()

# List flows programmatically
flows = FlowRegistry.list_flows()
for f in flows:
    print(f["id"], f["name"])

# Search by tag or author
qa_flows = FlowRegistry.search_flows(tag="question-generation")
author_flows = FlowRegistry.search_flows(author="SDG Hub Contributors")

# Browse by category
categories = FlowRegistry.get_flows_by_category()
for tag, flow_list in categories.items():
    print(f"{tag}: {len(flow_list)} flows")
```

## Step 2: Load a Flow

Resolve a flow name or ID to a YAML path, then load it. This example uses
the **Extractive Summary Knowledge Tuning** flow (`epic-jade-656`), which
generates QA pairs from documents by first extracting key passages, then
creating questions and answers grounded in those extracts.

```python
# get_flow_path returns Optional[str] (None if not found)
flow_path = FlowRegistry.get_flow_path("epic-jade-656")

# get_flow_path_safe raises ValueError with suggestions if not found
flow_path = FlowRegistry.get_flow_path_safe("epic-jade-656")

flow = Flow.from_yaml(flow_path)
print(flow)
```

Both `get_flow_path` and `get_flow_path_safe` accept either a flow ID or a
flow name.

## Step 3: Inspect Requirements

Before running anything, check what the flow needs.

### Default model and recommendations

```python
default_model = flow.get_default_model()
# Returns: Optional[str], e.g. "openai/gpt-oss-120b"

recommendations = flow.get_model_recommendations()
# Returns: {"default": "...", "compatible": [...], "experimental": [...]}
```

### Dataset schema and requirements

```python
# Get an empty DataFrame with the correct columns
schema_df = flow.get_dataset_schema()
print(schema_df.columns.tolist())

# Get full requirements (DatasetRequirements or None)
requirements = flow.get_dataset_requirements()
if requirements:
    print("Required columns:", requirements.required_columns)
    print("Min samples:", requirements.min_samples)
    print("Description:", requirements.description)
```

### Check model configuration status

```python
# True if the flow contains LLM blocks
print(flow.is_model_config_required())

# True if set_model_config() has been called (or no LLM blocks exist)
print(flow.is_model_config_set())
```

## Step 4: Configure the Model

Flows with LLM blocks require model configuration before execution. Calling
`generate()` without it raises `FlowValidationError`.

> **Tip:** LiteLLM auto-reads standard provider environment variables
> (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). If the appropriate env var
> is already set, you can skip the `api_key` parameter:
>
> ```python
> import os
> # LiteLLM reads standard env vars automatically
> # If OPENAI_API_KEY is set, no api_key needed:
> flow.set_model_config(model="openai/gpt-4o")
> ```

```python
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="your_key",
)
```

`set_model_config` auto-detects LLM blocks and applies the configuration. You
can target specific blocks with the `blocks` parameter:

```python
flow.set_model_config(
    model="openai/gpt-4o",
    api_key="your-openai-key",
    blocks=["gen_extractive_summary", "gen_qa_pairs"],
)
```

## Step 5: Dry Run

Always validate your pipeline before processing the full dataset. `dry_run`
executes the flow on a small subset and returns diagnostic results.

```python
# Prepare the dataset.
# This flow expects knowledge-tuning columns:
#   document         -- the source document to generate QA from
#   document_outline -- structural outline of the document
#   domain           -- subject area (e.g. "Computer Science")
#   icl_document     -- in-context learning example document
#   icl_query_1..3   -- in-context learning example questions
dataset = Dataset.from_dict({
    "document": ["Python is a high-level programming language..."],
    "document_outline": ["1. Introduction; 2. Features"],
    "domain": ["Computer Science"],
    "icl_document": ["Java is an object-oriented language..."],
    "icl_query_1": ["What type of language is Java?"],
    "icl_query_2": ["Where does Java run?"],
    "icl_query_3": ["What are the benefits of Java?"],
})

result = flow.dry_run(
    dataset,
    sample_size=2,
    enable_time_estimation=True,
    max_concurrency=10,
)

print(f"Tested {result['sample_size']} samples in {result['execution_time_seconds']:.2f}s")
print(f"Output columns: {result['final_dataset']['columns']}")
```

**Signature reference:**

```python
flow.dry_run(
    dataset,                          # pd.DataFrame or datasets.Dataset
    sample_size=2,                    # number of rows to test
    runtime_params=None,              # Optional[dict[str, dict[str, Any]]]
    max_concurrency=None,             # Optional[int]
    enable_time_estimation=False,     # runs extra dry runs for estimation
)
# Returns: dict[str, Any]
```

When `enable_time_estimation=True`, a Rich table with projected execution time
is printed automatically. The estimation data is displayed but not included in
the return dict.

## Step 6: Generate

Once the dry run succeeds, run the full pipeline.

```python
result = flow.generate(dataset)
print(f"Generated {len(result)} rows")
print(result.column_names)
```

**Signature reference:**

```python
flow.generate(
    dataset,                          # pd.DataFrame or datasets.Dataset
    runtime_params=None,              # Optional[dict[str, dict[str, Any]]]
    checkpoint_dir=None,              # Optional[str] - enables checkpointing
    save_freq=None,                   # Optional[int] - checkpoint every N samples
    log_dir=None,                     # Optional[str] - write logs to files
    max_concurrency=None,             # Optional[int]
)
# Returns: same type as input (DataFrame in -> DataFrame out, Dataset in -> Dataset out)
```

### Checkpointing

For large datasets, enable checkpointing to resume interrupted runs:

```python
result = flow.generate(
    dataset,
    checkpoint_dir="./checkpoints",
    save_freq=50,  # save progress every 50 completed samples
)
```

## Step 7: Runtime Parameters

Override block configuration at execution time without modifying flow YAML
files. Runtime parameters are organized by block name.

```python
result = flow.generate(
    dataset,
    runtime_params={
        "gen_qa_pairs": {"temperature": 0.9, "max_tokens": 200},
        "quality_filter": {"filter_value": 0.8, "operation": "ge"},
    },
)
```

The same `runtime_params` argument works with `dry_run`:

```python
result = flow.dry_run(
    dataset,
    sample_size=2,
    runtime_params={
        "gen_qa_pairs": {"temperature": 0.5},
    },
)
```

## Step 8: Error Handling

SDG Hub defines specific exception types for different failure modes. All
error classes live in `sdg_hub.core.utils.error_handling`.

### Common error pattern

```python
from sdg_hub import Flow
from sdg_hub.core.utils.error_handling import (
    FlowValidationError,
    MissingColumnError,
    EmptyDatasetError,
    OutputColumnCollisionError,
)

try:
    flow = Flow.from_yaml("path/to/flow.yaml")
    flow.set_model_config(model="openai/gpt-4o", api_key="your_key")
    result = flow.generate(dataset)
except FlowValidationError as e:
    # Invalid YAML, missing model config, empty dataset, or block execution failure
    print(f"Flow error: {e}")
except MissingColumnError as e:
    # Dataset is missing columns that a block requires
    print(f"Missing columns: {e.missing_columns}")
    print(f"Available columns: {e.available_columns}")
except EmptyDatasetError as e:
    # A block (often a filter) produced zero rows
    print(f"Block '{e.block_name}' received an empty dataset")
except OutputColumnCollisionError as e:
    # A block tried to create a column that already exists
    print(f"Collision columns: {e.collision_columns}")
```

### Error class hierarchy

All exceptions inherit from `SDGHubError`:

| Exception | When it is raised |
|-----------|-------------------|
| `FlowValidationError` | Invalid YAML, missing model config, empty input, block execution failure |
| `FlowExecutionError` | Flow execution failure |
| `FlowConfigurationError` | Invalid flow configuration |
| `MissingColumnError` | A block's `input_cols` are not present in the dataset |
| `EmptyDatasetError` | A block receives a dataset with zero rows |
| `OutputColumnCollisionError` | A block's `output_cols` already exist in the dataset |
| `BlockConfigurationError` | Invalid block configuration |
| `BlockExecutionError` | Block execution failure |
| `BlockValidationError` | Base class for column/dataset validation errors |
| `TemplateValidationError` | Prompt template references variables not in the dataset |
| `APIConnectionError` | API connection failure |
| `DataGenerationError` | Data generation failure |

### Prevention

Use `dry_run` to catch most errors before processing large datasets:

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")
flow.set_model_config(model="openai/gpt-4o", api_key="your_key")

# Catches MissingColumnError, EmptyDatasetError, and config issues early
result = flow.dry_run(dataset, sample_size=2)
```

## Discover Blocks

In addition to flows, you can browse available blocks.

```python
from sdg_hub import BlockRegistry

# Print a Rich table of all blocks
BlockRegistry.discover_blocks()

# List all block names
all_blocks = BlockRegistry.list_blocks()

# List blocks grouped by category
grouped = BlockRegistry.list_blocks(grouped=True)
# Returns: {"llm": ["LLMChatBlock", ...], "transform": [...], ...}

# List blocks in a specific category
llm_blocks = BlockRegistry.list_blocks(category="llm")

# Get available categories
categories = BlockRegistry.categories()
# Returns: ["agent", "filtering", "llm", "mcp", "parsing", "transform"]
```

## Next Steps

- [Core Concepts](concepts.md) -- understand blocks, flows, and registries in depth
- [Block System Overview](blocks/index.md) -- learn about each block category
- [Flow System Overview](flows/index.md) -- YAML structure and flow configuration
- [Custom Blocks](blocks/custom-blocks.md) -- build your own processing blocks
