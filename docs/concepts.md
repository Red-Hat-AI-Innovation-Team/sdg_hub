# Core Concepts

This page explains the foundational abstractions in SDG Hub: blocks, flows,
registries, column management, and dataset handling.

## Blocks

A **block** is the atomic unit of data processing. Every block takes a
`pd.DataFrame` as input, performs a transformation, and returns a
`pd.DataFrame` as output.

### The generate() contract

All blocks inherit from `BaseBlock` (defined in
`sdg_hub.core.blocks.base`) and must implement one method:

```python
from sdg_hub import BaseBlock
import pandas as pd
from typing import Any

class MyBlock(BaseBlock):
    def generate(self, samples: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        # Transform the DataFrame and return it
        ...
```

When a block is called (via `block(dataframe)`), the `__call__` method
runs validation and logging around the `generate` method automatically:

1. Log input data summary (columns, row count).
2. Validate the DataFrame is not empty.
3. Validate required `input_cols` exist.
4. Validate `output_cols` do not collide with existing columns.
5. Run any custom validation (`_validate_custom`).
6. Call `generate(samples, **kwargs)`.
7. Log output data summary (added/removed columns, row counts).

### Block categories

Blocks are organized into directories under `src/sdg_hub/core/blocks/`:

| Category | Directory | Purpose |
|----------|-----------|---------|
| `llm` | `blocks/llm/` | LLM-powered operations: chat completion, prompt building, text parsing |
| `parsing` | `blocks/parsing/` | Text parsing and extraction |
| `transform` | `blocks/transform/` | Data manipulation: column operations, text concatenation, duplication |
| `filtering` | `blocks/filtering/` | Row filtering based on value thresholds or conditions |
| `agent` | `blocks/agent/` | Agent framework integration (Langflow, LangGraph) |
| `mcp` | `blocks/mcp/` | Model Context Protocol tool calling |

### Block configuration

Every block has three standard fields defined on `BaseBlock`:

- `block_name` (str, required) -- unique identifier within a flow.
- `input_cols` (str, list, dict, or None) -- columns the block reads from.
- `output_cols` (str, list, dict, or None) -- columns the block writes to.

Subclasses add their own fields using Pydantic model definitions. All
configuration is validated at construction time.

## Flows

A **flow** is a YAML-defined pipeline that chains multiple blocks into a
sequential data processing workflow.

### Execution model

Flows execute blocks one at a time, in order:

```text
Input Dataset --> Block_1 --> Block_2 --> Block_3 --> Final Dataset
```

Each block receives the full DataFrame output of the previous block. The
final result is returned from `flow.generate()`.

### YAML structure

A flow YAML file has three top-level sections:

```yaml
metadata:
  name: "My Flow"
  version: "1.0.0"
  author: "Your Name"
  description: "What this flow does"
  recommended_models:
    default: "openai/gpt-oss-120b"
    compatible:
      - "meta-llama/Llama-3.3-70B-Instruct"
    experimental: []
  tags:
    - "knowledge-tuning"
    - "question-generation"
  dataset_requirements:
    required_columns:
      - document
      - domain
    description: "Input must contain document text and domain."
  output_columns:
    - question
    - response

parameters:
  param_name:
    type: "string"
    default: "default_value"
    description: "Parameter description"

blocks:
  - block_type: PromptBuilderBlock
    block_config:
      block_name: build_prompt
      input_cols: ["document"]
      output_cols: prompt
      prompt_config_path: prompts/my_prompt.yaml
      format_as_messages: true

  - block_type: LLMChatBlock
    block_config:
      block_name: generate_text
      input_cols: prompt
      output_cols: raw_output
```

**metadata** -- flow identity, model recommendations, dataset requirements,
and output column declarations. When `output_columns` is set, intermediate
columns are dropped from the final result (original input columns are always
preserved).

**parameters** -- optional flow-level parameters that can be referenced
in block configs.

**blocks** -- ordered list of blocks. Each entry specifies a `block_type`
(the registered block class name) and a `block_config` dict passed to the
block constructor.

### Model configuration

Flows with LLM blocks require model configuration before execution.
The `set_model_config` method auto-detects LLM blocks and applies settings:

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")

flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="your_key",
)
```

Flows with agent blocks have a parallel method:

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")

flow.set_agent_config(
    agent_framework="langflow",
    agent_url="http://localhost:7860/api/v1/run/my-flow",
    agent_api_key="your_key",
)
```

### Key Flow methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `Flow.from_yaml(yaml_path)` | `Flow` | Load a flow from a YAML file |
| `flow.generate(dataset, ...)` | `DataFrame` or `Dataset` | Execute the full pipeline |
| `flow.dry_run(dataset, ...)` | `dict[str, Any]` | Test with a small subset |
| `flow.set_model_config(...)` | `None` | Configure LLM model settings |
| `flow.set_agent_config(...)` | `None` | Configure agent framework settings |
| `flow.get_default_model()` | `Optional[str]` | Get the recommended default model |
| `flow.get_model_recommendations()` | `dict` | Get default, compatible, and experimental models |
| `flow.get_dataset_schema()` | `pd.DataFrame` | Empty DataFrame with correct columns |
| `flow.get_dataset_requirements()` | `Optional[DatasetRequirements]` | Full dataset requirements or None |
| `flow.is_model_config_required()` | `bool` | True if the flow has LLM blocks |
| `flow.is_model_config_set()` | `bool` | True if model config is set or not needed |
| `flow.validate_dataset(dataset)` | `list[str]` | List of validation error messages |
| `flow.print_info()` | `None` | Print a Rich summary to the console |

## Registries

SDG Hub uses three registries for auto-discovery and lookup of components.

### FlowRegistry

Defined in `sdg_hub.core.flow.registry`. Discovers flow YAML files in
registered search paths (the built-in `src/sdg_hub/flows/` directory is
auto-registered).

```python
from sdg_hub import FlowRegistry

# Display all flows in a Rich table
FlowRegistry.discover_flows()

# Programmatic access
flows = FlowRegistry.list_flows()           # [{"id": ..., "name": ...}]
results = FlowRegistry.search_flows(tag="question-generation")
path = FlowRegistry.get_flow_path("flow-id-or-name")       # Optional[str]
path = FlowRegistry.get_flow_path_safe("flow-id-or-name")  # str (raises ValueError)
metadata = FlowRegistry.get_flow_metadata("Flow Name")     # Optional[FlowMetadata]
by_cat = FlowRegistry.get_flows_by_category()               # dict[str, list[dict]]
```

Add custom search paths:

```python
from sdg_hub import FlowRegistry

FlowRegistry.register_search_path("/path/to/my/flows")
```

### BlockRegistry

Defined in `sdg_hub.core.blocks.registry`. Blocks register themselves
using the `@BlockRegistry.register(...)` decorator.

```python
from sdg_hub import BlockRegistry

# Display all blocks in a Rich table
BlockRegistry.discover_blocks()

# Programmatic access
all_blocks = BlockRegistry.list_blocks()                    # list[str]
grouped = BlockRegistry.list_blocks(grouped=True)           # dict[str, list[str]]
llm_blocks = BlockRegistry.list_blocks(category="llm")     # list[str]
categories = BlockRegistry.categories()                     # list[str]
```

### ConnectorRegistry

Defined in `sdg_hub.core.connectors.registry`. Connectors register
themselves with `@ConnectorRegistry.register("name")`.

```python
from sdg_hub.core.connectors import ConnectorRegistry

# List all registered connector names
names = ConnectorRegistry.list_all()     # list[str]

# Get a connector class by name
cls = ConnectorRegistry.get("langflow")  # type (raises ConnectorError if not found)
```

## Column Management

Blocks declare the columns they read and write. The framework validates
these declarations at runtime.

### input_cols

Specifies which columns a block needs. Before `generate()` runs, the
framework checks that every column in `input_cols` exists in the DataFrame.
If any are missing, `MissingColumnError` is raised.

### output_cols

Specifies which columns a block creates. Before `generate()` runs, the
framework checks that none of the `output_cols` already exist in the
DataFrame. If any collide, `OutputColumnCollisionError` is raised.

### Column formats

Both `input_cols` and `output_cols` accept multiple formats:

```python
# Single column as a string
input_cols="document"

# Multiple columns as a list
input_cols=["document", "domain"]

# Column mapping as a dict (renames or structured access)
input_cols={"document": "base_document"}
```

### Column flow through a pipeline

In a flow, columns accumulate. Block 1 adds columns to the DataFrame,
Block 2 sees the original columns plus Block 1's output, and so on.

If the flow's metadata specifies `output_columns`, intermediate columns
are automatically dropped from the final result. Original input columns
are always preserved.

## Dataset Handling

### Input types

`Flow.generate()` and `Flow.dry_run()` accept either:

- `pd.DataFrame` -- pandas DataFrame
- `datasets.Dataset` -- HuggingFace Dataset

The return type matches the input type. If you pass in a `datasets.Dataset`,
you get a `datasets.Dataset` back.

### Internal processing

Internally, all blocks operate on `pd.DataFrame`. When a `datasets.Dataset`
is provided, the flow converts it to a DataFrame before processing and
converts the result back afterward.

### Creating input datasets

```python
import pandas as pd
from datasets import Dataset

# From pandas
df = pd.DataFrame({
    "document": ["Text content here..."],
    "domain": ["science"],
})

# From HuggingFace datasets
ds = Dataset.from_dict({
    "document": ["Text content here..."],
    "domain": ["science"],
})
```

### Validating a dataset against a flow

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")

errors = flow.validate_dataset(dataset)
if errors:
    for error in errors:
        print(error)
```

## Best Practices

### Start with a dry run

Always run `dry_run` before `generate`. It executes the full pipeline on a
small subset and catches configuration errors, missing columns, and API
issues before you commit to processing the full dataset.

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")
flow.set_model_config(model="openai/gpt-4o", api_key="your_key")

result = flow.dry_run(dataset, sample_size=5, enable_time_estimation=True)
```

### Layer validation

Build pipelines with validation in mind. A common pattern chains
prompt building, LLM generation, text parsing, and quality filtering:

```text
PromptBuilder --> LLMChat --> TagParserBlock --> QualityFilter
```

Each block validates its inputs and outputs. If a filter removes all rows,
`EmptyDatasetError` is raised immediately rather than propagating empty
data through downstream blocks.

### Control concurrency

Use `max_concurrency` to limit parallel API requests and avoid rate limiting:

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")
flow.set_model_config(model="openai/gpt-4o", api_key="your_key")

result = flow.generate(dataset, max_concurrency=10)
```

Start with conservative values (5-10) and increase while monitoring error
rates and provider rate limits.

### Monitor execution

Pass `log_dir` to `generate()` to write execution logs and metrics to files:

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")
flow.set_model_config(model="openai/gpt-4o", api_key="your_key")

result = flow.generate(dataset, log_dir="./logs")
```

A JSON metrics file and a log file are saved with timestamps in the
specified directory.

### Design for reuse

Create modular flows with parameters for customization points. Use
`runtime_params` to adjust block behavior without editing YAML files.

## Next Steps

- [Quick Start](quickstart.md) -- step-by-step tutorial
- [Block System Overview](blocks/index.md) -- learn about each block category
- [Flow System Overview](flows/index.md) -- YAML structure and flow configuration
- [Custom Blocks](blocks/custom-blocks.md) -- build your own processing blocks
