# Flow System

Flows are YAML-defined pipelines that chain blocks into complete data generation
workflows. A flow loads from a single `flow.yaml` file, validates its metadata
and block sequence, and exposes a Python API for configuration and execution.

```
Input Dataset --> Block_1 --> Block_2 --> Block_3 --> Enriched Dataset
```

Each block receives the output dataset from the previous block, processes it, and
passes the result forward. The Flow class handles validation, model configuration,
checkpointing, and metrics collection automatically.

---

## Loading and Running a Flow

```python
from sdg_hub import Flow
from sdg_hub import FlowRegistry
from datasets import Dataset

# Discover built-in flows and load one by name or id
FlowRegistry.discover_flows()
flow_path = FlowRegistry.get_flow_path_safe("epic-jade-656")
flow = Flow.from_yaml(flow_path)

# Configure the LLM model for all LLM blocks
flow.set_model_config(
    model="hosted_vllm/meta-llama/Llama-3.3-70B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="your-key",
)

# Run the pipeline
dataset = Dataset.from_dict({
    "document": ["Your document text..."],
    "document_outline": ["Document Title"],
    "domain": ["articles"],
    "icl_document": ["Example doc..."],
    "icl_query_1": ["Q1?"],
    "icl_query_2": ["Q2?"],
    "icl_query_3": ["Q3?"],
})
result = flow.generate(dataset)
```

---

## Flow Class API

Source: `src/sdg_hub/core/flow/base.py`

### Construction

| Method | Signature | Description |
|--------|-----------|-------------|
| `Flow.from_yaml` | `from_yaml(yaml_path: str) -> Flow` | Load a flow from a YAML file. Validates structure, creates blocks from registry, and sets configuration flags. |
| `flow.to_yaml` | `to_yaml(output_path: str) -> None` | Save the flow to a YAML file. Creates a basic structure; for exact reproduction, keep the original file. |
| `flow.add_block` | `add_block(block: BaseBlock) -> Flow` | Return a new Flow with the block appended. Raises `ValueError` on duplicate block names. |

### Execution

| Method | Signature | Description |
|--------|-----------|-------------|
| `flow.generate` | `generate(dataset, runtime_params=None, checkpoint_dir=None, save_freq=None, log_dir=None, max_concurrency=None)` | Execute all blocks in sequence. Accepts `pd.DataFrame` or `datasets.Dataset` (return type matches input type). |
| `flow.dry_run` | `dry_run(dataset, sample_size=2, runtime_params=None, max_concurrency=None, enable_time_estimation=False) -> dict` | Run the pipeline on a small sample. Returns a dict with execution info, timing, and sample outputs. |
| `flow.validate_dataset` | `validate_dataset(dataset) -> list[str]` | Check dataset against flow requirements. Returns a list of error messages (empty if valid). |

### Model Configuration

Flows with LLM blocks require model configuration before `generate()`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `flow.set_model_config` | `set_model_config(model=None, api_base=None, api_key=None, blocks=None, **kwargs)` | Configure model settings for LLM blocks. Auto-detects LLM blocks unless `blocks` is specified. Extra kwargs (e.g., `temperature`, `max_tokens`) are forwarded. |
| `flow.is_model_config_required` | `is_model_config_required() -> bool` | True if the flow has LLM blocks. |
| `flow.is_model_config_set` | `is_model_config_set() -> bool` | True if model config has been set or is not required. |
| `flow.reset_model_config` | `reset_model_config() -> None` | Clear the config flag. `set_model_config()` must be called again before `generate()`. |
| `flow.get_default_model` | `get_default_model() -> Optional[str]` | Return the default model from `metadata.recommended_models`, or None. |
| `flow.get_model_recommendations` | `get_model_recommendations() -> dict[str, Any]` | Return `{"default": ..., "compatible": [...], "experimental": [...]}`. |

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")

# Check what the flow recommends
print(flow.get_default_model())          # e.g. "openai/gpt-oss-120b"
print(flow.get_model_recommendations())  # full dict

# Configure all LLM blocks
flow.set_model_config(
    model="hosted_vllm/openai/gpt-oss-120b",
    api_base="http://localhost:8101/v1",
    api_key="your-key",
    temperature=0.7,
    max_tokens=2048,
)

# Or configure specific blocks only
flow.set_model_config(
    model="openai/gpt-4o",
    api_key="your-openai-key",
    blocks=["gen_detailed_summary", "knowledge_generation"],
)
```

### Agent Configuration

Flows with agent blocks (e.g., `AgentBlock`) require agent configuration before
`generate()`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `flow.set_agent_config` | `set_agent_config(agent_framework=None, agent_url=None, agent_api_key=None, blocks=None, **kwargs)` | Configure agent settings for agent blocks. Auto-detects agent blocks unless `blocks` is specified. |
| `flow.is_agent_config_required` | `is_agent_config_required() -> bool` | True if the flow has agent blocks. |
| `flow.is_agent_config_set` | `is_agent_config_set() -> bool` | True if agent config has been set or is not required. |
| `flow.reset_agent_config` | `reset_agent_config() -> None` | Clear the agent config flag. |

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/agentic_flow.yaml")

# Configure agent blocks
flow.set_agent_config(
    agent_framework="langflow",
    agent_url="http://localhost:7860/api/v1/run/default",
    agent_api_key="your-langflow-key",
)

# If a flow has BOTH LLM and agent blocks, configure both:
flow.set_model_config(model="openai/gpt-5.2", api_key="llm-key")
flow.set_agent_config(
    agent_framework="langflow",
    agent_url="http://localhost:7860/api/v1/run/default",
)
result = flow.generate(dataset)
```

### Dataset Introspection

| Method | Signature | Description |
|--------|-----------|-------------|
| `flow.get_dataset_requirements` | `get_dataset_requirements() -> Optional[DatasetRequirements]` | Return the `DatasetRequirements` object from metadata, or None. |
| `flow.get_dataset_schema` | `get_dataset_schema() -> pd.DataFrame` | Return an empty DataFrame with the correct column names and types for the flow. |
| `flow.get_info` | `get_info() -> dict[str, Any]` | Return a dict with metadata, block list, total block count, and block names. |
| `flow.print_info` | `print_info() -> None` | Print a formatted summary to the console using Rich. |

---

## FlowRegistry API

Source: `src/sdg_hub/core/flow/registry.py`

The `FlowRegistry` discovers flows from `src/sdg_hub/flows/` automatically on first
use. You can also register custom search paths.

| Method | Signature | Description |
|--------|-----------|-------------|
| `FlowRegistry.discover_flows` | `discover_flows() -> None` | Scan search paths and display all discovered flows in a Rich table. |
| `FlowRegistry.list_flows` | `list_flows() -> list[dict[str, str]]` | Return `[{"id": "...", "name": "..."}, ...]` for all registered flows. |
| `FlowRegistry.search_flows` | `search_flows(tag=None, author=None) -> list[dict[str, str]]` | Filter flows by tag and/or author. Returns same format as `list_flows`. |
| `FlowRegistry.get_flow_path` | `get_flow_path(flow_name_or_id: str) -> Optional[str]` | Return the filesystem path for a flow by id (preferred) or name. Returns None if not found. |
| `FlowRegistry.get_flow_path_safe` | `get_flow_path_safe(flow_name_or_id: str) -> str` | Same as `get_flow_path` but raises `ValueError` with available flow list on failure. |
| `FlowRegistry.get_flow_metadata` | `get_flow_metadata(flow_name: str) -> Optional[FlowMetadata]` | Return the `FlowMetadata` object for a flow, or None. |
| `FlowRegistry.get_flows_by_category` | `get_flows_by_category() -> dict[str, list[dict[str, str]]]` | Return flows grouped by their first tag. |
| `FlowRegistry.register_search_path` | `register_search_path(path: str) -> None` | Add a directory to the list of paths scanned for flows. |

```python
from sdg_hub import FlowRegistry

# Display all flows in a formatted table
FlowRegistry.discover_flows()

# List flows programmatically
for flow_info in FlowRegistry.list_flows():
    print(f"ID: {flow_info['id']}, Name: {flow_info['name']}")

# Search by tag
qa_flows = FlowRegistry.search_flows(tag="question-generation")

# Get flows organized by category
categories = FlowRegistry.get_flows_by_category()
for category, flows in categories.items():
    print(f"\n{category}:")
    for f in flows:
        print(f"  {f['id']} - {f['name']}")

# Register a custom flows directory
FlowRegistry.register_search_path("/path/to/my/custom/flows")
```

---

## Checkpointing

Enable checkpointing to save progress during long runs. If execution is
interrupted, re-running with the same `checkpoint_dir` resumes from where it
left off.

Source: `src/sdg_hub/core/flow/checkpointer.py`

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")
flow.set_model_config(model="openai/gpt-oss-120b", api_key="key")

# Save a checkpoint every 100 completed samples
result = flow.generate(
    dataset,
    checkpoint_dir="./checkpoints",
    save_freq=100,
)
```

**How it works:**

1. Completed samples are saved to `checkpoint_NNNN.jsonl` files after every
   `save_freq` samples.
2. A `flow_metadata.json` file tracks progress and the `flow_id` to prevent
   mixing checkpoints from different flows.
3. On restart, the checkpointer loads completed samples, identifies remaining
   work, and processes only unfinished rows.

If `save_freq` is omitted, checkpoints are saved only at final completion.

---

## Dry Run and Time Estimation

Test a flow on a small sample before committing to a full run.

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")
flow.set_model_config(model="openai/gpt-oss-120b", api_key="key")

# Basic dry run with 2 samples
dry_result = flow.dry_run(dataset, sample_size=2)
print(f"Time: {dry_result['execution_time_seconds']:.2f}s")
print(f"Output columns: {dry_result['final_dataset']['columns']}")

# Dry run with time estimation for the full dataset
dry_result = flow.dry_run(
    dataset,
    sample_size=5,
    enable_time_estimation=True,
    max_concurrency=100,
)
```

When `enable_time_estimation=True`, the system runs two dry runs (with 1 and 5
samples) to separate fixed overhead from per-sample cost, then extrapolates to
the full dataset size. A 20% buffer is included for API variability.

---

## Runtime Parameters

Override block parameters at execution time without modifying the YAML.

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")
flow.set_model_config(model="openai/gpt-oss-120b", api_key="key")

result = flow.generate(
    dataset,
    runtime_params={
        "gen_detailed_summary": {
            "temperature": 0.9,
            "max_tokens": 4096,
            "n": 25,
        },
        "eval_faithful_filter": {
            "filter_value": "YES",
            "operation": "eq",
        },
    },
)
```

Parameters are keyed by `block_name`. Each dict is passed as kwargs to the
block's `generate()` call.

---

## Concurrency Control

Limit concurrent LLM requests to manage rate limits.

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")
flow.set_model_config(model="openai/gpt-oss-120b", api_key="key")

result = flow.generate(dataset, max_concurrency=10)
```

The `max_concurrency` parameter applies across all async LLM blocks. Omitting it
uses unbounded concurrency (fastest, but may hit rate limits).

---

## Logging and Metrics

Provide a `log_dir` to save execution logs and metrics to disk.

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")
flow.set_model_config(model="openai/gpt-oss-120b", api_key="key")

result = flow.generate(dataset, log_dir="./flow_logs")
# Creates:
#   ./flow_logs/<flow_name>_<timestamp>.log
#   ./flow_logs/<flow_name>_<timestamp>_metrics.json
```

A Rich metrics table is printed to the console after every run (success or
failure), showing per-block execution time, row counts, column changes, and
status.

---

## Flow Translation

Translate a flow and all its prompt YAMLs to another language using
`translate_flow()`. The function uses one LLM for translation and a second for
verification, with retry on failure.

Source: `src/sdg_hub/core/utils/translation.py`

```python
from sdg_hub.core.utils.translation import translate_flow

translated_flow = translate_flow(
    flow="epic-jade-656",           # flow id or name
    lang="Spanish",
    lang_code="es",
    translator_model="openai/gpt-4o",
    verifier_model="openai/gpt-4o",
    translator_api_key="your-key",
    max_retries=3,
)

# The returned Flow is ready to use
translated_flow.set_model_config(model="openai/gpt-oss-120b", api_key="key")
result = translated_flow.generate(spanish_dataset)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flow` | `str` | required | Flow id or name from FlowRegistry |
| `lang` | `str` | required | Target language name (e.g., "Spanish") |
| `lang_code` | `str` | required | ISO 639-1 code (e.g., "es") |
| `translator_model` | `str` | `"gpt-5.2"` | Model for translation |
| `verifier_model` | `str` | `"gpt-5.2"` | Model for verification |
| `output_dir` | `str` | `"./<flow_dir>_<lang_code>/"` | Output directory |
| `translator_api_key` | `str` | `None` | API key for translator |
| `translator_api_base` | `str` | `None` | API base for translator |
| `verifier_api_key` | `str` | `None` | API key for verifier |
| `verifier_api_base` | `str` | `None` | API base for verifier |
| `max_retries` | `int` | `3` | Retry count per prompt message |
| `verbose` | `bool` | `False` | Enable DEBUG logging |
| `register` | `bool` | `True` | Register translated flow with FlowRegistry |

The function preserves Jinja2 template variables and structural tags (from
`TagParserBlock` configs) during translation. If the translated flow already
exists in the registry or output directory, translation is skipped and the
existing flow is returned.
