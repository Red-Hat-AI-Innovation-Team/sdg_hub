# Custom Flows

This page covers how to create, structure, test, and register custom flows.

---

## Directory Structure

Place custom flows under `src/sdg_hub/flows/` in a category subdirectory.
Each flow lives in its own directory with a `flow.yaml` and an optional
`prompts/` folder.

```
src/sdg_hub/flows/
  {category}/
    {flow_name}/
      flow.yaml             # Required - flow definition
      prompts/              # Optional - prompt template files
        prompt_a.yaml
        prompt_b.yaml
```

Example for a custom sentiment classification flow:

```
src/sdg_hub/flows/
  text_analysis/
    sentiment_classification/
      flow.yaml
      prompts/
        classify_sentiment.yaml
```

---

## Complete flow.yaml Template

This template includes all metadata fields supported by `FlowMetadata` (source:
`src/sdg_hub/core/flow/metadata.py`).

```yaml
metadata:
  name: "My Custom Flow"
  # id is auto-generated from name if omitted.
  # To set it manually, use lowercase alphanumeric with hyphens:
  # id: "my-custom-flow"
  description: "What this flow does, in one or two sentences."
  version: "1.0.0"
  author: "Your Name"
  license: "Apache-2.0"

  recommended_models:
    default: "openai/gpt-oss-120b"
    compatible:
      - "meta-llama/Llama-3.3-70B-Instruct"
      - "microsoft/phi-4"
    experimental: []

  tags:
    - "my-category"
    - "my-purpose"

  dataset_requirements:
    required_columns:
      - "input_text"
    optional_columns:
      - "context"
    min_samples: 1
    max_samples: 10000
    column_types:
      input_text: "string"
      context: "string"
    description: "Dataset with text content for processing."

  # Optional: specify which columns to keep in the final output.
  # Original input columns are always preserved.
  # Omit this field entirely to keep all columns.
  output_columns:
    - "result"
    - "score"

blocks:
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_prompt"
      input_cols:
        - "input_text"
      output_cols: "messages"
      prompt_config_path: "prompts/classify_sentiment.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate"
      input_cols: "messages"
      output_cols: "raw_response"
      max_tokens: 512
      temperature: 0.3
      async_mode: true

  - block_type: "LLMResponseExtractorBlock"
    block_config:
      block_name: "extract"
      input_cols: "raw_response"
      extract_content: true
      expand_lists: true

  - block_type: "TagParserBlock"
    block_config:
      block_name: "parse"
      input_cols: "extract_content"
      output_cols:
        - "result"
        - "score"
      start_tags:
        - "[RESULT]"
        - "[SCORE]"
      end_tags:
        - "[/RESULT]"
        - "[/SCORE]"
```

---

## Adding Prompt Templates

Prompt templates are YAML files containing a list of message dicts. They are
loaded by `PromptBuilderBlock` and rendered with Jinja2, using input column
values as template variables.

Example `prompts/classify_sentiment.yaml`:

```yaml
- role: "system"
  content: |
    You are a sentiment classifier. Analyze the text and output:
    [RESULT]positive, negative, or neutral[/RESULT]
    [SCORE]confidence score between 0 and 1[/SCORE]

- role: "user"
  content: |
    Analyze the sentiment of this text:

    {{input_text}}
```

Template variables use double-brace syntax (`{{column_name}}`). The variable
names must match the column names specified in `input_cols` of the
`PromptBuilderBlock`.

The `prompt_config_path` in `block_config` is resolved relative to the directory
containing `flow.yaml`. So `prompts/classify_sentiment.yaml` resolves to
`{flow_dir}/prompts/classify_sentiment.yaml`.

---

## Block Naming Rules

All `block_name` values within a flow must be unique. The `Flow` class validates
this at construction time and raises `ValueError` on duplicates.

The common pattern for LLM-powered extraction is a four-block sequence:

1. `PromptBuilderBlock` -- builds the prompt messages
2. `LLMChatBlock` -- calls the LLM
3. `LLMResponseExtractorBlock` -- extracts the assistant content
4. `TagParserBlock` or `JSONParserBlock` -- parses structured output

Name blocks descriptively to indicate their purpose:

```yaml
blocks:
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_summary_prompt"    # what it builds
      # ...
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate_summary"        # what it generates
      # ...
  - block_type: "LLMResponseExtractorBlock"
    block_config:
      block_name: "extract_summary"         # what it extracts
      # ...
  - block_type: "TagParserBlock"
    block_config:
      block_name: "parse_summary"           # what it parses
      # ...
```

---

## Testing Flows

### Dry Run

Use `dry_run()` to test with a small sample before running on a full dataset.

```python
from sdg_hub import Flow
from datasets import Dataset

flow = Flow.from_yaml("src/sdg_hub/flows/text_analysis/sentiment_classification/flow.yaml")
flow.set_model_config(
    model="openai/gpt-oss-120b",
    api_key="your-key",
)

test_data = Dataset.from_dict({
    "input_text": ["This product is excellent.", "Terrible experience."],
})

dry_result = flow.dry_run(test_data, sample_size=2)
print(f"Success: {dry_result['execution_successful']}")
print(f"Time: {dry_result['execution_time_seconds']:.2f}s")
print(f"Output columns: {dry_result['final_dataset']['columns']}")
```

### Dataset Validation

Check that your dataset meets the flow's requirements without running any
blocks.

```python
from sdg_hub import Flow
from datasets import Dataset

flow = Flow.from_yaml("path/to/flow.yaml")

dataset = Dataset.from_dict({"wrong_column": ["data"]})
errors = flow.validate_dataset(dataset)
if errors:
    for err in errors:
        print(f"Validation error: {err}")
# Output: Validation error: Missing required columns: ['input_text']
```

### Dataset Schema

Get an empty DataFrame with the correct columns and types to use as a template.

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")
schema = flow.get_dataset_schema()
print(schema.columns.tolist())
# Output: ['input_text', 'context']
```

### Unit Tests

Write pytest tests that mock the LLM client. Verify that non-LLM blocks
(transform, filtering) work correctly with known input data.

```python
import pandas as pd
import pytest
from sdg_hub import Flow

def test_flow_loads():
    flow = Flow.from_yaml("src/sdg_hub/flows/text_analysis/sentiment_classification/flow.yaml")
    assert len(flow.blocks) > 0
    assert flow.metadata.name == "My Custom Flow"

def test_flow_validates_dataset():
    flow = Flow.from_yaml("src/sdg_hub/flows/text_analysis/sentiment_classification/flow.yaml")
    errors = flow.validate_dataset(pd.DataFrame({"input_text": ["test"]}))
    assert errors == []

def test_flow_rejects_bad_dataset():
    flow = Flow.from_yaml("src/sdg_hub/flows/text_analysis/sentiment_classification/flow.yaml")
    errors = flow.validate_dataset(pd.DataFrame({"wrong": ["test"]}))
    assert len(errors) > 0
```

---

## FlowRegistry Integration

### Automatic Discovery

Flows placed under `src/sdg_hub/flows/` are discovered automatically when
`FlowRegistry.discover_flows()` is called. No manual registration is needed.

Requirements for automatic discovery:

1. The file is named `*.yaml` (typically `flow.yaml`).
2. The YAML root contains both `metadata` and `blocks` keys.
3. The `metadata` section contains a `name` field.

### Custom Search Paths

To discover flows outside the built-in directory, register additional search
paths before calling `discover_flows()`.

```python
from sdg_hub import FlowRegistry

# Register a custom directory
FlowRegistry.register_search_path("/path/to/my/flows")

# Now discover_flows() will scan the custom directory too
FlowRegistry.discover_flows()
```

### Finding Your Flow

After discovery, look up your flow by id or name.

```python
from sdg_hub import Flow
from sdg_hub import FlowRegistry

FlowRegistry.discover_flows()

# By id (preferred)
path = FlowRegistry.get_flow_path_safe("my-custom-flow")
flow = Flow.from_yaml(path)

# By name (backward compatible)
path = FlowRegistry.get_flow_path("My Custom Flow")
if path:
    flow = Flow.from_yaml(path)

# Search by tag
matches = FlowRegistry.search_flows(tag="my-category")
for match in matches:
    print(f"Found: {match['id']} - {match['name']}")
```

### Flow ID Generation

If you omit `id` from the metadata, it is auto-generated from the flow name on
first load. The generated id is written back to the YAML file automatically.
To control the id, set it explicitly in `metadata.id`.

---

## Metadata Best Practices

### Tags

Choose 3-7 tags from these categories:

- **Purpose**: what the flow does (`question-generation`, `knowledge-tuning`,
  `text-analysis`, `summarization`, `sentiment-analysis`)
- **Output type**: what it produces (`qa-pairs`, `structured-output`,
  `key-facts`)
- **Domain**: where it is used (`educational`, `document-processing`, `nlp`)
- **Technical**: special characteristics (`multilingual`, `japanese`,
  `red-team`, `agentic`)

Tags are automatically lowercased.

### Recommended Models

Always specify a `default` model. Add `compatible` models that are tested and
`experimental` models that may work but are untested. The
`get_default_model()` and `get_model_recommendations()` methods on `Flow`
read these values.

### Output Columns

Set `output_columns` in metadata to control which columns appear in the final
output. This drops intermediate columns (like raw LLM responses and prompt
columns) while preserving the original input columns plus the specified output
columns. If omitted, all columns are kept.

```yaml
metadata:
  output_columns:
    - "question"
    - "response"
    - "faithfulness_judgment"
```
