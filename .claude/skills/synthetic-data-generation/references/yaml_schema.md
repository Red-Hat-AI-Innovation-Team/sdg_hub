# Flow YAML Schema

Complete reference for `flow.yaml` structure.

## Top-Level Structure

```yaml
metadata:      # Flow metadata (name required)
parameters:    # Optional runtime parameters
blocks:        # List of blocks to execute (required, non-empty)
```

## Metadata

```yaml
metadata:
  # Required
  name: "Human Readable Flow Name"

  # Recommended
  version: "1.0.0"
  author: "Your Name"
  description: "What this flow does"

  # Auto-generated if omitted
  id: "lowercase-kebab-id"

  # Model recommendations
  recommended_models:
    default: "openai/gpt-4"
    compatible:
      - "meta-llama/Llama-3.3-70B-Instruct"
    experimental:
      - "mistral/mistral-large"

  # Dataset requirements
  dataset_requirements:
    required_columns: ["document"]
    optional_columns: ["domain"]
    min_samples: 1
    max_samples: 10000
    column_types:
      document: "string"
    description: "Documents to process"

  # Categorization
  tags: ["qa-generation", "knowledge-infusion"]
  license: "Apache-2.0"
```

## Parameters (Optional)

Define runtime-configurable values:

```yaml
parameters:
  temperature:
    type: "float"
    default: 0.7
    description: "LLM temperature"
  max_tokens:
    type: "integer"
    default: 1024
    description: "Max tokens per response"
```

## Blocks

```yaml
blocks:
  - block_type: "RegisteredBlockName"
    block_config:
      block_name: "unique_id"        # Required, unique within flow
      input_cols: ...                 # Input column spec
      output_cols: ...               # Output column spec
      # ... block-specific params
    runtime_overrides: ["temperature", "max_tokens"]  # Optional
```

### Column Specification Formats

```yaml
# Single column (string)
input_cols: "text"
output_cols: "summary"

# Multiple columns (list)
input_cols: ["text", "context"]
output_cols: ["question", "answer"]

# Column mapping (dict) -- rename on input
input_cols:
  document: base_document      # Use base_document as "document"
```

## Complete Example

```yaml
metadata:
  name: "Document QA Generation"
  version: "1.0.0"
  author: "SDG Hub Contributors"
  description: "Generate QA pairs from documents"
  recommended_models:
    default: "openai/gpt-4"
    compatible: ["meta-llama/Llama-3.3-70B-Instruct"]
  dataset_requirements:
    required_columns: ["document"]
    optional_columns: ["domain"]
  tags: ["qa-generation"]

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
      max_tokens: 512
      async_mode: true

  - block_type: "TagParserBlock"
    block_config:
      block_name: "parse"
      input_cols: "raw_response"
      output_cols: ["question", "response"]
      start_tags: ["<question>", "<answer>"]
      end_tags: ["</question>", "</answer>"]

  - block_type: "ColumnValueFilterBlock"
    block_config:
      block_name: "filter_empty"
      input_cols: "question"
      filter_value: ""
      operation: "ne"
```

## Validation Rules

The flow validator checks:

1. `blocks` must exist and be a non-empty list
2. Each block must have `block_type` and `block_config` (both dicts)
3. Each `block_config` must have a `block_name`
4. Block names must be unique within the flow
5. `metadata.name` is required if metadata is present
6. `metadata.id` must be lowercase alphanumeric + hyphens
7. `prompt_config_path` files must exist relative to the flow.yaml directory

## Prompt Template Files

Referenced by `PromptBuilderBlock` via `prompt_config_path` (relative to flow.yaml).

```yaml
# prompts/qa.yaml
- role: system
  content: |
    You generate question-answer pairs.

- role: user
  content: |
    Document: {document}

    Generate a question and answer.
    Use <question>...</question> and <answer>...</answer> tags.
```

Template variables (`{document}`) are filled from `input_cols`.
