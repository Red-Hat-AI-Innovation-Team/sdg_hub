# Flow YAML Schema

Complete reference for flow.yaml structure.

## Top-Level Structure

```yaml
metadata:      # Flow metadata (name required)
  ...

parameters:    # Optional runtime parameters
  ...

blocks:        # List of blocks to execute (required)
  - ...
```

## Metadata Section

```yaml
metadata:
  # === Required ===
  name: "Human Readable Flow Name"

  # === Recommended ===
  version: "1.0.0"                    # Semantic versioning
  author: "Your Name"
  description: "What this flow does"

  # === Optional (auto-generated if not provided) ===
  id: "lowercase-kebab-case-id"       # Auto-generated from name

  # === Model Recommendations ===
  recommended_models:
    default: "openai/gpt-4"           # Primary model
    compatible:                        # Alternatives
      - "meta-llama/Llama-3.3-70B-Instruct"
      - "anthropic/claude-3-opus"
    experimental:                      # Untested
      - "mistral/mistral-large"

  # === Dataset Requirements ===
  dataset_requirements:
    required_columns:                  # Must be present
      - "text"
      - "domain"
    optional_columns:                  # Nice to have
      - "metadata"
    min_samples: 1                     # Minimum rows
    max_samples: 10000                 # Maximum rows
    column_types:                      # Expected types
      text: "string"
      domain: "string"
    description: "Input dataset description"

  # === Categorization ===
  tags:
    - "qa-generation"
    - "summarization"
  license: "Apache-2.0"
```

## Parameters Section (Optional)

Define runtime-configurable parameters:

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

  model_name:
    type: "string"
    default: "gpt-4"
    description: "Model to use"
```

Use in blocks: `${parameter_name}` (parameter substitution).

## Blocks Section

```yaml
blocks:
  - block_type: "RegisteredBlockName"   # From BlockRegistry
    block_config:
      block_name: "unique_id"           # Required, unique within flow
      input_cols: ...                   # Input column spec
      output_cols: ...                  # Output column spec
      # ... block-specific params
```

### Column Specifications

#### Input Columns

```yaml
# String (single column)
input_cols: "text"

# List (multiple columns)
input_cols:
  - "text"
  - "context"

# Dict (rename on input)
input_cols:
  document: base_document    # Use base_document as "document"
  query: user_question       # Use user_question as "query"
```

#### Output Columns

```yaml
# String (single column)
output_cols: "response"

# List (multiple columns)
output_cols:
  - "question"
  - "answer"
```

## Complete Example

```yaml
metadata:
  name: "Document QA Generation"
  version: "1.0.0"
  author: "SDG Hub Contributors"
  description: "Generate question-answer pairs from documents"

  recommended_models:
    default: "openai/gpt-4"
    compatible:
      - "meta-llama/Llama-3.3-70B-Instruct"

  dataset_requirements:
    required_columns:
      - "document"
    optional_columns:
      - "domain"
    min_samples: 1
    description: "Documents to generate QA from"

  tags:
    - "qa-generation"
    - "document-processing"

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
      async_mode: true

  - block_type: "TextParserBlock"
    block_config:
      block_name: "parse_qa"
      input_cols: "qa_response"
      output_cols:
        - "question"
        - "response"
      pattern: "Question:\\s*(.+?)\\s*Answer:\\s*(.+)"
      flags: "DOTALL"

  - block_type: "ColumnValueFilterBlock"
    block_config:
      block_name: "filter_empty"
      input_cols: "question"
      filter_value: ["", null]
      operation: "ne"
```

## Validation Rules

The flow validator checks:

1. **Required structure:**
   - `blocks` list must exist and be non-empty
   - Each block must have `block_type` and `block_config`
   - Each `block_config` must have `block_name`

2. **Uniqueness:**
   - Block names must be unique within the flow

3. **Metadata:**
   - `name` is required
   - `id` must be lowercase if provided

4. **References:**
   - `prompt_config_path` files must exist (relative to flow.yaml)
