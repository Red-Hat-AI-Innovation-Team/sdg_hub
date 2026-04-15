# YAML Reference

Every flow is defined in a single `flow.yaml` file. This page documents every
field, with types and defaults verified against source.

---

## Top-Level Structure

A flow YAML has three top-level sections: `metadata`, `parameters` (optional),
and `blocks`.

```yaml
metadata:
  # Flow identity and configuration (see below)

parameters:
  # Optional runtime parameter definitions (see below)

blocks:
  # Ordered list of block definitions (see below)
```

The `FlowValidator` (source: `src/sdg_hub/core/flow/validation.py`) enforces:

- `blocks` is required and must be a non-empty list.
- Each block must have `block_type` and `block_config` keys.
- Each `block_config` must contain `block_name`.
- `metadata`, if present, must be a dict with a non-empty `name` string.

---

## metadata Section

Source: `src/sdg_hub/core/flow/metadata.py` -- class `FlowMetadata`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | `str` | Yes | -- | Human-readable flow name. Minimum 1 character. |
| `id` | `str` | No | auto-generated | Unique identifier. Must be lowercase, alphanumeric with hyphens, no leading/trailing hyphens. Auto-generated from `name` if omitted. |
| `description` | `str` | No | `""` | What the flow does. |
| `version` | `str` | No | `"1.0.0"` | Semantic version matching `^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$`. |
| `author` | `str` | No | `""` | Author or contributor name. |
| `license` | `str` | No | `"Apache-2.0"` | License identifier. |
| `tags` | `list[str]` | No | `[]` | Tags for categorization. Automatically lowercased. |
| `recommended_models` | `RecommendedModels` | No | `None` | Model recommendations (see below). |
| `dataset_requirements` | `DatasetRequirements` | No | `None` | Input dataset validation rules (see below). |
| `output_columns` | `list[str]` | No | `None` | Columns to keep in final output. Original input columns are always preserved. When set, intermediate columns are dropped during and after execution. Must be non-empty if specified; omit entirely to keep all columns. |

### recommended_models

Source: `src/sdg_hub/core/flow/metadata.py` -- class `RecommendedModels`

```yaml
recommended_models:
  default: "openai/gpt-oss-120b"
  compatible:
    - "meta-llama/Llama-3.3-70B-Instruct"
    - "microsoft/phi-4"
  experimental:
    - "gpt-4o"
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `default` | `str` | Yes | -- | Primary recommended model. Cannot be empty. |
| `compatible` | `list[str]` | No | `[]` | Models known to work well. |
| `experimental` | `list[str]` | No | `[]` | Models not extensively tested. |

Model selection priority: `default` first, then `compatible` in order, then
`experimental` in order.

### dataset_requirements

Source: `src/sdg_hub/core/flow/metadata.py` -- class `DatasetRequirements`

```yaml
dataset_requirements:
  required_columns:
    - "document"
    - "document_outline"
  optional_columns:
    - "metadata"
  min_samples: 1
  max_samples: 10000
  column_types:
    document: "string"
  description: "Input documents for processing"
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `required_columns` | `list[str]` | No | `[]` | Columns that must be present. Flow fails if missing. |
| `optional_columns` | `list[str]` | No | `[]` | Columns that can enhance performance. |
| `min_samples` | `int` | No | `1` | Minimum row count. Must be >= 1. |
| `max_samples` | `int` | No | `None` | Maximum row count. Must be >= `min_samples` if set. |
| `column_types` | `dict[str, str]` | No | `{}` | Expected types for columns (documentation only). |
| `description` | `str` | No | `""` | Human-readable description of requirements. |

---

## parameters Section

Optional. Defines flow-level parameters that can be overridden at runtime. Each
parameter is a key-value pair where the key is the parameter name.

```yaml
parameters:
  temperature:
    type: "float"
    default: 0.7
    description: "Sampling temperature for generation"
    required: false

  max_tokens:
    type: "integer"
    default: 2048
    description: "Maximum token count"
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `default` | any | Yes | Default value for the parameter. |
| `type` | `str` | No | Type hint (e.g., "string", "float", "integer"). |
| `description` | `str` | No | Human-readable description. |
| `required` | `bool` | No | Whether the parameter must be provided at runtime. |

---

## blocks Section

An ordered list of block definitions. Each entry specifies a block type and its
configuration.

```yaml
blocks:
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_summary_prompt"
      input_cols:
        - "text"
      output_cols: "summary_prompt"
      prompt_config_path: "prompts/summarize.yaml"

  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate_summary"
      input_cols: "summary_prompt"
      output_cols: "raw_summary"
      max_tokens: 1024
      temperature: 0.3
      async_mode: true
```

### Block entry fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `block_type` | `str` | Yes | Class name of the block (must exist in BlockRegistry). Valid values: `AgentBlock`, `AgentResponseExtractorBlock`, `ColumnValueFilterBlock`, `DuplicateColumnsBlock`, `IndexBasedMapperBlock`, `JSONParserBlock`, `JSONStructureBlock`, `LLMChatBlock`, `LLMResponseExtractorBlock`, `MCPAgentBlock`, `MeltColumnsBlock`, `PromptBuilderBlock`, `RegexParserBlock`, `RenameColumnsBlock`, `RowMultiplierBlock`, `SamplerBlock`, `TagParserBlock`, `TextConcatBlock`, `TextParserBlock` (deprecated), `UniformColumnValueSetter`. |
| `block_config` | `dict` | Yes | Configuration passed to the block constructor. |
| `block_config.block_name` | `str` | Yes | Unique name within the flow. |

The `block_config` contents depend on the block type. Common fields include
`input_cols`, `output_cols`, and block-specific parameters. See the
[Block documentation](../blocks/index.md) for per-block configuration.

### Path resolution

Path fields in `block_config` (`config_path`, `config_paths`,
`prompt_config_path`) are resolved relative to the directory containing the
`flow.yaml` file. For example, if `flow.yaml` is at
`flows/my_flow/flow.yaml` and a block specifies
`prompt_config_path: prompts/summary.yaml`, it resolves to
`flows/my_flow/prompts/summary.yaml`.

---

## Complete Annotated Example

This is the Structured Text Insights Extraction Flow, taken from
`src/sdg_hub/flows/text_analysis/structured_insights/flow.yaml`:

```yaml
metadata:
  id: green-clay-812
  name: "Structured Text Insights Extraction Flow"
  description: >-
    Multi-step pipeline for extracting structured insights from text including
    summary, keywords, entities, and sentiment analysis combined into a JSON output
  version: "1.0.0"
  author: "SDG Hub Contributors"
  recommended_models:
    default: "openai/gpt-oss-120b"
    compatible:
      - "meta-llama/Llama-3.3-70B-Instruct"
      - "microsoft/phi-4"
      - "mistralai/Mixtral-8x7B-Instruct-v0.1"
    experimental:
      - "gpt-4o"
  tags:
    - "text-analysis"
    - "summarization"
    - "nlp"
    - "structured-output"
    - "insights"
    - "sentiment-analysis"
    - "entity-extraction"
    - "keyword-extraction"
  license: "Apache-2.0"
  dataset_requirements:
    required_columns:
      - "text"
    description: >-
      Input dataset should contain text content for analysis. Each text should be
      substantial enough for meaningful analysis (minimum 50 words recommended).

blocks:
  # Step 1: Build a prompt for summary extraction
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_summary_prompt"
      input_cols:
        - "text"
      output_cols: "summary_prompt"
      prompt_config_path: "prompts/summarize.yaml"

  # Step 2: Generate the summary via LLM
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate_summary"
      input_cols: "summary_prompt"
      output_cols: "raw_summary"
      max_tokens: 1024
      temperature: 0.3
      async_mode: true

  # Step 3: Extract the assistant message content
  - block_type: "LLMResponseExtractorBlock"
    block_config:
      block_name: "extract_summary"
      input_cols: "raw_summary"
      extract_content: true
      expand_lists: true

  # Step 4: Parse the summary from tagged output
  - block_type: "TagParserBlock"
    block_config:
      block_name: "parse_summary"
      input_cols: "extract_summary_content"
      output_cols: "summary"
      start_tags:
        - "[SUMMARY]"
      end_tags:
        - "[/SUMMARY]"

  # Steps 5-8: Same pattern for keywords extraction
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_keywords_prompt"
      input_cols:
        - "text"
      output_cols: "keywords_prompt"
      prompt_config_path: "prompts/extract_keywords.yaml"
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate_keywords"
      input_cols: "keywords_prompt"
      output_cols: "raw_keywords"
      max_tokens: 512
      temperature: 0.3
      async_mode: true
  - block_type: "LLMResponseExtractorBlock"
    block_config:
      block_name: "extract_keywords"
      input_cols: "raw_keywords"
      extract_content: true
      expand_lists: true
  - block_type: "TagParserBlock"
    block_config:
      block_name: "parse_keywords"
      input_cols: "extract_keywords_content"
      output_cols: "keywords"
      start_tags:
        - "[KEYWORDS]"
      end_tags:
        - "[/KEYWORDS]"

  # Steps 9-12: Entities extraction (same pattern)
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_entities_prompt"
      input_cols:
        - "text"
      output_cols: "entities_prompt"
      prompt_config_path: "prompts/extract_entities.yaml"
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate_entities"
      input_cols: "entities_prompt"
      output_cols: "raw_entities"
      max_tokens: 1024
      temperature: 0.3
      async_mode: true
  - block_type: "LLMResponseExtractorBlock"
    block_config:
      block_name: "extract_entities"
      input_cols: "raw_entities"
      extract_content: true
      expand_lists: true
  - block_type: "TagParserBlock"
    block_config:
      block_name: "parse_entities"
      input_cols: "extract_entities_content"
      output_cols: "entities"
      start_tags:
        - "[ENTITIES]"
      end_tags:
        - "[/ENTITIES]"

  # Steps 13-16: Sentiment analysis (same pattern)
  - block_type: "PromptBuilderBlock"
    block_config:
      block_name: "build_sentiment_prompt"
      input_cols:
        - "text"
      output_cols: "sentiment_prompt"
      prompt_config_path: "prompts/analyze_sentiment.yaml"
  - block_type: "LLMChatBlock"
    block_config:
      block_name: "generate_sentiment"
      input_cols: "sentiment_prompt"
      output_cols: "raw_sentiment"
      max_tokens: 256
      temperature: 0.1
      async_mode: true
  - block_type: "LLMResponseExtractorBlock"
    block_config:
      block_name: "extract_sentiment"
      input_cols: "raw_sentiment"
      extract_content: true
      expand_lists: true
  - block_type: "TagParserBlock"
    block_config:
      block_name: "parse_sentiment"
      input_cols: "extract_sentiment_content"
      output_cols: "sentiment"
      start_tags:
        - "[SENTIMENT]"
      end_tags:
        - "[/SENTIMENT]"

  # Step 17: Combine all analyses into a JSON structure
  - block_type: "JSONStructureBlock"
    block_config:
      block_name: "create_structured_insights"
      input_cols:
        - "summary"
        - "keywords"
        - "entities"
        - "sentiment"
      output_cols:
        - "structured_insights"
      ensure_json_serializable: true
```

This flow demonstrates the common pattern: `PromptBuilderBlock` builds messages,
`LLMChatBlock` generates, `LLMResponseExtractorBlock` extracts the assistant
content, and `TagParserBlock` parses tagged regions.
