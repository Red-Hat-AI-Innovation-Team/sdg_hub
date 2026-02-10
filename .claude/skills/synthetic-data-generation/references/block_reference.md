# Block Reference

Available blocks and their configurations.

## LLM Blocks

### LLMChatBlock

Call LLM APIs (100+ providers via LiteLLM).

```yaml
- block_type: "LLMChatBlock"
  block_config:
    block_name: "generate"
    input_cols: "messages"           # Column with chat messages
    output_cols: "response"          # Output column name

    # Generation params
    temperature: 0.7
    max_tokens: 1024
    top_p: 1.0
    n: 1

    # Operational params (can be set here or via set_model_config at runtime)
    # model: "openai/gpt-4"
    # api_key: "..."
    # api_base: "http://localhost:8000/v1"
    async_mode: true
```

**Input format:** Column must contain list of message dicts:
```python
[
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
]
```

### PromptBuilderBlock

Build chat messages from templates.

```yaml
- block_type: "PromptBuilderBlock"
  block_config:
    block_name: "build_prompt"
    input_cols: ["document", "query"]  # Variables for template
    output_cols: "messages"            # Output: list of messages
    prompt_config_path: "prompt.yaml"  # Relative to flow.yaml
```

**Prompt template format:**
```yaml
# prompt.yaml
system: |
  You are a helpful assistant.

user: |
  Document: {document}
  Query: {query}
  Please answer the query based on the document.
```

### LLMParserBlock

Parse LLM responses (extract JSON, fields).

```yaml
- block_type: "LLMParserBlock"
  block_config:
    block_name: "parse_response"
    input_cols: "raw_response"
    output_cols: "parsed"
    extract_content: true          # Extract from message structure
    field_prefix: "result_"        # Prefix for extracted fields
```

### TextParserBlock

Parse text with regex patterns.

```yaml
- block_type: "TextParserBlock"
  block_config:
    block_name: "parse_qa"
    input_cols: "response"
    output_cols:
      - "question"
      - "answer"
    pattern: "Question:\\s*(.+?)\\s*Answer:\\s*(.+)"
    flags: "DOTALL"              # DOTALL, IGNORECASE, MULTILINE
```

## Transform Blocks

### TextConcatBlock

Concatenate multiple columns.

```yaml
- block_type: "TextConcatBlock"
  block_config:
    block_name: "concat"
    input_cols: ["title", "body"]
    output_cols: "full_text"
    separator: "\n\n"
```

### RenameColumnsBlock

Rename columns.

```yaml
- block_type: "RenameColumnsBlock"
  block_config:
    block_name: "rename"
    input_cols:
      question: generated_question
      response: generated_answer
    # No output_cols needed
```

### DuplicateColumnsBlock

Copy columns.

```yaml
- block_type: "DuplicateColumnsBlock"
  block_config:
    block_name: "duplicate"
    input_cols: "document"
    output_cols: "base_document"
```

### MeltColumnsBlock

Reshape wide to long format.

```yaml
- block_type: "MeltColumnsBlock"
  block_config:
    block_name: "melt"
    input_cols:
      - "summary_a"
      - "summary_b"
      - "summary_c"
    output_cols: "summary"         # New column for values
    id_vars:                       # Columns to keep
      - "document"
      - "domain"
```

Before:
| document | summary_a | summary_b |
|----------|-----------|-----------|
| doc1     | sum_a1    | sum_b1    |

After:
| document | summary |
|----------|---------|
| doc1     | sum_a1  |
| doc1     | sum_b1  |

### JSONStructureBlock

Build JSON structures from columns.

```yaml
- block_type: "JSONStructureBlock"
  block_config:
    block_name: "build_json"
    input_cols: ["question", "answer"]
    output_cols: "qa_pair"
    structure:
      q: "{question}"
      a: "{answer}"
```

### UniformColValSetterBlock

Set column to constant value.

```yaml
- block_type: "UniformColValSetterBlock"
  block_config:
    block_name: "set_source"
    output_cols: "source"
    value: "generated"
```

### IndexBasedMapperBlock

Map values by index.

```yaml
- block_type: "IndexBasedMapperBlock"
  block_config:
    block_name: "map_labels"
    input_cols: "label_idx"
    output_cols: "label"
    mapping:
      0: "negative"
      1: "neutral"
      2: "positive"
```

## Filtering Blocks

### ColumnValueFilterBlock

Filter rows by column values.

```yaml
- block_type: "ColumnValueFilterBlock"
  block_config:
    block_name: "filter"
    input_cols: "score"
    filter_value: [3, 4, 5]        # Values to keep
    operation: "in"                # eq, ne, lt, gt, le, ge, in, contains
    convert_dtype: "int"           # Optional: float, int, str
```

**Operations:**
- `eq`: equals
- `ne`: not equals
- `lt`, `le`: less than, less or equal
- `gt`, `ge`: greater than, greater or equal
- `in`: value in list
- `contains`: list contains value

## Evaluation Blocks

### FaithfulnessEvalBlock

Evaluate response faithfulness to source.

```yaml
- block_type: "FaithfulnessEvalBlock"
  block_config:
    block_name: "eval_faithfulness"
    input_cols:
      - "response"
      - "source"
    output_cols: "faithfulness_score"
```

### RelevancyEvalBlock

Evaluate response relevancy to query.

```yaml
- block_type: "RelevancyEvalBlock"
  block_config:
    block_name: "eval_relevancy"
    input_cols:
      - "query"
      - "response"
    output_cols: "relevancy_score"
```

## Discovering Blocks

```python
# play.py - List available blocks
from sdg_hub.core.blocks import BlockRegistry

# All blocks
BlockRegistry.discover_blocks()

# By category
print(BlockRegistry.list_blocks(category="transform"))
print(BlockRegistry.list_blocks(category="llm"))
print(BlockRegistry.list_blocks(category="filtering"))

# Grouped
print(BlockRegistry.list_blocks(grouped=True))
```
