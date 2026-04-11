# Parsing Blocks

Parsing blocks extract structured data from text output, typically the text content produced by LLM blocks. This page covers four blocks: `TagParserBlock` for XML/HTML tag extraction, `RegexParserBlock` for regex pattern extraction, `JSONParserBlock` for JSON parsing and field expansion, and `TextParserBlock` (deprecated).

All parsing blocks operate on pandas DataFrames. They take a single input column of text and produce one or more output columns of extracted values.

---

## TagParserBlock

Parses text content using start/end tags. This is the recommended approach for extracting structured fields from LLM output that uses XML-style or custom delimiters.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `str \| list[str]` | required | Single input column containing text to parse |
| `output_cols` | `list[str]` | required | Output column names, one per tag pair |
| `start_tags` | `list[str]` | required | Start tags for extraction |
| `end_tags` | `list[str]` | required | End tags for extraction |
| `parser_cleanup_tags` | `Optional[list[str]]` | `None` | Tags to remove from extracted content |

The number of start/end tag pairs must equal the number of output columns. Exactly one input column is required. Both `start_tags` and `end_tags` accept a single string (auto-wrapped into a list) or a list of strings.

When the input column contains a list of strings instead of a single string, the block processes each list item and aggregates extracted values into lists.

### Python Example

```python
from sdg_hub.core.blocks import TagParserBlock
import pandas as pd

parser = TagParserBlock(
    block_name="extract_qa",
    input_cols="llm_content",
    output_cols=["question", "answer"],
    start_tags=["<question>", "<answer>"],
    end_tags=["</question>", "</answer>"],
)

dataset = pd.DataFrame({
    "llm_content": [
        "<question>What is Python?</question>\n<answer>A programming language.</answer>",
        "<question>What is AI?</question>\n<answer>Artificial intelligence.</answer>",
    ]
})

result = parser.generate(dataset)
# result["question"] -> ["What is Python?", "What is AI?"]
# result["answer"]   -> ["A programming language.", "Artificial intelligence."]
```

### Cleanup Tags

Remove unwanted markup from extracted content:

```python
from sdg_hub.core.blocks import TagParserBlock

parser = TagParserBlock(
    block_name="clean_extract",
    input_cols="llm_content",
    output_cols=["answer"],
    start_tags=["<answer>"],
    end_tags=["</answer>"],
    parser_cleanup_tags=["```", "**", "###"],
)
```

### Multiple Matches

When the text contains multiple occurrences of a tag pair, each match becomes a separate row in the output:

```python
from sdg_hub.core.blocks import TagParserBlock
import pandas as pd

parser = TagParserBlock(
    block_name="extract_items",
    input_cols="llm_content",
    output_cols=["item"],
    start_tags=["<item>"],
    end_tags=["</item>"],
)

dataset = pd.DataFrame({
    "llm_content": [
        "<item>First</item> <item>Second</item> <item>Third</item>"
    ]
})

result = parser.generate(dataset)
# Produces 3 rows, one for each <item>
```

### YAML Example

```yaml
blocks:
  - block_type: "TagParserBlock"
    block_config:
      block_name: "extract_qa"
      input_cols: "llm_content"
      output_cols:
        - "question"
        - "answer"
      start_tags:
        - "<question>"
        - "<answer>"
      end_tags:
        - "</question>"
        - "</answer>"
      parser_cleanup_tags:
        - "```"
```

---

## RegexParserBlock

Parses text content using regex patterns with capture groups. Use this when extraction patterns do not follow a simple tag structure.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `str \| list[str]` | required | Single input column containing text to parse |
| `output_cols` | `list[str]` | required | Output column names, one per capture group |
| `parsing_pattern` | `str` | required | Regex pattern with capture groups |
| `parser_cleanup_tags` | `Optional[list[str]]` | `None` | Tags to remove from extracted content |

Exactly one input column is required. The regex is applied with `re.DOTALL` so `.` matches newlines. If the pattern has multiple capture groups, each group maps to a corresponding output column. If only one capture group is used, only one output column is needed.

When the input column contains a list of strings, the block processes each item and aggregates results.

### Python Example

```python
from sdg_hub.core.blocks import RegexParserBlock
import pandas as pd

parser = RegexParserBlock(
    block_name="extract_answer",
    input_cols="llm_content",
    output_cols=["answer"],
    parsing_pattern=r"Answer:\s*(.+?)(?:\n|$)",
)

dataset = pd.DataFrame({
    "llm_content": [
        "Reasoning: AI is broad.\nAnswer: Artificial Intelligence is a field of CS.\n",
        "Let me explain.\nAnswer: Machine learning enables pattern recognition.\n",
    ]
})

result = parser.generate(dataset)
# result["answer"] -> ["Artificial Intelligence is a field of CS.",
#                       "Machine learning enables pattern recognition."]
```

### Multiple Capture Groups

```python
from sdg_hub.core.blocks import RegexParserBlock
import pandas as pd

parser = RegexParserBlock(
    block_name="extract_score_reason",
    input_cols="llm_content",
    output_cols=["score", "reason"],
    parsing_pattern=r"Score:\s*(\d+)\s*Reason:\s*(.+?)(?:\n|$)",
)

dataset = pd.DataFrame({
    "llm_content": [
        "Score: 8\nReason: Clear and accurate explanation.",
    ]
})

result = parser.generate(dataset)
# result["score"]  -> ["8"]
# result["reason"] -> ["Clear and accurate explanation."]
```

### YAML Example

```yaml
blocks:
  - block_type: "RegexParserBlock"
    block_config:
      block_name: "extract_answer"
      input_cols: "llm_content"
      output_cols:
        - "answer"
      parsing_pattern: "Answer:\\s*(.+?)(?:\\n|$)"
```

---

## JSONParserBlock

Parses JSON from text and expands fields into separate columns. Handles JSON embedded in surrounding text and fixes common LLM output issues such as trailing commas.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `str \| list[str]` | required | Single input column containing JSON text to parse |
| `output_cols` | `list[str]` | `[]` | Optional list of specific JSON fields to extract. If empty, all fields are extracted. |
| `field_prefix` | `str` | `""` | Prefix to add to extracted column names |
| `fix_trailing_commas` | `bool` | `True` | Whether to fix trailing commas in JSON (common LLM output issue) |
| `extract_embedded` | `bool` | `True` | Whether to extract JSON embedded in surrounding text |
| `drop_input` | `bool` | `False` | Whether to drop the input column after extraction |

Exactly one input column is required. When `extract_embedded=True`, the block finds JSON by locating the first `{` and last `}` in the text. JSON arrays are wrapped into `{"items": [...]}`. Non-dict/non-list JSON values are wrapped into `{"value": ...}`.

### Python Example

```python
from sdg_hub.core.blocks import JSONParserBlock
import pandas as pd

parser = JSONParserBlock(
    block_name="parse_json",
    input_cols="llm_content",
    output_cols=["topic", "summary"],
    drop_input=True,
)

dataset = pd.DataFrame({
    "llm_content": [
        'Here is the result: {"topic": "AI", "summary": "AI is transforming industries."}',
        '{"topic": "ML", "summary": "ML learns from data."}',
    ]
})

result = parser.generate(dataset)
# result["topic"]   -> ["AI", "ML"]
# result["summary"] -> ["AI is transforming industries.", "ML learns from data."]
# The "llm_content" column is dropped because drop_input=True
```

### Extract All Fields

When `output_cols` is empty (or not set), all JSON fields become columns:

```python
from sdg_hub.core.blocks import JSONParserBlock

parser = JSONParserBlock(
    block_name="parse_all",
    input_cols="llm_content",
    field_prefix="parsed_",
)

# If the JSON is {"name": "Alice", "age": 30}, the output has columns
# "parsed_name" and "parsed_age"
```

### Handling Embedded JSON

With `extract_embedded=True` (default), the block extracts JSON even when the LLM wraps it in explanatory text:

```python
from sdg_hub.core.blocks import JSONParserBlock
import pandas as pd

parser = JSONParserBlock(
    block_name="embedded_json",
    input_cols="llm_content",
    extract_embedded=True,
    fix_trailing_commas=True,
)

dataset = pd.DataFrame({
    "llm_content": [
        'Sure! Here is the data:\n{"key": "value",}\nHope this helps!',
    ]
})

result = parser.generate(dataset)
# The trailing comma is fixed, and the JSON is extracted from surrounding text
# result["key"] -> ["value"]
```

### YAML Example

```yaml
blocks:
  - block_type: "JSONParserBlock"
    block_config:
      block_name: "parse_json"
      input_cols: "llm_content"
      output_cols:
        - "topic"
        - "summary"
      field_prefix: ""
      fix_trailing_commas: true
      extract_embedded: true
      drop_input: true
```

---

## TextParserBlock

**DEPRECATED:** Use `TagParserBlock` or `RegexParserBlock` instead.

TextParserBlock combines tag-based and regex-based parsing into a single block. It is deprecated and emits a `DeprecationWarning` on instantiation. The block registry description reads: "DEPRECATED: Use TagParserBlock or RegexParserBlock".

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `str \| list[str]` | required | Single input column containing text to parse |
| `output_cols` | `list[str]` | required | Output column names for extracted content |
| `start_tags` | `list[str]` | `[]` | Start tags for tag-based extraction |
| `end_tags` | `list[str]` | `[]` | End tags for tag-based extraction |
| `parsing_pattern` | `Optional[str]` | `None` | Regex pattern with capture groups |
| `parser_cleanup_tags` | `Optional[list[str]]` | `None` | Tags to remove from extracted content |

Either `parsing_pattern` or `start_tags`/`end_tags` must be provided. If both are given, the regex pattern takes precedence.

### Migration Guide

Replace TextParserBlock with the appropriate specialized block:

**Tag-based parsing -- use TagParserBlock:**

```python
# Before (deprecated)
from sdg_hub.core.blocks import TextParserBlock

parser = TextParserBlock(
    block_name="old_parser",
    input_cols="text",
    output_cols=["answer"],
    start_tags=["<answer>"],
    end_tags=["</answer>"],
)

# After (recommended)
from sdg_hub.core.blocks import TagParserBlock

parser = TagParserBlock(
    block_name="new_parser",
    input_cols="text",
    output_cols=["answer"],
    start_tags=["<answer>"],
    end_tags=["</answer>"],
)
```

**Regex-based parsing -- use RegexParserBlock:**

```python
# Before (deprecated)
from sdg_hub.core.blocks import TextParserBlock

parser = TextParserBlock(
    block_name="old_parser",
    input_cols="text",
    output_cols=["answer"],
    parsing_pattern=r"Answer:\s*(.+?)(?:\n|$)",
)

# After (recommended)
from sdg_hub.core.blocks import RegexParserBlock

parser = RegexParserBlock(
    block_name="new_parser",
    input_cols="text",
    output_cols=["answer"],
    parsing_pattern=r"Answer:\s*(.+?)(?:\n|$)",
)
```

---

## Choosing a Parsing Block

| Block | Best For | Input Format |
|-------|----------|-------------|
| `TagParserBlock` | XML-style or custom tag delimiters (`<tag>...</tag>`) | Structured text with consistent delimiters |
| `RegexParserBlock` | Flexible patterns, key-value extraction, line-based formats | Text with identifiable patterns but no XML tags |
| `JSONParserBlock` | JSON output from LLMs, structured data extraction | Text containing JSON objects |
| `TextParserBlock` | Nothing -- deprecated | Use TagParserBlock or RegexParserBlock instead |

---

## Next Steps

- [LLM Blocks](llm-blocks.md) -- language model interaction and prompt building
- [Transform Blocks](transform-blocks.md) -- data manipulation and column operations
- [Filtering Blocks](filtering-blocks.md) -- quality control and row filtering
