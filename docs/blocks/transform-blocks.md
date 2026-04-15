# Transform Blocks

Transform blocks handle data manipulation, reshaping, and column operations. They operate on pandas DataFrames and do not call any external services. All transform blocks inherit common fields from `BaseBlock`: `block_name`, `input_cols`, and `output_cols`.

---

## TextConcatBlock

Concatenates values from multiple columns into a single output column using a configurable separator.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `list[str]` | required | Column names to concatenate (must be non-empty) |
| `output_cols` | `list[str]` | required | Single output column name (exactly one required) |
| `separator` | `str` | `"\n\n"` | String inserted between concatenated values |

### Python Example

```python
from sdg_hub.core.blocks import TextConcatBlock
import pandas as pd

block = TextConcatBlock(
    block_name="merge_context",
    input_cols=["title", "body"],
    output_cols=["full_text"],
    separator=" -- ",
)

dataset = pd.DataFrame({
    "title": ["Introduction to ML"],
    "body": ["Machine learning is a branch of AI."],
})

result = block(dataset)
print(result["full_text"].iloc[0])
# Output: "Introduction to ML -- Machine learning is a branch of AI."
```

### YAML Example

```yaml
- block_type: "TextConcatBlock"
  block_config:
    block_name: "merge_context"
    input_cols:
      - "title"
      - "body"
    output_cols:
      - "full_text"
    separator: "\n\n"
```

---

## DuplicateColumnsBlock

Creates copies of existing columns with new names according to a mapping provided through `input_cols` as a dictionary.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `dict[str, str]` | required | Mapping of existing column names to new column names |
| `output_cols` | `list[str]` | auto-derived | Defaults to the values from `input_cols` if not provided |

### Python Example

```python
from sdg_hub.core.blocks import DuplicateColumnsBlock
import pandas as pd

block = DuplicateColumnsBlock(
    block_name="backup_cols",
    input_cols={"question": "question_backup", "answer": "answer_backup"},
)

dataset = pd.DataFrame({
    "question": ["What is AI?"],
    "answer": ["Artificial Intelligence"],
})

result = block(dataset)
print(result.columns.tolist())
# Output: ['question', 'answer', 'question_backup', 'answer_backup']
```

### YAML Example

```yaml
- block_type: "DuplicateColumnsBlock"
  block_config:
    block_name: "backup_cols"
    input_cols:
      question: "question_backup"
      answer: "answer_backup"
```

---

## RenameColumnsBlock

Renames columns in a dataset according to a mapping provided through `input_cols` as a dictionary. Does not support chained or circular renames -- target names must not already exist in the dataset.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `dict[str, str]` | required | Mapping of existing column names to new column names |
| `output_cols` | `list[str]` | auto-derived | Defaults to the values from `input_cols` if not provided |

### Python Example

```python
from sdg_hub.core.blocks import RenameColumnsBlock
import pandas as pd

block = RenameColumnsBlock(
    block_name="standardize_names",
    input_cols={"q": "question", "a": "answer"},
)

dataset = pd.DataFrame({
    "q": ["What is ML?"],
    "a": ["Machine Learning"],
})

result = block(dataset)
print(result.columns.tolist())
# Output: ['question', 'answer']
```

### YAML Example

```yaml
- block_type: "RenameColumnsBlock"
  block_config:
    block_name: "standardize_names"
    input_cols:
      q: "question"
      a: "answer"
```

---

## MeltColumnsBlock

Transforms a wide-format dataset into long format by melting specified columns into rows. Columns not listed in `input_cols` are preserved as ID columns.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `list[str]` | required | Columns to melt into rows (variable columns) |
| `output_cols` | `list[str]` | required | Exactly two columns: `[value_column, variable_column]` |

### Python Example

```python
from sdg_hub.core.blocks import MeltColumnsBlock
import pandas as pd

block = MeltColumnsBlock(
    block_name="flatten_scores",
    input_cols=["math_score", "science_score"],
    output_cols=["score", "subject"],
)

dataset = pd.DataFrame({
    "student": ["Alice", "Bob"],
    "math_score": [95, 82],
    "science_score": [88, 91],
})

result = block(dataset)
print(result)
# Output:
#   student  score        subject
# 0   Alice     95     math_score
# 1     Bob     82     math_score
# 2   Alice     88  science_score
# 3     Bob     91  science_score
```

### YAML Example

```yaml
- block_type: "MeltColumnsBlock"
  block_config:
    block_name: "flatten_scores"
    input_cols:
      - "math_score"
      - "science_score"
    output_cols:
      - "score"
      - "subject"
```

---

## RowMultiplierBlock

Duplicates each row in the dataset a configurable number of times. Primary use case is expanding seed data before LLM processing.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `num_samples` | `int` | required | Number of times to duplicate each row (must be >= 1) |
| `shuffle` | `bool` | `false` | Whether to shuffle output rows after duplication |
| `random_seed` | `int` or `null` | `null` | Seed for reproducible shuffling |

### Python Example

```python
from sdg_hub.core.blocks import RowMultiplierBlock
import pandas as pd

block = RowMultiplierBlock(
    block_name="expand_seeds",
    num_samples=3,
    shuffle=True,
    random_seed=42,
)

dataset = pd.DataFrame({
    "topic": ["AI", "ML"],
    "difficulty": ["easy", "hard"],
})

result = block(dataset)
print(len(result))
# Output: 6 (2 rows * 3 duplicates)
```

### YAML Example

```yaml
- block_type: "RowMultiplierBlock"
  block_config:
    block_name: "expand_seeds"
    num_samples: 3
    shuffle: true
    random_seed: 42
```

---

## IndexBasedMapperBlock

Maps values from source columns to output columns based on a choice column's value and a shared mapping dictionary. The `choice_cols` and `output_cols` must have the same length -- `choice_cols[i]` determines the value for `output_cols[i]`.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `list[str]` | required | All columns involved (choice columns and mapped columns) |
| `output_cols` | `list[str]` | required | Output column names (must match length of `choice_cols`) |
| `choice_map` | `dict[str, str]` | required | Maps choice values to source column names |
| `choice_cols` | `list[str]` | required | Columns containing the choice values |

### Python Example

```python
from sdg_hub.core.blocks import IndexBasedMapperBlock
import pandas as pd

block = IndexBasedMapperBlock(
    block_name="select_answer",
    input_cols=["chosen", "response_a", "response_b"],
    output_cols=["selected_response"],
    choice_map={"A": "response_a", "B": "response_b"},
    choice_cols=["chosen"],
)

dataset = pd.DataFrame({
    "chosen": ["A", "B", "A"],
    "response_a": ["Answer from A1", "Answer from A2", "Answer from A3"],
    "response_b": ["Answer from B1", "Answer from B2", "Answer from B3"],
})

result = block(dataset)
print(result["selected_response"].tolist())
# Output: ['Answer from A1', 'Answer from B2', 'Answer from A3']
```

### YAML Example

```yaml
- block_type: "IndexBasedMapperBlock"
  block_config:
    block_name: "select_answer"
    input_cols:
      - "chosen"
      - "response_a"
      - "response_b"
    output_cols:
      - "selected_response"
    choice_map:
      A: "response_a"
      B: "response_b"
    choice_cols:
      - "chosen"
```

---

## SamplerBlock

Randomly samples a specified number of values from a list, set, or weighted dictionary column and outputs the sampled values to a new column.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `list[str]` | required | Single input column containing lists/sets/dicts to sample from |
| `output_cols` | `list[str]` | required | Single output column for sampled values |
| `num_samples` | `int` | `5` | Number of values to sample from each list |
| `random_seed` | `int` or `null` | `null` | Seed for reproducible sampling |
| `return_scalar` | `bool` | `false` | When `num_samples=1`, return a scalar instead of a single-element list |

When `input_cols` contains dictionaries, values are treated as weights for weighted sampling (without replacement).

### Python Example

```python
from sdg_hub.core.blocks import SamplerBlock
import pandas as pd

block = SamplerBlock(
    block_name="sample_keywords",
    input_cols=["all_keywords"],
    output_cols=["sampled_keywords"],
    num_samples=2,
    random_seed=42,
)

dataset = pd.DataFrame({
    "all_keywords": [
        ["python", "java", "rust", "go", "typescript"],
        ["react", "vue", "angular", "svelte"],
    ],
})

result = block(dataset)
print(result["sampled_keywords"].tolist())
# Output: Two randomly sampled keywords per row
```

### YAML Example

```yaml
- block_type: "SamplerBlock"
  block_config:
    block_name: "sample_keywords"
    input_cols:
      - "all_keywords"
    output_cols:
      - "sampled_keywords"
    num_samples: 2
    random_seed: 42
    return_scalar: false
```

---

## UniformColumnValueSetter

Replaces all values in a column with a single summary statistic computed from the column data. Modifies the column in-place and ignores any specified `output_cols`.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `list[str]` | required | Exactly one column to apply the reduction to |
| `reduction_strategy` | `"mode"` `"min"` `"max"` `"mean"` `"median"` | `"mode"` | Strategy for computing the replacement value |

### Python Example

```python
from sdg_hub.core.blocks import UniformColumnValueSetter
import pandas as pd

block = UniformColumnValueSetter(
    block_name="normalize_difficulty",
    input_cols=["difficulty"],
    reduction_strategy="mode",
)

dataset = pd.DataFrame({
    "question": ["Q1", "Q2", "Q3", "Q4"],
    "difficulty": ["easy", "hard", "easy", "easy"],
})

result = block(dataset)
print(result["difficulty"].tolist())
# Output: ['easy', 'easy', 'easy', 'easy']
```

### YAML Example

```yaml
- block_type: "UniformColumnValueSetter"
  block_config:
    block_name: "normalize_difficulty"
    input_cols:
      - "difficulty"
    reduction_strategy: "mode"
```

---

## JSONStructureBlock

Combines multiple columns into a single column containing a structured JSON object. Each input column name becomes a field key in the output JSON.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `list[str]` | required | Columns to include as JSON fields |
| `output_cols` | `list[str]` | required | Single output column name (exactly one required) |
| `ensure_json_serializable` | `bool` | `true` | Convert non-serializable values to strings |
| `pretty_print` | `bool` | `false` | Format JSON output with indentation |

### Python Example

```python
from sdg_hub.core.blocks import JSONStructureBlock
import pandas as pd

block = JSONStructureBlock(
    block_name="combine_results",
    input_cols=["summary", "keywords", "sentiment"],
    output_cols=["result_json"],
    pretty_print=True,
)

dataset = pd.DataFrame({
    "summary": ["AI adoption is accelerating across industries."],
    "keywords": [["AI", "adoption", "industry"]],
    "sentiment": ["positive"],
})

result = block(dataset)
print(result["result_json"].iloc[0])
```

Output:

```json
{
  "summary": "AI adoption is accelerating across industries.",
  "keywords": ["AI", "adoption", "industry"],
  "sentiment": "positive"
}
```

### YAML Example

```yaml
- block_type: "JSONStructureBlock"
  block_config:
    block_name: "combine_results"
    input_cols:
      - "summary"
      - "keywords"
      - "sentiment"
    output_cols:
      - "result_json"
    ensure_json_serializable: true
    pretty_print: true
```

Behavior notes:

- Missing input columns produce `null` values in the JSON output with a warning.
- Non-serializable values are converted to strings when `ensure_json_serializable` is enabled.
- Serialization errors return an empty JSON object (`{}`) rather than raising exceptions.
- Supports nested structures including lists, dictionaries, and `None` values.
