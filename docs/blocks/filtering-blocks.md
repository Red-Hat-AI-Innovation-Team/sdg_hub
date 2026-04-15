# Filtering Blocks

Filtering blocks remove rows from a dataset based on column values and comparison operations. They do not add new columns -- they reduce the row count by applying filter conditions.

---

## ColumnValueFilterBlock

Filters dataset rows by comparing a column's values against one or more filter values using a specified comparison operation. Supports optional type conversion before filtering.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `list[str]` | required | At least one column. The first column is used for filtering. |
| `filter_value` | `Any` or `list[Any]` | required | Value(s) to filter by. A single value is internally wrapped in a list. |
| `operation` | `str` | required | Comparison operation (see table below) |
| `convert_dtype` | `str` or `null` | `null` | Convert the filter column before comparison: `"float"` or `"int"` |

### Supported Operations

| Operation | Description | Example |
|-----------|-------------|---------|
| `"eq"` | Equal to | `score == 5` |
| `"ne"` | Not equal to | `status != "draft"` |
| `"lt"` | Less than | `score < 3` |
| `"le"` | Less than or equal to | `score <= 3` |
| `"gt"` | Greater than | `score > 7` |
| `"ge"` | Greater than or equal to | `score >= 7` |
| `"contains"` | Value contains the filter value | `text.contains("error")` |
| `"in"` | Value is a member of the filter value | `category in ["A", "B"]` |

When `filter_value` is a list, a row passes if **any** filter value matches the operation.

### Python Example -- Equality Filter

```python
from sdg_hub.core.blocks import ColumnValueFilterBlock
import pandas as pd

block = ColumnValueFilterBlock(
    block_name="keep_approved",
    input_cols=["status"],
    filter_value="approved",
    operation="eq",
)

dataset = pd.DataFrame({
    "status": ["approved", "draft", "approved", "rejected"],
    "content": ["Good text", "WIP", "Another good", "Bad text"],
})

result = block(dataset)
print(result["status"].tolist())
# Output: ['approved', 'approved']
```

### Python Example -- Numeric Comparison with Type Conversion

```python
from sdg_hub.core.blocks import ColumnValueFilterBlock
import pandas as pd

block = ColumnValueFilterBlock(
    block_name="high_quality",
    input_cols=["score"],
    filter_value=7,
    operation="ge",
    convert_dtype="float",
)

dataset = pd.DataFrame({
    "text": ["Great answer", "OK answer", "Poor answer", "Excellent"],
    "score": ["9.2", "5.1", "3.0", "8.5"],
})

result = block(dataset)
print(result["text"].tolist())
# Output: ['Great answer', 'Excellent']
```

### Python Example -- Contains

```python
from sdg_hub.core.blocks import ColumnValueFilterBlock
import pandas as pd

block = ColumnValueFilterBlock(
    block_name="find_python",
    input_cols=["description"],
    filter_value="Python",
    operation="contains",
)

dataset = pd.DataFrame({
    "description": [
        "Python web framework",
        "Java enterprise app",
        "Python data science",
    ],
})

result = block(dataset)
print(len(result))
# Output: 2
```

### Python Example -- Membership (in)

```python
from sdg_hub.core.blocks import ColumnValueFilterBlock
import pandas as pd

block = ColumnValueFilterBlock(
    block_name="select_categories",
    input_cols=["category"],
    filter_value=["science", "technology"],
    operation="in",
)

dataset = pd.DataFrame({
    "category": [["science", "math"], ["art"], ["technology", "science"]],
    "title": ["Physics paper", "Gallery exhibit", "AI research"],
})

result = block(dataset)
print(result["title"].tolist())
# Output: ['Physics paper', 'AI research']
```

### YAML Example

```yaml
- block_type: "ColumnValueFilterBlock"
  block_config:
    block_name: "high_quality"
    input_cols:
      - "score"
    filter_value: 7
    operation: "ge"
    convert_dtype: "float"
```

### YAML Example -- Multiple Filter Values

```yaml
- block_type: "ColumnValueFilterBlock"
  block_config:
    block_name: "keep_selected_topics"
    input_cols:
      - "topic"
    filter_value:
      - "machine_learning"
      - "deep_learning"
      - "nlp"
    operation: "eq"
```

Behavior notes:

- Rows with `None`/`NaN` values in the filter column are always removed.
- When `convert_dtype` is set, values that fail conversion are set to `None` and filtered out.
- Multiple `filter_value` entries use OR logic: a row passes if it matches any value.
