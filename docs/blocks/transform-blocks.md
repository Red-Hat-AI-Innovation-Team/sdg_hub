# Transform Blocks

Transform blocks handle data manipulation, reshaping, and column operations. These blocks provide essential data processing capabilities for preparing datasets, reformatting data structures, and performing common transformations.

## 🔄 Available Transform Blocks

### DuplicateColumnsBlock
Creates copies of existing columns with new names, useful for creating backup columns or preparing data for different processing paths.

### RenameColumnsBlock  
Renames existing columns to follow naming conventions or prepare data for downstream processing.

### TextConcatBlock
Concatenates text from multiple columns into a single column, with customizable separators and formatting.

### IndexBasedMapperBlock
Maps values based on their position/index, useful for applying transformations based on row order or position-dependent logic.

### MeltColumnsBlock
Reshapes data from wide format to long format, converting multiple columns into key-value pairs.

### SamplerBlock
Randomly samples values from list, set, or weighted dictionary columns. Use `num_samples=1` with `return_scalar=true` when downstream blocks expect a scalar value instead of a single-item list.

### UniformColumnValueSetter
Replaces all values in a column with a single statistical aggregate (mode, min, max, mean, or median) computed from the data. Modifies the column in-place, useful for data normalization, creating baseline comparisons, or extracting dominant values.

### JSONStructureBlock
Combines multiple columns into a single column containing a structured JSON object. Each input column becomes a field in the JSON output, using the column name as the field key.

**Configuration:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_cols` | `list[str]` | — | Columns to include as JSON fields |
| `output_cols` | `list[str]` | — | Single output column name (exactly one required) |
| `ensure_json_serializable` | `bool` | `true` | Convert non-serializable values to strings |
| `pretty_print` | `bool` | `false` | Format JSON with indentation |

**YAML Example:**

```yaml
- block_type: "JSONStructureBlock"
  block_config:
    block_name: "combine_insights"
    input_cols:
      - "summary"
      - "keywords"
      - "entities"
      - "sentiment"
    output_cols:
      - "structured_output"
    ensure_json_serializable: true
```

**Python Example:**

```python
from sdg_hub.core.blocks.transform import JSONStructureBlock
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

**Behavior Notes:**
- Missing input columns produce `null` values in the JSON output with a warning
- Non-serializable values are converted to strings when `ensure_json_serializable` is enabled
- Serialization errors return an empty JSON object (`{}`) rather than raising exceptions
- Supports nested structures including lists, dictionaries, and `None` values
- Handles unicode and emoji content correctly


## 🚀 Next Steps

- **[Filtering Blocks](filtering-blocks.md)** - Quality control and data validation
- **[LLM Blocks](llm-blocks.md)** - AI-powered text generation
- **[Flow Integration](../flows/overview.md)** - Combine transform blocks into complete pipelines