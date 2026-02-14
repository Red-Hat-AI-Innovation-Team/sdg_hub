# User Guide: Dataset Configuration

This guide covers loading and configuring datasets for data generation.

## Dataset Sources

SDG Hub UI supports multiple ways to provide seed data:

| Source | Description |
|--------|-------------|
| **Upload New File** | Upload a dataset file (JSONL, JSON, CSV, Parquet) |
| **Preprocessed PDF Data** | Use datasets created from the PDF preprocessing pipeline |
| **Existing Upload** | Reuse previously uploaded files from the uploads directory |

## Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| **JSONL** | `.jsonl` | JSON Lines — one JSON object per line |
| **JSON** | `.json` | JSON array of objects |
| **CSV** | `.csv` | Comma-separated values |
| **Parquet** | `.parquet`, `.pq` | Apache Parquet columnar format |

### Format Examples

**JSONL (Recommended):**

```jsonl
{"document": "First document text...", "domain": "science"}
{"document": "Second document text...", "domain": "history"}
{"document": "Third document text...", "domain": "technology"}
```

**JSON:**

```json
[
  {"document": "First document text...", "domain": "science"},
  {"document": "Second document text...", "domain": "history"},
  {"document": "Third document text...", "domain": "technology"}
]
```

**CSV:**

```csv
document,domain
"First document text...","science"
"Second document text...","history"
"Third document text...","technology"
```

## Uploading Files

### Drag and Drop

1. Click the upload area or drag a file onto it
2. Wait for upload confirmation
3. File appears in the "Uploaded Files" list

### File Size Limits

Default maximum: **512 MB**

Configure via environment variable:

```bash
export SDG_HUB_MAX_UPLOAD_MB=1024  # 1 GB
```

### Allowed Directories

By default, only files in `backend/uploads/` are accessible.

Add additional directories:

```bash
export SDG_HUB_ALLOWED_DATA_DIRS="/data/datasets:/shared/data"
```

## Using Preprocessed PDF Data

If you've created datasets through the PDF preprocessing pipeline:

1. In the Dataset Configuration step, select **Use Preprocessed Dataset**
2. Browse available preprocessed datasets (listed from `/api/preprocessing/datasets`)
3. Select a dataset — it will be automatically loaded
4. Proceed with the standard dataset settings (samples, shuffle, etc.)

Preprocessed datasets are JSONL files created from chunked PDF/Markdown documents, typically containing columns like `document`, `domain`, `document_outline`, and ICL template fields.

See [PDF Preprocessing](pdf-preprocessing.md) for details on creating these datasets.

## Dataset Settings

### Number of Samples

Limit rows to process:

| Setting | Behavior |
|---------|----------|
| Empty | Process all rows |
| 10 | Process first 10 rows |
| 100 | Process first 100 rows |

**Use cases:**

- **Testing:** 5-10 samples
- **Dry run:** 2-5 samples
- **Full run:** Leave empty or set to total

### Shuffle

Randomize row order before processing:

| Setting | Behavior |
|---------|----------|
| Off | Process in original order |
| On | Random order (reproducible with seed) |

**When to shuffle:**

- When you want diverse samples
- When testing with limited samples
- Not when order matters (sequential data)

### Random Seed

Control shuffle reproducibility:

```
Seed: 42  → Same random order every time
Seed: 123 → Different order, but consistent
```

**Tip:** Use the same seed for comparable runs.

### Added Columns

You can add extra columns with default values to the dataset:

```json
{
  "extra_col": "default_value",
  "category": "general"
}
```

These columns are appended to every row in the dataset.

### CSV-Specific Options

For CSV files, additional options appear:

| Option | Default | Description |
|--------|---------|-------------|
| **Delimiter** | `,` | Column separator (`,`, `;`, `\t`) |
| **Encoding** | `utf-8` | File encoding |

## Duplicate Checking

After loading a dataset, you can check for and remove duplicate rows:

### Check for Duplicates

Click **Check Duplicates** to scan the dataset. The UI reports:
- Whether duplicates exist
- How many duplicate rows were found
- Total row count

### Remove Duplicates

If duplicates are found, click **Remove Duplicates** to clean the dataset:
- Duplicate rows are removed from the loaded data
- The UI confirms how many rows were removed
- Remaining row count is updated

## Schema Validation

### Required Columns

Each flow specifies required input columns:

```
Required Columns for "QA Generation Flow":
✅ document    - Present in dataset
✅ domain      - Present in dataset
❌ outline     - MISSING
```

**If columns are missing:**

1. Modify your dataset to include them
2. Or choose a different flow
3. Or build a custom flow with different requirements
4. Or use "Added Columns" to provide default values

### Column Preview

After loading, review your data:

```
┌────────────┬─────────────────────────────┬──────────┐
│ Column     │ Sample Value                │ Type     │
├────────────┼─────────────────────────────┼──────────┤
│ document   │ "The solar system is..."    │ string   │
│ domain     │ "astronomy"                 │ string   │
│ created_at │ "2024-01-15"               │ string   │
└────────────┴─────────────────────────────┴──────────┘

Rows: 1,234 | Columns: 3
```

## Dataset Preview

### Preview Panel

Shows first 5 rows of loaded data:

```json
[
  {
    "document": "The solar system consists of...",
    "domain": "astronomy"
  },
  {
    "document": "Photosynthesis is the process...",
    "domain": "biology"
  }
]
```

### What to Check

1. **Column names** match flow requirements
2. **Data types** are correct (strings, numbers)
3. **Content quality** looks reasonable
4. **No missing values** in required columns
5. **No excessive duplicates** in the data

## Best Practices

### Data Quality

**Good seed data has:**

- Consistent formatting
- Complete required fields
- Representative examples
- Proper encoding (UTF-8)

**Avoid:**

- Null/empty values in required columns
- Inconsistent column names
- Mixed data types in columns
- Excessive whitespace or formatting issues

### File Organization

```
backend/uploads/
├── project_a/
│   ├── train_data.jsonl
│   └── eval_data.jsonl
├── project_b/
│   └── seed_data.jsonl
├── preprocessed/
│   └── pdf_dataset_20241127.jsonl
└── shared/
    └── common_examples.jsonl
```

### Sample Size Guidelines

| Phase | Samples | Purpose |
|-------|---------|---------|
| **Dry Run** | 2-5 | Validate configuration |
| **Testing** | 10-50 | Check output quality |
| **Small Run** | 100-500 | Initial data generation |
| **Full Run** | All | Complete generation |

### Memory Considerations

Large datasets load into memory. For very large files:

1. **Use Parquet** — More memory efficient
2. **Sample first** — Process subsets
3. **Batch process** — Multiple smaller runs

## Troubleshooting

### "File not found"

- Verify file path is correct
- Check file exists in allowed directory
- Ensure proper permissions

### "Invalid format"

- Verify file extension matches content
- Check for encoding issues
- Validate JSON/CSV syntax

### "Column not found"

- Check column name spelling (case-sensitive)
- Verify column exists in file
- Review first few rows manually

### "Out of memory"

- Reduce `num_samples`
- Use Parquet format
- Increase server memory

## Advanced Usage

### Creating Compatible Datasets

Use Python to prepare data:

```python
import pandas as pd

# Create dataset with required columns
data = [
    {"document": "Your text here...", "domain": "category"},
    # ... more rows
]

df = pd.DataFrame(data)

# Save as JSONL
df.to_json("seed_data.jsonl", orient="records", lines=True)
```

### Validating Schema

Check if dataset matches flow requirements:

```python
from sdg_hub import Flow

flow = Flow.from_yaml("path/to/flow.yaml")
requirements = flow.get_dataset_requirements()

print(f"Required columns: {requirements.required_columns}")
print(f"Optional columns: {requirements.optional_columns}")
```

### Transforming Existing Data

Rename columns to match requirements:

```python
df = pd.read_json("original_data.jsonl", lines=True)

# Rename columns
df = df.rename(columns={
    "text": "document",
    "category": "domain"
})

# Save transformed data
df.to_json("transformed_data.jsonl", orient="records", lines=True)
```

## Next Steps

- [PDF Preprocessing](pdf-preprocessing.md) — Create datasets from PDF documents
- [Running Generation](generation.md) — Execute your configured flow
- [Flow Configuration](flow-configuration.md) — Complete wizard guide
- [Run History](history.md) — Track generation runs
