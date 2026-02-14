# User Guide: PDF Preprocessing

This guide covers converting PDF and Markdown documents into structured datasets for data generation.

## Overview

The PDF Preprocessing pipeline allows you to:

1. **Upload** PDF and Markdown files
2. **Convert** PDFs to Markdown using Docling
3. **Preview** converted content
4. **Chunk** documents into manageable segments
5. **Configure** In-Context Learning (ICL) templates
6. **Create** ready-to-use JSONL datasets

## Accessing PDF Preprocessing

You can access PDF preprocessing in two ways:

### From the Wizard

During flow configuration, if you select a dataset source that involves PDFs, the wizard shows the PDF Preprocessing step.

### From the Dashboard

The Dashboard page shows a **Preprocessed Datasets** section where you can:

- View existing preprocessing jobs
- Access previously created datasets
- Delete old preprocessing jobs

## Step 1: Upload Files

### Supported File Types

| Type | Extension | Description |
|------|-----------|-------------|
| **PDF** | `.pdf` | PDF documents (converted via Docling) |
| **Markdown** | `.md` | Markdown files (used directly) |

### Uploading

1. Click the upload area or drag files onto it
2. Multiple files can be uploaded at once
3. Files are organized by preprocessing job ID
4. Upload status is shown for each file

### File Storage

Uploaded files are stored in:

```
backend/pdf_uploads/{job_id}/
├── document1.pdf
├── document2.pdf
└── notes.md
```

## Step 2: Convert PDFs to Markdown

### Starting Conversion

1. After uploading, click **Convert** to process PDFs
2. Docling converts each PDF to Markdown format
3. Markdown files (`.md`) are used as-is without conversion
4. Progress is shown during conversion

### Conversion Output

Converted files are saved to:

```
backend/pdf_converted/{job_id}/
├── document1.md
├── document2.md
└── notes.md
```

### Preview Converted Content

After conversion, preview the Markdown output:

1. Click on a converted file to view its content
2. Review the quality of the conversion
3. Check that text, headings, and structure are preserved
4. Identify any conversion issues

### Conversion Notes

- Complex PDF layouts may not convert perfectly
- Tables, images, and special formatting may be simplified
- Review converted content before proceeding
- Markdown files are passed through without modification

## Step 3: Chunk Documents

Split converted documents into smaller segments for dataset creation.

### Chunking Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Chunk Size** | 1000 | Maximum characters per chunk |
| **Overlap** | 200 | Characters of overlap between consecutive chunks |
| **Method** | recursive | Splitting algorithm (`recursive` uses langchain-text-splitters) |

### Per-File Configuration

You can customize chunking settings for individual files:

```
Global Settings: chunk_size=1000, overlap=200

File-Specific Overrides:
├── document1.md: chunk_size=500, overlap=100  (shorter chunks)
├── document2.md: (uses global settings)
└── notes.md: chunk_size=2000, overlap=300     (longer chunks)
```

### Selecting Files

Choose which converted files to include in chunking:

- Select all files or pick specific ones
- Deselected files are excluded from the dataset
- Useful for filtering out irrelevant documents

### Viewing Chunks

After chunking, review the results:

- Total number of chunks created
- Chunks per file breakdown
- Preview individual chunk content

## Step 4: Configure ICL Templates

In-Context Learning (ICL) templates provide example question-answer pairs that guide the LLM during generation.

### ICL Template Fields

| Field | Description |
|-------|-------------|
| **ICL Document** | An example document snippet for the LLM to reference |
| **ICL Query 1** | First example question about the document |
| **ICL Response 1** | First example answer |
| **ICL Query 2** | Second example question |
| **ICL Response 2** | Second example answer |
| **ICL Query 3** | Third example question |
| **ICL Response 3** | Third example answer |

### Loading Templates

Click **Load Template** to fill in ICL fields from pre-configured templates. Available templates provide domain-specific examples.

### Additional Configuration

| Field | Description |
|-------|-------------|
| **Domain** | Subject domain (e.g., "science", "technology", "finance") |
| **Document Outline** | Structural overview or summary of the document collection |
| **Dataset Name** | Name for the output dataset file |
| **Content Column Name** | Name of the main content column (default: "document") |
| **Include Domain** | Whether to add the domain column to the dataset |
| **Include Document Outline** | Whether to add the outline column to the dataset |

## Step 5: Create Dataset

Generate the final JSONL dataset from chunked and configured data.

### Dataset Structure

The created dataset is a JSONL file with the following columns (depending on configuration):

| Column | Always Present | Description |
|--------|---------------|-------------|
| Content column (e.g., `document`) | Yes | The chunked text content |
| `domain` | If "Include Domain" is checked | Subject domain |
| `document_outline` | If "Include Outline" is checked | Document structure overview |
| `icl_document` | If ICL template is configured | Example document |
| `icl_query_1` through `icl_query_3` | If ICL template is configured | Example questions |
| `icl_response_1` through `icl_response_3` | If ICL template is configured | Example answers |

### Example Output

```jsonl
{"document": "Chunk 1 text content...", "domain": "science", "icl_document": "Example...", "icl_query_1": "What is...?", "icl_response_1": "It is..."}
{"document": "Chunk 2 text content...", "domain": "science", "icl_document": "Example...", "icl_query_1": "What is...?", "icl_response_1": "It is..."}
```

### Dataset Location

Created datasets are saved to `backend/uploads/` and can be immediately used in the Dataset Configuration step.

## Managing Preprocessing Jobs

### Job Status

Each preprocessing job tracks its status:

| Status | Description |
|--------|-------------|
| **uploaded** | Files uploaded, awaiting conversion |
| **converting** | PDF conversion in progress |
| **converted** | Conversion complete, ready for chunking |
| **chunked** | Documents chunked, ready for dataset creation |
| **completed** | Dataset created successfully |

### Viewing Jobs

From the Dashboard or the preprocessing API:

- View all preprocessing jobs
- Check job status and file lists
- Access converted files and chunks

### Deleting Jobs

Delete preprocessing jobs when no longer needed:

- Removes the job record
- Cleans up uploaded and converted files
- Does not delete datasets already created from the job

### Deleting Datasets

Delete preprocessed datasets independently:

- Removes the JSONL dataset file
- Does not affect the preprocessing job

## Best Practices

### Document Preparation

- **Clean PDFs** — Well-formatted PDFs convert better
- **Text-based PDFs** — Scanned/image PDFs may not convert well
- **Reasonable size** — Very large PDFs may take longer to process
- **Consistent formatting** — Similar document structures produce better results

### Chunking Strategy

| Use Case | Chunk Size | Overlap |
|----------|-----------|---------|
| Short Q&A | 500-1000 | 100-200 |
| Detailed analysis | 1000-2000 | 200-400 |
| Long-form generation | 2000-4000 | 300-500 |

**Guidelines:**

- Larger chunks provide more context but may exceed model limits
- Overlap ensures continuity between chunks
- Adjust per-file when documents have different characteristics

### ICL Templates

- **Relevant examples** — ICL examples should match your target output style
- **Quality matters** — High-quality examples produce better results
- **Domain-specific** — Tailor examples to your document domain
- **Three examples** — Three ICL examples typically provide sufficient guidance

## Troubleshooting

### PDF Conversion Issues

**"Conversion failed"**

- Check Docling is installed (`pip install docling>=2.3.0`)
- Verify the PDF is not corrupted
- Try with a simpler PDF first

**Poor conversion quality**

- Review the Markdown output for issues
- Complex layouts, tables, and images may not convert perfectly
- Consider pre-processing PDFs to simplify layout

### Chunking Issues

**Too many/few chunks**

- Adjust chunk size up or down
- Review overlap settings
- Use per-file configuration for outliers

**Chunks splitting mid-sentence**

- The recursive splitter tries to respect sentence boundaries
- Increase chunk size to reduce mid-sentence splits
- Adjust overlap to capture split content

### Dataset Creation Issues

**Empty dataset**

- Verify files were selected for chunking
- Check that conversion completed successfully
- Review chunk count before creating dataset

## Next Steps

- [Dataset Configuration](dataset-configuration.md) — Use your preprocessed dataset in a flow
- [Flow Configuration](flow-configuration.md) — Complete the configuration wizard
- [Running Generation](generation.md) — Execute generation with your dataset
