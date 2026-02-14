# User Guide: Flow Configuration Wizard

The Flow Configuration Wizard guides you through setting up a complete data generation pipeline. The wizard adapts its steps based on your selections, providing a streamlined experience for each use case.

## Starting the Wizard

Click **Configure Flow** from the Data Generation Flows page to launch the wizard.

You can also reach the wizard by:
- Clicking **Edit** on an existing configuration
- Clicking **Clone** to duplicate and modify a configuration
- Clicking **Create New Flow** from the Home dashboard

## Step 1: Choose Source

Select how you want to create your flow:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  📦 Use Existing │  │  ➕ Start Blank  │  │  📋 Clone Flow  │
│     Flow        │  │                 │  │                 │
│                 │  │  Build from     │  │  Copy & modify  │
│  Select from    │  │  scratch using  │  │  an existing    │
│  SDG Hub library│  │  block builder  │  │  flow           │
└─────────────────┘  └─────────────────┘  └─────────────────┘

                     ┌─────────────────┐
                     │  ✏️ Continue     │  (Only if drafts exist)
                     │     Draft       │
                     │                 │
                     │  Resume saved   │
                     │  work-in-prog   │
                     └─────────────────┘
```

| Option | Best For |
|--------|----------|
| **Use Existing Flow** | Using pre-built SDG Hub flows as-is |
| **Start from Blank** | Creating completely custom pipelines |
| **Clone Existing Flow** | Modifying existing flows for your needs |
| **Continue Draft** | Resuming previous incomplete work |

## Step 2: Select or Build Flow

### Using Existing Flows

If you chose "Use Existing Flow":

1. **Browse the flow list** — Organized by SDG Hub flows and Custom flows
2. **Search** — Type to filter by name
3. **Filter by tags** — Select relevant tags (question-generation, etc.)
4. **Click to select** — View flow details on the right panel
5. **Review details** — Check description, required columns, recommended model

**Flow Details Panel shows:**

- Flow ID and version
- Author information
- Tags and description
- Default recommended model
- Required dataset columns

### Building Custom Flows

If you chose "Start from Blank" or "Clone Existing":

You'll enter the [Flow Builder](flow-builder.md) interface. See that guide for complete details on the visual node-based editor.

## Step 3: Configure Model

Set up the LLM that will power your generation:

### Basic Configuration

| Field | Description | Example |
|-------|-------------|---------|
| **Model** | Full model identifier | `hosted_vllm/meta-llama/Llama-3.3-70B-Instruct` |
| **API Base** | Model server endpoint | `http://localhost:8000/v1` |
| **API Key** | Authentication key | `your-key` or `env:OPENAI_API_KEY` |

### Model Naming Convention

```
provider/model_name

Examples:
- hosted_vllm/meta-llama/Llama-3.3-70B-Instruct  (Local vLLM)
- openai/gpt-4o                                   (OpenAI)
- anthropic/claude-3-opus                         (Anthropic)
```

### Using Environment Variables

You can reference environment variables for API keys:

```
env:OPENAI_API_KEY     → Resolves to $OPENAI_API_KEY
env:ANTHROPIC_API_KEY  → Resolves to $ANTHROPIC_API_KEY
```

**Note:** Direct API keys are not saved in configurations for security. Use `env:` references for persistent key configuration.

### Test Connection

Click **Test Connection** to verify your model configuration. The UI sends a test prompt to the model and reports:
- Whether the connection succeeded
- Response latency in milliseconds
- Any error details

### Advanced Parameters

Click "Show Advanced Parameters" to configure:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | Randomness (0=deterministic, 1=creative) |
| `max_tokens` | 2048 | Maximum response length |
| `top_p` | 0.95 | Nucleus sampling threshold |

See [Model Configuration](model-configuration.md) for detailed guidance.

## Step 4: Configure Dataset

Load and configure your seed data for generation. You have multiple dataset source options:

### Upload New File

1. **Drag-and-drop** or click the upload area
2. Supported formats: JSONL, JSON, CSV, Parquet
3. File appears in the available files list

### Use Preprocessed PDF Data

If you've already preprocessed PDFs (via the PDF Preprocessing step or the Dashboard):

1. Select **Use Preprocessed Dataset** as the source
2. Choose from available preprocessed datasets
3. The dataset is automatically loaded

### Reuse Existing Upload

Select from previously uploaded datasets in the uploads directory.

### Dataset Settings

| Setting | Description |
|---------|-------------|
| **Number of Samples** | Limit rows to process (blank = all) |
| **Shuffle** | Randomize row order |
| **Random Seed** | Seed for reproducible shuffling |
| **Added Columns** | Additional columns with default values |

### Schema Validation

The UI shows required columns for your selected flow:

```
Required Columns:
✅ document
✅ domain
⚠️ document_outline (missing)
```

### Duplicate Checking

After loading, the UI can check for duplicate rows:
- Click **Check Duplicates** to scan the dataset
- Click **Remove Duplicates** to clean the data

See [Dataset Configuration](dataset-configuration.md) for detailed guidance.

## Step 5: PDF Preprocessing (Optional)

This step appears when your dataset source involves PDF documents. It provides a complete pipeline for converting documents to structured data:

### Upload PDFs

1. Drag-and-drop or click to upload PDF and Markdown files
2. Multiple files can be uploaded at once
3. Files are stored per preprocessing job

### Convert to Markdown

1. Click **Convert** to process PDFs with Docling
2. View converted Markdown content with a live preview
3. Review and verify conversion quality

### Chunk Documents

Configure how documents are split into segments:

| Setting | Default | Description |
|---------|---------|-------------|
| **Chunk Size** | 1000 | Characters per chunk |
| **Overlap** | 200 | Overlapping characters between chunks |
| **Method** | recursive | Splitting algorithm |

Per-file chunk settings can be customized independently.

### Create Dataset

Generate a JSONL dataset from the chunked documents, optionally with ICL templates, domain, and document outline columns.

See [PDF Preprocessing](pdf-preprocessing.md) for a detailed walkthrough.

## Step 6: ICL Configuration (Optional)

This step appears when creating datasets from preprocessed PDFs. Configure In-Context Learning templates:

### ICL Template Fields

| Field | Description |
|-------|-------------|
| **ICL Document** | Example document content |
| **ICL Query 1-3** | Example questions for the document |
| **ICL Response 1-3** | Example answers to the questions |

### Additional Fields

| Field | Description |
|-------|-------------|
| **Domain** | Subject domain for the dataset |
| **Document Outline** | Structural overview of the documents |
| **Dataset Name** | Name for the generated dataset |
| **Content Column Name** | Name of the main content column |

## Step 7: Dry Run Settings

Configure test execution parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| **Sample Size** | 2 | Number of rows for dry run |
| **Enable Time Estimation** | Yes | Estimate full run duration |
| **Max Concurrency** | 10 | Parallel LLM requests |

### Why Dry Run?

- **Validate configuration** — Catch errors before full runs
- **Estimate time** — Plan for long-running jobs
- **Check output quality** — Review generated samples
- **Cost control** — Use minimal tokens for testing

## Step 8: Dry Run

Execute the dry run and view results in real-time:

- **Streaming output** — Watch progress via SSE streaming
- **Block-by-block status** — See each block execute
- **Cancel option** — Stop the dry run if issues are detected
- **Results preview** — Review generated samples before proceeding

## Step 9: Review & Save

Review all your settings before saving:

### Configuration Summary

```
Flow:     Advanced Document QA Generation
Model:    hosted_vllm/meta-llama/Llama-3.3-70B-Instruct
API Base: http://localhost:8000/v1
Dataset:  seed_data.jsonl (100 samples)
Shuffle:  Yes (seed: 42)
```

### Save Options

| Button | Action |
|--------|--------|
| **Save and Run** | Save configuration and immediately start generation |
| **Save to Flows List** | Save configuration for later use |

### After Saving

Your configuration appears in the Data Generation Flows table, ready for:

- Editing settings
- Cloning for variations
- Running generation
- Viewing in detail

## Editing Existing Configurations

Click **Edit** on any configuration to modify it:

- **Not Configured** — Opens at the step needing completion
- **Configured** — Opens at flow selection step
- **Custom Flows** — Opens in Flow Builder

Changes are auto-saved when you exit the wizard with unsaved modifications.

## Tips

1. **Start simple** — Use existing flows before building custom ones
2. **Test first** — Always run a dry run before full generation
3. **Use env vars** — Reference API keys via environment variables
4. **Check columns** — Verify dataset schema matches flow requirements
5. **Save progress** — The wizard auto-saves drafts periodically
6. **Test connection** — Use the model test feature to verify connectivity before proceeding

## Next Steps

- [Flow Builder](flow-builder.md) — Learn to build custom flows
- [Model Configuration](model-configuration.md) — Deep dive on model setup
- [PDF Preprocessing](pdf-preprocessing.md) — Convert PDFs to datasets
- [Running Generation](generation.md) — Execute your configured flow
