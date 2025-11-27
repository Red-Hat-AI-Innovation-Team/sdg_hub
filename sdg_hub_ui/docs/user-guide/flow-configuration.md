# User Guide: Flow Configuration Wizard

The Flow Configuration Wizard guides you through setting up a complete data generation pipeline in 6 steps.

## Starting the Wizard

Click **Configure Flow** from the Data Generation Flows page to launch the wizard.

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

You'll enter the [Flow Builder](flow-builder.md) interface. See that guide for complete details.

## Step 3: Configure Model

Set up the LLM that will power your generation:

### Basic Configuration

| Field | Description | Example |
|-------|-------------|---------|
| **Model** | Full model identifier | `hosted_vllm/meta-llama/Llama-3.3-70B-Instruct` |
| **API Base** | Model server endpoint | `http://localhost:8000/v1` |
| **API Key** | Authentication key | `sk-...` or `env:OPENAI_API_KEY` |

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

**Note:** Direct API keys are not saved in configurations. You'll need to re-enter them when loading saved configs.

### Advanced Parameters

Click "Show Advanced Parameters" to configure:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 0.7 | Randomness (0=deterministic, 1=creative) |
| `max_tokens` | 2048 | Maximum response length |
| `top_p` | 0.95 | Nucleus sampling threshold |

### Quick Setup Templates

**Local vLLM:**

```yaml
Model: hosted_vllm/meta-llama/Llama-3.3-70B-Instruct
API Base: http://localhost:8000/v1
API Key: EMPTY
```

**OpenAI:**

```yaml
Model: openai/gpt-4o
API Base: (leave empty)
API Key: env:OPENAI_API_KEY
```

## Step 4: Configure Dataset

Load and configure your seed data for generation.

### Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| JSONL | `.jsonl` | JSON Lines (one object per line) |
| JSON | `.json` | JSON array of objects |
| CSV | `.csv` | Comma-separated values |
| Parquet | `.parquet` | Apache Parquet columnar format |

### Upload or Select

1. **Upload New File** — Drag-and-drop or click to upload
2. **Select Existing** — Choose from previously uploaded files

### Dataset Settings

| Setting | Description |
|---------|-------------|
| **Number of Samples** | Limit rows to process (blank = all) |
| **Shuffle** | Randomize row order |
| **Random Seed** | Seed for reproducible shuffling |

### Schema Validation

The UI shows required columns for your selected flow:

```
Required Columns:
✅ document
✅ domain
⚠️ document_outline (missing)
```

Ensure your dataset contains all required columns before proceeding.

### Dataset Preview

After loading, review a preview of your data:

- First 5 rows displayed
- Column names and types shown
- Row count confirmed

## Step 5: Dry Run Settings

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

## Step 6: Review & Save

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

- **Not Configured** → Opens at the step needing completion
- **Configured** → Opens at flow selection step
- **Custom Flows** → Opens in Flow Builder

Changes are auto-saved when you exit the wizard with unsaved modifications.

## Tips

1. **Start simple** — Use existing flows before building custom ones
2. **Test first** — Always run a dry run before full generation
3. **Use env vars** — Reference API keys via environment variables
4. **Check columns** — Verify dataset schema matches flow requirements
5. **Save progress** — The wizard auto-saves drafts periodically

## Next Steps

- [Flow Builder](flow-builder.md) — Learn to build custom flows
- [Model Configuration](model-configuration.md) — Deep dive on model setup
- [Running Generation](generation.md) — Execute your configured flow

