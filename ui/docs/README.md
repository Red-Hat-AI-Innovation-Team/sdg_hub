# SDG Hub UI Documentation

A modern web interface for synthetic data generation using the SDG Hub framework.

> **⚠️ Local Use Only:** SDG Hub UI runs locally on your machine. All services run on localhost.

## Quick Start

```bash
cd ui
./start.sh
```

The script handles everything: dependencies, servers, and opens the UI at `http://localhost:3000`.

**Prerequisites:** Python 3.10+, Node.js 16+, SDG Hub installed

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](installation.md) | Setup and prerequisites |
| [User Guide](user-guide/overview.md) | Complete guide to using the UI |
| [API Reference](api-reference.md) | Backend REST API documentation |
| [Architecture](architecture.md) | System design and technical details |

### User Guide

- [Overview](user-guide/overview.md) — UI layout and navigation
- [Flow Configuration](user-guide/flow-configuration.md) — Using the configuration wizard
- [Flow Builder](user-guide/flow-builder.md) — Creating custom flows with the visual editor
- [Model Configuration](user-guide/model-configuration.md) — Setting up LLM models
- [Dataset Configuration](user-guide/dataset-configuration.md) — Loading and configuring datasets
- [PDF Preprocessing](user-guide/pdf-preprocessing.md) — Converting PDFs to structured datasets
- [Running Generation](user-guide/generation.md) — Executing flows and monitoring progress
- [Run History](user-guide/history.md) — Viewing past runs and outputs

## Key Features

### 🏠 Home Dashboard

A welcoming landing page with:

- Quick-start actions to create new flow configurations
- Pre-built SDG Hub flows organized by category
- Navigation shortcuts to all major sections
- Overview of the platform capabilities

### 📊 Dashboard

A central overview of your work:

- Your flow configurations with status summary
- Flow statistics (samples generated, run counts, durations)
- Preprocessed PDF datasets with management tools
- Custom flows library with download and delete options

### 🔄 Flow Configuration Wizard

A multi-step wizard guides you through:

1. **Choose Source** — Use existing flows, create from scratch, clone, or continue a draft
2. **Select/Build Flow** — Pick from the SDG Hub library or build custom with the visual editor
3. **Configure Model** — Set up your LLM (vLLM, OpenAI, Anthropic, etc.)
4. **Configure Dataset** — Upload files, use preprocessed PDF data, or reuse existing datasets
5. **PDF Preprocessing** — (Optional) Upload PDFs, convert to Markdown, chunk, and create datasets
6. **ICL Configuration** — (Optional) Configure In-Context Learning templates for PDF-based datasets
7. **Dry Run Settings** — Configure test execution parameters
8. **Dry Run** — Test with a small sample before full generation
9. **Review & Save** — Verify settings and save configuration

### 🛠️ Custom Flow Builder

Build data pipelines visually:

- Canvas-based node graph editor
- Four node types: LLM, Parser, Transform, and Eval
- Drag nodes from a sidebar palette onto the canvas
- Connect nodes with edges to define data flow
- Pre-built flow templates for common patterns (QA Generation, Summarization, etc.)
- Node configuration drawer for detailed settings
- Real-time prompt editing with Jinja2 templating
- Connection validation ensuring compatible inputs/outputs
- Guided tour for new users

### 🧪 Step-by-Step Flow Testing

Test your custom flows before saving:

- Execute blocks one at a time with sample data
- View input/output for each node in the pipeline
- Configure test model settings independently
- Validate the entire flow end-to-end with real LLM calls

### 📄 PDF Preprocessing

Convert documents into structured datasets:

- Upload PDF and Markdown files
- Convert PDFs to Markdown using Docling
- Preview converted Markdown content
- Chunk documents with configurable size and overlap
- Configure In-Context Learning (ICL) templates
- Create ready-to-use JSONL datasets from processed documents
- Manage preprocessing jobs with status tracking

### 📊 Live Monitoring

Track generation progress in real-time:

- Overall progress with percentage completion
- Block-by-block execution status
- Token usage statistics (input/output)
- Per-block timing metrics

### 📊 Multi-Flow Monitoring

Monitor multiple flows simultaneously:

- Side-by-side progress view for batch executions
- Unified monitoring modal for all running flows
- Per-flow status, progress, and terminal output

### 💾 Checkpoint & Resume

Never lose progress:

- Automatic checkpointing during generation
- Resume from last checkpoint after failures
- Clear checkpoints for fresh starts

### 📋 Configuration Management

Organize and manage your flow configurations:

- Save, load, clone, and delete configurations
- Import configurations from file
- Tag-based organization
- Batch run and stop operations
- Restore configurations from past runs

### 📜 Run History

Comprehensive run tracking:

- Filter and search past runs
- View detailed run information and logs
- Log analysis for debugging failed runs
- Preview output data before downloading
- Download generated JSONL output files
- Restore configuration from any historical run

## Architecture

The codebase follows a modular architecture:

**Backend:**
- 11 FastAPI APIRouter modules under `routers/` handling 80 endpoints
- Extracted `models/`, `utils/`, `workers/`, `state.py`, and `config.py` from the original monolithic `api_server.py`
- `api_server.py` is now a slim entry point (~150 lines) for app setup, middleware, and router registration

**Frontend:**
- `useReducer` patterns in UnifiedFlowWizard, DatasetConfigurationStep, and VisualFlowEditor for complex state
- Custom hooks (`usePDFProcessing`) to encapsulate component logic
- `ExecutionContext` for prop drilling elimination across execution-related components
- Component extraction (PDFPreprocessingStep sub-components, SimpleFlowCanvas, MissingColumnsModal, DuplicatesModal)

See [Architecture](architecture.md) for full details.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18, PatternFly 5, Axios |
| Frontend Patterns | useReducer, custom hooks, React Context, React.memo |
| Backend | FastAPI (11 APIRouter modules), Pydantic, Uvicorn |
| Core Engine | SDG Hub (Python) |
| PDF Processing | Docling |
| Data Format | JSONL, JSON, CSV, Parquet |

---
