# SDG Hub UI Documentation

A modern web interface for synthetic data generation using the SDG Hub framework.

> **⚠️ Local Use Only:** SDG Hub UI runs locally on your machine. All services run on localhost.

## Quick Start

```bash
cd sdg_hub_ui
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
- [Flow Builder](user-guide/flow-builder.md) — Creating custom flows
- [Model Configuration](user-guide/model-configuration.md) — Setting up LLM models
- [Dataset Configuration](user-guide/dataset-configuration.md) — Loading and configuring datasets
- [Running Generation](user-guide/generation.md) — Executing flows and monitoring progress
- [Run History](user-guide/history.md) — Viewing past runs and outputs

## Key Features

### 🔄 Flow Configuration Wizard

A 6-step wizard guides you through:

1. **Choose Source** — Use existing flows, create from scratch, or clone
2. **Select/Build Flow** — Pick from SDG Hub library or build custom
3. **Configure Model** — Set up your LLM (vLLM, OpenAI, Anthropic, etc.)
4. **Configure Dataset** — Upload and configure your seed data
5. **Dry Run** — Test with a small sample before full generation
6. **Review & Save** — Verify settings and save configuration

### 🛠️ Custom Flow Builder

Build data pipelines visually:

- Drag-and-drop block arrangement
- Pre-configured bundles for common patterns
- Real-time prompt editing with Jinja2 templating
- Automatic flow validation

### 📊 Live Monitoring

Track generation progress in real-time:

- Overall progress with percentage completion
- Block-by-block execution status

### 💾 Checkpoint & Resume

Never lose progress:

- Automatic checkpointing during generation
- Resume from last checkpoint after failures
- Clear checkpoints for fresh starts

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18, PatternFly 5, Axios |
| Backend | FastAPI, Pydantic, Uvicorn |
| Core Engine | SDG Hub (Python) |
| Data Format | JSONL, JSON, CSV, Parquet |

---

Built with ❤️ by the Red Hat AI Innovation Team
