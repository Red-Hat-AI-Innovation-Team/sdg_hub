# SDG Hub UI

A modern web interface for synthetic data generation using the SDG Hub framework.

> **⚠️ Local Use Only:** This UI is designed to run locally on your machine. All services run on localhost.

## AI-Assisted Development Disclaimer

This UI was primarily built using AI-assisted development (vibe coding). While the code has been reviewed and tested, users should be aware of this development approach.

## ✨ Features

- **Home Dashboard** — Quick-start actions and prebuilt flow catalog organized by category
- **Visual Flow Configuration** — Multi-step wizard for configuring generation pipelines
- **Custom Flow Builder** — Node-based visual editor for creating custom data flows
- **Visual Flow Editor** — Canvas-based node graph with LLM, Parser, Transform, and Eval node types
- **Step-by-Step Flow Testing** — Test custom flows block-by-block before saving
- **PDF Preprocessing** — Convert PDFs to Markdown, chunk documents, and create structured datasets
- **ICL Configuration** — In-Context Learning template setup for PDF-based datasets
- **Live Monitoring** — Real-time progress tracking with block-level metrics and token statistics
- **Multi-Flow Monitoring** — Side-by-side monitoring for batch executions
- **Configuration Management** — Save, load, clone, import, and organize flow configurations
- **Checkpoint & Resume** — Never lose progress on long-running jobs
- **Run History** — Track all generation runs with log analysis, output preview, and downloads
- **Dashboard** — Overview of flows, statistics, preprocessed datasets, and custom flows

## 🚀 Quick Start

### Prerequisites

- Python 3.10–3.12
- Node.js 16+

### Run the UI

```bash
cd ui
./start.sh
```

That's it! The script will:
1. ✅ Check prerequisites (Python, Node.js, SDG Hub)
2. ✅ Install backend dependencies (creates venv automatically)
3. ✅ Install frontend dependencies (npm install)
4. ✅ Start both servers
5. ✅ Open the UI in your browser

The UI opens at `http://localhost:3000`.

Press `Ctrl+C` to stop all servers.

### API Keys (Optional)

Copy `.env.example` to `backend/.env` and add your LLM provider API keys:

```bash
cp .env.example backend/.env
```

Reference keys in the UI using `env:VARIABLE_NAME` syntax.

## 📖 Documentation

Full documentation is available in the [`docs/`](docs/) folder:

| Document | Description |
|----------|-------------|
| [Installation](docs/installation.md) | Detailed setup instructions |
| [User Guide](docs/user-guide/overview.md) | Complete usage guide |
| [API Reference](docs/api-reference.md) | Backend REST API |
| [Architecture](docs/architecture.md) | System design |

### User Guide Topics

| Guide | Description |
|-------|-------------|
| [Overview](docs/user-guide/overview.md) | UI layout and navigation |
| [Flow Configuration](docs/user-guide/flow-configuration.md) | Configuration wizard walkthrough |
| [Flow Builder](docs/user-guide/flow-builder.md) | Building custom flows |
| [Model Configuration](docs/user-guide/model-configuration.md) | LLM setup |
| [Dataset Configuration](docs/user-guide/dataset-configuration.md) | Dataset loading and configuration |
| [PDF Preprocessing](docs/user-guide/pdf-preprocessing.md) | PDF to dataset pipeline |
| [Running Generation](docs/user-guide/generation.md) | Execution and monitoring |
| [Run History](docs/user-guide/history.md) | Tracking past runs |

## 🏗️ Project Structure

```
ui/
├── start.sh                # One-command setup & run
├── .env.example            # API key template
├── backend/
│   ├── api_server.py       # App setup, middleware, router registration (~150 lines)
│   ├── config.py           # Path constants, settings
│   ├── state.py            # Shared mutable state
│   ├── models/             # Pydantic request/response models
│   ├── utils/              # Security, file handling, dataset, API key utilities
│   ├── workers/            # Background workers (dry run, generation)
│   ├── routers/            # FastAPI APIRouter modules (11 routers, 80 endpoints)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── contexts/       # React contexts (Notification, Execution)
│   │   ├── hooks/          # Custom hooks (usePDFProcessing)
│   │   └── services/       # API client
│   └── package.json
├── docs/                   # Documentation
└── tests/                  # Backend (pytest) & Frontend (jest) tests
```

## 🧪 Testing

103 backend tests and 133 frontend tests covering API endpoints, components, and utilities.

```bash
# Frontend tests
cd tests/frontend && npm test

# Backend tests
cd tests/backend && pytest
```

## 📄 License

Apache License 2.0 — See [LICENSE](../LICENSE) for details.

--
