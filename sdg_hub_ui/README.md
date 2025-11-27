# SDG Hub UI

A modern web interface for synthetic data generation using the SDG Hub framework.

> **⚠️ Local Use Only:** This UI is designed to run locally on your machine. All services run on localhost.

## ✨ Features

- **Visual Flow Configuration** — Step-by-step wizard for configuring generation pipelines
- **Custom Flow Builder** — Drag-and-drop interface for creating custom data flows
- **Live Monitoring** — Real-time progress tracking with block-level metrics
- **Configuration Management** — Save, load, clone, and organize flow configurations
- **Checkpoint & Resume** — Never lose progress on long-running jobs
- **Run History** — Track all generation runs with downloadable outputs

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 16+

### Run the UI

```bash
cd sdg_hub_ui
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

## 📖 Documentation

Full documentation is available in the [`docs/`](docs/) folder:

| Document | Description |
|----------|-------------|
| [Installation](docs/installation.md) | Detailed setup instructions |
| [User Guide](docs/user-guide/overview.md) | Complete usage guide |
| [API Reference](docs/api-reference.md) | Backend REST API |
| [Architecture](docs/architecture.md) | System design |

## 🏗️ Project Structure

```
sdg_hub_ui/
├── start.sh                # One-command setup & run
├── backend/                # FastAPI server
│   ├── api_server.py       # Main API application
│   └── requirements.txt    # Python dependencies
├── frontend/               # React application
│   ├── src/                # Source code
│   └── package.json        # Node dependencies
├── docs/                   # Documentation
└── tests/                  # Test suites
```

## 🧪 Testing

```bash
# Frontend tests
cd tests/frontend && npm test

# Backend tests
cd tests/backend && pytest
```

## 📄 License

Apache License 2.0 — See [LICENSE](../LICENSE) for details.

---

Built with ❤️ by the Red Hat AI Innovation Team
