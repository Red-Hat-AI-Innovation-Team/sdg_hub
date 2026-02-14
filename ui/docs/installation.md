# Installation

> **⚠️ Local Use Only:** SDG Hub UI runs locally on your machine. All services run on localhost.

## Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.10–3.12 | `python3 --version` |
| Node.js | 16+ | `node --version` |
| npm | 8+ | `npm --version` |

> **Note:** SDG Hub is automatically installed by the start script from the parent repository.

## Quick Start

**One command to run everything:**

```bash
cd ui
./start.sh
```

The script automatically:
1. Checks all prerequisites (Python 3.10–3.12, Node.js 16+)
2. Checks that ports 8000 and 3000 are available
3. Creates a Python virtual environment
4. Installs backend dependencies (including Docling for PDF processing)
5. Installs sdg_hub from the parent repository (`pip install -e ../..`)
6. Installs frontend dependencies (npm install)
7. Starts the backend server (port 8000)
8. Starts the frontend server (port 3000)
9. Opens the UI in your browser

Press `Ctrl+C` to stop all servers.

## Environment Configuration

### API Keys

Copy the example environment file to configure API keys:

```bash
cp .env.example backend/.env
```

Edit `backend/.env` and add your provider keys:

```bash
# OpenAI
OPENAI_API_KEY=your-openai-key-here

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-key-here

# vLLM (typically no key needed for local)
VLLM_API_KEY=EMPTY

# Other providers
COHERE_API_KEY=your-cohere-key-here
HUGGINGFACE_API_KEY=your-hf-key-here
TOGETHER_AI_API_KEY=your-together-key-here
```

In the UI, reference these keys using `env:VARIABLE_NAME` syntax (e.g., `env:OPENAI_API_KEY`).

### Backend Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SDG_HUB_DATA_DIR` | `""` | Data directory for uploads, outputs, etc. (relative to backend/) |
| `SDG_HUB_MAX_UPLOAD_MB` | `512` | Maximum file upload size in MB |
| `SDG_HUB_ALLOWED_DATA_DIRS` | `""` | Additional allowed data directories (OS path-separated) |

### Frontend Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REACT_APP_API_URL` | `http://localhost:8000` | Backend API URL (for production builds) |

## Manual Setup (Optional)

If you prefer to run components separately:

### Backend

```bash
cd ui/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SDG Hub (from parent repository)
pip install -e ../..

# (Optional) Install python-dotenv for .env support
pip install python-dotenv

# Start server
python api_server.py
```

The API server runs at `http://localhost:8000`.

**Key backend dependencies:**

- `fastapi` + `uvicorn` — Web framework and ASGI server
- `pydantic` — Request/response validation
- `pandas` — Dataset processing
- `docling` — PDF to Markdown conversion
- `langchain-text-splitters` — Document chunking
- `PyYAML` — YAML flow parsing
- `python-multipart` — File upload handling
- `nest-asyncio` — Async compatibility

### Frontend

```bash
cd ui/frontend

# Install dependencies
npm install

# Start development server
npm start
```

The UI opens at `http://localhost:3000`.

**Key frontend dependencies:**

- React 18 — UI framework
- PatternFly 5 — Design system (core, icons, table, drag-drop, topology)
- Axios — HTTP client
- ansi-to-html — Terminal output formatting

## Verifying Installation

### Check Backend

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "healthy", "service": "sdg_hub_api"}
```

### Check Frontend

Open `http://localhost:3000` in your browser. You should see the SDG Hub UI with:
- Home dashboard with quick-start actions
- Navigation sidebar with Home, Dashboard, Flows, and Run History

## Troubleshooting

### "SDG Hub is not installed"

Install SDG Hub from the main repository:

```bash
cd /path/to/sdg_hub
pip install .
```

### "Port already in use"

The start script will prompt you to kill existing processes. Or manually:

```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>
```

### "npm install" fails

Try clearing the npm cache:

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Backend starts but frontend can't connect

Ensure both servers are running and check the browser console for CORS errors. The backend allows CORS from localhost ports 3000-3005.

### PDF conversion fails

Ensure Docling is installed correctly:

```bash
pip install docling>=2.3.0
```

Docling requires certain system dependencies for PDF parsing. Check the [Docling documentation](https://github.com/DS4SD/docling) for platform-specific requirements.

### "python-dotenv not found" warning

Install the optional dependency:

```bash
pip install python-dotenv
```

This is needed for `.env` file support but is not strictly required.

## Next Steps

- [User Guide Overview](user-guide/overview.md) — Learn the UI basics
- [Flow Configuration](user-guide/flow-configuration.md) — Create your first configuration
- [API Reference](api-reference.md) — Explore the backend API
