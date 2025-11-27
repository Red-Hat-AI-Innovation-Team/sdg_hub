# Installation

> **⚠️ Local Use Only:** SDG Hub UI runs locally on your machine. All services run on localhost.

## Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| Python | 3.10+ | `python3 --version` |
| Node.js | 16+ | `node --version` |
| npm | 8+ | `npm --version` |

> **Note:** SDG Hub is automatically installed by the start script from the parent repository.

## Quick Start

**One command to run everything:**

```bash
cd sdg_hub_ui
./start.sh
```

The script automatically:
1. Checks all prerequisites (Python, Node.js)
2. Creates a Python virtual environment
3. Installs backend dependencies
4. Installs sdg_hub from the parent repository
5. Installs frontend dependencies  
6. Starts the backend server (port 8000)
7. Starts the frontend server (port 3000)
8. Opens the UI in your browser

Press `Ctrl+C` to stop all servers.

## Manual Setup (Optional)

If you prefer to run components separately:

### Backend

```bash
cd sdg_hub_ui/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
python api_server.py
```

The API server runs at `http://localhost:8000`.

### Frontend

```bash
cd sdg_hub_ui/frontend

# Install dependencies
npm install

# Start development server
npm start
```

The UI opens at `http://localhost:3000`.

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
- Navigation sidebar
- "Data Generation Flows" page

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

Ensure both servers are running and check the browser console for CORS errors.

## Next Steps

- [User Guide Overview](user-guide/overview.md) — Learn the UI basics
- [Flow Configuration](user-guide/flow-configuration.md) — Create your first configuration
- [API Reference](api-reference.md) — Explore the backend API
