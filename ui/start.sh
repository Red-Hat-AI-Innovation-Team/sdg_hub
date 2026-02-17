#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SDG Hub UI - Start Script
# One-command setup: installs dependencies and starts the UI

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=3000
BACKEND_PID=""

echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                     SDG Hub UI                            ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Cleanup function to kill background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down...${NC}"
    
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${BLUE}   Stopping backend (PID: $BACKEND_PID)...${NC}"
        kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ Shutdown complete${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Function to check if a port is in use
# Uses lsof if available, falls back to nc or /dev/tcp for container environments
check_port() {
    local port=$1
    
    # Try lsof first (most reliable when available)
    if command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            return 0  # Port is in use
        else
            return 1  # Port is free
        fi
    fi
    
    # Fallback: try nc (netcat) if available
    if command -v nc >/dev/null 2>&1; then
        if nc -z 127.0.0.1 $port >/dev/null 2>&1; then
            return 0  # Port is in use
        else
            return 1  # Port is free
        fi
    fi
    
    # Fallback: try bash /dev/tcp (works in most bash environments)
    if (echo >/dev/tcp/127.0.0.1/$port) 2>/dev/null; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to gracefully kill process on a port
# Sends SIGTERM first, waits, then escalates to SIGKILL if needed
kill_process_on_port() {
    local port=$1
    local pids
    
    # Get PIDs using the port (silently return if none)
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -ti:$port 2>/dev/null) || true
    else
        # Can't determine PIDs without lsof, skip graceful shutdown
        echo -e "${YELLOW}   Warning: lsof not available, cannot kill process${NC}"
        return 1
    fi
    
    # No processes found
    if [ -z "$pids" ]; then
        return 0
    fi
    
    # Send graceful SIGTERM first
    echo "$pids" | xargs kill -TERM 2>/dev/null || true
    
    # Wait for graceful shutdown (up to 3 seconds)
    local waited=0
    while [ $waited -lt 3 ]; do
        sleep 1
        waited=$((waited + 1))
        # Check if any processes still exist
        local still_running=false
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then
                still_running=true
                break
            fi
        done
        if [ "$still_running" = false ]; then
            return 0
        fi
    done
    
    # Escalate to SIGKILL for remaining processes
    echo "$pids" | xargs kill -9 2>/dev/null || true
    sleep 1
    return 0
}

# ============================================================
# PREREQUISITES CHECK
# ============================================================
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check Python - prefer 3.10-3.12 (3.13+ may have async compatibility issues)
PYTHON_CMD=""
for py in python3.12 python3.11 python3.10 python3; do
    if command_exists "$py"; then
        PY_VER=$($py -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
        PY_MINOR=$($py -c 'import sys; print(sys.version_info.minor)' 2>/dev/null)
        if [ -n "$PY_VER" ] && [ "$PY_MINOR" -ge 10 ] && [ "$PY_MINOR" -le 12 ]; then
            PYTHON_CMD="$py"
            break
        elif [ -z "$PYTHON_CMD" ]; then
            PYTHON_CMD="$py"  # Fallback to any python3
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    echo -e "${YELLOW}   Please install Python 3.10, 3.11, or 3.12${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MINOR" -gt 12 ]; then
    echo -e "${YELLOW}   ⚠ Python $PYTHON_VERSION detected (3.13+ may have compatibility issues)${NC}"
    echo -e "${YELLOW}   Recommended: Python 3.10, 3.11, or 3.12${NC}"
else
    echo -e "${GREEN}   ✓ Python $PYTHON_VERSION${NC}"
fi

# Check Node.js
if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    echo -e "${YELLOW}   Please install Node.js 16 or higher${NC}"
    exit 1
fi

NODE_VERSION=$(node -v)
echo -e "${GREEN}   ✓ Node.js $NODE_VERSION${NC}"

# Check npm
if ! command_exists npm; then
    echo -e "${RED}❌ npm is not installed${NC}"
    echo -e "${YELLOW}   Please install npm${NC}"
    exit 1
fi

NPM_VERSION=$(npm -v)
echo -e "${GREEN}   ✓ npm $NPM_VERSION${NC}"
echo ""

# ============================================================
# PORT CHECK
# ============================================================
echo -e "${BLUE}📋 Checking ports...${NC}"

if check_port $BACKEND_PORT; then
    echo -e "${YELLOW}⚠️  Port $BACKEND_PORT is already in use.${NC}"
    echo -e "${YELLOW}   Kill existing process? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}   Killing process on port $BACKEND_PORT...${NC}"
        kill_process_on_port $BACKEND_PORT
    else
        echo -e "${RED}❌ Cannot start - port $BACKEND_PORT in use${NC}"
        exit 1
    fi
fi

if check_port $FRONTEND_PORT; then
    echo -e "${YELLOW}⚠️  Port $FRONTEND_PORT is already in use.${NC}"
    echo -e "${YELLOW}   Kill existing process? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}   Killing process on port $FRONTEND_PORT...${NC}"
        kill_process_on_port $FRONTEND_PORT
    else
        echo -e "${RED}❌ Cannot start - port $FRONTEND_PORT in use${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Ports are available${NC}"
echo ""

# ============================================================
# BACKEND SETUP
# ============================================================
echo -e "${BLUE}🔧 Setting up backend...${NC}"

cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}   Creating Python virtual environment...${NC}"
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies (including sdg_hub from parent repo)
# Use explicit venv pip path as fallback in case activate doesn't update PATH
PIP_CMD="./venv/bin/pip"
if [ ! -f "venv/.dependencies_installed" ] || [ "requirements.txt" -nt "venv/.dependencies_installed" ]; then
    echo -e "${YELLOW}   Installing Python dependencies...${NC}"
    $PIP_CMD install -q -r requirements.txt
    
    # Install sdg_hub from the parent repository
    echo -e "${YELLOW}   Installing sdg_hub...${NC}"
    $PIP_CMD install -q -e ../..
    
    touch venv/.dependencies_installed
    echo -e "${GREEN}   ✓ Backend dependencies installed${NC}"
else
    echo -e "${GREEN}   ✓ Backend dependencies up to date${NC}"
fi

cd ..

# ============================================================
# FRONTEND SETUP
# ============================================================
echo -e "${BLUE}🔧 Setting up frontend...${NC}"

cd frontend

# Install dependencies if needed
# Check if node_modules exists and has our marker file
if [ ! -d "node_modules" ] || [ ! -f "node_modules/.installed" ]; then
    echo -e "${YELLOW}   Installing Node.js dependencies (this may take a minute)...${NC}"
    if npm install --silent; then
        # Create marker file on successful install
        touch node_modules/.installed
        echo -e "${GREEN}   ✓ Frontend dependencies installed${NC}"
    else
        echo -e "${RED}❌ Failed to install frontend dependencies${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}   ✓ Frontend dependencies up to date${NC}"
fi

cd ..
echo ""

# ============================================================
# START SERVERS
# ============================================================

# Start Backend
echo -e "${BLUE}🚀 Starting backend server...${NC}"
cd backend
source venv/bin/activate
./venv/bin/python api_server.py &
BACKEND_PID=$!
cd ..

echo -e "${GREEN}   Backend started (PID: $BACKEND_PID)${NC}"

# Wait for backend to be ready
echo -e "${BLUE}⏳ Waiting for backend to initialize...${NC}"
MAX_WAIT=30
WAITED=0
while ! check_port $BACKEND_PORT && [ $WAITED -lt $MAX_WAIT ]; do
    sleep 1
    WAITED=$((WAITED + 1))
    echo -ne "\r   Waiting... ${WAITED}s"
done
echo ""

if ! check_port $BACKEND_PORT; then
    echo -e "${RED}❌ Backend failed to start within ${MAX_WAIT}s${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Backend is ready on http://localhost:$BACKEND_PORT${NC}"
echo ""

# Open browser after a short delay
echo -e "${BLUE}🌐 Opening browser in 5 seconds...${NC}"
(
    sleep 5
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://localhost:$FRONTEND_PORT"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "http://localhost:$FRONTEND_PORT" 2>/dev/null || \
        sensible-browser "http://localhost:$FRONTEND_PORT" 2>/dev/null || \
        echo "Please open http://localhost:$FRONTEND_PORT in your browser"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        start "http://localhost:$FRONTEND_PORT"
    fi
) &

# Start Frontend (runs in foreground)
echo -e "${BLUE}🚀 Starting frontend server...${NC}"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Backend:  http://localhost:$BACKEND_PORT${NC}"
echo -e "${GREEN}  Frontend: http://localhost:$FRONTEND_PORT${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

cd frontend
# Node.js 25.2.0 introduced a bug where html-webpack-plugin triggers a SecurityError
# by accessing the experimental localStorage global (nodejs/node#60704).
# Disable webstorage on Node 25+ where the bug exists. Append to any existing
# NODE_OPTIONS so user-defined flags (e.g. --max-old-space-size) are preserved.
NODE_MAJOR=$(node -e "console.log(process.versions.node.split('.')[0])")
if [ "$NODE_MAJOR" -ge 25 ] 2>/dev/null; then
    export NODE_OPTIONS="${NODE_OPTIONS:+$NODE_OPTIONS }--no-experimental-webstorage"
fi
BROWSER=none npm start
