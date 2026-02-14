#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SDG Hub UI - Test Environment Script
# Spins up a completely fresh, isolated UI instance for pre-merge testing.
# All test data is stored separately and cleaned up on exit.
#
# Usage:
#   ./start_test.sh            # Start fresh test environment
#   ./start_test.sh --clean    # Only clean up previous test artifacts
#   ./start_test.sh --keep     # Start test env, keep data after exit (don't auto-clean)

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
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=3000
BACKEND_PID=""
TEST_VENV="backend/test_venv"
TEST_DATA_DIR="test_data"
KEEP_DATA=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --clean)
            echo ""
            echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════╗${NC}"
            echo -e "${MAGENTA}║              SDG Hub UI - Test Cleanup                    ║${NC}"
            echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo -e "${BLUE}🧹 Cleaning up test environment artifacts...${NC}"
            
            if [ -d "$TEST_VENV" ]; then
                echo -e "${YELLOW}   Removing test virtual environment ($TEST_VENV)...${NC}"
                rm -rf "$TEST_VENV"
                echo -e "${GREEN}   ✓ Removed $TEST_VENV${NC}"
            fi
            
            if [ -d "backend/$TEST_DATA_DIR" ]; then
                echo -e "${YELLOW}   Removing test data directory (backend/$TEST_DATA_DIR)...${NC}"
                rm -rf "backend/$TEST_DATA_DIR"
                echo -e "${GREEN}   ✓ Removed backend/$TEST_DATA_DIR${NC}"
            fi

            if [ -d "frontend/node_modules_test_backup" ]; then
                echo -e "${YELLOW}   Removing frontend backup...${NC}"
                rm -rf "frontend/node_modules_test_backup"
                echo -e "${GREEN}   ✓ Removed frontend backup${NC}"
            fi
            
            echo ""
            echo -e "${GREEN}✅ Test cleanup complete${NC}"
            exit 0
            ;;
        --keep)
            KEEP_DATA=true
            ;;
    esac
done

echo ""
echo -e "${MAGENTA}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║              SDG Hub UI - Test Environment                ║${NC}"
echo -e "${MAGENTA}║                                                           ║${NC}"
echo -e "${MAGENTA}║   Fresh, isolated instance for pre-merge testing          ║${NC}"
echo -e "${MAGENTA}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}   Test venv:     $TEST_VENV${NC}"
echo -e "${CYAN}   Test data dir: backend/$TEST_DATA_DIR${NC}"
echo -e "${CYAN}   Backend port:  $BACKEND_PORT${NC}"
echo -e "${CYAN}   Frontend port: $FRONTEND_PORT${NC}"
if [ "$KEEP_DATA" = true ]; then
    echo -e "${CYAN}   Cleanup:       Manual (--keep mode)${NC}"
else
    echo -e "${CYAN}   Cleanup:       Automatic on exit${NC}"
fi
echo ""

# Cleanup function to kill background processes and remove test artifacts
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down test environment...${NC}"
    
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${BLUE}   Stopping backend (PID: $BACKEND_PID)...${NC}"
        kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
    fi
    
    if [ "$KEEP_DATA" = false ]; then
        echo -e "${BLUE}   Cleaning up test artifacts...${NC}"
        
        if [ -d "$SCRIPT_DIR/$TEST_VENV" ]; then
            echo -e "${YELLOW}   Removing test venv...${NC}"
            rm -rf "$SCRIPT_DIR/$TEST_VENV"
        fi
        
        if [ -d "$SCRIPT_DIR/backend/$TEST_DATA_DIR" ]; then
            echo -e "${YELLOW}   Removing test data...${NC}"
            rm -rf "$SCRIPT_DIR/backend/$TEST_DATA_DIR"
        fi
        
        echo -e "${GREEN}   ✓ Test artifacts cleaned up${NC}"
    else
        echo -e "${YELLOW}   --keep mode: Test artifacts preserved at:${NC}"
        echo -e "${YELLOW}     - $TEST_VENV${NC}"
        echo -e "${YELLOW}     - backend/$TEST_DATA_DIR${NC}"
        echo -e "${YELLOW}   Run './start_test.sh --clean' to remove them later.${NC}"
    fi
    
    echo -e "${GREEN}✅ Test environment shutdown complete${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Function to check if a port is in use
check_port() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    fi
    if command -v nc >/dev/null 2>&1; then
        if nc -z 127.0.0.1 $port >/dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    fi
    if (echo >/dev/tcp/127.0.0.1/$port) 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

kill_process_on_port() {
    local port=$1
    local pids
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -ti:$port 2>/dev/null) || true
    else
        echo -e "${YELLOW}   Warning: lsof not available, cannot kill process${NC}"
        return 1
    fi
    if [ -z "$pids" ]; then
        return 0
    fi
    echo "$pids" | xargs kill -TERM 2>/dev/null || true
    local waited=0
    while [ $waited -lt 3 ]; do
        sleep 1
        waited=$((waited + 1))
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
    echo "$pids" | xargs kill -9 2>/dev/null || true
    sleep 1
    return 0
}

# ============================================================
# PREREQUISITES CHECK
# ============================================================
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

PYTHON_CMD=""
for py in python3.12 python3.11 python3.10 python3; do
    if command_exists "$py"; then
        PY_VER=$($py -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
        PY_MINOR=$($py -c 'import sys; print(sys.version_info.minor)' 2>/dev/null)
        if [ -n "$PY_VER" ] && [ "$PY_MINOR" -ge 10 ] && [ "$PY_MINOR" -le 12 ]; then
            PYTHON_CMD="$py"
            break
        elif [ -z "$PYTHON_CMD" ]; then
            PYTHON_CMD="$py"
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
else
    echo -e "${GREEN}   ✓ Python $PYTHON_VERSION${NC}"
fi

if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}   ✓ Node.js $(node -v)${NC}"

if ! command_exists npm; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}   ✓ npm $(npm -v)${NC}"
echo ""

# ============================================================
# PORT CHECK - Must be free for testing
# ============================================================
echo -e "${BLUE}📋 Checking ports...${NC}"

for port in $BACKEND_PORT $FRONTEND_PORT; do
    if check_port $port; then
        echo -e "${YELLOW}⚠️  Port $port is already in use.${NC}"
        echo -e "${YELLOW}   Kill existing process? (y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}   Killing process on port $port...${NC}"
            kill_process_on_port $port
            sleep 1
        else
            echo -e "${RED}❌ Cannot start test env - port $port in use${NC}"
            echo -e "${YELLOW}   Stop existing services first or use different ports.${NC}"
            # Disable cleanup trap since we didn't create anything
            trap - EXIT
            exit 1
        fi
    fi
done

echo -e "${GREEN}✅ Ports are available${NC}"
echo ""

# ============================================================
# CLEAN PREVIOUS TEST DATA (fresh start every time)
# ============================================================
echo -e "${BLUE}🧹 Ensuring fresh test state...${NC}"

if [ -d "backend/$TEST_DATA_DIR" ]; then
    echo -e "${YELLOW}   Removing previous test data...${NC}"
    rm -rf "backend/$TEST_DATA_DIR"
fi

echo -e "${GREEN}   ✓ Clean test data directory${NC}"
echo ""

# ============================================================
# BACKEND SETUP (isolated test venv)
# ============================================================
echo -e "${BLUE}🔧 Setting up test backend...${NC}"

cd backend

# Always create a fresh test venv for clean testing
if [ -d "test_venv" ]; then
    echo -e "${YELLOW}   Removing old test venv for fresh install...${NC}"
    rm -rf test_venv
fi

echo -e "${YELLOW}   Creating fresh test virtual environment...${NC}"
$PYTHON_CMD -m venv test_venv

# Activate test virtual environment
source test_venv/bin/activate

echo -e "${YELLOW}   Installing Python dependencies (this may take a minute)...${NC}"
pip install -q -r requirements.txt

# Install sdg_hub from the parent repository
echo -e "${YELLOW}   Installing sdg_hub...${NC}"
pip install -q -e ../..

echo -e "${GREEN}   ✓ Test backend dependencies installed${NC}"

cd ..
echo ""

# ============================================================
# FRONTEND SETUP (reuse node_modules if available)
# ============================================================
echo -e "${BLUE}🔧 Setting up test frontend...${NC}"

cd frontend

# Reuse existing node_modules if available (no state in node_modules)
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}   Installing Node.js dependencies (this may take a minute)...${NC}"
    if npm install --silent; then
        echo -e "${GREEN}   ✓ Frontend dependencies installed${NC}"
    else
        echo -e "${RED}❌ Failed to install frontend dependencies${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}   ✓ Frontend dependencies already available${NC}"
fi

cd ..
echo ""

# ============================================================
# START TEST SERVERS
# ============================================================

# Start Backend with isolated test data directory
echo -e "${BLUE}🚀 Starting test backend server...${NC}"
cd backend
source test_venv/bin/activate

# Set the test data directory for complete state isolation
export SDG_HUB_DATA_DIR="$TEST_DATA_DIR"

./test_venv/bin/python api_server.py &
BACKEND_PID=$!
cd ..

echo -e "${GREEN}   Backend started (PID: $BACKEND_PID)${NC}"

# Wait for backend to be ready
echo -e "${BLUE}⏳ Waiting for backend to initialize...${NC}"
MAX_WAIT=60
WAITED=0
while ! check_port $BACKEND_PORT && [ $WAITED -lt $MAX_WAIT ]; do
    sleep 1
    WAITED=$((WAITED + 1))
    echo -ne "\r   Waiting... ${WAITED}s"
done
echo ""

if ! check_port $BACKEND_PORT; then
    echo -e "${RED}❌ Test backend failed to start within ${MAX_WAIT}s${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Test backend is ready on http://localhost:$BACKEND_PORT${NC}"
echo ""

# Open browser after a short delay
echo -e "${BLUE}🌐 Opening browser in 5 seconds...${NC}"
(
    sleep 5
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "http://localhost:$FRONTEND_PORT"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "http://localhost:$FRONTEND_PORT" 2>/dev/null || true
    fi
) &

# Start Frontend (runs in foreground)
echo -e "${BLUE}🚀 Starting test frontend server...${NC}"
echo ""
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}            TEST ENVIRONMENT - FRESH STATE                ${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Backend:   http://localhost:$BACKEND_PORT${NC}"
echo -e "${GREEN}  Frontend:  http://localhost:$FRONTEND_PORT${NC}"
echo -e "${CYAN}  Data dir:  backend/$TEST_DATA_DIR/ (isolated)${NC}"
echo -e "${CYAN}  Venv:      $TEST_VENV/ (isolated)${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""
if [ "$KEEP_DATA" = false ]; then
    echo -e "${YELLOW}Press Ctrl+C to stop and auto-clean test artifacts${NC}"
else
    echo -e "${YELLOW}Press Ctrl+C to stop (test data will be preserved)${NC}"
fi
echo ""

cd frontend
BROWSER=none npm start
