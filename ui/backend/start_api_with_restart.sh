#!/bin/bash
# Auto-restart wrapper for API server

SCRIPT_DIR="$(dirname "$0")"
if ! cd "$SCRIPT_DIR"; then
    echo "Error: Failed to change directory to $SCRIPT_DIR" >&2
    exit 1
fi

while true; do
    echo "Starting API server..."
    python api_server.py
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 143 ]; then
        echo "API server stopped cleanly (exit code: $EXIT_CODE)"
        echo "Restarting in 2 seconds..."
        sleep 2
    else
        echo "API server crashed (exit code: $EXIT_CODE)"
        echo "Restarting in 5 seconds..."
        sleep 5
    fi
done

