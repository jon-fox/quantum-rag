#!/bin/bash
# Kill script for quantum_work API
# This script finds and kills the running API process

echo "Searching for quantum_work API process..."

# Look for python running app.py
PID=$(pgrep -f "python app.py" || pgrep -f "python3 app.py")

echo "API process found with PID: $PID"
echo "Stopping process..."

# Kill the process
kill -9 $PID
echo "Process stopped."

# Also check for any process using port 8000
PORT_PID=$(lsof -ti:8000 2>/dev/null)
if [ ! -z "$PORT_PID" ]; then
    echo "Found process using port 8000 (PID: $PORT_PID), killing it..."
    kill -9 $PORT_PID 2>/dev/null
    echo "Port 8000 process stopped."
fi

echo "API has been shut down."
