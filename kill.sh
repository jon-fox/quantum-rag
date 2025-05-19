#!/bin/bash
# Kill script for quantum_work API
# This script finds and kills the running API process

echo "Searching for quantum_work API process..."

# Find if the app is running
PID=$(pgrep -f "python3 app.py" || pgrep -f "python app.py")

if [ -z "$PID" ]; then
    echo "No API process found running."
    exit 0
fi

echo "API process found with PID: $PID"
echo "Stopping process..."

# Try graceful shutdown first
kill -15 $PID

# Give it a moment to shut down
sleep 2

# Check if it's still running and force kill if needed
if ps -p $PID > /dev/null; then
    echo "Process still running, force killing..."
    kill -9 $PID
    echo "Process force killed."
else
    echo "Process stopped gracefully."
fi

echo "API has been shut down."
