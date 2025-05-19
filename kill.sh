#!/bin/bash
# Kill script for quantum_work API
# This script finds and kills the running API process

echo "Searching for quantum_work API process..."

# Look for python running app.py
PID=$(pgrep -f "python app.py" || pgrep -f "python3 app.py")

if [ -z "$PID" ]; then
    echo "No API process found running."
    exit 0
fi

echo "API process found with PID: $PID"
echo "Stopping process..."

if [ -z "$PID" ]; then
    echo "No API process found running."
    exit 0
fi

echo "API process found with PID: $PID"
echo "Stopping process..."

# Kill the process
kill $PID
echo "Process stopped."

echo "API has been shut down."
