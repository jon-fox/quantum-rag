#!/bin/bash
# Startup script for the quantum_work API
# This script starts app.py or restarts it if it's already running

echo "Starting up quantum_work API..."

# Set the Python path to include the current directory
CURRENT_DIR=$(pwd)
if [[ "${PYTHONPATH}" != *"${CURRENT_DIR}"* ]]; then
    echo "Adding current directory to PYTHONPATH"
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${CURRENT_DIR}"
else
    echo "Current directory already in PYTHONPATH"
fi

# Simple check if app is running
PID=$(pgrep -f "python app.py" || pgrep -f "python3 app.py")

if [ ! -z "$PID" ]; then
    echo "API is already running with PID: $PID"
    echo "Stopping existing process..."
    kill $PID
    sleep 2
    echo "Previous API process stopped"
fi

# Start the API the simple way
echo "Starting API..."
python3 app.py > api_logs.log 2>&1 &

# Store the new PID
NEW_PID=$!
echo "API started with PID: $NEW_PID"
echo "API is now running. Use 'kill $NEW_PID' to stop it manually."
echo "Logs are being written to api_logs.log"

# Print the localhost endpoint information
echo ""
echo "=============================================="
echo "API should be available at: http://localhost:8000"
echo "API documentation available at: http://localhost:8000/docs#"
echo "=============================================="
echo ""
