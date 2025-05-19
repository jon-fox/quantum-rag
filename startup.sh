#!/bin/bash
# Startup script for the quantum_work API
# This script starts app.py or restarts it if it's already running

echo "Starting up quantum_work API..."

# Find if the app is already running
PID=$(pgrep -f "python3 app.py" || pgrep -f "python app.py")

if [ ! -z "$PID" ]; then
    echo "API is already running with PID: $PID"
    echo "Stopping existing process..."
    kill -15 $PID
    # Give it a moment to shut down gracefully
    sleep 2
    
    # If it didn't shut down gracefully, force kill
    if ps -p $PID > /dev/null; then
        echo "Force killing process..."
        kill -9 $PID
    fi
    
    echo "Previous API process stopped"
fi

# Start the API
echo "Starting API..."
python3 app.py &

# Store the new PID
NEW_PID=$!
echo "API started with PID: $NEW_PID"
echo "API is now running. Use 'kill $NEW_PID' to stop it manually."
