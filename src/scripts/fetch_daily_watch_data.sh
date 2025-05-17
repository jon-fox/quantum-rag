#!/bin/bash
# Cron job script to fetch watch data daily
# Add to crontab with: 
# 0 6 * * * /path/to/watch_arb/src/scripts/fetch_daily_watch_data.sh >> /path/to/watch_arb/cron_execution.log 2>&1

# Navigate to project root directory
cd "$(dirname "$(dirname "$(dirname "$0")")")" || exit 1

# Load environment variables if .env exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "$(date): Loaded environment variables from .env"
fi

# Set Python path
export PYTHONPATH="$PYTHONPATH:$(pwd)"
echo "$(date): Set PYTHONPATH to include $(pwd)"

# Log execution start
echo "$(date): Starting daily watch data collection"

# Run the data collection script for all profiles
python -m src.scripts.fetch_watch_data --all-profiles

# Check exit status
if [ $? -eq 0 ]; then
  echo "$(date): Data collection completed successfully"
else
  echo "$(date): ERROR - Data collection failed with exit code $?"
fi