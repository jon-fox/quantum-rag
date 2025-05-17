#!/bin/bash
# Shell script to fetch watch data from WatchRecon for multiple brands
# Usage:
#   ./fetch_watchrecon_daily.sh              # Normal mode
#   ./fetch_watchrecon_daily.sh --test       # Test mode with limited data
#   ./fetch_watchrecon_daily.sh --test-store # Test mode that actually stores data

# Parse command line arguments
TEST_MODE=""
if [ "$1" == "--test" ]; then
  TEST_MODE="--test-mode --limit 3 --pages 1"
  echo "Running in TEST MODE (no database storage)"
elif [ "$1" == "--test-store" ]; then
  TEST_MODE="--limit 3 --pages 1"
  echo "Running in TEST-STORE MODE (limited records, with actual storage)"
fi

# Set logging
echo "Starting WatchRecon data fetch at $(date)"

# Define brands to fetch
BRANDS=("rolex" "omega" "tudor" "patek" "audemars" "grand seiko")

# For test mode, use fewer brands
if [ -n "$TEST_MODE" ]; then
  BRANDS=("rolex" "omega")
fi

# Set pages per brand and delay to be respectful to the server
PAGES=2
DELAY=3

# Environment setup - adjust path as needed
export PYTHONPATH=/mnt/c/Developer_Workspace/watch_arb

# Fetch data for each brand
for BRAND in "${BRANDS[@]}"
do
  echo "==============================================="
  echo "Fetching $BRAND data from WatchRecon..."
  echo "==============================================="
  python3 $PYTHONPATH/src/scripts/fetch_watchrecon_data.py --brand "$BRAND" --pages $PAGES --delay $DELAY $TEST_MODE
  
  # Add small delay between brands to be nice to the server
  if [ -z "$TEST_MODE" ]; then
    echo "Sleeping for 5 seconds before next brand..."
    sleep 5
  else
    echo "Test mode: skipping delay between brands"
  fi
done

echo "WatchRecon data fetch completed at $(date)"