#!/usr/bin/env bash

# Run act_client.py to test the deployed VLA server and measure performance.
for i in {1..10}; do
    echo "Running inference test $i..."
    python vla-scripts/act_client.py
done

# Send a request to the profiler endpoint to get the performance report.
echo "Fetching profiler report..."
python vla-scripts/profiler_client.py