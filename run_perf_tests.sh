#!/usr/bin/env bash

# Set up output directory and file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="performance_results/A100"
OUTPUT_FILE="$OUTPUT_DIR/results_${TIMESTAMP}.txt"

mkdir -p "$OUTPUT_DIR"
echo "Writing results to $OUTPUT_FILE"
echo "Performance Test Run - $(date)" | tee "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

# Run act_client.py to test the deployed VLA server and measure performance.
for i in {1..33}; do
    sleep 5
    echo "Running inference test $i..." | tee -a "$OUTPUT_FILE"
    python vla-scripts/act_client.py 2>&1 | tee -a "$OUTPUT_FILE"
done

# Send a request to the profiler endpoint to get the performance report.
echo "" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"
echo "Fetching profiler report..." | tee -a "$OUTPUT_FILE"
python vla-scripts/profiler_client.py 2>&1 | tee -a "$OUTPUT_FILE"

echo "" | tee -a "$OUTPUT_FILE"
echo "Test completed at $(date)" | tee -a "$OUTPUT_FILE"
echo "Results saved to $OUTPUT_FILE"
