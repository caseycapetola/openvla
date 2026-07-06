#!/usr/bin/env bash

set -euo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="perf_tests/7b/openvla_load_test/A100-2/${TIMESTAMP}"
OUTPUT_FILE="$OUTPUT_DIR/results_${TIMESTAMP}.txt"
SERVER_URL="${SERVER_URL:-http://0.0.0.0:8000/act}"
SERVER_PROFILER_URL="${SERVER_URL%/act}/profiler"
NUM_CLIENTS=5
REQUESTS_PER_CLIENT=10

export RUN_TIMESTAMP="$TIMESTAMP"
export RUN_OUTPUT_DIR="$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
echo "Writing results to $OUTPUT_FILE"
echo "Run directory: $OUTPUT_DIR" | tee "$OUTPUT_FILE"
echo "Clients: $NUM_CLIENTS, requests per client: $REQUESTS_PER_CLIENT" | tee -a "$OUTPUT_FILE"
echo "Performance Test Run - $(date)" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

if ! python - "$SERVER_PROFILER_URL" <<'PY'
import sys
import requests

try:
    requests.get(sys.argv[1], timeout=1).raise_for_status()
except Exception:
    raise SystemExit(1)
else:
    raise SystemExit(0)
PY
then
    echo "Server is not reachable at $SERVER_URL. Start it with run_openvla_server.sh first." | tee -a "$OUTPUT_FILE"
    exit 1
fi

CLIENT_PIDS=()
for client_id in $(seq 1 "$NUM_CLIENTS"); do
    CLIENT_DIR="$OUTPUT_DIR/client_$(printf '%02d' "$client_id")"
    mkdir -p "$CLIENT_DIR"
    echo "Launching client $client_id..." | tee -a "$OUTPUT_FILE"
    python vla-scripts/load_test_client.py \
        --client-id "$client_id" \
        --num-requests "$REQUESTS_PER_CLIENT" \
        --server-url "$SERVER_URL" \
        --output-dir "$OUTPUT_DIR" \
        --run-timestamp "$TIMESTAMP" \
        >"$CLIENT_DIR/client_stdout.log" 2>&1 &
    CLIENT_PIDS+=("$!")
done

for pid in "${CLIENT_PIDS[@]}"; do
    wait "$pid"
done

echo "" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"
echo "Fetching profiler report..." | tee -a "$OUTPUT_FILE"
python vla-scripts/get_perf.py 2>&1 | tee -a "$OUTPUT_FILE"

echo "" | tee -a "$OUTPUT_FILE"
echo "Test completed at $(date)" | tee -a "$OUTPUT_FILE"
echo "Results saved to $OUTPUT_FILE"
echo "Per-client client CSVs: $OUTPUT_DIR/client_*/client_metrics.csv"
echo "Per-client server CSVs: $OUTPUT_DIR/client_*/server_metrics.csv"
