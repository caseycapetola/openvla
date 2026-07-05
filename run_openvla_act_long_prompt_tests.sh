#!/usr/bin/env bash

set -euo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="perf_tests/v01/openvla_one_client_stress_test/A100/${TIMESTAMP}"
OUTPUT_FILE="$OUTPUT_DIR/results_${TIMESTAMP}.txt"
SERVER_URL="${SERVER_URL:-http://0.0.0.0:8000/act}"
SERVER_PROFILER_URL="${SERVER_URL%/act}/profiler"
CLIENT_SCRIPT="vla-scripts/act_client.py"
NUM_CLIENTS=1
REQUESTS_PER_CLIENT=33
SLEEP_SECONDS=0

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
    echo "Launching long-prompt client $client_id..." | tee -a "$OUTPUT_FILE"

    (
        for request_index in $(seq 1 "$REQUESTS_PER_CLIENT"); do
            RUN_ITERATION="$(printf '%02d-%03d' "$client_id" "$request_index")" \
                python "$CLIENT_SCRIPT"

            if [ "$SLEEP_SECONDS" -gt 0 ]; then
                sleep "$SLEEP_SECONDS"
            fi
        done
    ) >"$CLIENT_DIR/client_stdout.log" 2>&1 &

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
echo "Consolidated profiler CSV: $OUTPUT_DIR/profile_${TIMESTAMP}.csv"
echo "Per-client logs: $OUTPUT_DIR/client_*/client_stdout.log"
