#!/usr/bin/env bash

set -euo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="perf_tests/v01/openvla_load_test/${TIMESTAMP}"
OUTPUT_FILE="$OUTPUT_DIR/results_${TIMESTAMP}.txt"
SERVER_LOG="$OUTPUT_DIR/server_stdout.log"
SERVER_URL="${SERVER_URL:-http://0.0.0.0:8000/act}"
SERVER_PROFILER_URL="${SERVER_URL%/act}/profiler"
NUM_CLIENTS=5
REQUESTS_PER_CLIENT=10
START_SERVER="${START_SERVER:-1}"

export RUN_TIMESTAMP="$TIMESTAMP"
export RUN_OUTPUT_DIR="$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
echo "Writing results to $OUTPUT_FILE"
echo "Run directory: $OUTPUT_DIR" | tee "$OUTPUT_FILE"
echo "Clients: $NUM_CLIENTS, requests per client: $REQUESTS_PER_CLIENT" | tee -a "$OUTPUT_FILE"
echo "Performance Test Run - $(date)" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" || true
        wait "$SERVER_PID" || true
    fi
}

if [[ "$START_SERVER" == "1" ]]; then
    echo "Starting OpenVLA server..." | tee -a "$OUTPUT_FILE"
    python vla-scripts/deploy.py >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    trap cleanup EXIT

    echo "Waiting for server at $SERVER_PROFILER_URL ..." | tee -a "$OUTPUT_FILE"
    SERVER_READY=0
    for _ in {1..120}; do
        if python - "$SERVER_PROFILER_URL" <<'PY'
import sys
import requests

try:
    requests.get(sys.argv[1], timeout=1).raise_for_status()
except Exception:
    raise SystemExit(1)
PY
        then
            SERVER_READY=1
            break
        fi
        sleep 5
    done

    if [[ "$SERVER_READY" -ne 1 ]]; then
        echo "Server did not become ready in time. See $SERVER_LOG for details." | tee -a "$OUTPUT_FILE"
        exit 1
    fi
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

if [[ "$START_SERVER" == "1" ]]; then
    cleanup
    trap - EXIT
fi

echo "" | tee -a "$OUTPUT_FILE"
echo "Test completed at $(date)" | tee -a "$OUTPUT_FILE"
echo "Results saved to $OUTPUT_FILE"
echo "Per-client client CSVs: $OUTPUT_DIR/client_*/client_metrics.csv"
echo "Per-client server CSVs: $OUTPUT_DIR/client_*/server_metrics.csv"