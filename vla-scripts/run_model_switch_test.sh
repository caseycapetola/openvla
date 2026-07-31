#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/performance_results/openvla_oft_model_switch_test/$TIMESTAMP}"
OUTPUT_FILE="$OUTPUT_DIR/results_${TIMESTAMP}.txt"
SERVER_URL="${SERVER_URL:-http://0.0.0.0:8777}"
SERVER_PROFILER_URL="${SERVER_URL%/}/profiler"

PRE_SWAP_REQUESTS="${PRE_SWAP_REQUESTS:-20}"
POST_SWAP_REQUESTS="${POST_SWAP_REQUESTS:-20}"
MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-unknown}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:?Set MODEL_B_CHECKPOINT to the checkpoint path/HF repo id to swap to}"
MODEL_B_UNNORM_KEY="${MODEL_B_UNNORM_KEY:?Set MODEL_B_UNNORM_KEY to the unnorm_key for MODEL_B_CHECKPOINT}"

export RUN_TIMESTAMP="$TIMESTAMP"
export RUN_OUTPUT_DIR="$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
echo "Writing results to $OUTPUT_FILE"
echo "Run directory: $OUTPUT_DIR" | tee "$OUTPUT_FILE"
echo "Pre-swap requests: $PRE_SWAP_REQUESTS, post-swap requests: $POST_SWAP_REQUESTS" | tee -a "$OUTPUT_FILE"
echo "Model A (assumed already loaded): $MODEL_A_CHECKPOINT" | tee -a "$OUTPUT_FILE"
echo "Model B (swap target): $MODEL_B_CHECKPOINT" | tee -a "$OUTPUT_FILE"
echo "Model Switch Test Run - $(date)" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

if ! python - "$SERVER_PROFILER_URL" <<'PY'
import sys

import requests

try:
    requests.get(sys.argv[1], timeout=1).raise_for_status()
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
then
    echo "Server is not reachable at $SERVER_URL. Start it with deploy.py first." | tee -a "$OUTPUT_FILE"
    exit 1
fi

python "$REPO_ROOT/vla-scripts/model_switch_test_client.py" \
    --client-id 1 \
    --server-url "$SERVER_URL" \
    --output-dir "$OUTPUT_DIR" \
    --run-timestamp "$TIMESTAMP" \
    --pre-swap-requests "$PRE_SWAP_REQUESTS" \
    --post-swap-requests "$POST_SWAP_REQUESTS" \
    --model-a-checkpoint "$MODEL_A_CHECKPOINT" \
    --model-b-checkpoint "$MODEL_B_CHECKPOINT" \
    --model-b-unnorm-key "$MODEL_B_UNNORM_KEY" \
    "$@" \
    2>&1 | tee -a "$OUTPUT_FILE"

echo "" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"
echo "Fetching profiler report (includes model_load_*/model_teardown ops)..." | tee -a "$OUTPUT_FILE"
python "$REPO_ROOT/vla-scripts/get_perf.py" --server-url "$SERVER_URL" 2>&1 | tee -a "$OUTPUT_FILE"

echo "" | tee -a "$OUTPUT_FILE"
echo "Test completed at $(date)" | tee -a "$OUTPUT_FILE"
echo "Results saved to $OUTPUT_FILE"
echo "Client /act CSV: $OUTPUT_DIR/client_01/model_switch_client_metrics.csv"
echo "Client swap-event CSV: $OUTPUT_DIR/client_01/model_switch_client_events.csv"
echo "Server swap-event CSV: $OUTPUT_DIR/model_switch_metrics.csv"
