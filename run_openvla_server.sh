#!/usr/bin/env bash

set -euo pipefail

SERVER_RUN_TIMESTAMP="${SERVER_RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-performance_results/v01/openvla_server/${SERVER_RUN_TIMESTAMP}}"
SERVER_URL="${SERVER_URL:-http://0.0.0.0:8000/act}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8000}"
OPENVLA_PATH="${OPENVLA_PATH:-openvla/openvla-7b}"
PID_FILE="${PID_FILE:-$SERVER_LOG_DIR/openvla_server.pid}"
LOG_FILE="${LOG_FILE:-$SERVER_LOG_DIR/server_stdout.log}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --openvla_path)
            OPENVLA_PATH="$2"
            shift 2
            ;;
        --openvla_path=*)
            OPENVLA_PATH="${1#*=}"
            shift
            ;;
        --host)
            SERVER_HOST="$2"
            shift 2
            ;;
        --host=*)
            SERVER_HOST="${1#*=}"
            shift
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --port=*)
            SERVER_PORT="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

export RUN_TIMESTAMP="${RUN_TIMESTAMP:-$SERVER_RUN_TIMESTAMP}"
export RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$SERVER_LOG_DIR}"

mkdir -p "$SERVER_LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Server already running with PID $(cat "$PID_FILE")"
    exit 0
fi

echo "Starting OpenVLA server at $SERVER_URL using model $OPENVLA_PATH"
python vla-scripts/deploy.py --openvla_path "$OPENVLA_PATH" --host "$SERVER_HOST" --port "$SERVER_PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

echo "Waiting for server to become ready..."
for _ in {1..120}; do
    if python - "$SERVER_URL" <<'PY'
import sys
import requests

try:
    requests.get(sys.argv[1].replace("/act", "/profiler"), timeout=1).raise_for_status()
except Exception:
    raise SystemExit(1)
PY
    then
        echo "Server is ready. PID: $SERVER_PID"
        exit 0
    fi
    sleep 5
done

echo "Server did not become ready in time. See $LOG_FILE for details."
exit 1