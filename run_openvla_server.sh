#!/usr/bin/env bash

set -euo pipefail

SERVER_LOG_DIR="${SERVER_LOG_DIR:-performance_results/v01/openvla_server}"
SERVER_URL="${SERVER_URL:-http://0.0.0.0:8000/act}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
SERVER_PORT="${SERVER_PORT:-8000}"
PID_FILE="${PID_FILE:-$SERVER_LOG_DIR/openvla_server.pid}"
LOG_FILE="${LOG_FILE:-$SERVER_LOG_DIR/server_stdout.log}"

mkdir -p "$SERVER_LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Server already running with PID $(cat "$PID_FILE")"
    exit 0
fi

echo "Starting OpenVLA server at $SERVER_URL"
python vla-scripts/deploy.py --host "$SERVER_HOST" --port "$SERVER_PORT" >"$LOG_FILE" 2>&1 &
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