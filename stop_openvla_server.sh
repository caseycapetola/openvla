#!/usr/bin/env bash

set -euo pipefail

SERVER_LOG_DIR="${SERVER_LOG_DIR:-performance_results/v01/openvla_server}"
PID_FILE="${PID_FILE:-$SERVER_LOG_DIR/openvla_server.pid}"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found at $PID_FILE"
    exit 0
fi

SERVER_PID=$(cat "$PID_FILE")

if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
    echo "Stopped OpenVLA server PID $SERVER_PID"
else
    echo "OpenVLA server PID $SERVER_PID is not running"
fi

rm -f "$PID_FILE"