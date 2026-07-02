#!/usr/bin/env python3

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import json_numpy
import numpy as np
import requests

json_numpy.patch()


def build_payload(instruction: str, unnorm_key: str, image_height: int, image_width: int) -> dict:
    return {
        "image": np.zeros((image_height, image_width, 3), dtype=np.uint8),
        "instruction": instruction,
        "unnorm_key": unnorm_key,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated OpenVLA requests and log client-side latency.")
    parser.add_argument("--client-id", type=int, required=True, help="Numeric client identifier.")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of back-to-back requests to send.")
    parser.add_argument(
        "--server-url",
        default=os.environ.get("OPENVLA_SERVER_URL", "http://0.0.0.0:8000/act"),
        help="OpenVLA /act endpoint URL.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("RUN_OUTPUT_DIR", "performance_results/v01/openvla_load_test"),
        help="Base directory where client metrics should be written.",
    )
    parser.add_argument(
        "--run-timestamp",
        default=os.environ.get("RUN_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="Shared run timestamp used to tie client and server logs together.",
    )
    parser.add_argument(
        "--instruction",
        default="pick up the blue block",
        help="Instruction to send with each request.",
    )
    parser.add_argument(
        "--unnorm-key",
        default="bridge_orig",
        help="Normalization key to send with each request.",
    )
    parser.add_argument("--image-height", type=int, default=256, help="Height of the dummy image payload.")
    parser.add_argument("--image-width", type=int, default=256, help="Width of the dummy image payload.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request HTTP timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client_dir = Path(args.output_dir) / f"client_{args.client_id:02d}"
    client_dir.mkdir(parents=True, exist_ok=True)
    csv_path = client_dir / "client_metrics.csv"
    file_exists = csv_path.exists()

    payload = build_payload(args.instruction, args.unnorm_key, args.image_height, args.image_width)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "run_timestamp",
                    "client_id",
                    "request_id",
                    "request_index",
                    "status",
                    "status_code",
                    "client_latency_ms",
                    "response_bytes",
                    "error",
                    "started_at",
                    "finished_at",
                ]
            )

        for request_index in range(1, args.num_requests + 1):
            request_id = f"{args.client_id:02d}-{request_index:02d}"
            headers = {
                "X-Client-Id": str(args.client_id),
                "X-Request-Id": request_id,
                "X-Request-Index": str(request_index),
                "X-Run-Timestamp": args.run_timestamp,
            }

            request_start = time.perf_counter()
            started_at = datetime.now().isoformat(timespec="seconds")
            response_bytes = 0
            status = "ok"
            status_code = ""
            error_text = ""

            try:
                response = requests.post(args.server_url, json=payload, headers=headers, timeout=args.timeout)
                status_code = response.status_code
                response.raise_for_status()
                response_bytes = len(response.content)
                _ = response.json()
            except Exception as exc:  # noqa: BLE001
                status = "error"
                error_text = str(exc)

            finished_at = datetime.now().isoformat(timespec="seconds")
            client_latency_ms = round((time.perf_counter() - request_start) * 1000, 4)

            writer.writerow(
                [
                    args.run_timestamp,
                    args.client_id,
                    request_id,
                    request_index,
                    status,
                    status_code,
                    client_latency_ms,
                    response_bytes,
                    error_text,
                    started_at,
                    finished_at,
                ]
            )
            print(
                f"client={args.client_id} request={request_index}/{args.num_requests} "
                f"status={status} latency_ms={client_latency_ms:.2f}"
            )

    print(f"Wrote client metrics to {csv_path}")


if __name__ == "__main__":
    main()