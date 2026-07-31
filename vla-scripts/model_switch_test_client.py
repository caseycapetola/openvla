#!/usr/bin/env python3

"""
model_switch_test_client.py

Exercises `POST /load_model` on a running OpenVLA deploy server (see `vla-scripts/deploy.py`) to measure
the performance cost of switching the loaded model mid-session:

  1. Sends `--pre-swap-requests` baseline `/act` calls against whatever model is currently loaded ("model A").
  2. Calls `POST /load_model` with overrides describing "model B", and records the server's own timing
     breakdown (teardown + per-component load latency) from the response body.
  3. Sends `--post-swap-requests` `/act` calls against the newly-loaded model, tagging the very first one
     distinctly (`post_swap_first`) from the rest (`post_swap_steady`) so a cold first-inference-after-swap
     cost is visible separately from steady-state in the resulting CSV.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import json_numpy
import numpy as np
import requests

json_numpy.patch()


def str2bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "y")


def build_payload(instruction: str, unnorm_key: str, image_height: int, image_width: int, state_dim: int) -> dict:
    return {
        "full_image": np.zeros((image_height, image_width, 3), dtype=np.uint8),
        "left_wrist_image": np.zeros((image_height, image_width, 3), dtype=np.uint8),
        "state": np.zeros((state_dim,), dtype=np.float32),
        "instruction": instruction,
        "unnorm_key": unnorm_key,
    }


def header_float(response: requests.Response, header_name: str) -> str:
    value = response.headers.get(header_name, "")
    if value == "":
        return ""
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return ""


def build_model_b_overrides(args: argparse.Namespace) -> dict:
    overrides = {
        "pretrained_checkpoint": args.model_b_checkpoint,
        "unnorm_key": args.model_b_unnorm_key,
        "use_l1_regression": args.model_b_use_l1_regression,
        "use_diffusion": args.model_b_use_diffusion,
        "use_film": args.model_b_use_film,
        "num_images_in_input": args.model_b_num_images_in_input,
        "use_proprio": args.model_b_use_proprio,
        "center_crop": args.model_b_center_crop,
        "lora_rank": args.model_b_lora_rank,
        "load_in_8bit": args.model_b_load_in_8bit,
        "load_in_4bit": args.model_b_load_in_4bit,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    if args.model_b_config_json:
        with open(args.model_b_config_json, "r") as f:
            overrides.update(json.load(f))

    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise POST /load_model and measure pre/post-swap /act latency."
    )
    parser.add_argument("--client-id", type=int, default=1, help="Numeric client identifier.")
    parser.add_argument(
        "--server-url",
        default=os.environ.get("OPENVLA_SERVER_URL", "http://0.0.0.0:8777"),
        help="Base OpenVLA server URL (without /act or /load_model).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("RUN_OUTPUT_DIR", "performance_results/openvla_oft_model_switch_test"),
        help="Base directory where client metrics should be written.",
    )
    parser.add_argument(
        "--run-timestamp",
        default=os.environ.get("RUN_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="Shared run timestamp used to tie client and server logs together.",
    )
    parser.add_argument("--pre-swap-requests", type=int, default=20, help="Baseline /act calls before the swap.")
    parser.add_argument(
        "--post-swap-requests",
        type=int,
        default=20,
        help="/act calls after the swap, including the first (cold) one.",
    )
    parser.add_argument(
        "--model-a-checkpoint",
        default="unknown",
        help="Label only -- the checkpoint assumed to already be loaded on the server (for CSV logging).",
    )
    parser.add_argument("--model-b-checkpoint", required=True, help="Checkpoint path/HF repo id to swap to.")
    parser.add_argument("--model-b-unnorm-key", required=True, help="unnorm_key for the model-B checkpoint.")
    parser.add_argument("--model-b-use-l1-regression", type=str2bool, default=None)
    parser.add_argument("--model-b-use-diffusion", type=str2bool, default=None)
    parser.add_argument("--model-b-use-film", type=str2bool, default=None)
    parser.add_argument("--model-b-num-images-in-input", type=int, default=None)
    parser.add_argument("--model-b-use-proprio", type=str2bool, default=None)
    parser.add_argument("--model-b-center-crop", type=str2bool, default=None)
    parser.add_argument("--model-b-lora-rank", type=int, default=None)
    parser.add_argument("--model-b-load-in-8bit", type=str2bool, default=None)
    parser.add_argument("--model-b-load-in-4bit", type=str2bool, default=None)
    parser.add_argument(
        "--model-b-config-json",
        default=None,
        help="Path to a JSON file of raw DeployConfig overrides, merged on top of the named --model-b-* flags.",
    )
    parser.add_argument("--instruction", default="pick up the blue block", help="Instruction sent with each /act call.")
    parser.add_argument("--unnorm-key", default="bridge_orig", help="unnorm_key sent with pre-swap /act payloads.")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-/act-request HTTP timeout in seconds.")
    parser.add_argument("--swap-timeout", type=float, default=600.0, help="HTTP timeout for the /load_model call.")
    return parser.parse_args()


def send_act_request(
    writer: csv.writer,
    act_url: str,
    payload: dict,
    args: argparse.Namespace,
    phase: str,
    request_index: int,
    model_checkpoint: str,
) -> None:
    request_id = f"{args.client_id:02d}-{phase}-{request_index:02d}"
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
    server_total_latency_ms = ""
    server_decode_latency_ms = ""
    server_parse_latency_ms = ""
    server_inference_latency_ms = ""
    server_encode_latency_ms = ""
    server_overhead_latency_ms = ""

    try:
        response = requests.post(act_url, json=payload, headers=headers, timeout=args.timeout)
        status_code = response.status_code
        response_bytes = len(response.content)
        server_total_latency_ms = header_float(response, "X-Server-Total-Latency-Ms")
        server_decode_latency_ms = header_float(response, "X-Server-Decode-Latency-Ms")
        server_parse_latency_ms = header_float(response, "X-Server-Parse-Latency-Ms")
        server_inference_latency_ms = header_float(response, "X-Server-Inference-Latency-Ms")
        server_encode_latency_ms = header_float(response, "X-Server-Encode-Latency-Ms")
        server_overhead_latency_ms = header_float(response, "X-Server-Overhead-Latency-Ms")
        response.raise_for_status()
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
            phase,
            request_id,
            request_index,
            model_checkpoint,
            status,
            status_code,
            client_latency_ms,
            server_total_latency_ms,
            server_decode_latency_ms,
            server_parse_latency_ms,
            server_inference_latency_ms,
            server_encode_latency_ms,
            server_overhead_latency_ms,
            response_bytes,
            error_text,
            started_at,
            finished_at,
        ]
    )
    print(
        f"client={args.client_id} phase={phase} request={request_index} "
        f"status={status} client_ms={client_latency_ms:.2f} server_ms={server_total_latency_ms or 'n/a'}"
    )


def send_load_model_request(
    switch_writer: csv.writer, load_model_url: str, overrides: dict, args: argparse.Namespace
) -> None:
    print(f"Swapping model -> {overrides.get('pretrained_checkpoint')} ...")
    swap_start = time.perf_counter()
    status = "ok"
    error_text = ""
    body = {}
    try:
        response = requests.post(load_model_url, json=overrides, timeout=args.swap_timeout)
        body = response.json()
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error_text = str(exc)
    client_swap_ms = round((time.perf_counter() - swap_start) * 1000, 4)

    switch_writer.writerow(
        [
            args.run_timestamp,
            args.client_id,
            body.get("switch_id", ""),
            status if status == "error" else body.get("status", "ok"),
            body.get("old_checkpoint", args.model_a_checkpoint),
            body.get("new_checkpoint", overrides.get("pretrained_checkpoint", "")),
            body.get("teardown_ms", ""),
            body.get("load_vla_ms", ""),
            body.get("load_proprio_projector_ms", ""),
            body.get("load_action_head_ms", ""),
            body.get("load_processor_ms", ""),
            body.get("load_resize_size_ms", ""),
            body.get("load_total_ms", ""),
            body.get("swap_total_ms", ""),
            client_swap_ms,
            error_text or body.get("error", ""),
        ]
    )
    print(f"Swap finished: status={status} client_swap_ms={client_swap_ms:.2f} server_swap_ms={body.get('swap_total_ms', 'n/a')}")

    if status == "error":
        raise RuntimeError(f"/load_model failed: {error_text or body}")


def main() -> None:
    args = parse_args()
    server_url = args.server_url.rstrip("/")
    act_url = f"{server_url}/act"
    load_model_url = f"{server_url}/load_model"

    client_dir = Path(args.output_dir) / f"client_{args.client_id:02d}"
    client_dir.mkdir(parents=True, exist_ok=True)

    act_csv_path = client_dir / "model_switch_client_metrics.csv"
    act_file_exists = act_csv_path.exists()
    swap_csv_path = client_dir / "model_switch_client_events.csv"
    swap_file_exists = swap_csv_path.exists()

    payload = build_payload(args.instruction, args.unnorm_key, args.image_height, args.image_width, args.state_dim)
    model_b_overrides = build_model_b_overrides(args)

    with open(act_csv_path, "a", newline="") as act_f, open(swap_csv_path, "a", newline="") as swap_f:
        act_writer = csv.writer(act_f)
        if not act_file_exists:
            act_writer.writerow(
                [
                    "run_timestamp",
                    "client_id",
                    "phase",
                    "request_id",
                    "request_index",
                    "model_checkpoint",
                    "status",
                    "status_code",
                    "client_latency_ms",
                    "server_total_latency_ms",
                    "server_decode_latency_ms",
                    "server_parse_latency_ms",
                    "server_inference_latency_ms",
                    "server_encode_latency_ms",
                    "server_overhead_latency_ms",
                    "response_bytes",
                    "error",
                    "started_at",
                    "finished_at",
                ]
            )

        swap_writer = csv.writer(swap_f)
        if not swap_file_exists:
            swap_writer.writerow(
                [
                    "run_timestamp",
                    "client_id",
                    "switch_id",
                    "status",
                    "old_checkpoint",
                    "new_checkpoint",
                    "teardown_ms",
                    "load_vla_ms",
                    "load_proprio_projector_ms",
                    "load_action_head_ms",
                    "load_processor_ms",
                    "load_resize_size_ms",
                    "load_total_ms",
                    "server_swap_total_ms",
                    "client_swap_total_ms",
                    "error",
                ]
            )

        # Phase 1: baseline requests against whatever model is currently loaded ("model A")
        for i in range(1, args.pre_swap_requests + 1):
            send_act_request(act_writer, act_url, payload, args, "pre_swap", i, args.model_a_checkpoint)
            act_f.flush()

        # Phase 2: swap to model B
        send_load_model_request(swap_writer, load_model_url, model_b_overrides, args)
        swap_f.flush()

        # Phase 3: post-swap requests against model B; tag the first one distinctly
        post_swap_payload = dict(payload)
        post_swap_payload["unnorm_key"] = args.model_b_unnorm_key
        for i in range(1, args.post_swap_requests + 1):
            phase = "post_swap_first" if i == 1 else "post_swap_steady"
            send_act_request(
                act_writer, act_url, post_swap_payload, args, phase, i, args.model_b_checkpoint
            )
            act_f.flush()

    print(f"Wrote client /act metrics to {act_csv_path}")
    print(f"Wrote model-switch events to {swap_csv_path}")


if __name__ == "__main__":
    main()
