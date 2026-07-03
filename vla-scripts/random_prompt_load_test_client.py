#!/usr/bin/env python3

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import json_numpy
import numpy as np
import requests

json_numpy.patch()


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    category: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)


PROMPT_POOL: list[PromptSpec] = [
    PromptSpec("simple_01", "simple", "pick up the block"),
    PromptSpec("simple_02", "simple", "move the red cube"),
    PromptSpec("simple_03", "simple", "push the object forward"),
    PromptSpec("simple_04", "simple", "grasp the nearest item"),
    PromptSpec("simple_05", "simple", "place the toy in the bin"),
    PromptSpec("medium_01", "medium", "Pick up the blue block and place it next to the green block."),
    PromptSpec("medium_02", "medium", "Move the small object to the left side of the workspace without knocking over anything."),
    PromptSpec(
        "medium_03",
        "medium",
        "Carefully grasp the red object, lift it a few centimeters, and set it down near the center of the table.",
    ),
    PromptSpec(
        "medium_04",
        "medium",
        "Slide the closest item toward the top edge of the workspace, then stop and wait for the next instruction.",
    ),
    PromptSpec(
        "complex_01",
        "complex",
        "Locate the most clearly visible block, align the gripper above it, close the gripper gently, lift the block without dragging it across the surface, and place it in the empty area marked by the open space near the back of the table.",
    ),
    PromptSpec(
        "complex_02",
        "complex",
        "First inspect the arrangement of objects, choose the item that is farthest from the robot arm, move in a straight line to avoid contact with nearby objects, pick it up, carry it to the target zone, and release it only after confirming the gripper is centered over the drop location.",
    ),
    PromptSpec(
        "complex_03",
        "complex",
        "Reach for the object with the highest contrast against the background, approach from above, maintain a slow and steady descent, secure the object without bumping neighboring items, lift it slightly to verify a stable grasp, then transport it to the open region on the opposite side of the workspace and place it down carefully.",
    ),
    PromptSpec(
        "complex_04",
        "complex",
        "If there are multiple blocks visible, select the one closest to the front edge, avoid objects that are already touching each other, use a deliberate approach path that keeps the gripper clear of obstacles, and complete the pick-and-place motion by setting the block down in the largest available empty area.",
    ),
    PromptSpec(
        "complex_05",
        "complex",
        "Before acting, briefly assess the workspace, identify a safe path to the target object, move the arm slowly until the gripper is directly above the item, close the gripper with enough force to secure the object but not crush it, lift it clear of the surrounding objects, translate to the destination area, and release it smoothly so it lands inside the free space without sliding away.",
    ),
]


def build_payload(instruction: str, unnorm_key: str, image_height: int, image_width: int) -> dict:
    return {
        "image": np.zeros((image_height, image_width, 3), dtype=np.uint8),
        "instruction": instruction,
        "unnorm_key": unnorm_key,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated OpenVLA requests with randomly selected prompts and log client-side latency."
    )
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
        "--unnorm-key",
        default="bridge_orig",
        help="Normalization key to send with each request.",
    )
    parser.add_argument("--image-height", type=int, default=256, help="Height of the dummy image payload.")
    parser.add_argument("--image-width", type=int, default=256, help="Width of the dummy image payload.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request HTTP timeout in seconds.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic prompt selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    client_dir = Path(args.output_dir) / f"client_{args.client_id:02d}"
    client_dir.mkdir(parents=True, exist_ok=True)
    csv_path = client_dir / "client_metrics.csv"
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "run_timestamp",
                    "client_id",
                    "request_id",
                    "request_index",
                    "prompt_id",
                    "prompt_category",
                    "prompt_word_count",
                    "prompt_char_count",
                    "prompt_text",
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
            prompt_spec = rng.choice(PROMPT_POOL)
            payload = build_payload(prompt_spec.text, args.unnorm_key, args.image_height, args.image_width)
            request_id = f"{args.client_id:02d}-{request_index:02d}"
            headers = {
                "X-Client-Id": str(args.client_id),
                "X-Request-Id": request_id,
                "X-Request-Index": str(request_index),
                "X-Run-Timestamp": args.run_timestamp,
                "X-Prompt-Id": prompt_spec.prompt_id,
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
                    prompt_spec.prompt_id,
                    prompt_spec.category,
                    prompt_spec.word_count,
                    prompt_spec.char_count,
                    prompt_spec.text,
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
                f"prompt={prompt_spec.prompt_id} words={prompt_spec.word_count} "
                f"status={status} latency_ms={client_latency_ms:.2f}"
            )

    print(f"Wrote client metrics to {csv_path}")


if __name__ == "__main__":
    main()