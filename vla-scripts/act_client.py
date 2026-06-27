"""
act_client.py

Tiny client for hitting the OpenVLA `/act` endpoint with a lightweight payload.

Example:

    python vla-scripts/act_client.py \
        --host 0.0.0.0 \
        --port 8000 \
        --instruction "pick up the block"
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import json_numpy
import numpy as np
import requests

json_numpy.patch()
get_time = time.perf_counter


@dataclass
class ActClientConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    timeout: float = 30.0
    instruction: str = "do something"
    unnorm_key: Optional[str] = None
    image_size: int = 224


def make_payload(cfg: ActClientConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "image": np.zeros((cfg.image_size, cfg.image_size, 3), dtype=np.uint8),
        "instruction": cfg.instruction,
        "unnorm_key": "bridge_orig",
    }
    if cfg.unnorm_key is not None:
        payload["unnorm_key"] = cfg.unnorm_key
    return payload


def call_act_endpoint(cfg: ActClientConfig) -> Dict[str, Any]:
    payload = make_payload(cfg)
    url = f"http://{cfg.host}:{cfg.port}/act"

    try:
        start = get_time()
        response = requests.post(url, json=payload, timeout=cfg.timeout)
        elapsed_ms = (get_time() - start) * 1000.0
        response.raise_for_status()

        print(f"Latency: {elapsed_ms:.2f} ms")
        return response.json()
    except requests.exceptions.HTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        raise RuntimeError(f"/act request failed with HTTP {status}") from error
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"/act request failed: {error}") from error


def parse_args() -> ActClientConfig:
    parser = argparse.ArgumentParser(description="Call a running OpenVLA /act endpoint and print request latency.")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", default=8000, type=int, help="Server port")
    parser.add_argument("--timeout", default=30.0, type=float, help="Request timeout in seconds")
    parser.add_argument("--instruction", default="do something", help="Instruction sent to the model")
    parser.add_argument("--unnorm_key", default=None, help="Optional normalization key for model action de-normalization")
    parser.add_argument("--image_size", default=256, type=int, help="Square side length for the dummy RGB image")
    args = parser.parse_args()

    return ActClientConfig(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        instruction=args.instruction,
        unnorm_key=args.unnorm_key,
        image_size=args.image_size,
    )


def main() -> None:
    cfg = parse_args()
    action = call_act_endpoint(cfg)
    print("Response:")
    print(action)


if __name__ == "__main__":
    main()
