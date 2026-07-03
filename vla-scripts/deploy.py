"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import csv
import json
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

import time
import statistics
import contextlib

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# Performance testing utilities
@dataclass
class LatencyStats:
    operation: str
    samples: list[float] = field(default_factory=list)

    def record(self, elapsed_ms: float):
        self.samples.append(elapsed_ms)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def p95(self) -> float:
        if len(self.samples) < 2:
            return self.samples[0] if self.samples else 0.0
        if len(self.samples) < 20:
            # Not enough data for quantiles — use sorted index instead
            sorted_samples = sorted(self.samples)
            idx = int(len(sorted_samples) * 0.95)
            return sorted_samples[min(idx, len(sorted_samples) - 1)]
        return statistics.quantiles(self.samples, n=20)[18]

    def summary(self, discard_first_n: int = 5) -> dict:
        # Remove the first `discard_first_n` samples if they exist
        if len(self.samples) > discard_first_n:
            samples = self.samples[discard_first_n:]
        else:
            samples = self.samples

        return {
            "operation": self.operation,
            "calls": len(samples),
            "mean_ms": round(statistics.mean(samples) if samples else 0.0, 4),
            "min_ms": round(min(samples) if samples else 0.0, 4),
            "max_ms": round(max(samples) if samples else 0.0, 4),
            "p95_ms": round(
                statistics.quantiles(samples, n=20)[18] if len(samples) >= 2 else (samples[0] if samples else 0.0), 4
            ),
        }


class PerformanceProfiler:

    def __init__(self, sample_writer: Optional["OperationSampleWriter"] = None):
        self._stats: dict[str, LatencyStats] = {}
        self._sample_writer = sample_writer

    @contextlib.contextmanager
    def measure(self, operation: str, context: Optional[dict[str, Any]] = None):
        """Context manager to time any block of code."""
        stats = self._stats.setdefault(operation, LatencyStats(operation))
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            stats.record(elapsed_ms)
            if self._sample_writer is not None:
                self._sample_writer.append(
                    {
                        "operation": operation,
                        "sample_index": len(stats.samples),
                        "elapsed_ms": round(elapsed_ms, 4),
                        **(context or {}),
                    }
                )

    def report(self) -> list[dict]:
        return [s.summary() for s in self._stats.values()]

    def full_report(self) -> list[dict]:
        rows: list[dict] = []
        for operation, stats in self._stats.items():
            for sample_index, elapsed_ms in enumerate(stats.samples, start=1):
                rows.append(
                    {
                        "operation": operation,
                        "sample_index": sample_index,
                        "elapsed_ms": round(elapsed_ms, 4),
                    }
                )
        return rows

    def print_report(self):
        print(f"\n{'Operation':<30} {'Calls':>6} {'Mean ms':>10} {'Min ms':>10} {'Max ms':>10} {'P95 ms':>10}")
        print("-" * 80)
        for s in self.report():
            print(
                f"{s['operation']:<30} {s['calls']:>6} {s['mean_ms']:>10} {s['min_ms']:>10} {s['max_ms']:>10} {s['p95_ms']:>10}"
            )

    def print_full_report(self):
        print(f"\n{'Operation':<30} {'Sample':>6} {'Elapsed ms':>12}")
        print("-" * 54)
        for s in self.full_report():
            print(f"{s['operation']:<30} {s['sample_index']:>6} {s['elapsed_ms']:>12}")


@dataclass
class OperationSampleWriter:
    output_dir: Path
    run_timestamp: str
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _csv_path(self) -> Path:
        return self.output_dir / "server_operation_metrics.csv"

    def append(self, row: dict[str, Any]) -> None:
        csv_path = self._csv_path()
        file_exists = csv_path.exists()

        with self._lock, open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "run_timestamp",
                        "operation",
                        "sample_index",
                        "elapsed_ms",
                        "client_id",
                        "request_id",
                        "request_index",
                        "client_host",
                        "logged_at",
                    ]
                )
            writer.writerow(
                [
                    self.run_timestamp,
                    row.get("operation", ""),
                    row.get("sample_index", ""),
                    row.get("elapsed_ms", 0.0),
                    row.get("client_id", "unknown"),
                    row.get("request_id", ""),
                    row.get("request_index", ""),
                    row.get("client_host", ""),
                    datetime.now().isoformat(timespec="seconds"),
                ]
            )


@dataclass
class RequestLogWriter:
    output_dir: Path
    run_timestamp: str
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_client_id(self, client_id: str) -> str:
        if client_id.isdigit():
            return f"{int(client_id):02d}"
        return client_id

    def _csv_path(self, client_id: str) -> Path:
        client_dir = self.output_dir / f"client_{self._normalize_client_id(client_id)}"
        client_dir.mkdir(parents=True, exist_ok=True)
        return client_dir / "server_metrics.csv"

    def append(self, row: dict[str, Any]) -> None:
        client_id = str(row.get("client_id", "unknown"))
        csv_path = self._csv_path(client_id)
        file_exists = csv_path.exists()

        with self._lock, open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "run_timestamp",
                        "client_id",
                        "request_id",
                        "request_index",
                        "status",
                        "server_latency_ms",
                        "client_host",
                        "logged_at",
                    ]
                )
            writer.writerow(
                [
                    self.run_timestamp,
                    row.get("client_id", "unknown"),
                    row.get("request_id", ""),
                    row.get("request_index", ""),
                    row.get("status", "ok"),
                    row.get("server_latency_ms", 0.0),
                    row.get("client_host", ""),
                    datetime.now().isoformat(timespec="seconds"),
                ]
            )


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # Optional request / operation loggers used for performance tests.
        perf_output_dir = os.environ.get("RUN_OUTPUT_DIR")
        perf_run_timestamp = os.environ.get("RUN_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.request_logger = None
        self.sample_logger = None
        if perf_output_dir is not None:
            perf_output_path = Path(perf_output_dir)
            self.request_logger = RequestLogWriter(perf_output_path, perf_run_timestamp)
            self.sample_logger = OperationSampleWriter(perf_output_path, perf_run_timestamp)

        self.profiler = PerformanceProfiler(sample_writer=self.sample_logger)

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    def predict_action(self, payload: Dict[str, Any], request: Request) -> str:
        client_id = request.headers.get("x-client-id", "unknown")
        request_id = request.headers.get("x-request-id", "")
        request_index = request.headers.get("x-request-index", "")
        client_host = request.client.host if request.client is not None else ""
        request_start = time.perf_counter()
        request_status = "ok"
        operation_context = {
            "client_id": client_id,
            "request_id": request_id,
            "request_index": request_index,
            "client_host": client_host,
        }
        try:
            if double_encode := "encoded" in payload:
                with self.profiler.measure("0_decode_payload", context=operation_context):
                    # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                    assert len(payload.keys()) == 1, "Only uses encoded payload!"
                    payload = json.loads(payload["encoded"])

            # Parse payload components
            with self.profiler.measure("1_parse_payload", context=operation_context):
                image, instruction = payload["image"], payload["instruction"]
                unnorm_key = payload.get("unnorm_key", None)

            # Run VLA Inference
            with self.profiler.measure("2_run_inference", context=operation_context):
                prompt = get_openvla_prompt(instruction, self.openvla_path)
                inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(
                    self.device, dtype=torch.bfloat16
                )
                action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            if double_encode:
                with self.profiler.measure("3_encode_and_return_action", context=operation_context):
                    return JSONResponse(json_numpy.dumps(action))
            else:
                with self.profiler.measure("3_return_action", context=operation_context):
                    return JSONResponse(action)
        except:  # noqa: E722
            request_status = "error"
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"
        finally:
            if self.request_logger is not None:
                self.request_logger.append(
                    {
                        "client_id": client_id,
                        "request_id": request_id,
                        "request_index": request_index,
                        "status": request_status,
                        "server_latency_ms": round((time.perf_counter() - request_start) * 1000, 4),
                        "client_host": client_host,
                    }
                )

    def get_profiler_report(self) -> list[dict]:
        return self.profiler.report()

    def get_profiler_full_report(self) -> list[dict]:
        return self.profiler.full_report()

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        self.app.get("/profiler")(self.get_profiler_report)
        self.app.get("/profiler/full")(self.get_profiler_full_report)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = "openvla/openvla-7b"               # HF Hub Path (or path to local run directory)

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    print(f"[INIT] OPENVLA_PATH: {cfg.openvla_path}")
    server = OpenVLAServer(cfg.openvla_path)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
