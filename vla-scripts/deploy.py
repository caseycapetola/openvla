"""
deploy.py

Starts a VLA server which the client can query to get robot actions.
Also exposes lightweight profiling endpoints used for inference performance tests.
"""

import contextlib
import csv
import dataclasses
import json
import logging
import os.path
import statistics
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import json_numpy

# ruff: noqa: E402

json_numpy.patch()

import draccus
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    STOP_INDEX,
)

def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


@dataclass
class LatencyStats:
    operation: str
    samples: list[float] = field(default_factory=list)

    def record(self, elapsed_ms: float) -> None:
        self.samples.append(elapsed_ms)

    def summary(self, discard_first_n: int = 5) -> dict[str, Any]:
        samples = self.samples[discard_first_n:] if len(self.samples) > discard_first_n else self.samples
        return {
            "operation": self.operation,
            "calls": len(samples),
            "mean_ms": round(statistics.mean(samples) if samples else 0.0, 4),
            "min_ms": round(min(samples) if samples else 0.0, 4),
            "max_ms": round(max(samples) if samples else 0.0, 4),
            "p95_ms": round(
                statistics.quantiles(samples, n=20)[18] if len(samples) >= 2 else (samples[0] if samples else 0.0),
                4,
            ),
        }


class PerformanceProfiler:
    def __init__(self, sample_writer: Optional["OperationSampleWriter"] = None):
        self._stats: dict[str, LatencyStats] = {}
        self._sample_writer = sample_writer

    @contextlib.contextmanager
    def measure(self, operation: str, context: Optional[dict[str, Any]] = None):
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

    def report(self) -> list[dict[str, Any]]:
        return [stats.summary() for stats in self._stats.values()]

    def full_report(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for operation, stats in self._stats.items():
            for sample_index, elapsed_ms in enumerate(stats.samples, start=1):
                rows.append({"operation": operation, "sample_index": sample_index, "elapsed_ms": round(elapsed_ms, 4)})
        return rows


@dataclass
class OperationSampleWriter:
    output_dir: Path
    run_timestamp: str
    filename: str = "server_operation_metrics.csv"
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def csv_path(self) -> Path:
        return self.output_dir / self.filename

    def _csv_path(self) -> Path:
        return self.csv_path()

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
                        "server_decode_ms",
                        "server_parse_ms",
                        "server_inference_ms",
                        "server_encode_ms",
                        "server_overhead_ms",
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
                    row.get("server_decode_ms", 0.0),
                    row.get("server_parse_ms", 0.0),
                    row.get("server_inference_ms", 0.0),
                    row.get("server_encode_ms", 0.0),
                    row.get("server_overhead_ms", 0.0),
                    row.get("client_host", ""),
                    datetime.now().isoformat(timespec="seconds"),
                ]
            )


@dataclass
class ModelSwitchLogWriter:
    output_dir: Path
    run_timestamp: str
    filename: str = "model_switch_metrics.csv"
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def csv_path(self) -> Path:
        return self.output_dir / self.filename

    def append(self, row: dict[str, Any]) -> None:
        csv_path = self.csv_path()
        file_exists = csv_path.exists()

        with self._lock, open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "run_timestamp",
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
                        "swap_total_ms",
                        "client_host",
                        "error",
                        "logged_at",
                    ]
                )
            writer.writerow(
                [
                    self.run_timestamp,
                    row.get("switch_id", ""),
                    row.get("status", "ok"),
                    row.get("old_checkpoint", ""),
                    row.get("new_checkpoint", ""),
                    row.get("teardown_ms", 0.0),
                    row.get("load_vla_ms", 0.0),
                    row.get("load_proprio_projector_ms", 0.0),
                    row.get("load_action_head_ms", 0.0),
                    row.get("load_processor_ms", 0.0),
                    row.get("load_resize_size_ms", 0.0),
                    row.get("load_total_ms", 0.0),
                    row.get("swap_total_ms", 0.0),
                    row.get("client_host", ""),
                    row.get("error", ""),
                    datetime.now().isoformat(timespec="seconds"),
                ]
            )


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.cfg = cfg

        # Guards self.vla/self.action_head/self.proprio_projector/self.processor/self.cfg so that `/act`
        # and `/load_model` can't run concurrently. A plain `threading.Lock` (not `asyncio.Lock`) is correct
        # here because `get_server_action`/`load_model` are sync `def` handlers, and Starlette/uvicorn runs
        # sync FastAPI handlers in a thread-pool executor -- concurrent requests genuinely run on separate
        # OS threads.
        self._model_lock = threading.Lock()
        self._switch_counter = 0

        perf_output_dir = os.environ.get("RUN_OUTPUT_DIR")
        perf_run_timestamp = os.environ.get("RUN_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.request_logger = None
        self.sample_logger = None
        self.switch_logger = None
        if perf_output_dir is not None:
            perf_output_path = Path(perf_output_dir)
            self.request_logger = RequestLogWriter(perf_output_path, perf_run_timestamp)
            self.sample_logger = OperationSampleWriter(perf_output_path, perf_run_timestamp)
            self.switch_logger = ModelSwitchLogWriter(perf_output_path, perf_run_timestamp)

        self.profiler = PerformanceProfiler(sample_writer=self.sample_logger)

        # Load model + associated components
        self._load_model_components(cfg, context={"phase": "startup"})

    def _load_model_components(self, cfg, context: Optional[dict[str, Any]] = None) -> None:
        """
        Loads the VLA model, proprio projector, action head, processor, and resize size for `cfg` onto
        `self`. Instrumented per sub-stage so both server startup and a runtime model swap (via
        `/load_model`) share identical profiling.
        """
        ctx = context or {}

        # Load model
        with self.profiler.measure("model_load_vla", context=ctx):
            self.vla = get_vla(cfg)

        # Load proprio projector
        self.proprio_projector = None
        if cfg.use_proprio:
            with self.profiler.measure("model_load_proprio_projector", context=ctx):
                self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

        # Load continuous action head
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            with self.profiler.measure("model_load_action_head", context=ctx):
                self.action_head = get_action_head(cfg, self.vla.llm_dim)

        # Check that the model contains the action un-normalization key
        assert cfg.unnorm_key in self.vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # Get Hugging Face processor
        with self.profiler.measure("model_load_processor", context=ctx):
            self.processor = get_processor(cfg)

        # Get expected image dimensions
        with self.profiler.measure("model_load_resize_size", context=ctx):
            self.resize_size = get_image_resize_size(cfg)

        self.cfg = cfg

    def _teardown_model_components(self, context: Optional[dict[str, Any]] = None) -> None:
        """Frees the currently-loaded model components from GPU memory, timed as its own profiled span."""
        with self.profiler.measure("model_teardown", context=context or {}):
            del self.vla, self.action_head, self.proprio_projector, self.processor
            torch.cuda.empty_cache()

    def get_server_action(self, payload: Dict[str, Any], request: Request) -> str:
        client_id = request.headers.get("x-client-id", "unknown")
        request_id = request.headers.get("x-request-id", "")
        request_index = request.headers.get("x-request-index", "")
        client_host = request.client.host if request.client is not None else ""
        request_start = time.perf_counter()
        request_status = "ok"
        decode_ms = 0.0
        parse_ms = 0.0
        inference_ms = 0.0
        encode_ms = 0.0
        operation_context = {
            "client_id": client_id,
            "request_id": request_id,
            "request_index": request_index,
            "client_host": client_host,
        }

        def make_timing_headers(total_ms: float) -> dict[str, str]:
            overhead_ms = max(total_ms - inference_ms, 0.0)
            return {
                "X-Server-Total-Latency-Ms": f"{total_ms:.4f}",
                "X-Server-Decode-Latency-Ms": f"{decode_ms:.4f}",
                "X-Server-Parse-Latency-Ms": f"{parse_ms:.4f}",
                "X-Server-Inference-Latency-Ms": f"{inference_ms:.4f}",
                "X-Server-Encode-Latency-Ms": f"{encode_ms:.4f}",
                "X-Server-Overhead-Latency-Ms": f"{overhead_ms:.4f}",
            }

        try:
            if double_encode := "encoded" in payload:
                with self.profiler.measure("0_decode_payload", context=operation_context):
                    # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                    decode_start = time.perf_counter()
                    assert len(payload.keys()) == 1, "Only uses encoded payload!"
                    payload = json.loads(payload["encoded"])
                    decode_ms = round((time.perf_counter() - decode_start) * 1000, 4)

            with self.profiler.measure("1_parse_payload", context=operation_context):
                parse_start = time.perf_counter()
                observation = payload
                instruction = observation["instruction"]
                parse_ms = round((time.perf_counter() - parse_start) * 1000, 4)

            with self.profiler.measure("2_run_inference", context=operation_context):
                inference_start = time.perf_counter()
                try:
                    # Block while a `/load_model` swap is in-flight, and vice versa.
                    with self._model_lock:
                        action = get_vla_action(
                            self.cfg,
                            self.vla,
                            self.processor,
                            observation,
                            instruction,
                            action_head=self.action_head,
                            proprio_projector=self.proprio_projector,
                            use_film=self.cfg.use_film,
                        )
                finally:
                    inference_ms = round((time.perf_counter() - inference_start) * 1000, 4)

            if double_encode:
                with self.profiler.measure("3_encode_and_return_action", context=operation_context):
                    encode_start = time.perf_counter()
                    response_content = json_numpy.dumps(action)
                    encode_ms = round((time.perf_counter() - encode_start) * 1000, 4)
            else:
                with self.profiler.measure("3_return_action", context=operation_context):
                    encode_start = time.perf_counter()
                    response_content = action
                    encode_ms = round((time.perf_counter() - encode_start) * 1000, 4)

            total_ms = round((time.perf_counter() - request_start) * 1000, 4)
            return JSONResponse(content=response_content, headers=make_timing_headers(total_ms))
        except:  # noqa: E722
            request_status = "error"
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            total_ms = round((time.perf_counter() - request_start) * 1000, 4)
            return JSONResponse(content={"error": "error"}, status_code=500, headers=make_timing_headers(total_ms))
        finally:
            if self.request_logger is not None:
                total_ms = round((time.perf_counter() - request_start) * 1000, 4)
                self.request_logger.append(
                    {
                        "client_id": client_id,
                        "request_id": request_id,
                        "request_index": request_index,
                        "status": request_status,
                        "server_latency_ms": total_ms,
                        "server_decode_ms": decode_ms,
                        "server_parse_ms": parse_ms,
                        "server_inference_ms": inference_ms,
                        "server_encode_ms": encode_ms,
                        "server_overhead_ms": round(max(total_ms - inference_ms, 0.0), 4),
                        "client_host": client_host,
                    }
                )

    def load_model(self, payload: Dict[str, Any], request: Request) -> JSONResponse:
        """
        Hot-swaps the currently-loaded model for a different one, described as a partial override of the
        server's `DeployConfig` (any field may be omitted, in which case it keeps its current value).
        Tears down the old model (freeing GPU memory) before loading the new one, with both phases timed
        and broken down into sub-stages via the existing `PerformanceProfiler` machinery.
        """
        client_host = request.client.host if request.client is not None else ""
        swap_start = time.perf_counter()

        with self._model_lock:
            self._switch_counter += 1
            switch_id = f"switch-{self._switch_counter:03d}"
            ctx = {
                "client_id": "admin",
                "request_id": switch_id,
                "request_index": self._switch_counter,
                "client_host": client_host,
            }

            old_checkpoint = str(self.cfg.pretrained_checkpoint)
            overrides = {k: v for k, v in payload.items() if v is not None}

            try:
                new_cfg = dataclasses.replace(self.cfg, **overrides)
            except TypeError as exc:
                return JSONResponse(content={"error": f"invalid override keys: {exc}"}, status_code=400)

            status = "ok"
            error_text = ""
            teardown_ms = 0.0
            load_total_ms = 0.0
            try:
                teardown_start = time.perf_counter()
                self._teardown_model_components(context=ctx)
                teardown_ms = round((time.perf_counter() - teardown_start) * 1000, 4)

                load_start = time.perf_counter()
                self._load_model_components(new_cfg, context=ctx)
                load_total_ms = round((time.perf_counter() - load_start) * 1000, 4)
            except Exception:  # noqa: BLE001
                status = "error"
                error_text = traceback.format_exc()
                logging.error(error_text)
            finally:
                swap_total_ms = round((time.perf_counter() - swap_start) * 1000, 4)

                def last_sample(operation: str) -> float:
                    stats = self.profiler._stats.get(operation)
                    return stats.samples[-1] if stats and stats.samples else 0.0

                row = {
                    "switch_id": switch_id,
                    "status": status,
                    "old_checkpoint": old_checkpoint,
                    "new_checkpoint": str(new_cfg.pretrained_checkpoint) if status == "ok" else old_checkpoint,
                    "teardown_ms": round(teardown_ms, 4),
                    "load_vla_ms": round(last_sample("model_load_vla"), 4),
                    "load_proprio_projector_ms": round(last_sample("model_load_proprio_projector"), 4),
                    "load_action_head_ms": round(last_sample("model_load_action_head"), 4),
                    "load_processor_ms": round(last_sample("model_load_processor"), 4),
                    "load_resize_size_ms": round(last_sample("model_load_resize_size"), 4),
                    "load_total_ms": round(load_total_ms, 4),
                    "swap_total_ms": swap_total_ms,
                    "client_host": client_host,
                    "error": error_text[-500:],
                }
                if self.switch_logger is not None:
                    self.switch_logger.append(row)

        status_code = 200 if status == "ok" else 500
        return JSONResponse(content=row, status_code=status_code)

    def get_profiler_report(self) -> list[dict[str, Any]]:
        return self.profiler.report()

    def get_profiler_full_report(self) -> list[dict[str, Any]]:
        return self.profiler.full_report()

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        self.app.post("/load_model")(self.load_model)
        self.app.get("/profiler")(self.get_profiler_report)
        self.app.get("/profiler/full")(self.get_profiler_full_report)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8777                                                    # Host Port

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    print(f"[INIT] OPENVLA CHECKPOINT: {cfg.pretrained_checkpoint}")
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
