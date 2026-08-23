"""
telemetry.py

Latency / step accounting for the robot evaluation loops.

Ported from the server-side profiler in `vla-scripts/deploy.py` so that in-process eval numbers use the same
vocabulary (`*_latency_ms`, `calls/mean/min/max/p95`) as the deploy server's `/profiler` report, and the two can be
compared directly.

This module deliberately does **not** import `wandb`: it emits plain dicts and row lists, so it stays importable and
testable in environments without W&B installed. Telemetry is a pure observer of the eval loop -- nothing here touches
action computation, control flow, or success determination.
"""

import contextlib
import csv
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Measured operations. Warm-up (`num_steps_wait`) sim steps are timed under their own key so they do not pollute the
# distribution of real post-warm-up environment steps.
OP_IMAGE_PREP = "image_prep"
OP_INFERENCE = "inference"
OP_ENV_STEP = "env_step"
OP_ENV_STEP_WAIT = "env_step_wait"
OPERATIONS = (OP_IMAGE_PREP, OP_INFERENCE, OP_ENV_STEP, OP_ENV_STEP_WAIT)

# Columns of the per-episode record (consumed as `wandb.Table(columns=EPISODE_COLUMNS, data=...)`).
EPISODE_COLUMNS = [
    "task_id",
    "task_description",
    "episode_idx",
    "global_episode_idx",
    "success",
    "steps",
    "inference_calls",
    "wait_steps",
    "noisy_steps",
    "duration_s",
    "control_hz",
    "inference_mean_ms",
    "inference_p50_ms",
    "inference_p95_ms",
    "inference_p99_ms",
    "inference_max_ms",
    "env_step_mean_ms",
    "env_step_total_ms",
    "image_prep_mean_ms",
    "error",
]

# Cap on retained samples per operation at run scope. A `libero_90` sweep at 50 trials/task is ~4500 episodes of up to
# ~410 steps, i.e. ~1.8M samples per operation; retaining all of them across every operation would cost hundreds of MB.
# Exact call/total/min/max are always tracked -- only the percentile buffer is bounded.
DEFAULT_MAX_SAMPLES = 200_000


@dataclass
class LatencyStats:
    """Running latency statistics for a single named operation, in milliseconds."""

    operation: str
    max_samples: int = DEFAULT_MAX_SAMPLES

    calls: int = 0
    total_ms: float = 0.0
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None
    first_ms: Optional[float] = None

    # Bounded reservoir used only for percentiles; see DEFAULT_MAX_SAMPLES.
    samples: List[float] = field(default_factory=list, repr=False)

    # Private RNG for reservoir sampling. This must NOT be the global `random` module: `set_seed_everywhere()`
    # (robot_utils.py) seeds it, and consuming from that stream would perturb the eval's own RNG.
    _rng: random.Random = field(default_factory=lambda: random.Random(0), repr=False, compare=False)

    def record(self, elapsed_ms: float) -> None:
        self.calls += 1
        self.total_ms += elapsed_ms
        if self.first_ms is None:
            self.first_ms = elapsed_ms
        self.min_ms = elapsed_ms if self.min_ms is None else min(self.min_ms, elapsed_ms)
        self.max_ms = elapsed_ms if self.max_ms is None else max(self.max_ms, elapsed_ms)

        if len(self.samples) < self.max_samples:
            self.samples.append(elapsed_ms)
        else:
            # Reservoir sampling: keeps the retained subset unbiased instead of truncating to the (warm-up heavy)
            # earliest samples.
            j = self._rng.randrange(self.calls)
            if j < self.max_samples:
                self.samples[j] = elapsed_ms

    def reset(self) -> None:
        self.calls = 0
        self.total_ms = 0.0
        self.min_ms = None
        self.max_ms = None
        self.first_ms = None
        self.samples.clear()

    def summary(self, discard_first_n: int = 0) -> Dict[str, Any]:
        """Summary dict. `discard_first_n` drops the leading samples (deploy.py's warm-up handling); the default of 0
        keeps every sample, with the warm-up cost surfaced separately via `first_ms`."""
        if discard_first_n and len(self.samples) > discard_first_n:
            # Derived wholly from the trimmed reservoir; approximate if the reservoir ever filled.
            samples = self.samples[discard_first_n:]
            calls, total_ms = len(samples), float(np.sum(samples))
            min_ms, max_ms = float(np.min(samples)), float(np.max(samples))
        else:
            samples = self.samples
            calls, total_ms = self.calls, self.total_ms
            min_ms = self.min_ms if self.min_ms is not None else 0.0
            max_ms = self.max_ms if self.max_ms is not None else 0.0

        if not samples:
            percentiles = (0.0, 0.0, 0.0)
        else:
            percentiles = tuple(float(p) for p in np.percentile(samples, [50, 95, 99]))

        return {
            "operation": self.operation,
            "calls": calls,
            "mean_ms": round(total_ms / calls if calls else 0.0, 4),
            "min_ms": round(min_ms, 4),
            "max_ms": round(max_ms, 4),
            "p50_ms": round(percentiles[0], 4),
            "p95_ms": round(percentiles[1], 4),
            "p99_ms": round(percentiles[2], 4),
            "total_ms": round(total_ms, 4),
            "first_ms": round(self.first_ms if self.first_ms is not None else 0.0, 4),
        }


class PerformanceProfiler:
    """Collection of `LatencyStats` keyed by operation name."""

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES):
        self.max_samples = max_samples
        self._stats: Dict[str, LatencyStats] = {}

    def stats(self, operation: str) -> LatencyStats:
        """Fetch (creating on demand) the stats for an operation, so summaries exist even with zero samples."""
        if operation not in self._stats:
            self._stats[operation] = LatencyStats(operation, max_samples=self.max_samples)
        return self._stats[operation]

    def record(self, operation: str, elapsed_ms: float) -> None:
        self.stats(operation).record(elapsed_ms)

    @contextlib.contextmanager
    def measure(self, operation: str):
        start = time.perf_counter()
        yield
        # Only successful operations are recorded. A call that raised has no meaningful latency, and counting it would
        # both inflate step counts and pollute the percentiles. Note there is no `except`/`finally` here, so exceptions
        # propagate untouched -- the eval loop's own error handling must keep working.
        self.record(operation, (time.perf_counter() - start) * 1000)

    def report(self, discard_first_n: int = 0) -> List[Dict[str, Any]]:
        return [stats.summary(discard_first_n) for stats in self._stats.values()]

    def reset(self) -> None:
        for stats in self._stats.values():
            stats.reset()


class EvalTelemetry:
    """Episode / task / run scoped telemetry for a LIBERO-style eval loop.

    Every measurement is recorded into all three scopes at once, so each scope has exact statistics without any
    reservoir-merging math. Only the run scope is bounded (see DEFAULT_MAX_SAMPLES).
    """

    def __init__(self, max_run_samples: int = DEFAULT_MAX_SAMPLES, csv_path: Optional[str] = None):
        self._run = PerformanceProfiler(max_samples=max_run_samples)
        self._task = PerformanceProfiler(max_samples=max_run_samples)
        self._episode = PerformanceProfiler(max_samples=max_run_samples)
        self._profilers = (self._episode, self._task, self._run)

        self._rows: List[List[Any]] = []
        self._last_summary = ""

        # Optional machine-readable sidecar. Opened eagerly (like the eval script's own log file) so a bad path fails
        # immediately rather than after the first episode, and flushed per row so a killed run keeps its rows.
        self._csv_file = None
        self._csv_writer = None
        if csv_path is not None:
            self._csv_file = open(csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(EPISODE_COLUMNS)
            self._csv_file.flush()

        self._episode_start: Optional[float] = None
        self._episode_error = ""
        self._episode_noisy_steps = 0

        self._task_episodes, self._task_steps, self._task_inference_calls, self._task_duration_s = 0, 0, 0, 0.0
        self._run_episodes, self._run_steps, self._run_inference_calls, self._run_duration_s = 0, 0, 0, 0.0

    # === Loop hooks ===

    def start_task(self) -> None:
        self._task.reset()
        self._task_episodes, self._task_steps, self._task_inference_calls, self._task_duration_s = 0, 0, 0, 0.0

    def start_episode(self) -> None:
        self._episode.reset()
        self._episode_error = ""
        self._episode_noisy_steps = 0
        self._episode_start = time.perf_counter()

    def note_error(self, message: str) -> None:
        self._episode_error = message

    def note_noisy_step(self) -> None:
        """Record that the most recent env step had action noise injected (see `cfg.noise_mode` in the eval loop).
        Pure counter -- like the rest of this class, it observes what the eval loop did rather than computing it."""
        self._episode_noisy_steps += 1

    @contextlib.contextmanager
    def measure(self, operation: str):
        """Time a block and fan the sample out to the episode, task, and run scopes.

        Records only on success (see `PerformanceProfiler.measure`); exceptions propagate untouched so the eval loop's
        own `except` handler still fires.
        """
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        for profiler in self._profilers:
            profiler.record(operation, elapsed_ms)

    def end_episode(
        self,
        task_id: int,
        task_description: str,
        episode_idx: int,
        global_episode_idx: int,
        success: bool,
    ) -> None:
        duration_s = (time.perf_counter() - self._episode_start) if self._episode_start is not None else 0.0
        self._episode_start = None

        inference = self._episode.stats(OP_INFERENCE).summary()
        env_step = self._episode.stats(OP_ENV_STEP).summary()
        image_prep = self._episode.stats(OP_IMAGE_PREP).summary()
        wait_steps = self._episode.stats(OP_ENV_STEP_WAIT).calls

        # Counts come from the recorded samples rather than the loop's `t`, which increments after the success `break`
        # and also folds in the warm-up steps.
        steps, inference_calls = env_step["calls"], inference["calls"]
        control_hz = steps / duration_s if duration_s > 0 else 0.0

        row = [
            task_id,
            task_description,
            episode_idx,
            global_episode_idx,
            bool(success),
            steps,
            inference_calls,
            wait_steps,
            self._episode_noisy_steps,
            round(duration_s, 3),
            round(control_hz, 3),
            inference["mean_ms"],
            inference["p50_ms"],
            inference["p95_ms"],
            inference["p99_ms"],
            inference["max_ms"],
            env_step["mean_ms"],
            env_step["total_ms"],
            image_prep["mean_ms"],
            self._episode_error,
        ]
        self._rows.append(row)
        if self._csv_writer is not None:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        self._task_episodes += 1
        self._task_steps += steps
        self._task_inference_calls += inference_calls
        self._task_duration_s += duration_s
        self._run_episodes += 1
        self._run_steps += steps
        self._run_inference_calls += inference_calls
        self._run_duration_s += duration_s

        self._last_summary = (
            f"Steps: {steps} | Inference calls: {inference_calls} | Duration: {duration_s:.2f}s "
            f"| Control rate: {control_hz:.2f} Hz "
            f"| Inference ms mean/p50/p95/p99: {inference['mean_ms']:.1f}/{inference['p50_ms']:.1f}/"
            f"{inference['p95_ms']:.1f}/{inference['p99_ms']:.1f} "
            f"| Env step ms mean: {env_step['mean_ms']:.1f} | Image prep ms mean: {image_prep['mean_ms']:.1f}"
        )

    def close(self) -> None:
        """Close the CSV sidecar, if one is open. Safe to call more than once."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # === Reporting ===

    def episode_rows(self) -> List[List[Any]]:
        return self._rows

    def last_episode_summary(self) -> str:
        return self._last_summary

    def _scope_metrics(
        self, profiler: PerformanceProfiler, suffix: str, episodes: int, steps: int, calls: int, duration_s: float
    ) -> Dict[str, Any]:
        inference = profiler.stats(OP_INFERENCE).summary()
        env_step = profiler.stats(OP_ENV_STEP).summary()
        image_prep = profiler.stats(OP_IMAGE_PREP).summary()
        return {
            f"episode/mean_steps/{suffix}": steps / episodes if episodes else 0.0,
            f"episode/mean_inference_calls/{suffix}": calls / episodes if episodes else 0.0,
            f"episode/mean_duration_s/{suffix}": duration_s / episodes if episodes else 0.0,
            f"latency/inference_mean_ms/{suffix}": inference["mean_ms"],
            f"latency/inference_p50_ms/{suffix}": inference["p50_ms"],
            f"latency/inference_p95_ms/{suffix}": inference["p95_ms"],
            f"latency/inference_p99_ms/{suffix}": inference["p99_ms"],
            f"latency/inference_max_ms/{suffix}": inference["max_ms"],
            f"latency/env_step_mean_ms/{suffix}": env_step["mean_ms"],
            f"latency/image_prep_mean_ms/{suffix}": image_prep["mean_ms"],
            f"throughput/control_hz/{suffix}": steps / duration_s if duration_s > 0 else 0.0,
        }

    def task_metrics(self, task_description: str) -> Dict[str, Any]:
        return self._scope_metrics(
            self._task,
            task_description,
            self._task_episodes,
            self._task_steps,
            self._task_inference_calls,
            self._task_duration_s,
        )

    def run_metrics(self) -> Dict[str, Any]:
        metrics = self._scope_metrics(
            self._run,
            "total",
            self._run_episodes,
            self._run_steps,
            self._run_inference_calls,
            self._run_duration_s,
        )
        metrics.update(
            {
                # Logged rather than discarded: the first inference call absorbs CUDA/kernel warm-up and is a large
                # outlier, so it is reported explicitly instead of being silently dropped from the statistics.
                "latency/inference_first_call_ms": self._run.stats(OP_INFERENCE).summary()["first_ms"],
                "counts/total_inference_calls": self._run_inference_calls,
                "counts/total_steps": self._run_steps,
                "counts/total_eval_duration_s": round(self._run_duration_s, 3),
            }
        )
        return metrics

    def run_summary(self) -> str:
        inference = self._run.stats(OP_INFERENCE).summary()
        env_step = self._run.stats(OP_ENV_STEP).summary()
        image_prep = self._run.stats(OP_IMAGE_PREP).summary()
        mean_steps = self._run_steps / self._run_episodes if self._run_episodes else 0.0
        control_hz = self._run_steps / self._run_duration_s if self._run_duration_s > 0 else 0.0
        return (
            "\n=== Telemetry summary (run) ===\n"
            f"Episodes: {self._run_episodes}\n"
            f"Total env steps: {self._run_steps}\n"
            f"Total inference calls: {self._run_inference_calls}\n"
            f"Mean steps per episode: {mean_steps:.1f}\n"
            f"Total time in timestep loop: {self._run_duration_s:.1f}s\n"
            f"Mean control rate: {control_hz:.2f} Hz\n"
            f"Inference ms mean/p50/p95/p99/max: {inference['mean_ms']:.1f}/{inference['p50_ms']:.1f}/"
            f"{inference['p95_ms']:.1f}/{inference['p99_ms']:.1f}/{inference['max_ms']:.1f}\n"
            f"Inference first call (warm-up) ms: {inference['first_ms']:.1f}\n"
            f"Env step ms mean/p95: {env_step['mean_ms']:.1f}/{env_step['p95_ms']:.1f}\n"
            f"Image prep ms mean/p95: {image_prep['mean_ms']:.1f}/{image_prep['p95_ms']:.1f}\n"
        )
