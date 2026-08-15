# LIBERO Eval Telemetry

Latency / step-count / inference-invocation metrics for the LIBERO evaluation pipeline
(`experiments/robot/libero/run_libero_eval.py`).

**Status:** §3a (episode step count + duration), §3b (per-step inference latency), and §3c
(sim vs. model split) are **implemented**. §3d (action statistics) is not. Line numbers below
refer to this repo's current `run_libero_eval.py`.

## 1. What gets logged today

W&B logging is optional (`cfg.use_wandb`, default `False`) and is not on the critical path —
turning it off does not change eval behavior, only what gets reported. See
[`libero_design.md`](./libero_design.md) for how this fits into the overall control flow.

Telemetry lives in [`experiments/robot/telemetry.py`](../experiments/robot/telemetry.py)
(`LatencyStats`, `PerformanceProfiler`, `EvalTelemetry`), ported from the server-side profiler in
`vla-scripts/deploy.py` so that in-process eval numbers use the same vocabulary (`*_latency_ms`,
`calls/mean/min/max/p95`) as the deploy server's `/profiler` report and the two can be compared
directly. That module does not import `wandb` — it emits plain dicts and row lists.

### Pre-existing metrics

| Metric | Where computed | Where logged |
|---|---|---|
| `success_rate/{task_description}` | `task_successes / task_episodes` | `run_libero_eval.py:291-298`, after each task |
| `num_episodes/{task_description}` | `task_episodes` | same block |
| `success_rate/total` | `total_successes / total_episodes` | `run_libero_eval.py:306-316`, once at the end |
| `num_episodes/total` | `total_episodes` | same block |
| local log file (`.txt`) | mirrors the console prints | uploaded via `wandb.save(local_log_filepath)` (`:317`) |

### Telemetry metrics

Four operations are timed: `image_prep`, `inference`, `env_step`, and `env_step_wait`. Warm-up
(`num_steps_wait`) sim steps are timed under their own key so they do not pollute the
distribution of real post-warm-up environment steps.

Per task, merged into the existing `wandb.log` block (`:291-298`):

```
episode/mean_steps/{task}            episode/mean_inference_calls/{task}
episode/mean_duration_s/{task}
latency/inference_mean_ms/{task}     latency/inference_p50_ms/{task}
latency/inference_p95_ms/{task}      latency/inference_p99_ms/{task}
latency/inference_max_ms/{task}
latency/env_step_mean_ms/{task}      latency/image_prep_mean_ms/{task}
throughput/control_hz/{task}
```

Run level, merged into the final `wandb.log` (`:306-316`) — the same keys under `/total`, plus:

```
latency/inference_first_call_ms   # warm-up outlier, reported rather than discarded
counts/total_inference_calls
counts/total_steps
counts/total_eval_duration_s
```

Per-episode detail goes to W&B as a single `wandb.Table` under the key `episodes` (`:314`), with
one row per episode and the columns in `telemetry.EPISODE_COLUMNS`: task/episode identifiers,
`success`, `steps`, `inference_calls`, `wait_steps`, `duration_s`, `control_hz`, the inference
latency summary, `env_step_*`, `image_prep_mean_ms`, and `error`.

`wandb.init` now also receives `config=vars(cfg)` (`:131-138`) — latency numbers are not
comparable across runs without the checkpoint / quantization / crop settings. `vars()` rather
than `dataclasses.asdict()` so the dynamically-attached `cfg.unnorm_key` is included.
`wandb.finish()` (`:318`) ensures the table flushes.

### Local output

Two files land in `cfg.local_log_dir`, both written unconditionally — nothing here depends on
`cfg.use_wandb`:

- `EVAL-<suite>-<family>-<timestamp>.txt` — the existing log, plus a one-line telemetry summary
  per episode and a run summary block written before `log_file.close()`.
- `EVAL-<suite>-<family>-<timestamp>.csv` — one row per episode with the full `EPISODE_COLUMNS`
  header, i.e. the same data as the W&B table in a form `pandas.read_csv` can load directly.

The CSV is opened eagerly (so a bad `local_log_dir` fails immediately rather than after the first
episode) and flushed per row, and the `.txt` is flushed per episode. An interrupted run — a
cluster wall-time kill, say — therefore keeps every completed episode in both files. Only the run
summary block and the W&B upload are lost, since those happen after the last task.

Rollout MP4s (`save_rollout_video`, `libero_utils.py`) go to local disk (`./rollouts/{DATE}/`)
and are still **not** pushed to W&B.

## 2. Where the instrumentation sits

Inside the per-timestep loop (`while t < max_steps + cfg.num_steps_wait:`, `:196`):

- `telemetry.start_episode()` (`:195`) starts the episode clock immediately before the loop, so
  duration covers exactly the timestep loop — not `env.reset()` / `set_init_state` setup.
- `with telemetry.measure("env_step_wait"):` (`:201`) wraps the warm-up dummy step.
- `with telemetry.measure("image_prep"):` (`:207`) wraps `get_libero_image`, whose TF
  jpeg/lanczos round trip (`libero_utils.py:33-47`) runs every real step.
- `with telemetry.measure("inference"):` (`:225`) wraps `get_action`.
- `with telemetry.measure("env_step"):` (`:243`) wraps `env.step`, with the `if done:` / `break`
  left outside the `with` block.
- `telemetry.note_error(repr(e))` (`:254`) in the existing `except` handler.
- `telemetry.end_episode(...)` (`:261`) closes the episode out before `save_rollout_video`
  (`:270`), so MP4 disk I/O is excluded from episode duration.

### Step counts do not come from `t`

`t` is unusable as a step count: `t += 1` sits *after* the success `break`, so a successful
episode under-counts its final step by one, and `t` also folds in the `num_steps_wait` warm-up
steps. Counts are instead derived from the number of recorded samples — `steps` from `env_step`
calls, `inference_calls` from `inference` calls.

These two are 1:1 today (`predict_action`, `prismatic/extern/hf/modeling_prismatic.py:506-536`,
returns a single 7-dim action per call with no chunking) but are logged separately because they
diverge the moment action chunking (OFT) lands — and they already diverge on the error path,
where `get_action` succeeds but `env.step` raises.

### Failed operations are not recorded

`measure()` records only on success. A call that raised has no meaningful latency, and counting
it would inflate step counts and pollute the percentiles. There is no `except`/`finally` in
`measure()`, so exceptions propagate untouched and the eval loop's own error handling still works.

## 3. Design constraints

- **Additive and opt-in.** Every new metric is gated the same way the existing ones are
  (`if cfg.use_wandb: ...`) and does not change control flow, action computation, or success
  determination. Telemetry is a pure observer of the loop (see `libero_design.md` §1 — the
  network hop is fire-and-forget by design).
- **No per-step `wandb.log` in the hot loop.** Samples accumulate locally and flush once per task
  plus once at the end; per-episode detail rides along in the end-of-run table. This keeps the
  network hop count essentially unchanged, and sidesteps a real problem: every `wandb.log` in the
  script is unstepped, so per-episode logs would interleave with task metrics on W&B's implicit
  step counter.
- **Reservoir sampling uses a private RNG.** `LatencyStats` caps its percentile buffer
  (`DEFAULT_MAX_SAMPLES = 200_000`) because a `libero_90` sweep at 50 trials/task is ~1.8M
  samples per operation. It samples with a private `random.Random(0)`, never the global `random`
  module — `set_seed_everywhere` (`robot_utils.py:29`) seeds that, and consuming from it would
  perturb the eval's own RNG stream. Exact `calls/total/min/max/first` are always tracked; only
  the percentile buffer is bounded.
- **Instrumented at the family-agnostic seam.** `get_action` (`robot_utils.py:63-72`) is the
  model-family dispatcher, so timing at that call site covers any future model family for free.
  The trade-off is that the inference number folds in prompt construction and the optional
  center-crop (`openvla_utils.py:135-163`); see §4.
- **Local log file stays the source of truth.** `log_file` (opened at `run_libero_eval.py:126`)
  and console prints are written unconditionally, independent of `cfg.use_wandb`.
- **CUDA timing needs no explicit sync.** `predict_action`
  (`prismatic/extern/hf/modeling_prismatic.py:521`) calls `.cpu()`, which forces a device sync
  inside the measured region.

## 4. Not done / possible next steps

1. **Forward-pass-only latency.** `get_vla_action` (`openvla_utils.py:127-170`) does prompt
   construction and the optional TF center-crop (CPU) *and* the model forward pass (GPU)
   together. To compare against reported OpenVLA throughput numbers you would need to time around
   `openvla_utils.py:166` (processor) and `:169` (`predict_action`) separately. Instrumentation
   there needs a parallel counterpart for each new model family, unlike the `get_action` seam.
2. **Rollout videos to W&B.** `save_rollout_video` (`libero_utils.py:61-74`) already returns
   `mp4_path`, and the return value is currently discarded at `run_libero_eval.py:270` — capturing
   it is the free path to `wandb.log({"rollout": wandb.Video(mp4_path)})`.
3. **Action statistics (§3d in the original scoping notes).** `action` after
   `normalize_gripper_action` / `invert_gripper_action` (`run_libero_eval.py:235-241`) is a 7-dim
   array; per-dimension mean/std or a histogram would help debug degenerate policies (e.g. a model
   that always outputs near-zero actions). Lower value than latency — skip unless a specific
   debugging need comes up.
4. **`run_bridgev2_eval.py`.** The real-robot eval script has no W&B config fields at all, but is
   where latency telemetry is most actionable: if inference exceeds `step_duration`
   (0.2 s at 5 Hz, `run_bridgev2_eval.py:115`) the control loop silently misses its rate and
   nothing currently detects it. `EvalTelemetry` is model- and env-agnostic and can be reused
   there as-is.
