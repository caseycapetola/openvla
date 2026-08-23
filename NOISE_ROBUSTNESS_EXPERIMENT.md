# Noise robustness experiment: action chunking vs. per-step generation

## Motivation

OpenVLA-OFT predicts `NUM_ACTIONS_CHUNK` (8, for LIBERO) actions per model call and,
by default, executes all 8 open-loop before requerying the model
(`experiments/robot/libero/run_libero_eval.py`, `run_episode()`). If one action
vector inside that chunk is bad, the rest of the chunk still plays out — the model
gets no chance to react until the queue drains.

This experiment tests whether generating one action vector per environment step
(fully closed-loop, requerying the model every step) is more robust to a bad action
than chunked open-loop execution, by deliberately injecting noise into the executed
action stream and comparing success rates.

## The two axes

**1. Chunking granularity — `--num_open_loop_steps`** (pre-existing flag,
`run_libero_eval.py:101`)

- `--num_open_loop_steps 8` (default, matches `NUM_ACTIONS_CHUNK`): full open-loop
  chunk execution — today's normal OFT behavior.
- `--num_open_loop_steps 1`: the model still predicts an 8-action chunk each call,
  but only the first action is ever used before the queue drains and a fresh chunk is
  requested from a fresh observation — i.e., true "one action vector per environment
  step."

No model or inference changes were needed for this axis — it was already a config
knob.

**2. Action noise — `--noise_mode`, `--noise_scale`, `--noise_period`, `--noise_dims`,
`--noise_seed`** (new)

Two independently-selectable modes:

- `outlier` — every `noise_period` real env steps (counted from the first
  post-warm-up step, so it lines up the same way regardless of chunking granularity),
  the action about to execute gets a large, randomly-signed corruption. This is the
  targeted test of the "one bad vector" hypothesis: default `noise_period=8` injects
  roughly once per native chunk length.
- `gaussian` — small random noise added to every executed action. A secondary,
  general robustness sweep (not chunk-position-specific).

Noise is injected **per real environment step**, keyed off the actual step counter,
not chunk-relative position. That keeps the timing/frequency of "bad" actions
identical across different `--num_open_loop_steps` settings — the only thing that
differs between conditions is how fast the model gets to observe and react to the
consequences.

By default only the pose dimensions are perturbed — `dx, dy, dz, drx, dry, drz`
(indices 0–5). The gripper (index 6) is left alone by default, since gripper
corruption (dropping/failing to grasp) is a qualitatively different failure mode that
would confound the pose-tracking-robustness question. Override with `--noise_dims`.

Magnitude is expressed as a **fraction of each action dimension's real scale**, not an
arbitrary constant, so `--noise_scale` means the same thing across dims of different
units (meters vs. radians): `action_scale = q99 - q01`, the per-dimension 1st–99th
percentile range from the training data's normalization stats
(`model.norm_stats[cfg.unnorm_key]["action"]`). `outlier` noise is
`±U(0.5, 1.5) * noise_scale * action_scale`; `gaussian` noise is
`N(0, noise_scale * action_scale)`.

Noise draws use a private `np.random.default_rng(noise_seed + episode_idx)`, kept
independent of the global seeded stream (`set_seed_everywhere`) that model/env
determinism relies on — so enabling noise doesn't change the model's own inference
determinism or the environment's initial-state sampling.

## New / changed files

- `experiments/robot/libero/run_libero_eval.py` — `GenerateConfig` fields
  (`noise_mode`, `noise_scale`, `noise_period`, `noise_dims`, `noise_seed`),
  validation, per-dim `action_scale` computed once after model load, noise RNG per
  episode, injection call site between `process_action()` and `env.step()`. Run IDs
  (log filenames / W&B run names) now auto-include `--ol{num_open_loop_steps}` and,
  when enabled, `-noise_{mode}_{scale}`, so a comparison sweep's outputs are
  self-documenting without hand-set `--run_id_note` per run.
- `experiments/robot/libero/libero_utils.py` — `apply_action_noise()`, the pure
  function implementing both noise modes.
- `experiments/robot/telemetry.py` — `noisy_steps` column (per-episode count of how
  many steps actually had noise applied), via `EvalTelemetry.note_noisy_step()`.

`--noise_mode none` (the default) is a no-op — behavior is unchanged from before this
feature existed.

## Running the comparison

Core 2×3 matrix (6 conditions), per task suite of interest:

```bash
for ol in 1 8; do
  for noise in none outlier gaussian; do
    extra_args=""
    if [ "$noise" = "outlier" ]; then
      extra_args="--noise_mode outlier --noise_scale 1.0 --noise_period 8"
    elif [ "$noise" = "gaussian" ]; then
      extra_args="--noise_mode gaussian --noise_scale 0.15"
    fi
    python experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint <CHECKPOINT_PATH> \
      --task_suite_name libero_spatial \
      --num_open_loop_steps $ol \
      --num_trials_per_task 10 \
      $extra_args
  done
done
```

Start with a small `--num_trials_per_task` (e.g. 10) to check the noise magnitudes
land in a sensible range (success rate not saturated at 0% or 100%) before committing
to a full run (e.g. 50).

## Reading the results

Each run's local log (`experiments/logs/EVAL-...--ol{N}[-noise_...].txt`) and CSV
telemetry sidecar report `final_success_rate` / per-episode `success` and
`noisy_steps`. With W&B enabled (`--use_wandb`), compare `success_rate/total` across
the 6 run names directly in the dashboard.

The core hypothesis test: does `num_open_loop_steps=1` degrade less than
`num_open_loop_steps=8` under `outlier` noise, relative to their respective
`noise_mode=none` baselines? If per-step replanning is more robust to a single bad
action, its success-rate drop (baseline → outlier) should be smaller than the
chunked condition's drop.
