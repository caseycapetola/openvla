# Model-Switch Performance Testing

This document explains how to run and interpret performance tests for the runtime model-switching
feature (`POST /load_model`) added to `vla-scripts/deploy.py`. See [ALOHA.md](ALOHA.md#runtime-model-switching-perf-testing)
for the feature's request/response shape and general usage.

The goal of this testing is to measure the cost of swapping the loaded model mid-session (broken down
into teardown + per-component load latency) and to see whether the first inference after a swap is
slower than steady-state.

## Pre-downloading checkpoints

For repeatable testing it's best to download the checkpoints you want to swap between ahead of time and
reference their local paths, rather than pointing `--pretrained_checkpoint`/`MODEL_B_CHECKPOINT` at a
Hugging Face repo ID and re-downloading (or hitting the Hub) on every run.

This also sidesteps a real limitation: `model_is_on_hf_hub()`
(`experiments/robot/openvla_utils.py:46-53`) decides "local vs. Hub" by calling
`HfApi().model_info(pretrained_checkpoint)`, which only succeeds for a valid repo ID string, not a
filesystem path. Once `--pretrained_checkpoint` points at a local directory, the loader takes the
"local" branch everywhere, including in `get_proprio_projector`/`get_action_head` — which otherwise only
know how to fetch components for 5 hardcoded `moojink/openvla-7b-oft-finetuned-libero-*` repo IDs. The
local branch instead uses `find_checkpoint_file()`, which pattern-matches any file in the directory
containing `"proprio_projector"`/`"action_head"`/`"vision_backbone"` + `"checkpoint"`. **Downloading
locally lets you swap between any two checkpoints, not just those 5 known ones.**

Download with the `hf` CLI (ships with `huggingface_hub`; `huggingface-cli download` is the older alias
for the same command):

```bash
mkdir -p ~/vla_checkpoints

hf download moojink/openvla-7b-oft-finetuned-libero-spatial \
  --local-dir ~/vla_checkpoints/libero-spatial

hf download moojink/openvla-7b-oft-finetuned-libero-goal \
  --local-dir ~/vla_checkpoints/libero-goal
```

This pulls down `config.json`, the model weights, processor files, `dataset_statistics.json`, and the
`proprio_projector--*.pt`/`action_head--*.pt` component files — everything `get_vla`/
`get_proprio_projector`/`get_action_head` need — as real files directly under `--local-dir` (recent
`huggingface_hub` versions don't symlink into the cache by default). Each OpenVLA-7B checkpoint is
roughly 15-16 GB in bf16, so budget disk space accordingly; `du -sh ~/vla_checkpoints/*` after
downloading is a good sanity check. The `check_model_logic_mismatch`/`update_auto_map` steps in `get_vla`
run automatically for local checkpoints (syncing `modeling_prismatic.py`/`configuration_prismatic.py`
into the checkpoint dir if needed) — nothing extra required there.

**Verify the `unnorm_key` before running**: it must exist in that checkpoint's `dataset_statistics.json`.

```bash
python -c "import json; print(list(json.load(open('~/vla_checkpoints/libero-spatial/dataset_statistics.json')).keys()))"
```

Once downloaded, reference the local paths directly in place of `/PATH/TO/MODEL_A/`,
`/PATH/TO/MODEL_B/`, `MODEL_A_CHECKPOINT`, and `MODEL_B_CHECKPOINT` in the steps below — no code changes
needed, the client/server scripts just pass whatever string you give them straight through to
`pretrained_checkpoint`.

## Running the test

**1. Start the server with model A** (on the GPU machine, `openvla-oft` conda env). Set `RUN_OUTPUT_DIR`
so all the CSVs described below get written:

```bash
export RUN_OUTPUT_DIR=performance_results/openvla_oft_model_switch_test/manual_run
export RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

python vla-scripts/deploy.py \
  --pretrained_checkpoint /storage/scratch1/5/ccapetola3/vla_checkpoints/libero-goal \
  --use_l1_regression True \
  --use_proprio True \
  --center_crop True \
  --unnorm_key libero_spatial_no_noops
```

**2. Run the switch test** from the client machine/shell against that server:

```bash
MODEL_B_CHECKPOINT=/storage/scratch1/5/ccapetola3/vla_checkpoints/libero-spatial \
MODEL_B_UNNORM_KEY=libero_spatial_no_noops \
MODEL_A_CHECKPOINT=/storage/scratch1/5/ccapetola3/vla_checkpoints/libero-goal \
SERVER_URL=http://<server-host>:8777 \
PRE_SWAP_REQUESTS=30 \
POST_SWAP_REQUESTS=30 \
./vla-scripts/run_model_switch_test.sh
```

Use 30+ requests on each side of the swap so you get a stable steady-state mean rather than noise. If
model B needs different architecture flags than model A (different action head, FiLM, num images, etc.),
pass them through, e.g.:

```bash
./vla-scripts/run_model_switch_test.sh --model-b-use-film true --model-b-num-images-in-input 3
```

**3. Repeat** for however many trials you want (swap direction, different checkpoint pairs, etc.) — each
run gets its own timestamped subfolder under `$OUTPUT_DIR`, so nothing overwrites.

### Output files

Under `$OUTPUT_DIR` (a timestamped subfolder is created automatically if `OUTPUT_DIR` isn't set):

| File | Contents |
|---|---|
| `client_01/model_switch_client_metrics.csv` | Every `/act` call, tagged `pre_swap` / `post_swap_first` / `post_swap_steady` |
| `client_01/model_switch_client_events.csv` | The one swap event, from the client's perspective (client + server timing) |
| `model_switch_metrics.csv` (server-side, in `$OUTPUT_DIR` root) | The same swap event, as recorded by the server |
| `results_<timestamp>.txt` | The aggregate `GET /profiler` dump plus stdout log from the run |

## Interpreting the output metrics

### Swap cost (`model_switch_client_events.csv` / `model_switch_metrics.csv`, one row per swap)

- **`teardown_ms`** — time to free model A's GPU memory. Should be small (ms-scale); if it's large,
  that's worth investigating (lingering references keeping tensors alive longer than expected).
- **`load_vla_ms`** — the dominant cost. This is `AutoModelForVision2Seq.from_pretrained(...)` reading
  and moving weights to GPU. Scales with checkpoint size and disk/network speed (local disk vs. HF Hub
  download).
- **`load_proprio_projector_ms` / `load_action_head_ms` / `load_processor_ms` / `load_resize_size_ms`**
  — should each be tiny; these are small components, not the multi-GB backbone.
- **`load_total_ms`** — sum of the load-stage columns above.
- **`swap_total_ms`** (server-reported) vs. **`client_swap_total_ms`** (client-observed) — the gap
  between these two is network/HTTP overhead for the `/load_model` round-trip, the same idea as the
  client/server latency gap on `/act`.

### Inference cost around the swap (`model_switch_client_metrics.csv`, filter by `phase`)

- Compare mean `server_inference_latency_ms` for `pre_swap` vs. `post_swap_steady` — these should be
  close if model B has similar architecture/size to model A. A persistent gap points to a real
  difference between the models (e.g. different action head, different sequence length), not a swap
  artifact.
- Compare `post_swap_first` against `post_swap_steady` — this isolates "cold first inference after a
  swap." If `post_swap_first` is meaningfully slower than the `post_swap_steady` mean, that's evidence
  of a cold-start cost beyond the raw weight-load time already captured in `load_vla_ms` — likely CUDA
  kernel/cuBLAS autotuning, memory allocator warm-up, or first-call JIT/graph-capture effects that only
  show up on the very first forward pass.
- Watch the `status`/`error` columns — any `error` rows (especially clustered right after the swap)
  suggest the swap left the server in a bad state rather than being a genuine latency signal. A failed
  `/load_model` call can leave the server's model components partially loaded, requiring a restart —
  this shows up as sustained `/act` errors post-swap, not just a slow first request.

### Aggregate check (`GET /profiler`, dumped into `results_<timestamp>.txt`)

Confirms `model_load_vla`, `model_teardown`, etc. appear with sample counts matching your swap count,
and gives mean/min/max/p95 if you run multiple swaps in one server session.

**Caveat**: `PerformanceProfiler.summary()` discards the first 5 samples per operation as "warmup" by
default. With only one swap per server run, that means the load-latency summary in `/profiler` will show
`calls: 0` for the `model_load_*`/`model_teardown` operations. For swap timings specifically, don't rely
on `/profiler`'s summary stats unless you run several swaps back-to-back in the same server session —
instead read the raw numbers directly from `model_switch_metrics.csv` or `GET /profiler/full`.
