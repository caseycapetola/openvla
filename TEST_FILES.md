# Test Files

Used to track relevant test files and the corresponding test configurations.

- `experiments/logs/EVAL-libero_spatial-openvla-2026_08_24-09_06_09--ol8-noise_outlier_0.5-5offset.csv`
  - Achieved by running the following command:

```bash
# Normal chunking, but space the noise period so that different action vector in chunk gets affected
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl
export PYTHONPATH=/home/caseycapetola/omscs/8903/forks/openvla
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --load_in_8bit True \
  --noise_mode outlier \
  --noise_period 5
```

- `experiments/logs/EVAL-libero_spatial-openvla-2026_08_24-10_27_46--ol1-noise_outlier_0.5-1-step.csv`
  - Achieved by running the following command:

```bash
# Test performance of simply taking one action vector from each chunk.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl
export PYTHONPATH=/home/caseycapetola/omscs/8903/forks/openvla
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --load_in_8bit True \
  --noise_mode outlier \
  --noise_period 5 \
  --num_open_loop_steps 1
```

- `experiments/logs/EVAL-libero_spatial-openvla-2026_08_21-18_36_26-8BIT.csv`
  - Primary CSV data used for presentation
  - Configured to use 8-bit quantization
  - Run with the following command:

```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MUJOCO_GL=egl
export PYTHONPATH=/home/caseycapetola/omscs/8903/forks/openvla
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --load_in_8bit True
```
