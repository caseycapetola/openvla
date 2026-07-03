import csv
import os
from datetime import datetime

import json_numpy
import requests

json_numpy.patch()
import numpy as np
from deploy import PerformanceProfiler

profiler = PerformanceProfiler()

LONG_PROMPT = (
    "Perform a careful pick-and-place routine while prioritizing safety, stability, and repeatability. "
    "Start by scanning the entire workspace and identifying all visible objects, then choose a single target object that is "
    "isolated enough to manipulate without touching neighboring items. Before moving, estimate a collision-free approach path "
    "from the current arm pose to a point directly above the object. Keep motions smooth and conservative, avoiding sudden "
    "acceleration. Align the gripper over the center of the target, descend slowly, and pause briefly just before contact so "
    "the grasp can be adjusted if needed. Close the gripper with moderate force, then lift the object a short distance to "
    "confirm a stable hold before translating. Move the object toward an open placement zone near the back-right side of the "
    "workspace while maintaining sufficient clearance from all other objects and workspace boundaries. If the path appears "
    "constrained, prefer a higher vertical clearance and slower lateral motion. Once over the placement zone, descend steadily, "
    "release the object gently, and retract upward without dragging. After placement, verify that the object remains upright "
    "and that no adjacent objects have been displaced. If any instability is detected, stop motion and hold position for a "
    "follow-up instruction instead of attempting a corrective action. Throughout the routine, favor precision over speed, "
    "minimize unnecessary arm travel, and maintain a consistent trajectory that can be reproduced across repeated trials."
)

print(f"Using long prompt with {len(LONG_PROMPT.split())} words and {len(LONG_PROMPT)} characters")

with profiler.measure("inference_invocation"):
    action = requests.post(
        "http://0.0.0.0:8000/act",
        json={
            "image": np.zeros((256, 256, 3), dtype=np.uint8),  # replace with your real image
            "instruction": LONG_PROMPT,
            "unnorm_key": "bridge_orig",
        },
    ).json()

print(action)
profiler.print_report()

# --- Append this invocation's stats to a CSV consolidated for the whole bash run ---
# RUN_TIMESTAMP / RUN_OUTPUT_DIR are exported by the driver shell script so that all
# 33 (or however many) subprocess invocations append to the SAME file. If they're not
# set (e.g. running this script standalone), fall back to a fresh timestamp/dir.
run_timestamp = os.environ.get("RUN_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
output_dir = os.environ.get("RUN_OUTPUT_DIR", "profiler_results")
iteration = os.environ.get("RUN_ITERATION", "")

os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, f"profile_{run_timestamp}.csv")
file_exists = os.path.isfile(csv_path)

# Each process only ever has ONE sample, so the LatencyStats.summary() default of
# discard_first_n=3 would always zero it out. Bypass that by summarizing with
# discard_first_n=0 directly on the underlying stats objects.
rows = [s.summary(discard_first_n=0) for s in profiler._stats.values()]

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["date_timestamp", "iteration", "operation", "calls", "mean_ms", "min_ms", "max_ms", "p95_ms"])
    for s in rows:
        writer.writerow(
            [
                run_timestamp,
                iteration,
                s["operation"],
                s["calls"],
                s["mean_ms"],
                s["min_ms"],
                s["max_ms"],
                s["p95_ms"],
            ]
        )

print(f"Appended to {csv_path}")
