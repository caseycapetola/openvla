import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

from deploy import OperationSampleWriter, PerformanceProfiler
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated OpenVLA inference calls and log latency.")
    parser.add_argument("--num-calls", type=int, default=61, help="Number of times to call get_vla_action.")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("RUN_OUTPUT_DIR", "performance_results/openvla_oft_simple_inference"),
        help="Base directory where inference metrics should be written.",
    )
    parser.add_argument(
        "--run-timestamp",
        default=os.environ.get("RUN_TIMESTAMP", datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="Shared run timestamp used to label the CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_filename = f"simple_inference_operation_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"

    # Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
    cfg = GenerateConfig(
        pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key="libero_spatial_no_noops",
    )

    # Load OpenVLA-OFT policy and inputs processor
    vla = get_vla(cfg)
    processor = get_processor(cfg)

    # Load MLP action head to generate continuous actions (via L1 regression)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

    # Load proprio projector to map proprio to language embedding space
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    # Load sample observation:
    #   observation (dict): {
    #     "full_image": primary third-person image,
    #     "wrist_image": wrist-mounted camera image,
    #     "state": robot proprioceptive state,
    #     "task_description": task description,
    #   }
    with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
        observation = pickle.load(file)

    sample_writer = OperationSampleWriter(
        output_dir=Path(args.output_dir),
        run_timestamp=args.run_timestamp,
        filename=metrics_filename,
    )
    profiler = PerformanceProfiler(sample_writer=sample_writer)

    actions = None
    for call_index in range(1, args.num_calls + 1):
        with profiler.measure(
            "get_vla_action",
            context={
                "client_id": "simple_inference",
                "request_id": f"call-{call_index:03d}",
                "request_index": call_index,
                "client_host": "local",
            },
        ):
            actions = get_vla_action(
                cfg,
                vla,
                processor,
                observation,
                observation["task_description"],
                action_head,
                proprio_projector,
            )
        print(f"call={call_index}/{args.num_calls} recorded")

    if actions is not None:
        print("Generated action chunk from final call:")
        for act in actions:
            print(act)

    print(f"Wrote inference metrics to {sample_writer.csv_path()}")


if __name__ == "__main__":
    main()
