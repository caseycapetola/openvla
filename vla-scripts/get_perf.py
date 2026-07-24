#!/usr/bin/env python3

"""Fetch server-side inference performance data from deploy.py."""

import argparse
import json

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch deploy.py profiler output.")
    parser.add_argument(
        "--server-url",
        default="http://0.0.0.0:8777",
        help="Base server URL (without /profiler suffix).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Fetch the per-sample profiler output instead of the summary report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint = "/profiler/full" if args.full else "/profiler"
    response = requests.get(f"{args.server_url.rstrip('/')}{endpoint}", timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()