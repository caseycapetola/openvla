"""
ping_client.py

Tiny client for hitting the OpenVLA `/ping` endpoint.

Example:

    python vla-scripts/ping_client.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

get_time = time.perf_counter


@dataclass
class PingClientConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    timeout: float = 5.0


def ping_server(cfg: PingClientConfig) -> Optional[str]:
    request = Request(f"http://{cfg.host}:{cfg.port}/ping", method="POST", data=b"")

    try:
        with urlopen(request, timeout=cfg.timeout) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
            return body if body else None
    except HTTPError as error:
        raise RuntimeError(f"Ping request failed with HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError(f"Ping request failed: {error.reason}") from error


def parse_args() -> PingClientConfig:
    parser = argparse.ArgumentParser(description="Ping a running OpenVLA deploy server.")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", default=8000, type=int, help="Server port")
    parser.add_argument("--timeout", default=5.0, type=float, help="Request timeout in seconds")
    args = parser.parse_args()
    return PingClientConfig(host=args.host, port=args.port, timeout=args.timeout)


def main() -> None:
    cfg = parse_args()
    response = ping_server(cfg)
    print(f"Ping succeeded at http://{cfg.host}:{cfg.port}/ping")
    if response is not None:
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
