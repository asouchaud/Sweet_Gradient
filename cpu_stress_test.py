#!/usr/bin/env python3
"""
CPU stress test utility.

This script is intended to saturate available CPU cores so you can verify
high utilization in Task Manager while benchmarking your laptop.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from multiprocessing import get_context


def _cpu_worker(duration_seconds: float, worker_id: int) -> tuple[int, float]:
    """
    Run a heavy floating-point loop for the requested duration.
    Returns (worker_id, pseudo-iteration-count) for a tiny completion summary.
    """
    end_time = time.perf_counter() + duration_seconds
    x = 0.123456789 + worker_id
    n = 0

    # Keep Python busy with math operations that prevent trivial optimization.
    while time.perf_counter() < end_time:
        for _ in range(200_000):
            x = math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
            n += 1

    return worker_id, float(n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CPU stress test using all available logical CPUs."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Stress duration in seconds (default: 60).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (default: logical CPU count).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be > 0.")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0.")

    logical_cpus = os.cpu_count() or 1
    print(f"Logical CPUs detected: {logical_cpus}")
    print(f"Starting stress test with {args.workers} workers for {args.duration:.1f}s...")
    print("Open Task Manager -> Performance -> CPU to monitor usage.")

    ctx = get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        results = pool.starmap(_cpu_worker, [(args.duration, i) for i in range(args.workers)])

    total_ops = sum(count for _, count in results)
    print("Stress test finished.")
    print(f"Approximate total math-ops: {total_ops:,.0f}")


if __name__ == "__main__":
    main()
