#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a Megatron-Bridge training log with the llmb-run parser.

Step time and TFLOPS come from cli/llmb-run/src/llmb_run/pretrain_log_parser.py, so results match
runs submitted through llmb-run. The global batch size and loss are read from the same lines.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import types
from pathlib import Path

LLMB_RUN_SRC = Path(
    os.environ.get("LLMB_RUN_SRC", Path(__file__).resolve().parents[3] / "cli" / "llmb-run" / "src" / "llmb_run")
)

_NUMBER = r"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
GLOBAL_BATCH_RE = re.compile(r"global batch size:\s*(\d+)")
LOSS_RE = re.compile(rf"lm loss:\s*{_NUMBER}")

EXIT_CODES = {"success": 0, "invalid_grad_norm": 2}


def _load_parser():
    """Import the parser module without llmb_run/__init__.py, which needs the CLI's dependencies."""
    if not (LLMB_RUN_SRC / "pretrain_log_parser.py").is_file():
        sys.exit(f"llmb-run parser not found in {LLMB_RUN_SRC}; run from the benchmark repository or set LLMB_RUN_SRC")
    package = types.ModuleType("llmb_run")
    package.__path__ = [str(LLMB_RUN_SRC)]
    sys.modules.setdefault("llmb_run", package)
    from llmb_run import pretrain_log_parser

    return pretrain_log_parser


def summarize(logs: list[Path], start: int, end: int, seq_length: int | None) -> dict:
    """Parse the logs, concatenated in order, and average iterations start..end inclusive.

    With pipeline parallelism across nodes, the last rank prints the iteration lines and rank 0 the
    throughput lines, so a multi-node run needs the logs of both nodes.
    """
    parser = _load_parser()
    batch_size = None
    losses = []
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as combined:
        for log in logs:
            with log.open("r", errors="replace") as f:
                for line in f:
                    combined.write(line)
                    if match := GLOBAL_BATCH_RE.search(line):
                        batch_size = int(match.group(1))
                    if match := LOSS_RE.search(line):
                        losses.append(float(match.group(1)))
    try:
        result = parser.parse_pretrain_log(combined.name, "megatron_bridge", start, end)
    finally:
        os.unlink(combined.name)

    metrics = result.metrics
    step_time = metrics.time_mean_seconds if metrics else None
    return {
        "status": result.status.value,
        "log_path": ", ".join(str(log) for log in logs),
        "window_start": start,
        "window_end": end,
        "iterations_seen": result.max_iteration_seen,
        "step_time_mean_s": step_time,
        "step_time_std_s": metrics.time_std_seconds if metrics else None,
        "tflops_per_gpu_mean": metrics.tflops_per_gpu_mean if metrics else None,
        "global_batch_size": batch_size,
        "seq_length": seq_length,
        "tokens_per_second": seq_length * batch_size / step_time if seq_length and batch_size and step_time else None,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "nan_grad_norm_iteration": result.invalid_grad_norm_iteration,
    }


def _format(value: float | int | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def print_summary(summary: dict) -> None:
    print("=" * 60)
    print(" DeepSeek V3 benchmark summary")
    print("=" * 60)
    print(f" Status:            {summary['status']}")
    print(f" Log:               {summary['log_path']}")
    print(f" Averaging window:  iterations {summary['window_start']}-{summary['window_end']}")
    step_time = _format(summary["step_time_mean_s"], ".3f")
    print(f" Step time:         {step_time} s/iter (std {_format(summary['step_time_std_s'], '.3f')})")
    print(f" TFLOPS per GPU:    {_format(summary['tflops_per_gpu_mean'], '.2f')}")
    print(f" Global batch size: {_format(summary['global_batch_size'], 'd')}")
    print(f" Tokens per second: {_format(summary['tokens_per_second'], ',.0f')}")
    print(f" LM loss:           {_format(summary['first_loss'], '.4f')} -> {_format(summary['last_loss'], '.4f')}")
    if summary["nan_grad_norm_iteration"] is not None:
        print(f" NaN grad norm at iteration {summary['nan_grad_norm_iteration']}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "log",
        type=Path,
        nargs="+",
        help="Training logs (log-*.out). For multi-node runs, pass the last node's log and node 0's log.",
    )
    parser.add_argument("--start", type=int, default=35, help="First iteration to average (default: 35).")
    parser.add_argument("--end", type=int, default=44, help="Last iteration to average (default: 44).")
    parser.add_argument("--seq-length", type=int, default=None, help="Sequence length, used to report tokens/s.")
    parser.add_argument("--json", type=Path, default=None, help="Optional path for a JSON copy of the summary.")
    args = parser.parse_args(argv)

    if args.start < 1 or args.end < args.start:
        parser.error("--start must be at least 1 and no greater than --end")
    for log in args.log:
        if not log.is_file():
            parser.error(f"log file not found: {log}")

    summary = summarize(args.log, args.start, args.end, args.seq_length)
    print_summary(summary)
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    return EXIT_CODES.get(summary["status"], 1)


if __name__ == "__main__":
    sys.exit(main())
