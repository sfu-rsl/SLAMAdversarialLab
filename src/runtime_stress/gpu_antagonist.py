"""Tunable GPU load antagonist for SAL load-based runtime stress.

Runs inside a CUDA-capable container (default: the droidslam image) as a
sibling of the SLAM under test and generates calibrated GPU contention with
three knobs, reconfigurable at runtime:

- ``vram_mb``: VRAM ballast held in 256 MB chunks (allocator pressure).
- ``matmul_n``: square matmul operand size (SM occupancy per launch).
- ``duty_cycle``: fraction of each 100 ms period spent launching matmuls
  (time-slice contention; 0.0 = idle).

Protocol (the SAL_PROGRESS_PATH pattern, JSON files in a bind-mounted dir):
the controller writes an epoch-stamped control file; this script polls it,
applies changes, and echoes the epoch into an atomically-renamed status file.
The controller treats a missing/late echo as failure (fail loud). Any CUDA
error writes an error status and exits nonzero so the container leaves the
Running state and the controller notices.

Deliberately stdlib+torch only: it must run unmodified in any torch image.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PERIOD_S = 0.1
CHUNK_BYTES = 256 * 1024 * 1024


def read_json(path):
    """Read a JSON file, returning None on missing/partial/corrupt content.

    The writer may be mid-rename or mid-write; like the deadline progress
    reader, a failed read means "keep last known state", never a crash.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json_atomic(path, payload):
    """Write JSON via tmp-file + os.replace so readers never see partials."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def ballast_chunks_for(vram_mb: int) -> int:
    """Number of 256 MB chunks needed to hold ~vram_mb of ballast."""
    if vram_mb <= 0:
        return 0
    return max(1, round(vram_mb * 1024 * 1024 / CHUNK_BYTES))


def parse_control(payload) -> dict:
    """Normalize a control-file payload into applied settings.

    Unknown/missing fields fall back to idle values; epoch is required for
    a payload to be actionable.
    """
    if not isinstance(payload, dict) or "epoch" not in payload:
        return {}
    try:
        return {
            "epoch": int(payload["epoch"]),
            "vram_mb": max(0, int(payload.get("vram_mb", 0))),
            "matmul_n": max(0, int(payload.get("matmul_n", 0))),
            "duty_cycle": min(1.0, max(0.0, float(payload.get("duty_cycle", 0.0)))),
        }
    except (TypeError, ValueError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, help="control JSON path (read)")
    parser.add_argument("--status", required=True, help="status JSON path (written)")
    args = parser.parse_args()

    import torch  # noqa: PLC0415 — deferred so --help works without torch

    if not torch.cuda.is_available():
        write_json_atomic(args.status, {"state": "error", "message": "CUDA not available"})
        return 2

    # Force full CUDA context + cuBLAS init now, so the controller's ready
    # gate covers ALL first-use costs and reconfiguration acks are fast.
    warm = torch.ones(8, 8, device="cuda")
    (warm @ warm).sum().item()

    state = {"epoch": 0, "vram_mb": 0, "matmul_n": 0, "duty_cycle": 0.0}
    ballast = []
    operands = {}  # matmul_n -> (a, b), allocated once per size

    write_json_atomic(
        args.status,
        {"state": "ready", "epoch": 0, **{k: state[k] for k in ("vram_mb", "matmul_n", "duty_cycle")}},
    )

    try:
        while True:
            period_start = time.monotonic()

            control = parse_control(read_json(args.control))
            if control and control["epoch"] != state["epoch"]:
                want_chunks = ballast_chunks_for(control["vram_mb"])
                if want_chunks != len(ballast):
                    ballast.clear()
                    torch.cuda.empty_cache()
                    for _ in range(want_chunks):
                        ballast.append(
                            torch.empty(CHUNK_BYTES, dtype=torch.uint8, device="cuda")
                        )
                n = control["matmul_n"]
                if control["duty_cycle"] > 0 and n > 0 and n not in operands:
                    operands[n] = (
                        torch.randn(n, n, device="cuda"),
                        torch.randn(n, n, device="cuda"),
                    )
                state = control
                write_json_atomic(
                    args.status,
                    {
                        "state": "ready",
                        "epoch": state["epoch"],
                        "vram_mb": state["vram_mb"],
                        "matmul_n": state["matmul_n"],
                        "duty_cycle": state["duty_cycle"],
                    },
                )

            busy_budget = state["duty_cycle"] * PERIOD_S
            if busy_budget > 0 and state["matmul_n"] in operands:
                a, b = operands[state["matmul_n"]]
                while time.monotonic() - period_start < busy_budget:
                    (a @ b).sum()
                    torch.cuda.synchronize()

            remaining = PERIOD_S - (time.monotonic() - period_start)
            if remaining > 0:
                time.sleep(remaining)
    except RuntimeError as exc:  # CUDA errors surface as torch RuntimeError
        write_json_atomic(args.status, {"state": "error", "message": str(exc)[:500]})
        return 3


if __name__ == "__main__":
    sys.exit(main())
