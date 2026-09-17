#!/usr/bin/env python3
"""SAM 3 segmentation as a realistic co-tenant workload beside a SLAM.

Run by the load controller, not by hand:

    conda run -n sam3 python -m src.runtime_stress.sam3_antagonist \
        --frames-dir <dir> --status <path> --prompt person

WHY A REAL MODEL. The other load antagonists are synthetic -- stress-ng burn
threads and a CUDA matmul spinner. They are defensible as controlled contention
and indefensible as deployment, because nobody runs stress-ng beside their SLAM.
Segmentation is what a robot actually runs: labelling the SLAM's map downstream,
and masking dynamic objects upstream so the SLAM does not track a walking person
and conclude the camera moved.

NO RATE KNOB, BY MEASUREMENT. One instance flat out holds 88-100% SM on a 3090
(12 samples), so this is not one rung of a ladder, it is the ceiling. A second
instance would add no compute pressure -- there is none left -- and would only
double VRAM. So the treatment is binary: segmentation running, or not.

THE SEQUENCE LOOPS, DELIBERATELY. A 500-frame sequence at ~6.5 fps is exhausted
in about 77 s while the SLAM may still be running. Stopping there would let the
load silently disappear part-way through a cell that still reports healthy --
exactly the "ran uncontended but looks stressed" failure the harness exists to
prevent. It wraps instead, and reports `passes` so partial coverage is visible
rather than assumed.

STATUS FILE, NOT A CONTROL FILE. The GPU antagonist polls a control file because
its dose is reconfigured per phase. This has no dose to reconfigure, so the only
signal needed flows the other way: the controller must not start a phase until
the model is loaded. Loading takes ~5.8 s, and a phase that began during it would
measure an unstressed prefix. The controller gates on `state: ready` here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import time

_IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")


def _write_status(path: str, payload: dict) -> None:
    """Atomic status write: the controller polls this while we write it."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle)
    os.replace(tmp, path)


def _find_frames(frames_dir: str, max_frames: int | None) -> list:
    """Frames in deterministic order, honouring the same cap the SLAM uses.

    Sorted so a run is reproducible, and capped so `source: dataset` means the
    same frame budget rather than merely the same directory.
    """
    found: list = []
    for pattern in _IMAGE_EXTS:
        found.extend(glob.glob(os.path.join(frames_dir, "**", pattern), recursive=True))
    found.sort()
    if max_frames:
        found = found[:max_frames]
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--prompt", default="person")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    frames = _find_frames(args.frames_dir, args.max_frames or None)
    if not frames:
        _write_status(args.status, {
            "state": "error",
            "message": f"no images under {args.frames_dir}",
        })
        return 2

    try:
        import torch
        from PIL import Image
    except Exception as exc:
        _write_status(args.status, {"state": "error", "message": f"import: {exc}"[:400]})
        return 2

    if not torch.cuda.is_available():
        _write_status(args.status, {"state": "error", "message": "CUDA not available"})
        return 2

    # SAM 3 weights are bf16 and activations arrive fp32; without this the
    # forward pass dies on a dtype mismatch. Every SAM 3 example enters it
    # globally, so it is the intended invocation rather than a workaround.
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        t0 = time.monotonic()
        processor = Sam3Processor(build_sam3_image_model())
        # One inference before declaring ready. The first frame costs 2-5x a
        # steady-state one (kernel autotuning, lazy init), so a controller that
        # unblocked on load alone would still race that cost into the phase.
        warm = Image.open(frames[0]).convert("RGB")
        st = processor.set_image(warm)
        processor.set_text_prompt(state=st, prompt=args.prompt)
        torch.cuda.synchronize()
        load_s = time.monotonic() - t0
    except Exception as exc:
        _write_status(args.status, {"state": "error", "message": f"load: {exc}"[:400]})
        return 3

    stopping = {"now": False}

    def _stop(_signum, _frame):
        stopping["now"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    # ORPHAN GUARD. The controller idles this process with SIGSTOP between
    # phases, and a SIGSTOPped process cannot handle SIGTERM. If the harness
    # dies abruptly while we are stopped -- a killed campaign, a crashed
    # parent -- the queued SIGTERM is never delivered and this survives as a
    # stopped orphan still holding ~6 GB of VRAM, which would then contend with
    # every later run while those runs record no antagonist at all.
    #
    # Watching the parent pid closes that: once re-parented to init (ppid 1)
    # there is no harness left to serve, so exit. Checked in the frame loop
    # below, which only runs while we are NOT stopped -- exactly when we are
    # able to act on it.
    parent_pid = os.getppid()

    started = time.monotonic()
    _write_status(args.status, {
        "state": "ready", "load_s": round(load_s, 2),
        "frames_in_sequence": len(frames), "prompt": args.prompt,
        "frames_done": 0, "passes": 0,
    })

    done = 0
    last_report = time.monotonic()
    try:
        while not stopping["now"]:
            for path in frames:
                if stopping["now"]:
                    break
                if os.getppid() != parent_pid:
                    # Re-parented: the harness that owns us is gone.
                    stopping["now"] = True
                    break
                image = Image.open(path).convert("RGB")
                state = processor.set_image(image)
                processor.set_text_prompt(state=state, prompt=args.prompt)
                done += 1
                # Report about once a second. Frequent enough that a stalled
                # antagonist is visible in telemetry, rare enough that the
                # status write is not part of the measured workload.
                now = time.monotonic()
                if now - last_report >= 1.0:
                    _write_status(args.status, {
                        "state": "running", "load_s": round(load_s, 2),
                        "frames_in_sequence": len(frames), "prompt": args.prompt,
                        "frames_done": done, "passes": done // len(frames),
                        "elapsed_s": round(now - started, 1),
                        "fps": round(done / max(now - started, 1e-6), 2),
                    })
                    last_report = now
    except Exception as exc:
        _write_status(args.status, {
            "state": "error", "message": str(exc)[:400], "frames_done": done,
        })
        return 4

    elapsed = time.monotonic() - started
    _write_status(args.status, {
        "state": "stopped", "load_s": round(load_s, 2),
        "frames_in_sequence": len(frames), "prompt": args.prompt,
        "frames_done": done, "passes": done // len(frames),
        "elapsed_s": round(elapsed, 1),
        "fps": round(done / max(elapsed, 1e-6), 2),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
