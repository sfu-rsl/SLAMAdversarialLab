"""Helpers for mapping SLAM-internal frame indices back to dataset
positions when the SAL real-time deadline harness was active.

When ``DeadlineIterator`` skips frames mid-run, the SLAM's internal
sequential frame counter (0, 1, 2, ...) no longer matches positions
in the original sampled stream. The iterator records the mapping in
``deadline_drops.json`` (the survivor list). This module loads that
log and applies the mapping so per-SLAM trajectory-conversion code
doesn't need to copy the same ~20 lines.

SLAM-agnostic: a wrapper for any algorithm that produces a
sequential frame counter can call ``load_drop_log`` and
``remap_internal_indices`` to get back the dataset-ordered positions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

DROP_LOG_FILENAME = "deadline_drops.json"
# Live frame-progress file the DeadlineIterator updates each frame, read by
# the orchestrator to advance frame-anchored control phases.
PROGRESS_FILENAME = "deadline_progress.json"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DropLog:
    """Parsed contents of a deadline_drops.json file.

    ``survivors[i]`` is the position in the original sampled stream
    of the i-th item the SLAM actually consumed. The first
    ``warmup_frames`` survivors were yielded un-deadlined, the rest
    were yielded subject to the wall-clock deadline.
    """

    survivors: List[int]
    dropped: List[int]
    target_fps: float
    total_items: int
    warmup_frames: int = 0


def load_drop_log(output_dir: Path) -> Optional[DropLog]:
    """Load deadline_drops.json from ``output_dir``.

    Returns None ONLY when the file is absent -- the legitimate "the SLAM ran
    without the deadline harness" case, where the caller's identity remap is
    correct.

    A file that EXISTS but is unreadable, malformed, or carries no survivors is
    treated as corruption and raises. Silently returning None there would let a
    counter-keyed SLAM (DROID, DPVO) fall back to an identity remap and emit a
    misaligned trajectory with no warning. Fail loud instead: a present-but-bad
    log always signals a real problem worth debugging.

    A non-None return guarantees ``len(survivors) > 0``, so callers can use it
    directly without further checks.

    Raises
    ------
    ValueError
        If the drop log file exists but cannot be read or parsed, or if it
        parses but has a missing/empty/non-list ``survivors`` field.
    """
    path = Path(output_dir) / DROP_LOG_FILENAME
    if not path.exists():
        return None

    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Deadline drop log {path} exists but is unreadable/malformed: {exc}. "
            f"Refusing to fall back to an identity remap (would misalign the "
            f"trajectory). Delete it to force an unstressed identity run, or "
            f"re-run the SLAM."
        ) from exc

    survivors = payload.get("survivors")
    if not isinstance(survivors, list) or not survivors:
        raise ValueError(
            f"Deadline drop log {path} exists but has no usable 'survivors' list "
            f"(got {type(survivors).__name__}). A written log always records at "
            f"least the warmup survivors, so this indicates a truncated or "
            f"corrupt log; refusing to identity-remap."
        )

    return DropLog(
        survivors=[int(s) for s in survivors],
        dropped=[int(d) for d in payload.get("dropped") or []],
        target_fps=float(payload.get("target_fps") or 0.0),
        total_items=int(payload.get("total_items") or len(survivors)),
        warmup_frames=int(payload.get("warmup_frames") or 0),
    )


def remap_internal_indices(
    internal_indices: Sequence[int],
    drop_log: Optional[DropLog],
) -> List[int]:
    """Map SLAM-internal frame counters to sampled-stream positions.

    Parameters
    ----------
    internal_indices:
        The sequential frame counters the SLAM emitted (e.g. DROID's
        ``data["tstamps"]`` cast to int, or VGGT's submap frame ids).
        Each value must be a non-negative integer.
    drop_log:
        Result of ``load_drop_log``. When None, this function is the
        identity (returns ``list(internal_indices)``).

    Returns
    -------
    List[int]
        The same length as ``internal_indices``. Without a drop log,
        unchanged. With a drop log, ``drop_log.survivors[i]`` for
        each ``i`` in the input.

    Raises
    ------
    ValueError
        If any internal index is out of range for the survivor list,
        or if any index is negative.
    """
    indices = [int(i) for i in internal_indices]
    if drop_log is None:
        return indices

    survivors = drop_log.survivors
    n = len(survivors)
    out: List[int] = []
    for raw in indices:
        if raw < 0:
            raise ValueError(f"internal index must be non-negative, got {raw}")
        if raw >= n:
            raise ValueError(
                f"internal index {raw} out of range for survivor list "
                f"of length {n}"
            )
        out.append(survivors[raw])
    return out


def remap_internal_indices_from_dir(
    internal_indices: Sequence[int],
    output_dir: Path,
) -> List[int]:
    """Convenience: load the drop log from ``output_dir`` and apply the
    remap in one call. Falls back to the identity when no log is found.
    """
    return remap_internal_indices(internal_indices, load_drop_log(output_dir))
