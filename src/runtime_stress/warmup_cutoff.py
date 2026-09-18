"""Timestamp cutoff for excluding deadline-warmup frames from metrics.

The deadline harness delivers the first ``warmup_frames`` frames unpaced and
undropped so one-time startup costs (model load, CUDA JIT) do not count
against the frame budget. Those frames therefore measure the UNSTRESSED
system, and scoring them lets a run that collapsed right after warmup
masquerade as accurate (the scored "trajectory" is just the easy warmup
prefix). Metrics for deadline conditions exclude them, for stressed runs and
their baselines alike, so every cell of a condition scores the same
post-warmup frame set.

This module answers one question: what is the timestamp of the first
post-warmup frame, in the dataset's native trajectory timebase (TUM unix
seconds, EuRoC nanoseconds, KITTI seconds from times.txt)?
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compute_warmup_cutoff_ts(
    dataset_type: str,
    dataset_path: Path,
    warmup_frames: int,
    timestamps_path: Optional[Path] = None,
    stride: int = 1,
) -> Optional[float]:
    """Timestamp of sampled frame ``warmup_frames`` (the first paced frame).

    Poses with a timestamp >= the returned value are post-warmup. Returns
    None (with a warning) when the timestamp source is missing or too short,
    so callers degrade to unfiltered metrics instead of crashing.

    ``warmup_frames`` counts frames the SLAM actually processes, i.e. the
    stride-sampled stream: the SLAM sees every ``stride``-th dataset frame, so
    sampled frame ``warmup_frames`` is raw dataset frame
    ``warmup_frames * stride``. For a stride-2 SLAM (DROID-SLAM, DPVO on TUM)
    the cutoff must be raw frame ``2 * warmup_frames``, not ``warmup_frames``,
    or the last ``(stride-1)/stride`` of the warmup prefix leaks back into the
    post-warmup metrics.
    """
    if warmup_frames <= 0:
        return None
    stride = max(1, int(stride))
    raw_index = warmup_frames * stride
    dataset_path = Path(dataset_path)
    kind = (dataset_type or "").lower()
    try:
        if kind == "tum":
            lines = _data_lines(dataset_path / "rgb.txt")
            return float(lines[raw_index].split()[0])
        if kind == "euroc":
            lines = _data_lines(dataset_path / "mav0" / "cam0" / "data.csv")
            return float(lines[raw_index].split(",")[0])
        if kind == "kitti":
            if timestamps_path is None or not Path(timestamps_path).exists():
                logger.warning(
                    "warmup cutoff: no times.txt for KITTI (%s); metrics stay unfiltered",
                    timestamps_path,
                )
                return None
            lines = _data_lines(Path(timestamps_path))
            return float(lines[raw_index].split()[0])
        logger.warning(
            "warmup cutoff: unsupported dataset type %r; metrics stay unfiltered", dataset_type
        )
        return None
    except (OSError, IndexError, ValueError) as exc:
        logger.warning("warmup cutoff: could not compute (%s); metrics stay unfiltered", exc)
        return None


def _data_lines(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as handle:
        return [
            ln for ln in (line.strip() for line in handle)
            if ln and not ln.startswith("#") and not ln.lower().startswith("timestamp")
        ]
