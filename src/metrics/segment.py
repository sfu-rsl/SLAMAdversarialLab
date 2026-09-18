"""Segment-windowed trajectory error, for the recovery experiment.

The recovery experiment releases a stress pulse mid-run and asks which designs come back. The
question is about the stretch AFTER release, so whole-trajectory ATE cannot
answer it: the pulse's own damage sits in the same number and swamps whatever
happened afterwards.

This restricts both the estimate and the ground truth to one time window and
scores that window alone, against the same window of the system's clean run.

Alignment is a choice, and it changes the answer. Two readings exist:

- ``segment`` (default): align on the window itself. Measures how well the
  system tracks after release, independent of any constant offset it carries
  out of the pulse. This is the recovery question.
- ``global``: align on the whole trajectory, then score the window. Keeps the
  accumulated offset in the number, so a system that tracks perfectly after
  release but never returns to the true frame still scores badly.

Default is ``segment`` because the experiment's primary metric already handles the case
this could otherwise flatter. A system that abandoned its map and started a
fresh, internally-consistent one would look "recovered" under segment
alignment, and that is exactly what the binary map-continuity verdict catches
first: the segment ATE is only computed where continuity SURVIVED. The two
metrics are load-bearing together, not separately.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SEGMENT, GLOBAL = "segment", "global"


def window_trajectory(traj, t_start: float, t_end: float):
    """Restrict a trajectory to [t_start, t_end], returning a copy.

    Uses evo's own reducer where present so windowing matches how evo treats
    time ranges elsewhere, and falls back to an explicit timestamp mask.
    """
    out = copy.deepcopy(traj)
    reducer = getattr(out, "reduce_to_time_range", None)
    if callable(reducer):
        reducer(t_start, t_end)
        return out

    import numpy as np  # noqa: PLC0415 — only needed on the fallback path
    mask = (out.timestamps >= t_start) & (out.timestamps <= t_end)
    if not mask.any():
        return None
    out.reduce_to_ids(np.where(mask)[0])
    return out


def segment_ape(
    trajectory_path,
    ground_truth_path,
    dataset_type: str,
    t_start: float,
    t_end: float,
    align: str = SEGMENT,
) -> Optional[Dict[str, Any]]:
    """APE over one time window, or None when the window holds too little.

    Returns the statistics plus the pose count behind them, because a window
    scored on a handful of poses is not comparable to one scored on hundreds
    and the caller must be able to see that rather than infer it.
    """
    from evo.core import metrics as evo_metrics, sync as evo_sync  # noqa: PLC0415

    from .trajectory import _get_max_diff_for_trajectory, _load_trajectory

    traj_est = _load_trajectory(Path(trajectory_path), dataset_type)
    traj_ref = _load_trajectory(Path(ground_truth_path), dataset_type)

    if align == GLOBAL:
        # Align on everything, THEN cut, so the window keeps its offset.
        max_diff = _get_max_diff_for_trajectory(traj_ref)
        ref_sync, est_sync = evo_sync.associate_trajectories(
            traj_ref, traj_est, max_diff=max_diff
        )
        est_aligned = copy.deepcopy(est_sync)
        est_aligned.align(ref_sync, correct_scale=True)
        ref_w = window_trajectory(ref_sync, t_start, t_end)
        est_w = window_trajectory(est_aligned, t_start, t_end)
    else:
        # Cut first, THEN align on the window alone.
        ref_w = window_trajectory(traj_ref, t_start, t_end)
        est_w = window_trajectory(traj_est, t_start, t_end)
        if ref_w is None or est_w is None:
            return None
        max_diff = _get_max_diff_for_trajectory(ref_w)
        ref_w, est_w = evo_sync.associate_trajectories(ref_w, est_w, max_diff=max_diff)
        est_w = copy.deepcopy(est_w)
        est_w.align(ref_w, correct_scale=True)

    if ref_w is None or est_w is None or ref_w.num_poses < 5:
        logger.warning(
            "segment [%.2f, %.2f] holds %s poses: too few to score",
            t_start, t_end, "0" if ref_w is None else ref_w.num_poses,
        )
        return None

    metric = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)
    metric.process_data((ref_w, est_w))
    stats = metric.get_all_statistics()
    return {
        "rmse": stats.get("rmse"),
        "mean": stats.get("mean"),
        "median": stats.get("median"),
        "std": stats.get("std"),
        "num_poses": ref_w.num_poses,
        "window": [t_start, t_end],
        "alignment": align,
    }


def recovery_ratio(
    stressed_traj, clean_traj, ground_truth, dataset_type: str,
    t_start: float, t_end: float, align: str = SEGMENT,
) -> Optional[Dict[str, Any]]:
    """Post-release error against the SAME window of that system's clean run.

    The ratio is the point. An absolute post-release ATE says little on its own,
    because systems differ by an order of magnitude when nothing is wrong: the
    comparison that means something is against what this system does on this
    window when unstressed.
    """
    stressed = segment_ape(stressed_traj, ground_truth, dataset_type, t_start, t_end, align)
    clean = segment_ape(clean_traj, ground_truth, dataset_type, t_start, t_end, align)
    if not stressed or not clean or not clean.get("rmse"):
        return None
    return {
        "stressed_rmse": stressed["rmse"],
        "clean_rmse": clean["rmse"],
        "ratio": stressed["rmse"] / clean["rmse"],
        "stressed_poses": stressed["num_poses"],
        "clean_poses": clean["num_poses"],
        "window": [t_start, t_end],
        "alignment": align,
    }
