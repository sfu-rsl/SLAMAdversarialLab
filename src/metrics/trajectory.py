"""Trajectory metrics evaluation using evo."""

import json
import numpy as np
import copy
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import matplotlib
matplotlib.use('Agg')

# evo imports for direct API usage
from evo.core import metrics as evo_metrics
from evo.core import sync as evo_sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.tools import plot as evo_plot
from evo.core import lie_algebra

from ..utils import get_logger

logger = get_logger(__name__)


def _get_max_diff_for_trajectory(traj: PoseTrajectory3D, base_max_diff: float = 10.0) -> float:
    """
    Get appropriate max_diff for timestamp association based on timestamp scale.

    EuRoC uses nanosecond timestamps (~1e18), while TUM/KITTI use seconds (~1e9).
    We detect the scale from the first timestamp and adjust max_diff accordingly.

    Args:
        traj: Trajectory to check timestamp scale
        base_max_diff: Base max_diff in seconds (default 10s)

    Returns:
        Appropriate max_diff for the trajectory's timestamp scale
    """
    if traj.num_poses == 0:
        return base_max_diff

    first_ts = traj.timestamps[0]

    # EuRoC nanosecond timestamps are around 1.4e18 (year 2014 in ns)
    # Standard second timestamps are around 1.4e9 (year 2014 in s)
    # Threshold: if first timestamp > 1e15, assume nanoseconds
    if first_ts > 1e15:
        # Nanosecond timestamps - scale max_diff accordingly
        return base_max_diff * 1e9
    else:
        # Second timestamps
        return base_max_diff


# Paper-mode severity colors (consistent across all experiments)
SEVERITY_COLORS = {
    'baseline': '#2ca02c',    # green
    'reference': '#7f7f7f',   # gray
    'light': '#6baed6',       # light blue
    'moderate': '#fed976',    # yellow
    'heavy': '#fd8d3c',       # orange
    'severe': '#d73027',      # red
}

# Default fallback color for unknown severities
DEFAULT_SEVERITY_COLOR = '#1f77b4'  # blue


def get_severity_color(name: str) -> str:
    """Extract severity from trajectory name and return corresponding color.

    Args:
        name: Trajectory name (e.g., 'rain_heavy', 'fog_light', 'baseline')

    Returns:
        Hex color code for the severity level
    """
    name_lower = name.lower()

    if name_lower == 'baseline':
        return SEVERITY_COLORS['baseline']

    for severity in ['light', 'moderate', 'heavy', 'severe']:
        if severity in name_lower:
            return SEVERITY_COLORS[severity]

    # Fallback to default
    return DEFAULT_SEVERITY_COLOR


def plot_trajectories(
    trajectories: Dict[str, Path],
    output_path: Path,
    reference_path: Optional[Path] = None,
    format_type: str = "tum",
    plot_mode: str = "xy",
    align: bool = True,
    correct_scale: bool = True,
    paper_mode: bool = False
) -> bool:
    """
    Plot multiple trajectories using evo Python API.

    Matches behavior of: evo_traj tum ... --ref gt.txt -a -s --save_plot

    Args:
        trajectories: Dictionary mapping names to trajectory file paths
        output_path: Path to save plot (without extension, .png added)
        reference_path: Optional ground truth trajectory path
        format_type: Trajectory format ('tum' or 'kitti')
        plot_mode: Plot mode ('xy', 'xz', 'yz', 'xyz')
        align: If True, align trajectories to reference (matches -a flag)
        correct_scale: If True, correct scale during alignment (matches -s flag)
        paper_mode: If True, use severity-based colors and hide legend/title

    Returns:
        True if plot was generated successfully
    """
    import matplotlib.pyplot as plt

    try:
        ref_traj = None
        if reference_path and reference_path.exists():
            ref_traj = _load_trajectory(reference_path, format_type)

        traj_dict = {}
        ref_transformed = None
        for name, path in trajectories.items():
            traj = _load_trajectory(path, format_type)

            if ref_traj and align:
                max_diff = _get_max_diff_for_trajectory(ref_traj)
                ref_sync, traj_sync = evo_sync.associate_trajectories(
                    ref_traj, traj, max_diff=max_diff
                )

                traj_aligned = copy.deepcopy(traj_sync)
                r, t, s = traj_aligned.align(ref_sync, correct_scale=correct_scale)
                logger.debug(f"  Aligned {name}: scale={s:.4f}")
                traj_dict[name] = traj_aligned

                if ref_transformed is None:
                    ref_transformed = ref_traj
            else:
                traj_dict[name] = traj

        ref_for_plot = ref_transformed if ref_transformed is not None else ref_traj

        fig = plt.figure(figsize=(10, 8))

        mode_map = {
            'xy': evo_plot.PlotMode.xy,
            'xz': evo_plot.PlotMode.xz,
            'yz': evo_plot.PlotMode.yz,
            'xyz': evo_plot.PlotMode.xyz
        }
        mode = mode_map.get(plot_mode, evo_plot.PlotMode.xy)

        if plot_mode == 'xyz':
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        # Plot reference first (if available, skip in paper mode since we have baseline)
        if ref_for_plot and not paper_mode:
            evo_plot.traj(ax, mode, ref_for_plot, '--', 'gray', 'reference')

        # Plot each trajectory with different colors
        if paper_mode:
            # Paper mode: use severity-based colors, no labels, thicker lines
            original_linewidth = plt.rcParams['lines.linewidth']
            plt.rcParams['lines.linewidth'] = 5.0  # Thicker lines for paper
            for name, traj in traj_dict.items():
                color = get_severity_color(name)
                evo_plot.traj(ax, mode, traj, '-', color, '')  # Empty label
            plt.rcParams['lines.linewidth'] = original_linewidth  # Restore
        else:
            # Normal mode: use tab10 colormap with labels
            colors = plt.cm.tab10(np.linspace(0, 1, len(traj_dict)))
            for (name, traj), color in zip(traj_dict.items(), colors):
                evo_plot.traj(ax, mode, traj, '-', color, name)

        # Only show legend and title in normal mode
        if not paper_mode:
            ax.legend()
            ax.set_title(f"Trajectory Comparison ({plot_mode})")
        else:
            # Paper mode: simplify axis ticks to min, middle, max with larger font
            tick_fontsize = 40
            ax.tick_params(axis='both', pad=25)  # Add padding to avoid overlap in corner
            for axis_getter in [ax.get_xlim, ax.get_ylim]:
                lim = axis_getter()
                mid = (lim[0] + lim[1]) / 2
                ticks = [lim[0], mid, lim[1]]
                if axis_getter == ax.get_xlim:
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([f'{t:.1f}' for t in ticks], fontsize=tick_fontsize)
                else:
                    ax.set_yticks(ticks)
                    ax.set_yticklabels([f'{t:.1f}' for t in ticks], fontsize=tick_fontsize)
            # For 3D plots, also simplify z-axis
            if plot_mode == 'xyz':
                zlim = ax.get_zlim()
                zmid = (zlim[0] + zlim[1]) / 2
                ax.set_zticks([zlim[0], zmid, zlim[1]])
                ax.set_zticklabels([f'{t:.1f}' for t in [zlim[0], zmid, zlim[1]]], fontsize=tick_fontsize)

        fig.savefig(f"{output_path}.png", dpi=150)
        plt.close(fig)

        logger.debug(f"  Saved trajectory plot: {output_path}.png")
        return True

    except Exception as e:
        logger.warning(f"  Failed to generate trajectory plot: {e}")
        return False


def plot_metric_comparison(
    metrics_data: Dict[str, Dict[str, float]],
    output_path: Path,
    metric_name: str = "APE"
) -> bool:
    """
    Plot metric comparison bar chart using matplotlib.

    Args:
        metrics_data: Dictionary mapping trajectory names to their metrics
        output_path: Path to save plot (without extension)
        metric_name: Name of metric for title ('APE' or 'RPE')

    Returns:
        True if plot was generated successfully
    """
    import matplotlib.pyplot as plt

    try:
        names = list(metrics_data.keys())
        rmse_values = [m.get('rmse', 0) for m in metrics_data.values()]

        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

        colors = ['green' if 'baseline' in n.lower() else 'steelblue' for n in names]
        x_positions = range(len(names))
        bars = ax.bar(x_positions, rmse_values, color=colors)

        ax.set_ylabel(f'{metric_name} RMSE (m)')
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(names, rotation=45, ha='right')

        for bar, val in zip(bars, rmse_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        fig.savefig(f"{output_path}.png", dpi=150)
        plt.close(fig)

        logger.debug(f"  Saved {metric_name} comparison plot: {output_path}.png")
        return True

    except Exception as e:
        logger.warning(f"  Failed to generate {metric_name} comparison plot: {e}")
        return False


def count_valid_trajectory_poses(trajectory_path: Path) -> int:
    """
    Count valid poses in a trajectory file (excluding comments, empty lines, and headers).

    Args:
        trajectory_path: Path to trajectory file

    Returns:
        Number of valid pose lines
    """
    count = 0
    with open(trajectory_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and CSV headers
            if line and not line.startswith('#') and not line.startswith('timestamp'):
                count += 1
    return count


def _load_trajectory(trajectory_path: Path, format_type: str) -> PoseTrajectory3D:
    """
    Load trajectory file using evo's file interface.

    Args:
        trajectory_path: Path to trajectory file
        format_type: Format type ('tum', 'kitti', or 'euroc')

    Returns:
        PoseTrajectory3D object
    """
    if format_type == "tum":
        return file_interface.read_tum_trajectory_file(str(trajectory_path))
    elif format_type == "euroc":
        # EuRoC ground truth is CSV; SLAM output is TUM format.
        if is_csv_trajectory(trajectory_path):
            return _load_euroc_trajectory(trajectory_path)
        else:
            logger.debug(f"  EuRoC mode: detected TUM format for {trajectory_path.name}")
            return file_interface.read_tum_trajectory_file(str(trajectory_path))
    elif format_type == "kitti":
        # KITTI loader returns PosePath3D, so add synthetic frame-index timestamps.
        path = file_interface.read_kitti_poses_file(str(trajectory_path))
        timestamps = np.arange(path.num_poses, dtype=float)
        return PoseTrajectory3D(
            timestamps=timestamps,
            positions_xyz=path.positions_xyz,
            orientations_quat_wxyz=path.orientations_quat_wxyz
        )
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def _load_euroc_trajectory(trajectory_path: Path) -> PoseTrajectory3D:
    """
    Load EuRoC ground truth trajectory.

    EuRoC CSV stores quaternions as w,x,y,z and timestamps in nanoseconds.

    Args:
        trajectory_path: Path to EuRoC ground truth CSV

    Returns:
        PoseTrajectory3D object
    """
    import pandas as pd

    df = pd.read_csv(trajectory_path, comment='#', header=None)

    # EuRoC includes additional columns (velocity, biases); only the first 8 are needed here.
    timestamps = df.iloc[:, 0].values.astype(float)
    positions = df.iloc[:, 1:4].values.astype(float)
    quaternions_wxyz = df.iloc[:, 4:8].values.astype(float)

    return PoseTrajectory3D(
        timestamps=timestamps,
        positions_xyz=positions,
        orientations_quat_wxyz=quaternions_wxyz
    )


def convert_euroc_to_tum(euroc_path: Path, tum_path: Path) -> None:
    """
    Convert EuRoC ground truth to TUM format.

    EuRoC uses quaternion order w,x,y,z; TUM uses x,y,z,w.

    Args:
        euroc_path: Path to EuRoC ground truth CSV
        tum_path: Path to output TUM trajectory file
    """
    import pandas as pd

    logger.debug(f"  Converting EuRoC -> TUM: {euroc_path.name} -> {tum_path.name}")

    df = pd.read_csv(euroc_path, comment='#', header=None)

    timestamps_ns = df.iloc[:, 0].values.astype(float)
    positions = df.iloc[:, 1:4].values.astype(float)
    quaternions_wxyz = df.iloc[:, 4:8].values.astype(float)

    quaternions_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]

    # Write TUM format (space-separated, no header)
    with open(tum_path, 'w') as f:
        for i in range(len(timestamps_ns)):
            f.write(f"{timestamps_ns[i]:.6f} "
                   f"{positions[i, 0]:.9f} {positions[i, 1]:.9f} {positions[i, 2]:.9f} "
                   f"{quaternions_xyzw[i, 0]:.9f} {quaternions_xyzw[i, 1]:.9f} "
                   f"{quaternions_xyzw[i, 2]:.9f} {quaternions_xyzw[i, 3]:.9f}\n")

    logger.debug(f"    Conversion complete: {len(timestamps_ns)} poses -> {tum_path}")


def is_csv_trajectory(trajectory_path: Path) -> bool:
    """
    Check if trajectory file is CSV format (comma-separated with header).

    Args:
        trajectory_path: Path to trajectory file

    Returns:
        True if CSV format, False otherwise
    """
    with open(trajectory_path, 'r') as f:
        first_line = f.readline().strip()
        # CSV files typically have a header with column names
        if ',' in first_line and 'timestamp' in first_line.lower():
            return True
        # Also check if first data line is comma-separated
        if ',' in first_line and not first_line.startswith('#'):
            return True
    return False


def convert_csv_to_tum(csv_path: Path, tum_path: Path) -> None:
    """
    Convert CSV trajectory to TUM format, sorted by timestamp.

    CSV format:
        timestamp,tx,ty,tz,qx,qy,qz,qw

    TUM format (evo):
        timestamp tx ty tz qx qy qz qw (space-separated, no header)

    Args:
        csv_path: Path to CSV trajectory file
        tum_path: Path to output TUM trajectory file
    """
    import pandas as pd

    logger.debug(f"  Converting CSV -> TUM: {csv_path.name} -> {tum_path.name}")

    # Read CSV and sort by timestamp for deterministic output
    df = pd.read_csv(csv_path)

    # Sort by first column (timestamp)
    df_sorted = df.sort_values(by=df.columns[0])

    # Write as space-separated TUM format (no header)
    df_sorted.to_csv(tum_path, header=False, index=False, sep=' ')

    logger.debug(f"    Conversion complete: {tum_path}")


def detect_trajectory_format(trajectory_path: Path) -> str:
    """
    Detect trajectory format by counting fields in first data line.

    Args:
        trajectory_path: Path to trajectory file

    Returns:
        'tum' for 8-field format (timestamp + 7DOF pose)
        'kitti' for 12-field format (3x4 matrix flattened)
        'csv' for CSV format (comma-separated with header)

    Raises:
        RuntimeError: If file is empty or format unrecognized
    """
    # First check if it's CSV format
    if is_csv_trajectory(trajectory_path):
        return "csv"

    with open(trajectory_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                num_fields = len(line.split())
                if num_fields == 8:
                    return "tum"
                elif num_fields == 12:
                    return "kitti"
                else:
                    raise RuntimeError(
                        f"Unknown trajectory format in {trajectory_path}: "
                        f"expected 8 (TUM) or 12 (KITTI) fields, got {num_fields}"
                    )
        else:
            raise RuntimeError(f"Empty trajectory file: {trajectory_path}")


def trajectory_uses_frame_indices(trajectory_path: Path) -> bool:
    """
    Detect if a TUM trajectory uses frame indices (0, 1, 2, ...) as timestamps.

    Some SLAM algorithms (like GigaSLAM) output frame indices instead of real
    timestamps. This function detects this by checking if the first few
    timestamps are sequential integers.

    Args:
        trajectory_path: Path to TUM trajectory file

    Returns:
        True if timestamps appear to be frame indices, False otherwise

    Raises:
        FileNotFoundError: If trajectory file doesn't exist
        ValueError: If trajectory file is malformed
    """
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    timestamps = []
    with open(trajectory_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Check first 10 lines
                break
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if not parts:
                    continue
                ts = float(parts[0])
                timestamps.append(ts)

    if len(timestamps) < 2:
        logger.debug(f"  Only {len(timestamps)} timestamps found, assuming real timestamps")
        return False

    for ts in timestamps:
        if abs(ts - round(ts)) > 0.01:
            return False

    avg_diff = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    return avg_diff >= 0.8


def convert_kitti_to_tum(
    kitti_poses_path: Path,
    timestamps_path: Path,
    tum_path: Path,
    use_frame_indices: bool = False
) -> None:
    """
    Convert KITTI trajectory to TUM format using timestamps.

    Uses official evo library functions for conversion:
    https://github.com/MichaelGrupp/evo/blob/master/contrib/kitti_poses_and_timestamps_to_trajectory.py

    KITTI format:
        - Poses file (12 values per line): r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
        - Timestamps file (1 value per line): timestamp in seconds

    TUM format (8 values per line):
        timestamp tx ty tz qx qy qz qw

    Args:
        kitti_poses_path: Path to KITTI poses file (12 values per line)
        timestamps_path: Path to timestamps file (1 value per line)
        tum_path: Path to output TUM trajectory file
        use_frame_indices: If True, use frame indices (0, 1, 2, ...) instead of
                          real timestamps from the timestamps file. Useful when
                          the estimated trajectory uses frame indices as timestamps.
    """
    from evo.tools import file_interface
    from evo.core.trajectory import PoseTrajectory3D

    logger.debug(f"  Converting KITTI → TUM: {kitti_poses_path.name} + {timestamps_path.name} → {tum_path.name}")

    pose_path = file_interface.read_kitti_poses_file(str(kitti_poses_path))

    if use_frame_indices:
        logger.debug("    Using frame indices as timestamps (matching estimated trajectory)")
        timestamps_mat = np.arange(pose_path.num_poses, dtype=float).reshape(-1, 1)
    else:
        raw_timestamps_mat = file_interface.csv_read_matrix(str(timestamps_path))

        error_msg = "timestamp file must have one column of timestamps and same number of rows as the KITTI poses file"
        if (
            len(raw_timestamps_mat) > 0
            and len(raw_timestamps_mat[0]) != 1
            or len(raw_timestamps_mat) != pose_path.num_poses
        ):
            raise file_interface.FileInterfaceException(error_msg)

        try:
            timestamps_mat = np.array(raw_timestamps_mat).astype(float)
        except ValueError:
            raise file_interface.FileInterfaceException(error_msg)

    trajectory = PoseTrajectory3D(
        poses_se3=pose_path.poses_se3,
        timestamps=timestamps_mat
    )

    # Write TUM format using evo
    file_interface.write_tum_trajectory_file(str(tum_path), trajectory)

    logger.debug(f"    Conversion complete: {tum_path}")


class MetricsEvaluator:
    """
    Evaluates SLAM trajectories using evo CLI tools.

    Computes APE (Absolute Pose Error) and RPE (Relative Pose Error) metrics
    by comparing estimated trajectories against ground truth.

    Output structure (new):
        metrics/
        ├── summary.json
        ├── aggregated.png
        └── run_X/
            ├── baseline/
            │   ├── ape.zip
            │   ├── rpe.zip
            │   └── *.png (plots)
            ├── {perturbation}/
            │   ├── ape.zip
            │   ├── rpe.zip
            │   ├── *.png (plots)
            │   └── vs_baseline.json
            └── comparison/
                └── *.png (comparison plots)
    """

    def __init__(
        self,
        output_dir: Path,
        dataset_type: str = "tum",
        max_frames: Optional[int] = None,
        timestamps_path: Optional[Path] = None
    ):
        """
        Initialize metrics evaluator.

        Args:
            output_dir: Directory to save metrics outputs (JSON, CSV, plots)
            dataset_type: Dataset type for evo format (kitti, tum, euroc, etc.)
            max_frames: If set, truncate ground truth to this many frames for comparison
            timestamps_path: Path to timestamps file (for KITTI to TUM conversion).
                             If None and dataset_type is 'kitti', will try to find
                             timestamps based on ground truth path structure.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_type = dataset_type
        self.max_frames = max_frames
        self.timestamps_path = timestamps_path

    def _get_perturbation_dir(self, run_id: int, perturbation_name: str) -> Path:
        """Get or create directory for a perturbation's metrics.

        Args:
            run_id: Run number (0, 1, 2, ...)
            perturbation_name: Name of perturbation (e.g., 'baseline', 'fog', 'rain')

        Returns:
            Path to perturbation directory (e.g., metrics/run_0/fog/)
        """
        perturbation_dir = self.output_dir / f"run_{run_id}" / perturbation_name
        perturbation_dir.mkdir(parents=True, exist_ok=True)
        return perturbation_dir

    def _get_comparison_dir(self, run_id: int) -> Path:
        """Get or create comparison directory for a run.

        Args:
            run_id: Run number

        Returns:
            Path to comparison directory (e.g., metrics/run_0/comparison/)
        """
        comparison_dir = self.output_dir / f"run_{run_id}" / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        return comparison_dir

    def _extract_run_id_and_name(self, trajectory_name: str) -> Tuple[int, str]:
        """Extract run_id and perturbation name from trajectory_name.

        Args:
            trajectory_name: e.g., "fog_run_0" or "baseline_run_1"

        Returns:
            Tuple of (run_id, perturbation_name)
        """
        if '_run_' in trajectory_name:
            parts = trajectory_name.rsplit('_run_', 1)
            return int(parts[1]), parts[0]
        return 0, trajectory_name

    def compute_metrics(
        self,
        trajectory_path: Path,
        ground_truth_path: Path,
        trajectory_name: str = "trajectory"
    ) -> Dict[str, Any]:
        """
        Compute APE and RPE metrics for a trajectory.

        Args:
            trajectory_path: Path to estimated trajectory (TUM or KITTI format)
            ground_truth_path: Path to ground truth trajectory
            trajectory_name: Name prefix for output files

        Returns:
            Dictionary containing APE and RPE metrics

        Raises:
            RuntimeError: If evo command fails
        """
        logger.info(f"Computing metrics for {trajectory_name}")
        logger.info(f"  Trajectory: {trajectory_path}")
        logger.info(f"  Ground truth: {ground_truth_path}")

        # Preserve timestamped formats for sparse trajectories; only synthesize
        # frame-index timestamps for KITTI pose files.

        converted_traj_path = None
        traj_to_use = trajectory_path
        format_to_use = self.dataset_type  # Default to configured dataset type

        detected_format = detect_trajectory_format(trajectory_path)

        if detected_format == "csv":
            logger.info(f"  Detected CSV format - converting to TUM format")
            converted_traj_path = trajectory_path.parent / f".{trajectory_path.stem}_tum.txt"
            convert_csv_to_tum(trajectory_path, converted_traj_path)
            traj_to_use = converted_traj_path
            format_to_use = "tum"
        elif self.dataset_type == "kitti":
            if detected_format == "tum":
                logger.info(f"  Detected TUM format (8 values) - using TUM mode for timestamp association")
                format_to_use = "tum"
            else:
                logger.debug(f"  Detected KITTI format (12 values) - using KITTI mode")
                format_to_use = "kitti"
        else:
            format_to_use = self.dataset_type

        original_dataset_type = self.dataset_type
        self.dataset_type = format_to_use

        converted_gt_path = None
        gt_to_use_for_format = ground_truth_path

        if format_to_use == "tum" and original_dataset_type == "kitti":
            logger.info(f"  Converting ground truth from KITTI to TUM format...")

            use_frame_indices = trajectory_uses_frame_indices(traj_to_use)
            if use_frame_indices:
                logger.info(f"  Detected frame indices in trajectory - using matching timestamps for ground truth")

            timestamps_path = self.timestamps_path
            if timestamps_path is None:
                sequence_num = ground_truth_path.stem
                dataset_root = ground_truth_path.parent.parent
                timestamps_path = dataset_root / "sequences" / sequence_num / "times.txt"

            if not timestamps_path.exists():
                raise FileNotFoundError(
                    f"Cannot convert ground truth to TUM: timestamps file not found at {timestamps_path}\n"
                    f"Provide timestamps_path to MetricsEvaluator or ensure KITTI structure: sequences/<seq>/times.txt"
                )

            converted_gt_path = ground_truth_path.parent / f".{ground_truth_path.stem}_tum.txt"
            convert_kitti_to_tum(
                ground_truth_path, timestamps_path, converted_gt_path,
                use_frame_indices=use_frame_indices
            )
            gt_to_use_for_format = converted_gt_path

        tracking_info = self._check_tracking_completeness(traj_to_use, gt_to_use_for_format, trajectory_name)

        metrics = {
            'tracking_completeness': tracking_info['completeness_percent'],
            'poses_tracked': tracking_info['trajectory_poses'],
            'total_poses': tracking_info['ground_truth_poses'],
            'tracking_lost': tracking_info['poses_lost']
        }

        # TUM trajectories keep timestamps, while KITTI trajectories may require
        # ground-truth truncation when evaluating subsets or partial tracks.
        truncated_gt_path = None
        gt_to_use = gt_to_use_for_format

        gt_full_len = count_valid_trajectory_poses(gt_to_use_for_format)
        need_truncation = False
        truncate_to = gt_full_len

        if format_to_use == "kitti":
            if self.max_frames and self.max_frames < gt_full_len:
                # max_frames is set and less than full ground truth
                need_truncation = True
                truncate_to = self.max_frames
                logger.info(f"  max_frames={self.max_frames} set, will truncate ground truth from {gt_full_len} to {truncate_to} poses")
            elif tracking_info['poses_lost'] > 0:
                # Tracking loss - truncate to match trajectory length
                need_truncation = True
                truncate_to = tracking_info['trajectory_poses']

        if need_truncation and format_to_use == "kitti":
            logger.info(f"  Creating truncated ground truth ({truncate_to} poses) for metrics computation")
            truncated_gt_path = self._create_truncated_ground_truth(
                gt_to_use_for_format,
                truncate_to,
                trajectory_name
            )
            gt_to_use = truncated_gt_path
        elif tracking_info['poses_lost'] > 0:
            logger.info(f"  Using timestamp-based matching (TUM format) - no truncation needed")

        try:
            try:
                ape_metrics = self._compute_ape(
                    traj_to_use,
                    gt_to_use,
                    trajectory_name
                )
                metrics['ape'] = ape_metrics
            except Exception as e:
                logger.error(f"APE computation failed: {e}")
                metrics['ape'] = None
                metrics['ape_error'] = str(e)

            try:
                rpe_metrics = self._compute_rpe(
                    traj_to_use,
                    gt_to_use,
                    trajectory_name
                )
                metrics['rpe'] = rpe_metrics
            except Exception as e:
                logger.error(f"RPE computation failed: {e}")
                metrics['rpe'] = None
                metrics['rpe_error'] = str(e)

        finally:
            # Restore original dataset_type
            self.dataset_type = original_dataset_type

            # Clean up temporary files
            if truncated_gt_path and truncated_gt_path.exists():
                truncated_gt_path.unlink()
                logger.debug(f"  Cleaned up truncated ground truth: {truncated_gt_path}")
            if converted_gt_path and converted_gt_path.exists():
                converted_gt_path.unlink()
                logger.debug(f"  Cleaned up converted ground truth: {converted_gt_path}")
            if converted_traj_path and converted_traj_path.exists():
                converted_traj_path.unlink()
                logger.debug(f"  Cleaned up converted trajectory: {converted_traj_path}")

        logger.info(f"Metrics computed successfully for {trajectory_name}")
        return metrics

    def _compute_ape(
        self,
        trajectory_path: Path,
        ground_truth_path: Path,
        name: str
    ) -> Dict[str, float]:
        """
        Compute Absolute Pose Error using evo Python API.

        Args:
            trajectory_path: Path to estimated trajectory
            ground_truth_path: Path to ground truth
            name: Trajectory name (e.g., "fog_run_0" or "baseline_run_0")

        Returns:
            Dictionary with APE statistics (rmse, mean, median, std, min, max)
        """
        run_id, perturbation_name = self._extract_run_id_and_name(name)
        perturbation_dir = self._get_perturbation_dir(run_id, perturbation_name)

        traj_est = _load_trajectory(trajectory_path, self.dataset_type)
        traj_ref = _load_trajectory(ground_truth_path, self.dataset_type)

        max_diff = _get_max_diff_for_trajectory(traj_ref)
        traj_ref_sync, traj_est_sync = evo_sync.associate_trajectories(
            traj_ref, traj_est, max_diff=max_diff
        )

        logger.debug(f"  Synchronized {traj_est_sync.num_poses} poses (from {traj_est.num_poses} estimated, {traj_ref.num_poses} reference)")

        traj_est_aligned = copy.deepcopy(traj_est_sync)
        r, t, s = traj_est_aligned.align(traj_ref_sync, correct_scale=True)
        logger.debug(f"  Alignment: scale={s:.4f}")

        ape_metric = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)
        ape_metric.process_data((traj_ref_sync, traj_est_aligned))

        stats = ape_metric.get_all_statistics()
        metrics = {
            'rmse': stats.get('rmse', None),
            'mean': stats.get('mean', None),
            'median': stats.get('median', None),
            'std': stats.get('std', None),
            'min': stats.get('min', None),
            'max': stats.get('max', None)
        }

        json_path = perturbation_dir / "ape.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"APE RMSE: {metrics['rmse']:.4f} m")
        return metrics

    def _compute_rpe(
        self,
        trajectory_path: Path,
        ground_truth_path: Path,
        name: str
    ) -> Dict[str, float]:
        """
        Compute Relative Pose Error using evo Python API.

        Args:
            trajectory_path: Path to estimated trajectory
            ground_truth_path: Path to ground truth
            name: Trajectory name (e.g., "fog_run_0" or "baseline_run_0")

        Returns:
            Dictionary with RPE statistics (rmse, mean, median, std, min, max)
        """
        run_id, perturbation_name = self._extract_run_id_and_name(name)
        perturbation_dir = self._get_perturbation_dir(run_id, perturbation_name)

        traj_est = _load_trajectory(trajectory_path, self.dataset_type)
        traj_ref = _load_trajectory(ground_truth_path, self.dataset_type)

        max_diff = _get_max_diff_for_trajectory(traj_ref)
        traj_ref_sync, traj_est_sync = evo_sync.associate_trajectories(
            traj_ref, traj_est, max_diff=max_diff
        )

        logger.debug(f"  Synchronized {traj_est_sync.num_poses} poses for RPE")

        traj_est_aligned = copy.deepcopy(traj_est_sync)
        r, t, s = traj_est_aligned.align(traj_ref_sync, correct_scale=True)
        logger.debug(f"  Alignment: scale={s:.4f}")

        delta = 5
        rpe_metric = evo_metrics.RPE(
            evo_metrics.PoseRelation.translation_part,
            delta=delta,
            delta_unit=evo_metrics.Unit.frames,
            all_pairs=False  # consecutive pairs only (faster than all_pairs)
        )

        try:
            rpe_metric.process_data((traj_ref_sync, traj_est_aligned))
        except evo_metrics.MetricsException as e:
            logger.warning(f"  RPE with delta={delta} failed: {e}")
            logger.warning(f"  Trying with delta=1 for sparse trajectory...")

            rpe_metric = evo_metrics.RPE(
                evo_metrics.PoseRelation.translation_part,
                delta=1,
                delta_unit=evo_metrics.Unit.frames,
                all_pairs=False
            )
            rpe_metric.process_data((traj_ref_sync, traj_est_aligned))

        stats = rpe_metric.get_all_statistics()
        metrics = {
            'rmse': stats.get('rmse', None),
            'mean': stats.get('mean', None),
            'median': stats.get('median', None),
            'std': stats.get('std', None),
            'min': stats.get('min', None),
            'max': stats.get('max', None)
        }

        json_path = perturbation_dir / "rpe.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"RPE RMSE: {metrics['rmse']:.4f} m")
        return metrics

    def compare_trajectories(
        self,
        baseline_path: Path,
        perturbed_path: Path,
        ground_truth_path: Path,
        baseline_name: str = "baseline",
        perturbed_name: str = "perturbed"
    ) -> Dict[str, Any]:
        """
        Compare baseline and perturbed trajectories against ground truth.

        Args:
            baseline_path: Path to baseline trajectory
            perturbed_path: Path to perturbed trajectory
            ground_truth_path: Path to ground truth trajectory
            baseline_name: Name for baseline trajectory (for unique file naming)
            perturbed_name: Name for perturbed trajectory (for unique file naming)

        Returns:
            Dictionary with metrics for both trajectories and degradation analysis
        """
        logger.info("Starting trajectory comparison")

        baseline_metrics = self.compute_metrics(
            baseline_path,
            ground_truth_path,
            trajectory_name=baseline_name
        )

        perturbed_metrics = self.compute_metrics(
            perturbed_path,
            ground_truth_path,
            trajectory_name=perturbed_name
        )

        degradation = self._compute_degradation(baseline_metrics, perturbed_metrics)

        # Summary
        results = {
            'baseline': baseline_metrics,
            'perturbed': perturbed_metrics,
            'degradation': degradation
        }

        run_id, perturbation_name = self._extract_run_id_and_name(perturbed_name)
        perturbation_dir = self._get_perturbation_dir(run_id, perturbation_name)
        summary_path = perturbation_dir / "vs_baseline.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Metrics summary saved to {summary_path}")

        # Skip per-pair plots here; evaluation.py generates the grouped comparisons.

        # Log degradation summary (absolute values, not percentages)
        logger.info("\nPerformance Comparison:")
        ape_perturbed = degradation['ape_perturbed_rmse']
        ape_baseline = degradation['ape_baseline_rmse']
        rpe_perturbed = degradation['rpe_perturbed_rmse']
        rpe_baseline = degradation['rpe_baseline_rmse']

        if ape_perturbed is not None and ape_baseline is not None:
            logger.info(f"  APE RMSE: {ape_perturbed:.4f} m (baseline: {ape_baseline:.4f} m)")
        else:
            logger.warning(f"  APE RMSE: N/A (computation failed)")

        if rpe_perturbed is not None and rpe_baseline is not None:
            logger.info(f"  RPE RMSE: {rpe_perturbed:.4f} m (baseline: {rpe_baseline:.4f} m)")
        else:
            logger.warning(f"  RPE RMSE: N/A (computation failed)")

        logger.info(f"  Tracking: {degradation['perturbed_tracking_completeness']:.1f}% ({degradation['tracking_loss']} frames lost)")

        return results

    def _compute_degradation(
        self,
        baseline_metrics: Dict[str, Any],
        perturbed_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute performance degradation metrics.

        Args:
            baseline_metrics: Metrics for baseline trajectory
            perturbed_metrics: Metrics for perturbed trajectory

        Returns:
            Dictionary with percentage increases for each metric
        """
        def percent_increase(baseline_val: Optional[float], perturbed_val: Optional[float]) -> Optional[float]:
            """Calculate percentage increase, handling None values."""
            if baseline_val is None or perturbed_val is None:
                return None
            if baseline_val == 0:
                return float('inf') if perturbed_val > 0 else 0.0
            return ((perturbed_val - baseline_val) / baseline_val) * 100.0

        def safe_get_rmse(metrics: Dict[str, Any], key: str) -> Optional[float]:
            """Safely get RMSE value from metrics dict."""
            metric_dict = metrics.get(key)
            if metric_dict is None:
                return None
            return metric_dict.get('rmse')

        ape_baseline = safe_get_rmse(baseline_metrics, 'ape')
        ape_perturbed = safe_get_rmse(perturbed_metrics, 'ape')

        rpe_baseline = safe_get_rmse(baseline_metrics, 'rpe')
        rpe_perturbed = safe_get_rmse(perturbed_metrics, 'rpe')

        degradation = {
            'ape_rmse_increase': percent_increase(ape_baseline, ape_perturbed),
            'ape_baseline_rmse': ape_baseline,
            'ape_perturbed_rmse': ape_perturbed,
            'rpe_rmse_increase': percent_increase(rpe_baseline, rpe_perturbed),
            'rpe_baseline_rmse': rpe_baseline,
            'rpe_perturbed_rmse': rpe_perturbed,
            'baseline_tracking_completeness': baseline_metrics.get('tracking_completeness', 100.0),
            'perturbed_tracking_completeness': perturbed_metrics.get('tracking_completeness', 100.0),
            'tracking_loss': perturbed_metrics.get('tracking_lost', 0)
        }

        return degradation

    def _check_tracking_completeness(
        self,
        trajectory_path: Path,
        ground_truth_path: Path,
        trajectory_name: str
    ) -> Dict[str, Any]:
        """
        Check tracking completeness and report tracking loss.

        Args:
            trajectory_path: Path to estimated trajectory
            ground_truth_path: Path to ground truth
            trajectory_name: Name for logging

        Returns:
            Dictionary with tracking statistics
        """
        traj_len = count_valid_trajectory_poses(trajectory_path)
        gt_len = count_valid_trajectory_poses(ground_truth_path)

        expected_gt_len = self.max_frames if self.max_frames else gt_len

        poses_lost = max(0, expected_gt_len - traj_len)
        completeness_percent = (traj_len / expected_gt_len * 100) if expected_gt_len > 0 else 0

        info = {
            'trajectory_poses': traj_len,
            'ground_truth_poses': expected_gt_len,
            'poses_lost': poses_lost,
            'completeness_percent': completeness_percent
        }

        if poses_lost > 0:
            logger.warning(
                f"  SLAM tracking incomplete for {trajectory_name}: "
                f"{traj_len}/{expected_gt_len} poses ({completeness_percent:.1f}%)"
            )
            logger.warning(
                f"  Tracking lost on {poses_lost} frames ({poses_lost/expected_gt_len*100:.1f}% tracking failure)"
            )
        else:
            logger.info(f"  Complete tracking: {traj_len}/{expected_gt_len} poses (100%)")

        return info

    def _create_truncated_ground_truth(
        self,
        ground_truth_path: Path,
        num_poses: int,
        trajectory_name: str
    ) -> Path:
        """
        Create a temporary truncated copy of ground truth to match trajectory length.

        Args:
            ground_truth_path: Path to original ground truth
            num_poses: Number of poses to keep (should match trajectory length)
            trajectory_name: Name for temp file

        Returns:
            Path to truncated ground truth file
        """
        temp_path = ground_truth_path.parent / f".gt_truncated_{trajectory_name}_{num_poses}.txt"

        with open(ground_truth_path, 'r') as f_in:
            lines = [next(f_in) for _ in range(num_poses)]

        with open(temp_path, 'w') as f_out:
            f_out.writelines(lines)

        logger.debug(f"  Created truncated ground truth: {temp_path}")
        return temp_path
