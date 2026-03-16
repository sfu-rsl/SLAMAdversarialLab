#!/usr/bin/env python3
"""
Depth Anything V2 Calibration Analysis

Compare DA V2 metric depth predictions against stereo-computed depth from KITTI
to determine if scale calibration is needed.

Usage:
    python scripts/analyze_dav2_calibration.py

Output:
    - Scale factor statistics
    - Visualization plots saved to results/depth_calibration/
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import json
import argparse


# Base paths
REPO_ROOT = Path(__file__).resolve().parents[1]
KITTI_BASE = REPO_ROOT / "datasets" / "kitti" / "sequences"
OUTPUT_BASE = REPO_ROOT / "results" / "depth_calibration"


def parse_kitti_calib(calib_path: Path) -> dict:
    """Parse KITTI calibration file and extract stereo parameters."""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, values = line.split(':', 1)
            calib[key.strip()] = [float(v) for v in values.strip().split()]

    # Extract P2 (left camera) and P3 (right camera) projection matrices
    # P format: [fx, 0, cx, tx, 0, fy, cy, ty, 0, 0, 1, tz]
    p2 = np.array(calib['P2']).reshape(3, 4)
    p3 = np.array(calib['P3']).reshape(3, 4)

    fx = p2[0, 0]
    cx = p2[0, 2]

    # Baseline calculation: tx values are -fx * baseline_offset
    # For rectified stereo: baseline = (tx_P2 - tx_P3) / fx
    tx_p2 = p2[0, 3]  # 46.89
    tx_p3 = p3[0, 3]  # -333.46

    # The baseline in KITTI is defined as the horizontal distance between cameras
    # tx = -fx * b_x, so b_x = -tx / fx
    # Baseline = distance between cam2 and cam3
    baseline = (tx_p2 - tx_p3) / fx

    return {
        'fx': fx,
        'cx': cx,
        'baseline': baseline,
        'P2': p2,
        'P3': p3
    }


def compute_stereo_depth(left_img: np.ndarray, right_img: np.ndarray,
                         fx: float, baseline: float,
                         min_disparity: int = 0,
                         num_disparities: int = 128,
                         block_size: int = 5) -> np.ndarray:
    """
    Compute depth from stereo pair using StereoSGBM.

    Args:
        left_img: Left grayscale image
        right_img: Right grayscale image
        fx: Focal length in pixels
        baseline: Stereo baseline in meters

    Returns:
        Depth map in meters (0 = invalid)
    """
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_img

    if len(right_img.shape) == 3:
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = right_img

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    # depth = (fx * baseline) / disparity
    depth = np.zeros_like(disparity)
    valid = disparity > 0
    depth[valid] = (fx * baseline) / disparity[valid]

    return depth


def load_dav2_depth(depth_path: Path) -> np.ndarray:
    """Load Depth Anything V2 depth map (encoded as uint16, depth * 256)."""
    depth_encoded = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    if depth_encoded is None:
        raise ValueError(f"Could not load depth: {depth_path}")

    # Decode: depth_meters = depth_value / 256
    depth_metric = depth_encoded.astype(np.float32) / 256.0
    return depth_metric


def compute_scale_statistics(stereo_depth: np.ndarray, dav2_depth: np.ndarray,
                            min_depth: float = 1.0, max_depth: float = 50.0,
                            min_stereo_conf: float = 0.1) -> dict:
    """
    Compute scale factor statistics between stereo and DA V2 depth.

    Args:
        stereo_depth: Stereo-computed depth (meters)
        dav2_depth: DA V2 depth (meters)
        min_depth: Minimum valid depth (meters)
        max_depth: Maximum valid depth (meters)

    Returns:
        Dictionary with scale statistics
    """
    valid_stereo = (stereo_depth > min_depth) & (stereo_depth < max_depth)
    valid_dav2 = (dav2_depth > min_depth) & (dav2_depth < max_depth)
    valid = valid_stereo & valid_dav2

    if np.sum(valid) < 100:
        return {'valid_pixels': int(np.sum(valid)), 'error': 'Insufficient valid pixels'}

    stereo_valid = stereo_depth[valid]
    dav2_valid = dav2_depth[valid]

    ratios = stereo_valid / dav2_valid

    # Remove outliers (ratios outside 0.1 to 10)
    valid_ratios = ratios[(ratios > 0.1) & (ratios < 10)]

    if len(valid_ratios) < 100:
        return {'valid_pixels': len(valid_ratios), 'error': 'Too many outliers'}

    return {
        'valid_pixels': int(len(valid_ratios)),
        'scale_mean': float(np.mean(valid_ratios)),
        'scale_median': float(np.median(valid_ratios)),
        'scale_std': float(np.std(valid_ratios)),
        'scale_min': float(np.min(valid_ratios)),
        'scale_max': float(np.max(valid_ratios)),
        'scale_q25': float(np.percentile(valid_ratios, 25)),
        'scale_q75': float(np.percentile(valid_ratios, 75)),
    }


def analyze_frame(frame_idx: int, calib: dict,
                  image_left_dir: Path, image_right_dir: Path,
                  depth_dav2_dir: Path) -> Tuple[dict, Optional[dict]]:
    """
    Analyze a single frame: compute stereo depth and compare with DA V2.

    Returns:
        (statistics_dict, visualization_data) or (error_dict, None)
    """
    frame_name = f"{frame_idx:06d}.png"

    left_path = image_left_dir / frame_name
    right_path = image_right_dir / frame_name
    dav2_path = depth_dav2_dir / frame_name

    if not left_path.exists():
        return {'frame': frame_idx, 'error': f'Left image not found: {left_path}'}, None
    if not right_path.exists():
        return {'frame': frame_idx, 'error': f'Right image not found: {right_path}'}, None
    if not dav2_path.exists():
        return {'frame': frame_idx, 'error': f'DA V2 depth not found: {dav2_path}'}, None

    left_img = cv2.imread(str(left_path))
    right_img = cv2.imread(str(right_path))
    dav2_depth = load_dav2_depth(dav2_path)

    stereo_depth = compute_stereo_depth(
        left_img, right_img,
        fx=calib['fx'],
        baseline=calib['baseline']
    )

    stats = compute_scale_statistics(stereo_depth, dav2_depth)
    stats['frame'] = frame_idx

    # Prepare visualization data
    vis_data = {
        'left_img': left_img,
        'stereo_depth': stereo_depth,
        'dav2_depth': dav2_depth,
        'frame_idx': frame_idx
    }

    return stats, vis_data


def create_visualization(vis_data: dict, stats: dict, output_path: Path):
    """Create comparison visualization for a frame."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    left_img = vis_data['left_img']
    stereo_depth = vis_data['stereo_depth']
    dav2_depth = vis_data['dav2_depth']
    frame_idx = vis_data['frame_idx']

    # Top row: images and depth maps
    axes[0, 0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Left Image')
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(stereo_depth, cmap='plasma', vmin=0, vmax=50)
    axes[0, 1].set_title('Stereo Depth (meters)')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(dav2_depth, cmap='plasma', vmin=0, vmax=50)
    axes[0, 2].set_title('DA V2 Depth (meters)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Bottom row: difference and scatter
    # Depth difference
    min_depth, max_depth = 1.0, 50.0
    valid = (stereo_depth > min_depth) & (stereo_depth < max_depth) & \
            (dav2_depth > min_depth) & (dav2_depth < max_depth)

    diff = np.zeros_like(stereo_depth)
    diff[valid] = stereo_depth[valid] - dav2_depth[valid]

    im3 = axes[1, 0].imshow(diff, cmap='RdBu', vmin=-10, vmax=10)
    axes[1, 0].set_title('Depth Difference (Stereo - DA V2)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # Scatter plot (subsample for speed)
    stereo_valid = stereo_depth[valid]
    dav2_valid = dav2_depth[valid]

    # Subsample if too many points
    if len(stereo_valid) > 10000:
        idx = np.random.choice(len(stereo_valid), 10000, replace=False)
        stereo_sample = stereo_valid[idx]
        dav2_sample = dav2_valid[idx]
    else:
        stereo_sample = stereo_valid
        dav2_sample = dav2_valid

    axes[1, 1].scatter(dav2_sample, stereo_sample, alpha=0.1, s=1)
    axes[1, 1].plot([0, 50], [0, 50], 'r--', label='y=x (perfect)')
    if 'scale_median' in stats:
        scale = stats['scale_median']
        axes[1, 1].plot([0, 50], [0, 50 * scale], 'g--', label=f'y={scale:.2f}x (median)')
    axes[1, 1].set_xlabel('DA V2 Depth (m)')
    axes[1, 1].set_ylabel('Stereo Depth (m)')
    axes[1, 1].set_title('Depth Correlation')
    axes[1, 1].set_xlim(0, 50)
    axes[1, 1].set_ylim(0, 50)
    axes[1, 1].legend()
    axes[1, 1].set_aspect('equal')

    # Scale ratio histogram
    ratios = stereo_valid / dav2_valid
    ratios = ratios[(ratios > 0.1) & (ratios < 10)]

    axes[1, 2].hist(ratios, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 2].axvline(x=1.0, color='r', linestyle='--', label='Scale = 1.0')
    if 'scale_median' in stats:
        axes[1, 2].axvline(x=stats['scale_median'], color='g', linestyle='--',
                          label=f"Median = {stats['scale_median']:.3f}")
    axes[1, 2].set_xlabel('Scale (Stereo / DA V2)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Scale Factor Distribution')
    axes[1, 2].legend()

    if 'scale_median' in stats:
        stats_text = (
            f"Frame {frame_idx}\n"
            f"Valid pixels: {stats['valid_pixels']:,}\n"
            f"Scale median: {stats['scale_median']:.4f}\n"
            f"Scale mean: {stats['scale_mean']:.4f}\n"
            f"Scale std: {stats['scale_std']:.4f}\n"
            f"Scale range: [{stats['scale_min']:.2f}, {stats['scale_max']:.2f}]"
        )
    else:
        stats_text = f"Frame {frame_idx}\nError: {stats.get('error', 'Unknown')}"

    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Depth Anything V2 vs Stereo Depth - KITTI Seq 04, Frame {frame_idx}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_plot(all_stats: list, output_path: Path, seq: str = "04"):
    """Create summary visualization across all frames."""
    # Filter valid results
    valid_stats = [s for s in all_stats if 'scale_median' in s]

    if not valid_stats:
        print("No valid frames to summarize")
        return

    frames = [s['frame'] for s in valid_stats]
    medians = [s['scale_median'] for s in valid_stats]
    means = [s['scale_mean'] for s in valid_stats]
    stds = [s['scale_std'] for s in valid_stats]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scale over frames
    axes[0, 0].plot(frames, medians, 'b-', label='Median', linewidth=2)
    axes[0, 0].plot(frames, means, 'g--', label='Mean', alpha=0.7)
    axes[0, 0].fill_between(frames,
                            [m - s for m, s in zip(medians, stds)],
                            [m + s for m, s in zip(medians, stds)],
                            alpha=0.3, label='Std dev')
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Scale = 1.0')
    axes[0, 0].set_xlabel('Frame Index')
    axes[0, 0].set_ylabel('Scale Factor')
    axes[0, 0].set_title('Scale Factor vs Frame')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Scale histogram
    axes[0, 1].hist(medians, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=1.0, color='r', linestyle='--', label='Scale = 1.0')
    axes[0, 1].axvline(x=np.median(medians), color='g', linestyle='--',
                       label=f'Overall median: {np.median(medians):.4f}')
    axes[0, 1].set_xlabel('Median Scale Factor')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Per-Frame Median Scale')
    axes[0, 1].legend()

    # Std deviation
    axes[1, 0].bar(frames, stds, alpha=0.7)
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('Scale Std Dev')
    axes[1, 0].set_title('Per-Frame Scale Std Deviation')
    axes[1, 0].grid(True, alpha=0.3)

    # Summary statistics text
    overall_median = np.median(medians)
    overall_mean = np.mean(medians)
    overall_std = np.std(medians)

    summary_text = (
        f"KITTI Sequence {seq} - Depth Calibration Summary\n"
        f"{'=' * 50}\n\n"
        f"Frames analyzed: {len(valid_stats)}\n\n"
        f"Overall Scale Factor (Stereo / DA V2):\n"
        f"  Median: {overall_median:.4f}\n"
        f"  Mean:   {overall_mean:.4f}\n"
        f"  Std:    {overall_std:.4f}\n\n"
        f"Interpretation:\n"
        f"  Scale > 1: DA V2 underestimates depth\n"
        f"  Scale < 1: DA V2 overestimates depth\n"
        f"  Scale = 1: Perfect calibration\n\n"
        f"Deviation from 1.0: {abs(overall_median - 1.0):.4f} "
        f"({abs(overall_median - 1.0) * 100:.1f}%)\n\n"
        f"Recommendation:\n"
    )

    if abs(overall_median - 1.0) < 0.05:
        summary_text += "  DA V2 is WELL CALIBRATED for KITTI (error < 5%)"
    elif abs(overall_median - 1.0) < 0.15:
        summary_text += "  DA V2 needs MINOR calibration (error 5-15%)\n"
        summary_text += f"  Multiply DA V2 depth by {overall_median:.3f}"
    else:
        summary_text += "  DA V2 needs SIGNIFICANT calibration (error > 15%)\n"
        summary_text += f"  Multiply DA V2 depth by {overall_median:.3f}"

    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=10, family='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.suptitle(f'Depth Anything V2 Calibration Analysis - KITTI Sequence {seq}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'frames_analyzed': len(valid_stats),
        'overall_scale_median': overall_median,
        'overall_scale_mean': overall_mean,
        'overall_scale_std': overall_std,
        'deviation_from_1': abs(overall_median - 1.0),
        'deviation_percent': abs(overall_median - 1.0) * 100
    }


def main():
    """Run the calibration analysis."""
    parser = argparse.ArgumentParser(description='Analyze DA V2 depth calibration against stereo')
    parser.add_argument('--sequence', '-s', type=str, default='04',
                        help='KITTI sequence number (default: 04)')
    parser.add_argument('--num-frames', '-n', type=int, default=200,
                        help='Number of frames to analyze (default: 200)')
    args = parser.parse_args()

    seq = args.sequence
    num_frames_requested = args.num_frames

    seq_dir = KITTI_BASE / seq
    image_left_dir = seq_dir / "image_2"
    image_right_dir = seq_dir / "image_3"
    depth_dav2_dir = seq_dir / "image_2_depth"
    calib_file = seq_dir / "calib.txt"
    output_dir = OUTPUT_BASE / f"seq_{seq}"

    print("=" * 60)
    print("Depth Anything V2 Calibration Analysis")
    print(f"Dataset: KITTI Sequence {seq}")
    print("=" * 60)

    if not seq_dir.exists():
        print(f"ERROR: Sequence directory not found: {seq_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse calibration
    print("\n[1] Parsing calibration...")
    calib = parse_kitti_calib(calib_file)
    print(f"    Focal length (fx): {calib['fx']:.2f} px")
    print(f"    Stereo baseline: {calib['baseline']:.4f} m ({calib['baseline']*100:.1f} cm)")

    frame_files = sorted(image_left_dir.glob("*.png"))
    total_frames = len(frame_files)
    print(f"\n[2] Found {total_frames} frames in sequence")

    # Analyze requested frames (or all if fewer)
    num_frames_to_analyze = min(num_frames_requested, total_frames)
    sample_indices = list(range(num_frames_to_analyze))

    print(f"    Analyzing {len(sample_indices)} frames (0 to {num_frames_to_analyze - 1})")

    # Analyze frames
    print("\n[3] Analyzing frames...")
    all_stats = []
    vis_frames = [0, 50, 100, 150, 199]  # Frames to visualize
    vis_frames = [f for f in vis_frames if f < total_frames]

    for i, frame_idx in enumerate(sample_indices):
        print(f"    Frame {frame_idx:03d} ({i+1}/{len(sample_indices)})...", end=" ")

        stats, vis_data = analyze_frame(frame_idx, calib, image_left_dir, image_right_dir, depth_dav2_dir)
        all_stats.append(stats)

        if 'scale_median' in stats:
            print(f"scale={stats['scale_median']:.4f}, pixels={stats['valid_pixels']:,}")

            if frame_idx in vis_frames:
                vis_path = output_dir / f"frame_{frame_idx:06d}_comparison.png"
                create_visualization(vis_data, stats, vis_path)
                print(f"           -> Saved visualization to {vis_path.name}")
        else:
            print(f"ERROR: {stats.get('error', 'Unknown')}")

    print("\n[4] Creating summary...")
    summary_path = output_dir / "calibration_summary.png"
    summary_stats = create_summary_plot(all_stats, summary_path, seq)

    if summary_stats:
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print("=" * 60)
        print(f"Frames analyzed: {summary_stats['frames_analyzed']}")
        print(f"Overall scale factor (median): {summary_stats['overall_scale_median']:.4f}")
        print(f"Overall scale factor (mean):   {summary_stats['overall_scale_mean']:.4f}")
        print(f"Scale std deviation:           {summary_stats['overall_scale_std']:.4f}")
        print(f"Deviation from 1.0:            {summary_stats['deviation_percent']:.1f}%")
        print()

        if summary_stats['deviation_percent'] < 5:
            print("CONCLUSION: DA V2 is WELL CALIBRATED for KITTI")
            print("            No scale correction needed for fog/rain modules")
        elif summary_stats['deviation_percent'] < 15:
            print("CONCLUSION: DA V2 needs MINOR calibration")
            print(f"            Recommend multiplying DA V2 depth by {summary_stats['overall_scale_median']:.3f}")
        else:
            print("CONCLUSION: DA V2 needs SIGNIFICANT calibration")
            print(f"            Recommend multiplying DA V2 depth by {summary_stats['overall_scale_median']:.3f}")

    results_path = output_dir / "calibration_results.json"
    results = {
        'dataset': f'KITTI Sequence {seq}',
        'calibration': {
            'focal_length_px': calib['fx'],
            'baseline_m': calib['baseline']
        },
        'per_frame_stats': all_stats,
        'summary': summary_stats
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Summary plot saved to: {summary_path}")
    print(f"Frame visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
