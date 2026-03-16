#!/usr/bin/env python3
"""Create a showcase video cycling through perturbation types and severity levels.

Usage:
    python scripts/create_showcase_video.py <config.yaml>
    python scripts/create_showcase_video.py <config.yaml> --dry-run
    python scripts/create_showcase_video.py <config.yaml> --keep-staging
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from slamadverseriallab.config.parser import expand_env_vars
from slamadverseriallab.utils.io import read_image, write_image
from slamadverseriallab.utils.paths import create_temp_dir


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate showcase video configuration.

    Args:
        path: Path to YAML config file.

    Returns:
        Parsed config dictionary with env vars expanded.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If required fields are missing or clip directories don't exist.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError("Config file is empty")

    config = expand_env_vars(config)

    for key in ("video", "clips"):
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

    if not config["clips"]:
        raise ValueError("'clips' list is empty")

    errors = []
    for i, clip in enumerate(config["clips"]):
        if "label" not in clip:
            errors.append(f"clips[{i}]: missing 'label'")
            continue
        if "experiment_dir" not in clip:
            errors.append(f"clips[{i}] ({clip['label']}): missing 'experiment_dir'")
            continue
        if "severities" not in clip or not clip["severities"]:
            errors.append(f"clips[{i}] ({clip['label']}): missing or empty 'severities'")
            continue

        exp_dir = Path(clip["experiment_dir"])
        if not exp_dir.exists():
            errors.append(f"clips[{i}] ({clip['label']}): experiment_dir not found: {exp_dir}")
            continue

        images_dir = exp_dir / "images"
        if not images_dir.exists():
            errors.append(f"clips[{i}] ({clip['label']}): images/ dir not found in {exp_dir}")
            continue

        for j, sev in enumerate(clip["severities"]):
            if "name" not in sev:
                errors.append(f"clips[{i}].severities[{j}] ({clip['label']}): missing 'name'")
                continue
            sev_dir = images_dir / sev["name"] / "image_2"
            if not sev_dir.exists():
                errors.append(
                    f"clips[{i}].severities[{j}] ({clip['label']}, {sev['name']}): "
                    f"directory not found: {sev_dir}"
                )

    if errors:
        error_msg = "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)

    return config


def _load_font(font_size: int, config: Dict[str, Any]) -> ImageFont.FreeTypeFont:
    """Load a font, trying config path, DejaVu fallback, then PIL default.

    Args:
        font_size: Desired font size.
        config: Config dict (may contain title_card.font_path).

    Returns:
        PIL font object.
    """
    # Try config-specified font
    font_path = config.get("title_card", {}).get("font_path")
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            print(f"Warning: Could not load font from config path: {font_path}")

    # Try DejaVu Sans (common on Linux)
    dejavu_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for dp in dejavu_paths:
        if Path(dp).exists():
            try:
                return ImageFont.truetype(dp, font_size)
            except (OSError, IOError):
                continue

    print(f"Warning: No TrueType font found, using PIL default (text may be small)")
    return ImageFont.load_default()


def create_title_card(
    label: str,
    width: int,
    height: int,
    config: Dict[str, Any],
) -> np.ndarray:
    """Create a title card image with centered text.

    Args:
        label: Main title text.
        width: Image width in pixels.
        height: Image height in pixels.
        config: Full config dict for styling options.

    Returns:
        RGB numpy array of shape (height, width, 3).
    """
    tc_cfg = config.get("title_card", {})
    bg_color = tuple(tc_cfg.get("background_color", [20, 20, 30]))
    text_color = tuple(tc_cfg.get("text_color", [255, 255, 255]))
    font_size = tc_cfg.get("font_size", 60)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    title_font = _load_font(font_size, config)

    # Center the title
    title_bbox = draw.textbbox((0, 0), label, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]

    title_x = (width - title_w) // 2
    title_y = (height - title_h) // 2

    draw.text((title_x, title_y), label, fill=text_color, font=title_font)

    return np.array(img, dtype=np.uint8)


def fit_to_resolution(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize image to target resolution with letterboxing/pillarboxing.

    Preserves aspect ratio by adding black bars as needed.

    Args:
        image: RGB numpy array of shape (H, W, 3).
        target_w: Target width.
        target_h: Target height.

    Returns:
        RGB numpy array of shape (target_h, target_w, 3).
    """
    h, w = image.shape[:2]

    if w == target_w and h == target_h:
        return image

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Choose interpolation based on whether we're up- or down-scaling
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

    # OpenCV resize expects (width, height)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    return canvas


def add_severity_overlay(
    image: np.ndarray, label: str, config: Dict[str, Any]
) -> np.ndarray:
    """Add a severity label overlay to an image.

    Args:
        image: RGB numpy array of shape (H, W, 3).
        label: Severity label text.
        config: Full config dict for overlay styling.

    Returns:
        RGB numpy array with overlay applied.
    """
    ov_cfg = config.get("severity_overlay", {})
    if not ov_cfg.get("enabled", True):
        return image

    font_size = ov_cfg.get("font_size", 28)
    text_color = tuple(ov_cfg.get("text_color", [255, 255, 255]))
    bg_color = tuple(ov_cfg.get("background_color", [0, 0, 0]))
    bg_alpha = ov_cfg.get("background_alpha", 0.6)
    padding = ov_cfg.get("padding", 10)
    position = ov_cfg.get("position", "top_left")

    pil_img = Image.fromarray(image)
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = _load_font(font_size, config)

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    w, h = pil_img.size
    rect_w = text_w + 2 * padding
    rect_h = text_h + 2 * padding

    if position == "top_left":
        rx, ry = 10, 10
    elif position == "top_right":
        rx, ry = w - rect_w - 10, 10
    elif position == "bottom_left":
        rx, ry = 10, h - rect_h - 10
    elif position == "bottom_right":
        rx, ry = w - rect_w - 10, h - rect_h - 10
    else:
        rx, ry = 10, 10

    # Draw semi-transparent background rectangle
    bg_rgba = bg_color + (int(255 * bg_alpha),)
    draw.rectangle([rx, ry, rx + rect_w, ry + rect_h], fill=bg_rgba)
    draw.text((rx + padding, ry + padding), label, fill=text_color + (255,), font=font)

    # Composite overlay onto image
    pil_img = pil_img.convert("RGBA")
    composited = Image.alpha_composite(pil_img, overlay)
    return np.array(composited.convert("RGB"), dtype=np.uint8)


def stage_frames(
    clips: List[Dict[str, Any]], config: Dict[str, Any], staging_dir: Path
) -> int:
    """Stage all frames (title cards + severity frames) into a sequential directory.

    Args:
        clips: List of clip configurations.
        config: Full config dict.
        staging_dir: Directory to write staged frames into.

    Returns:
        Total number of frames staged.
    """
    video_cfg = config.get("video", {})
    target_w, target_h = video_cfg.get("resolution", [1226, 370])
    fps = video_cfg.get("fps", 10)

    tc_cfg = config.get("title_card", {})
    tc_duration = tc_cfg.get("duration_sec", 2.0)
    tc_frame_count = int(tc_duration * fps)

    fs_cfg = config.get("frame_selection", {})
    frame_count = fs_cfg.get("count", 30)

    staging_dir.mkdir(parents=True, exist_ok=True)
    counter = 0

    total_severity_clips = sum(len(clip.get("severities", [])) for clip in clips)
    total_expected = len(clips) * tc_frame_count + total_severity_clips * frame_count

    with tqdm(total=total_expected, desc="Staging frames", unit="frame") as pbar:
        for clip in clips:
            label = clip["label"]
            exp_dir = Path(clip["experiment_dir"])

            # Write title card frames
            title_card = create_title_card(label, target_w, target_h, config)
            for _ in range(tc_frame_count):
                out_path = staging_dir / f"frame_{counter:06d}.png"
                write_image(title_card, out_path)
                counter += 1
                pbar.update(1)

            # Process each severity
            for sev in clip["severities"]:
                sev_name = sev["name"]
                display_label = sev.get("display", sev_name)
                sev_dir = exp_dir / "images" / sev_name / "image_2"

                # Gather and sort available frames
                frame_files = sorted(sev_dir.glob("*.png"))
                if not frame_files:
                    frame_files = sorted(sev_dir.glob("*.jpg"))

                available = len(frame_files)
                if available == 0:
                    print(f"Warning: No frames found in {sev_dir}, skipping")
                    pbar.update(frame_count)
                    continue

                if available < frame_count:
                    print(
                        f"Warning: {sev_dir} has {available} frames, "
                        f"requested {frame_count}"
                    )

                # Uniform sampling via np.linspace
                n_select = min(frame_count, available)
                indices = np.linspace(0, available - 1, n_select, dtype=int)

                for idx in indices:
                    frame = read_image(str(frame_files[idx]))
                    frame = fit_to_resolution(frame, target_w, target_h)
                    frame = add_severity_overlay(frame, display_label, config)
                    out_path = staging_dir / f"frame_{counter:06d}.png"
                    write_image(frame, out_path)
                    counter += 1
                    pbar.update(1)

                # If fewer frames available than requested, update progress for the gap
                gap = frame_count - n_select
                if gap > 0:
                    pbar.update(gap)

    return counter


def encode_video(staging_dir: Path, output_path: Path, config: Dict[str, Any]) -> None:
    """Encode staged frames into a video using ffmpeg.

    Args:
        staging_dir: Directory containing frame_000000.png, frame_000001.png, ...
        output_path: Path for the output video file.
        config: Full config dict for encoding parameters.

    Raises:
        FileNotFoundError: If ffmpeg is not found.
        RuntimeError: If ffmpeg encoding fails.
    """
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError(
            "ffmpeg not found on PATH. Install ffmpeg to encode videos."
        )

    video_cfg = config.get("video", {})
    fps = video_cfg.get("fps", 10)
    codec = video_cfg.get("codec", "libx264")
    preset = video_cfg.get("preset", "medium")
    crf = video_cfg.get("crf", 18)
    pix_fmt = video_cfg.get("pixel_format", "yuv420p")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(staging_dir / "frame_%06d.png"),
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", pix_fmt,
        str(output_path),
    ]

    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
    }

    print(f"Encoding video: {output_path}")
    result = subprocess.run(cmd, capture_output=True, env=env)

    if result.returncode != 0:
        stderr = result.stderr.decode() if result.stderr else "Unknown error"
        raise RuntimeError(f"ffmpeg encoding failed (exit {result.returncode}):\n{stderr}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Video saved: {output_path} ({file_size_mb:.1f} MB)")


def print_dry_run_summary(config: Dict[str, Any]) -> None:
    """Print a summary of what the video would contain without producing it.

    Args:
        config: Full validated config dict.
    """
    video_cfg = config.get("video", {})
    fps = video_cfg.get("fps", 10)
    res = video_cfg.get("resolution", [1226, 370])

    tc_cfg = config.get("title_card", {})
    tc_duration = tc_cfg.get("duration_sec", 2.0)

    fs_cfg = config.get("frame_selection", {})
    frame_count = fs_cfg.get("count", 30)

    clips = config["clips"]

    print("=" * 60)
    print("SHOWCASE VIDEO - DRY RUN")
    print("=" * 60)
    print(f"Output:     {video_cfg.get('output_path', 'showcase.mp4')}")
    print(f"Resolution: {res[0]}x{res[1]}")
    print(f"FPS:        {fps}")
    print(f"Clips:      {len(clips)}")
    print()

    total_frames = 0
    for i, clip in enumerate(clips):
        label = clip["label"]
        sevs = clip.get("severities", [])
        tc_frames = int(tc_duration * fps)
        clip_frames = tc_frames + len(sevs) * frame_count
        total_frames += clip_frames

        print(f"  [{i + 1}] {label}")
        print(f"      Source:   {clip['experiment_dir']}")
        print(f"      Title card: {tc_duration}s ({tc_frames} frames)")
        for sev in sevs:
            display = sev.get("display", sev["name"])
            sev_dir = Path(clip["experiment_dir"]) / "images" / sev["name"] / "image_2"
            n_available = len(list(sev_dir.glob("*.png")))
            if n_available == 0:
                n_available = len(list(sev_dir.glob("*.jpg")))
            print(f"        - {display}: {min(frame_count, n_available)}/{n_available} frames")
        print()

    total_duration = total_frames / fps
    minutes = int(total_duration) // 60
    seconds = total_duration - minutes * 60
    print(f"Total frames: {total_frames}")
    print(f"Estimated duration: {minutes}m {seconds:.0f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create a showcase video from perturbation experiment frames."
    )
    parser.add_argument("config", help="Path to showcase video YAML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print plan without encoding",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep the staging directory after encoding",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Config loaded: {len(config['clips'])} clips")

    if args.dry_run:
        print_dry_run_summary(config)
        return

    staging_dir = create_temp_dir(prefix="showcase_")
    print(f"Staging dir: {staging_dir}")

    try:
        # Stage all frames
        total = stage_frames(config["clips"], config, staging_dir)
        print(f"Staged {total} frames")

        if total == 0:
            print("Error: No frames were staged, nothing to encode.")
            sys.exit(1)

        # Encode video
        output_path = Path(config["video"].get("output_path", "./results/showcase.mp4"))
        encode_video(staging_dir, output_path, config)

    finally:
        if not args.keep_staging:
            print(f"Cleaning up staging dir: {staging_dir}")
            shutil.rmtree(staging_dir, ignore_errors=True)
        else:
            print(f"Staging dir kept at: {staging_dir}")


if __name__ == "__main__":
    main()
