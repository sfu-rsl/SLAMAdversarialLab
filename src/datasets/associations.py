"""Shared TUM association-file resolution and generation helpers."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_tum_association_for_orbslam3(
    dataset_path: Path,
    generate_if_missing: bool = True,
    max_diff: float = 0.02,
    *,
    log: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Find or generate TUM association file using ORB-SLAM3 behavior."""
    active_logger = log or logger

    association_names = [
        "associations.txt",
        "associate.txt",
        "association.txt",
        "rgb_depth_associations.txt",
    ]
    for name in association_names:
        assoc_path = dataset_path / name
        if assoc_path.exists():
            return assoc_path

    for assoc_file in dataset_path.glob("*assoc*.txt"):
        return assoc_file

    if not generate_if_missing:
        return None

    rgb_file = dataset_path / "rgb.txt"
    depth_file = dataset_path / "depth.txt"
    if rgb_file.exists() and depth_file.exists():
        active_logger.info("  Generating association file from rgb.txt and depth.txt...")
        return generate_tum_association_with_associate_py(
            dataset_path=dataset_path,
            rgb_file=rgb_file,
            depth_file=depth_file,
            max_diff=max_diff,
            log=active_logger,
        )

    return None


def generate_tum_association_with_associate_py(
    dataset_path: Path,
    rgb_file: Path,
    depth_file: Path,
    max_diff: float = 0.02,
    *,
    log: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Generate associations using the TUM associate.py script."""
    active_logger = log or logger
    script_path = Path(__file__).parent.parent.parent / "scripts" / "associate.py"
    if not script_path.exists():
        active_logger.error(f"  associate.py script not found at {script_path}")
        return None

    assoc_path = dataset_path / "associations.txt"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(rgb_file),
                str(depth_file),
                "--max_difference",
                str(max_diff),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            active_logger.error(f"  associate.py failed: {result.stderr}")
            return None

        with open(assoc_path, "w") as file_handle:
            file_handle.write(result.stdout)

        num_matches = len([line for line in result.stdout.strip().split("\n") if line])
        active_logger.info(f"  Generated associations.txt with {num_matches} pairs")
        return assoc_path

    except subprocess.TimeoutExpired:
        active_logger.error("  associate.py timed out")
        return None
    except Exception as exc:
        active_logger.error(f"  Failed to generate association file: {exc}")
        return None
