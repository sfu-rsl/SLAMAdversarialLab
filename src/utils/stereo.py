"""Stereo calibration parsing utilities for KITTI and EuRoC datasets."""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import yaml

from .logging import get_logger

logger = get_logger(__name__)

def parse_kitti_calibration(calib_path: Path) -> Dict[str, float]:
    """Extract focal length and baseline from KITTI calib.txt.

    KITTI calib.txt format contains projection matrices P0-P3:
    P0: 7.070912e+02 0.000000e+00 6.018873e+02 0.000000e+00 ...
    P1: ...
    P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 ...
    P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 ...

    P2 is left color camera, P3 is right color camera.
    fx = P[0,0], baseline = (P2[0,3] - P3[0,3]) / fx

    Args:
        calib_path: Path to KITTI calib.txt file

    Returns:
        Dictionary with:
        - 'fx': focal length in pixels
        - 'fy': focal length in pixels (usually same as fx)
        - 'cx': principal point x
        - 'cy': principal point y
        - 'baseline': stereo baseline in meters

    Raises:
        FileNotFoundError: If calib file doesn't exist
        ValueError: If calib file format is invalid
    """
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"KITTI calibration file not found: {calib_path}")

    calibration = {}

    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse "Px: value value value ..." format
            if ':' in line:
                key, values_str = line.split(':', 1)
                key = key.strip()
                values = [float(v) for v in values_str.strip().split()]

                if key in ['P0', 'P1', 'P2', 'P3']:
                    # Reshape to 3x4 matrix
                    if len(values) == 12:
                        calibration[key] = np.array(values).reshape(3, 4)

    if 'P2' not in calibration:
        raise ValueError(f"P2 matrix not found in calibration file: {calib_path}")

    P2 = calibration['P2']

    # Extract intrinsics from P2 (left color camera)
    fx = P2[0, 0]
    fy = P2[1, 1]
    cx = P2[0, 2]
    cy = P2[1, 2]

    if 'P3' in calibration:
        P3 = calibration['P3']
        baseline = abs(P2[0, 3] - P3[0, 3]) / fx
    else:
        # Fallback: use P0/P1 for grayscale cameras
        if 'P0' in calibration and 'P1' in calibration:
            P0 = calibration['P0']
            P1 = calibration['P1']
            baseline = abs(P0[0, 3] - P1[0, 3]) / fx
        else:
            raise ValueError(
                f"Cannot compute baseline: need P2+P3 or P0+P1 in {calib_path}"
            )

    logger.info(
        f"KITTI calibration: fx={fx:.1f}px, fy={fy:.1f}px, "
        f"cx={cx:.1f}, cy={cy:.1f}, baseline={baseline:.4f}m"
    )

    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'baseline': baseline,
    }


def parse_euroc_calibration(
    cam0_sensor_yaml: Path,
    cam1_sensor_yaml: Optional[Path] = None,
) -> Dict[str, float]:
    """Extract focal length and baseline from EuRoC sensor.yaml files.

    EuRoC sensor.yaml format:
    ```yaml
    sensor_type: camera
    camera:
      image_width: 752
      image_height: 480
      intrinsics: [458.654, 457.296, 367.215, 248.375]  # fu, fv, cu, cv
    T_BS:
      cols: 4
      rows: 4
      data: [...]  # 4x4 transformation from body to sensor
    ```

    Baseline is computed from the translation between cam0 and cam1 T_BS matrices.

    Args:
        cam0_sensor_yaml: Path to cam0/sensor.yaml
        cam1_sensor_yaml: Path to cam1/sensor.yaml (optional, for baseline)

    Returns:
        Dictionary with:
        - 'fx': focal length in pixels
        - 'fy': focal length in pixels
        - 'cx': principal point x
        - 'cy': principal point y
        - 'baseline': stereo baseline in meters (0 if only cam0 provided)
        - 'width': image width
        - 'height': image height

    Raises:
        FileNotFoundError: If sensor.yaml doesn't exist
        ValueError: If format is invalid
    """
    cam0_sensor_yaml = Path(cam0_sensor_yaml)
    if not cam0_sensor_yaml.exists():
        raise FileNotFoundError(f"EuRoC sensor.yaml not found: {cam0_sensor_yaml}")

    with open(cam0_sensor_yaml, 'r') as f:
        cam0_data = yaml.safe_load(f)

    # Extract intrinsics
    intrinsics = cam0_data.get('intrinsics')
    if intrinsics is None:
        # Try nested camera format
        camera_data = cam0_data.get('camera', {})
        intrinsics = camera_data.get('intrinsics')

    if intrinsics is None or len(intrinsics) < 4:
        raise ValueError(
            f"Invalid intrinsics in {cam0_sensor_yaml}. "
            f"Expected [fu, fv, cu, cv], got: {intrinsics}"
        )

    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

    resolution = cam0_data.get('resolution')
    if resolution:
        width, height = resolution[0], resolution[1]
    else:
        camera_data = cam0_data.get('camera', {})
        width = camera_data.get('image_width', 752)
        height = camera_data.get('image_height', 480)

    baseline = 0.0

    if cam1_sensor_yaml is not None:
        cam1_sensor_yaml = Path(cam1_sensor_yaml)
        if cam1_sensor_yaml.exists():
            with open(cam1_sensor_yaml, 'r') as f:
                cam1_data = yaml.safe_load(f)

            T_BS_0 = _parse_euroc_transform(cam0_data.get('T_BS'))
            T_BS_1 = _parse_euroc_transform(cam1_data.get('T_BS'))

            if T_BS_0 is not None and T_BS_1 is not None:
                # Baseline is the distance between camera origins
                # T_BS gives sensor in body frame, so translation is the position
                t0 = T_BS_0[:3, 3]
                t1 = T_BS_1[:3, 3]
                baseline = np.linalg.norm(t1 - t0)
                logger.debug(
                    f"EuRoC baseline from T_BS: cam0={t0}, cam1={t1}, baseline={baseline:.4f}m"
                )

    logger.info(
        f"EuRoC calibration: fx={fx:.1f}px, fy={fy:.1f}px, "
        f"cx={cx:.1f}, cy={cy:.1f}, baseline={baseline:.4f}m, "
        f"resolution={width}x{height}"
    )

    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'baseline': baseline,
        'width': width,
        'height': height,
    }


def _parse_euroc_transform(t_bs_data: Optional[Dict]) -> Optional[np.ndarray]:
    """Parse EuRoC T_BS transformation matrix from YAML data.

    Args:
        t_bs_data: Dictionary with 'rows', 'cols', 'data' keys

    Returns:
        4x4 transformation matrix, or None if parsing fails
    """
    if t_bs_data is None:
        return None

    try:
        rows = t_bs_data.get('rows', 4)
        cols = t_bs_data.get('cols', 4)
        data = t_bs_data.get('data', [])

        if len(data) != rows * cols:
            return None

        return np.array(data).reshape(rows, cols)
    except (KeyError, ValueError, TypeError):
        return None
