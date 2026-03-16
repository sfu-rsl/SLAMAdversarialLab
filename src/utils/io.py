"""Image I/O utilities for SLAMAdverserialLab."""

import os
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
from PIL import Image
import cv2

from . import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory

    Raises:
        OSError: If directory cannot be created
    """
    path = Path(path)

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise

    return path


def read_image(path: Union[str, Path]) -> np.ndarray:
    """
    Read an image file and return as RGB numpy array.

    Args:
        path: Path to image file

    Returns:
        RGB image as numpy array with shape (H, W, 3) and dtype uint8

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If image cannot be loaded or converted to RGB
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        image = Image.open(path)

        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            else:
                image = image.convert('RGB')

        image_array = np.array(image, dtype=np.uint8)

        logger.debug(f"Loaded image from {path}: shape={image_array.shape}, dtype={image_array.dtype}")
        return image_array

    except Exception as e:
        try:
            logger.debug(f"PIL failed, trying OpenCV for {path}")
            image_bgr = cv2.imread(str(path))

            if image_bgr is None:
                raise ValueError(f"Failed to load image with OpenCV: {path}")

            image_array = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            logger.debug(f"Loaded image with OpenCV from {path}: shape={image_array.shape}")
            return image_array

        except Exception as cv_error:
            logger.error(f"Failed to load image {path}: PIL error: {e}, OpenCV error: {cv_error}")
            raise ValueError(f"Cannot load image from {path}: {e}")


def read_depth(path: Union[str, Path], scale: float = 5000.0) -> np.ndarray:
    """
    Read a depth image and convert to meters.

    Args:
        path: Path to depth image file
        scale: Scale factor to convert to meters (e.g., 5000 for TUM)

    Returns:
        Depth map as numpy array with shape (H, W) and dtype float32 in meters

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If depth image cannot be loaded
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Depth file not found: {path}")

    try:
        depth_image = Image.open(path)

        if depth_image.mode == 'I':
            depth_raw = np.array(depth_image, dtype=np.float32)
        elif depth_image.mode in ['L', 'P']:
            depth_raw = np.array(depth_image, dtype=np.float32)
        else:
            depth_raw = np.array(depth_image, dtype=np.float32)

        depth_meters = depth_raw / scale

        depth_meters[depth_raw == 0] = 0.0

        logger.debug(f"Loaded depth from {path}: shape={depth_meters.shape}, "
                    f"range=[{depth_meters[depth_meters>0].min() if np.any(depth_meters>0) else 0:.3f}, "
                    f"{depth_meters.max():.3f}] meters")

        return depth_meters.astype(np.float32)

    except Exception as e:
        try:
            logger.debug(f"PIL failed for depth, trying OpenCV for {path}")
            depth_raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)

            if depth_raw is None:
                raise ValueError(f"Failed to load depth with OpenCV: {path}")

            depth_meters = depth_raw.astype(np.float32) / scale

            depth_meters[depth_raw == 0] = 0.0

            logger.debug(f"Loaded depth with OpenCV from {path}: shape={depth_meters.shape}")
            return depth_meters

        except Exception as cv_error:
            logger.error(f"Failed to load depth {path}: PIL error: {e}, OpenCV error: {cv_error}")
            raise ValueError(f"Cannot load depth from {path}: {e}")


def write_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """
    Write an image to file (supports both RGB and grayscale).

    Args:
        image: Image as numpy array - either (H, W, 3) for RGB or (H, W) for grayscale
        path: Output file path

    Raises:
        ValueError: If image format is invalid
        OSError: If file cannot be written
    """
    path = Path(path)

    is_grayscale = image.ndim == 2
    is_rgb = image.ndim == 3 and image.shape[2] == 3

    if not (is_grayscale or is_rgb):
        raise ValueError(f"Expected grayscale (H, W) or RGB (H, W, 3) image, got {image.shape}")

    if image.dtype != np.uint8:
        logger.warning(f"Converting image from {image.dtype} to uint8")
        # Clip and convert
        if image.dtype in [np.float32, np.float64]:
            # Assume float images are in [0, 1]
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)

    # Ensure parent directory exists
    ensure_dir(path.parent)

    try:
        extension = path.suffix.lower()

        if extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            pil_mode = 'L' if is_grayscale else 'RGB'
            pil_image = Image.fromarray(image, mode=pil_mode)

            save_kwargs = {}
            if extension in ['.jpg', '.jpeg']:
                save_kwargs['quality'] = 95
                save_kwargs['optimize'] = True
            elif extension == '.png':
                save_kwargs['compress_level'] = 6

            pil_image.save(path, **save_kwargs)
            logger.debug(f"Saved {'grayscale' if is_grayscale else 'RGB'} image to {path} using PIL")

        else:
            # Fallback to OpenCV for other formats
            if is_rgb:
                image_out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_out = image  # Grayscale stays as-is
            success = cv2.imwrite(str(path), image_out)

            if not success:
                raise OSError(f"OpenCV failed to write image to {path}")

            logger.debug(f"Saved image to {path} using OpenCV")

    except Exception as e:
        logger.error(f"Failed to write image to {path}: {e}")
        raise OSError(f"Cannot write image to {path}: {e}")


def write_depth(depth: np.ndarray, path: Union[str, Path], scale: float = 5000.0) -> None:
    """
    Write a depth map to file.

    Args:
        depth: Depth map as numpy array (H, W) in meters with dtype float32
        path: Output file path
        scale: Scale factor to convert from meters (e.g., 5000 for TUM format)

    Raises:
        ValueError: If depth format is invalid
        OSError: If file cannot be written
    """
    path = Path(path)

    if depth.ndim != 2:
        raise ValueError(f"Expected depth map with shape (H, W), got shape {depth.shape}")

    ensure_dir(path.parent)

    try:
        depth_scaled = depth * scale

        depth_scaled = np.where(depth > 0, depth_scaled, 0)

        depth_uint16 = np.clip(depth_scaled, 0, 65535).astype(np.uint16)

        extension = path.suffix.lower()

        if extension == '.png':
            pil_image = Image.fromarray(depth_uint16, mode='I;16')
            pil_image.save(path)
            logger.debug(f"Saved depth to {path} using PIL (16-bit PNG)")

        else:
            success = cv2.imwrite(str(path), depth_uint16)

            if not success:
                raise OSError(f"OpenCV failed to write depth to {path}")

            logger.debug(f"Saved depth to {path} using OpenCV")

    except Exception as e:
        logger.error(f"Failed to write depth to {path}: {e}")
        raise OSError(f"Cannot write depth to {path}: {e}")


def copy_or_truncate_text_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    max_lines: Optional[int] = None,
    preserve_comments: bool = True
) -> None:
    """Copy a text file, optionally truncating to max_lines data lines.

    Args:
        src: Source file path
        dst: Destination file path
        max_lines: Maximum number of data lines to keep (None = copy all)
        preserve_comments: If True, preserve comment lines (starting with #)
    """
    import shutil

    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        return

    # Ensure parent directory exists
    ensure_dir(dst.parent)

    if max_lines is None:
        # Simple copy
        shutil.copy2(src, dst)
        return

    with open(src, 'r') as f:
        all_lines = f.readlines()

    if preserve_comments:
        comment_lines = []
        data_lines = []
        for line in all_lines:
            if line.strip().startswith('#') or not line.strip():
                comment_lines.append(line)
            else:
                data_lines.append(line)
        truncated = comment_lines + data_lines[:max_lines]
    else:
        truncated = all_lines[:max_lines]

    with open(dst, 'w') as f:
        f.writelines(truncated)
