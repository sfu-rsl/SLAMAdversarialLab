"""Noise generation utilities for realistic perturbations."""

import numpy as np
from typing import Tuple, Optional


def generate_perlin_noise_2d(
    shape: Tuple[int, int],
    scale: float = 100.0,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate 2D Perlin noise using a simple implementation.

    Args:
        shape: Output shape (height, width)
        scale: Scale of the base noise (larger = smoother)
        octaves: Number of noise layers to combine
        persistence: Amplitude multiplier per octave
        lacunarity: Frequency multiplier per octave
        seed: Random seed for reproducibility

    Returns:
        2D array with Perlin noise values in range [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    height, width = shape

    # Generate base random noise
    def interpolate(a, b, x):
        """Cosine interpolation"""
        ft = x * np.pi
        f = (1 - np.cos(ft)) * 0.5
        return a * (1 - f) + b * f

    def generate_smooth_noise(shape, scale):
        """Generate smoothed noise at given scale"""
        h, w = shape
        noise_h = int(h / scale) + 2
        noise_w = int(w / scale) + 2

        noise = np.random.random((noise_h, noise_w))

        result = np.zeros((h, w))

        for y in range(h):
            for x in range(w):
                x_frac = (x / scale)
                y_frac = (y / scale)

                x0 = int(x_frac)
                y0 = int(y_frac)
                x1 = min(x0 + 1, noise_w - 1)
                y1 = min(y0 + 1, noise_h - 1)

                # Interpolation weights
                sx = x_frac - x0
                sy = y_frac - y0

                # Bilinear interpolation
                n00 = noise[y0, x0]
                n10 = noise[y0, x1]
                n01 = noise[y1, x0]
                n11 = noise[y1, x1]

                nx0 = interpolate(n00, n10, sx)
                nx1 = interpolate(n01, n11, sx)
                result[y, x] = interpolate(nx0, nx1, sy)

        return result

    # Generate octaves
    noise = np.zeros(shape)
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0

    for _ in range(octaves):
        noise += generate_smooth_noise(shape, scale / frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize to [0, 1]
    noise = noise / max_value
    return np.clip(noise, 0, 1)
