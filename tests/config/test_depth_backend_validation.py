"""Validation tests for explicit depth backend module parameters."""

from __future__ import annotations

import pytest

from slamadverseriallab.config.schema import PerturbationConfig


@pytest.mark.parametrize(
    "value",
    ["auto", "existing", "foundation_stereo", "da3", "da2", "DA2"],
)
def test_fog_depth_backend_accepts_supported_values(value: str) -> None:
    cfg = PerturbationConfig(
        name="fog_depth_backend",
        type="none",
        parameters={"visibility_m": 20.0, "depth_backend": value},
    )
    cfg._validate_fog_params()


def test_fog_depth_backend_rejects_unknown_value() -> None:
    cfg = PerturbationConfig(
        name="fog_depth_backend_invalid",
        type="none",
        parameters={"visibility_m": 20.0, "depth_backend": "foo_backend"},
    )
    with pytest.raises(ValueError, match="Invalid depth_backend"):
        cfg._validate_fog_params()


def test_rain_depth_backend_rejects_non_string() -> None:
    cfg = PerturbationConfig(
        name="rain_depth_backend_invalid",
        type="none",
        parameters={"intensity": 25, "depth_backend": 123},
    )
    with pytest.raises(ValueError, match="depth_backend must be a non-empty string"):
        cfg._validate_rain_params()


@pytest.mark.parametrize("value", ["auto", "simplex", "perlin", "AUTO", "Perlin"])
def test_fog_noise_backend_accepts_supported_values(value: str) -> None:
    cfg = PerturbationConfig(
        name="fog_noise_backend_valid",
        type="none",
        parameters={"visibility_m": 20.0, "noise_backend": value},
    )
    cfg._validate_fog_params()


def test_fog_noise_backend_rejects_unknown_value() -> None:
    cfg = PerturbationConfig(
        name="fog_noise_backend_invalid",
        type="none",
        parameters={"visibility_m": 20.0, "noise_backend": "foo"},
    )
    with pytest.raises(ValueError, match="Invalid noise_backend"):
        cfg._validate_fog_params()


def test_fog_strict_simplex_rejected_with_migration_hint() -> None:
    cfg = PerturbationConfig(
        name="fog_strict_simplex_removed",
        type="none",
        parameters={"visibility_m": 20.0, "strict_simplex": True},
    )
    with pytest.raises(ValueError, match="strict_simplex is no longer supported"):
        cfg._validate_fog_params()
