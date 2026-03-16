"""Tests for robustness-boundary output directory naming."""

from pathlib import Path

import yaml

from slamadverseriallab.pipelines.robustness_boundary import RobustnessBoundaryPipeline


def _write_boundary_config(
    path: Path,
    *,
    base_dir: Path,
    boundary_name: str | None,
) -> None:
    rb_section = {
        "enabled": True,
        "module": "fog",
        "parameter": "visibility_m",
        "lower_bound": 10.0,
        "upper_bound": 200.0,
        "tolerance": 20.0,
        "max_iters": 2,
        "ape_rmse_fail": 1.5,
    }
    if boundary_name is not None:
        rb_section["name"] = boundary_name

    config = {
        "experiment": {
            "name": "rb_output_path_test",
            "description": "Boundary output naming test",
        },
        "dataset": {
            "type": "mock",
            "max_frames": 1,
        },
        "perturbations": [
            {
                "name": "fog_boundary",
                "type": "fog",
                "enabled": True,
                "parameters": {
                    "visibility_m": 100.0,
                },
            }
        ],
        "robustness_boundary": rb_section,
        "output": {
            "base_dir": str(base_dir),
            "save_images": True,
            "create_timestamp_dir": False,
        },
    }

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_boundary_output_path_uses_explicit_name(tmp_path: Path) -> None:
    config_path = tmp_path / "rb_named.yaml"
    _write_boundary_config(
        config_path,
        base_dir=tmp_path / "results",
        boundary_name="fog_visibility_named_test",
    )

    pipeline = RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")

    expected = (
        (tmp_path / "results").resolve()
        / "rb_output_path_test"
        / "robustness_boundary"
        / "orbslam3"
        / "fog_visibility_named_test"
    )
    assert pipeline.boundary_dir == expected


def test_boundary_output_path_falls_back_to_module_parameter(tmp_path: Path) -> None:
    config_path = tmp_path / "rb_default.yaml"
    _write_boundary_config(
        config_path,
        base_dir=tmp_path / "results",
        boundary_name=None,
    )

    pipeline = RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")

    expected = (
        (tmp_path / "results").resolve()
        / "rb_output_path_test"
        / "robustness_boundary"
        / "orbslam3"
        / "fog"
        / "visibility_m"
    )
    assert pipeline.boundary_dir == expected
