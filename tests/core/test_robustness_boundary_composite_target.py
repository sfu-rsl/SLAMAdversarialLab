"""Tests for robustness-boundary targeting nested modules inside composite perturbations."""

from pathlib import Path

import pytest
import yaml

from slamadverseriallab.pipelines.robustness_boundary import RobustnessBoundaryPipeline


def _write_composite_boundary_config(
    path: Path,
    *,
    base_dir: Path,
    target_perturbation: str = "",
    include_second_composite: bool = False,
) -> None:
    perturbations = [
        {
            "name": "night_fog_boundary",
            "type": "composite",
            "enabled": True,
            "parameters": {
                "mode": "sequential",
                "modules": [
                    {
                        "name": "night",
                        "type": "daynight",
                        "parameters": {
                            "direction": "day_to_night",
                        },
                    },
                    {
                        "name": "fog",
                        "type": "fog",
                        "parameters": {
                            "visibility_m": 80,
                            "encoder": "vitl",
                        },
                    },
                ],
            },
        }
    ]

    if include_second_composite:
        perturbations.append(
            {
                "name": "night_fog_boundary_b",
                "type": "composite",
                "enabled": True,
                "parameters": {
                    "mode": "sequential",
                    "modules": [
                        {
                            "name": "night",
                            "type": "daynight",
                            "parameters": {
                                "direction": "day_to_night",
                            },
                        },
                        {
                            "name": "fog",
                            "type": "fog",
                            "parameters": {
                                "visibility_m": 120,
                                "encoder": "vitl",
                            },
                        },
                    ],
                },
            }
        )

    rb = {
        "enabled": True,
        "module": "fog",
        "parameter": "visibility_m",
        "lower_bound": 10,
        "upper_bound": 200,
        "tolerance": 5,
        "max_iters": 3,
        "ate_rmse_fail": 1.5,
    }
    if target_perturbation:
        rb["target_perturbation"] = target_perturbation

    config = {
        "experiment": {
            "name": "rb_composite_target_test",
            "description": "Boundary composite-target test",
        },
        "dataset": {
            "type": "mock",
            "max_frames": 1,
        },
        "perturbations": perturbations,
        "robustness_boundary": rb,
        "output": {
            "base_dir": str(base_dir),
            "save_images": True,
            "create_timestamp_dir": False,
        },
    }

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_resolve_target_perturbation_supports_nested_fog_in_composite(tmp_path: Path) -> None:
    config_path = tmp_path / "rb_composite.yaml"
    _write_composite_boundary_config(config_path, base_dir=tmp_path / "results")

    pipeline = RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")

    assert pipeline.target_perturbation.name == "night_fog_boundary"
    assert pipeline.target_perturbation.type == "composite"
    assert pipeline._target_nested_module_index == 1


def test_build_trial_config_updates_nested_fog_parameter(tmp_path: Path) -> None:
    config_path = tmp_path / "rb_composite.yaml"
    _write_composite_boundary_config(config_path, base_dir=tmp_path / "results")

    pipeline = RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")
    trial_cfg = pipeline._build_trial_config("trial_001", parameter_value=125)

    assert trial_cfg.robustness_boundary is None
    assert trial_cfg.experiment.name == "trial_001"

    selected = [p for p in trial_cfg.perturbations if p.enabled]
    assert len(selected) == 1
    selected_pert = selected[0]
    assert selected_pert.name == "night_fog_boundary"

    modules = selected_pert.parameters["modules"]
    assert modules[1]["type"] == "fog"
    assert modules[1]["parameters"]["visibility_m"] == 125


def test_target_perturbation_selects_specific_composite(tmp_path: Path) -> None:
    config_path = tmp_path / "rb_composite_select.yaml"
    _write_composite_boundary_config(
        config_path,
        base_dir=tmp_path / "results",
        target_perturbation="night_fog_boundary_b",
        include_second_composite=True,
    )

    pipeline = RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")

    assert pipeline.target_perturbation.name == "night_fog_boundary_b"
    assert pipeline._target_nested_module_index == 1


def test_multiple_composites_without_target_are_ambiguous(tmp_path: Path) -> None:
    config_path = tmp_path / "rb_composite_ambiguous.yaml"
    _write_composite_boundary_config(
        config_path,
        base_dir=tmp_path / "results",
        include_second_composite=True,
    )

    with pytest.raises(ValueError, match="target_perturbation"):
        RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")
