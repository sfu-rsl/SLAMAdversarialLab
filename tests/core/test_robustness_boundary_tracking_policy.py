"""Tests for tracking-failure policy in robustness-boundary classification."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from slamadverseriallab.pipelines.robustness_boundary import RobustnessBoundaryPipeline


def _write_config(path: Path, *, base_dir: Path, fail_on_tracking_failure: bool) -> None:
    config = {
        "experiment": {"name": "rb_tracking_policy_test"},
        "dataset": {"type": "mock", "max_frames": 1},
        "perturbations": [
            {
                "name": "fog_boundary",
                "type": "fog",
                "enabled": True,
                "parameters": {"visibility_m": 100.0},
            }
        ],
        "robustness_boundary": {
            "enabled": True,
            "module": "fog",
            "parameter": "visibility_m",
            "lower_bound": 10,
            "upper_bound": 200,
            "ate_rmse_fail": 1.5,
            "fail_on_tracking_failure": fail_on_tracking_failure,
        },
        "output": {
            "base_dir": str(base_dir),
            "save_images": True,
            "create_timestamp_dir": False,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_metrics_summary(path: Path, *, ate_rmse_mean: float, tracking_mean: float) -> None:
    path.write_text(
        json.dumps(
            {
                "perturbed_modules": {
                    "fog_boundary": {
                        "ate_rmse": {"mean": ate_rmse_mean},
                        "tracking_completeness": {"mean": tracking_mean},
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_tracking_failure_can_be_ignored_for_pass_fail(tmp_path: Path) -> None:
    config_path = tmp_path / "rb_tracking_ignored.yaml"
    _write_config(
        config_path,
        base_dir=tmp_path / "results",
        fail_on_tracking_failure=False,
    )
    pipeline = RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")

    metrics_path = tmp_path / "summary.json"
    _write_metrics_summary(metrics_path, ate_rmse_mean=0.08, tracking_mean=90.0)

    trial_eval = pipeline._classify_trial(
        trajectories={"fog_boundary_run_0": tmp_path / "traj.txt"},
        metrics_summary_path=metrics_path,
    )

    assert trial_eval["tracking_failure"] is True
    assert trial_eval["passed"] is True
    assert trial_eval["failed"] is False
    assert "tracking_failure_ignored" in trial_eval["reason"]


def test_tracking_failure_remains_fatal_by_default(tmp_path: Path) -> None:
    config_path = tmp_path / "rb_tracking_fatal.yaml"
    _write_config(
        config_path,
        base_dir=tmp_path / "results",
        fail_on_tracking_failure=True,
    )
    pipeline = RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")

    metrics_path = tmp_path / "summary.json"
    _write_metrics_summary(metrics_path, ate_rmse_mean=0.08, tracking_mean=90.0)

    trial_eval = pipeline._classify_trial(
        trajectories={"fog_boundary_run_0": tmp_path / "traj.txt"},
        metrics_summary_path=metrics_path,
    )

    assert trial_eval["tracking_failure"] is True
    assert trial_eval["passed"] is False
    assert trial_eval["failed"] is True
    assert "tracking_failure" in trial_eval["reason"]
