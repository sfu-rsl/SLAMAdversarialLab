"""Tests for robustness-boundary search loop behavior."""

from __future__ import annotations

from pathlib import Path

import yaml

from slamadverseriallab.pipelines.robustness_boundary import (
    BoundaryTrialResult,
    RobustnessBoundaryPipeline,
)


def _write_boundary_config(
    path: Path,
    *,
    base_dir: Path,
    lower_bound: float,
    upper_bound: float,
    tolerance: float,
    max_iters: int,
) -> None:
    config = {
        "experiment": {
            "name": "rb_search_behavior_test",
            "description": "Boundary search behavior test",
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
        "robustness_boundary": {
            "enabled": True,
            "module": "fog",
            "parameter": "visibility_m",
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "tolerance": tolerance,
            "max_iters": max_iters,
            "ate_rmse_fail": 1.5,
        },
        "output": {
            "base_dir": str(base_dir),
            "save_images": True,
            "create_timestamp_dir": False,
        },
    }

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _make_trial_result(
    pipeline: RobustnessBoundaryPipeline,
    *,
    label: str,
    search_value: float | int,
    passed: bool,
) -> BoundaryTrialResult:
    return BoundaryTrialResult(
        label=label,
        search_value=search_value,
        parameter_value=search_value,
        passed=passed,
        failed=not passed,
        tracking_failure=not passed,
        ate_rmse=0.5 if passed else 2.5,
        reason="pass" if passed else "fail",
        trial_config_path=pipeline.configs_dir / f"{label}.yaml",
        trial_output_dir=pipeline.trials_dir / label,
        metrics_summary_path=None,
        error=None,
    )


def test_boundary_search_converges_with_bracket(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "rb_bracketed.yaml"
    _write_boundary_config(
        config_path,
        base_dir=tmp_path / "results",
        lower_bound=0.0,
        upper_bound=10.0,
        tolerance=0.5,
        max_iters=12,
    )
    pipeline = RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")

    def _fake_run_trial(self, search_value, label):  # noqa: ANN001
        # Pass up to threshold 4.0, fail above it.
        return _make_trial_result(
            self,
            label=label,
            search_value=search_value,
            passed=float(search_value) <= 4.0,
        )

    monkeypatch.setattr(RobustnessBoundaryPipeline, "_run_trial", _fake_run_trial)

    summary = pipeline.run()["summary"]

    assert summary["termination_reason"] == "tolerance"
    assert summary["iterations"] > 0
    assert summary["trial_count"] > 2
    assert summary["pass_bound"] is not None
    assert summary["fail_bound"] is not None
    assert summary["pass_bound"]["search_value"] < summary["fail_bound"]["search_value"]


def test_boundary_search_all_pass_is_not_bracketed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "rb_all_pass.yaml"
    _write_boundary_config(
        config_path,
        base_dir=tmp_path / "results",
        lower_bound=0.0,
        upper_bound=10.0,
        tolerance=0.5,
        max_iters=8,
    )
    pipeline = RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")

    def _fake_run_trial(self, search_value, label):  # noqa: ANN001
        return _make_trial_result(
            self,
            label=label,
            search_value=search_value,
            passed=True,
        )

    monkeypatch.setattr(RobustnessBoundaryPipeline, "_run_trial", _fake_run_trial)

    summary = pipeline.run()["summary"]

    assert summary["termination_reason"] == "not_bracketed"
    assert summary["iterations"] == 0
    assert summary["trial_count"] == 2
    assert summary["pass_bound"] is None
    assert summary["fail_bound"] is None


def test_boundary_search_all_fail_is_not_bracketed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "rb_all_fail.yaml"
    _write_boundary_config(
        config_path,
        base_dir=tmp_path / "results",
        lower_bound=0.0,
        upper_bound=10.0,
        tolerance=0.5,
        max_iters=8,
    )
    pipeline = RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")

    def _fake_run_trial(self, search_value, label):  # noqa: ANN001
        return _make_trial_result(
            self,
            label=label,
            search_value=search_value,
            passed=False,
        )

    monkeypatch.setattr(RobustnessBoundaryPipeline, "_run_trial", _fake_run_trial)

    summary = pipeline.run()["summary"]

    assert summary["termination_reason"] == "not_bracketed"
    assert summary["iterations"] == 0
    assert summary["trial_count"] == 2
    assert summary["pass_bound"] is None
    assert summary["fail_bound"] is None


def test_boundary_search_stops_on_max_iters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "rb_max_iters.yaml"
    _write_boundary_config(
        config_path,
        base_dir=tmp_path / "results",
        lower_bound=0.0,
        upper_bound=100.0,
        tolerance=1e-12,
        max_iters=2,
    )
    pipeline = RobustnessBoundaryPipeline(config_path, slam_algorithm="orbslam3")

    def _fake_run_trial(self, search_value, label):  # noqa: ANN001
        # Keep valid bracket, but with tiny tolerance and low iteration budget.
        return _make_trial_result(
            self,
            label=label,
            search_value=search_value,
            passed=float(search_value) <= 60.0,
        )

    monkeypatch.setattr(RobustnessBoundaryPipeline, "_run_trial", _fake_run_trial)

    summary = pipeline.run()["summary"]

    assert summary["termination_reason"] == "max_iters"
    assert summary["iterations"] == 2
    assert summary["trial_count"] == 4
    labels = [trial["label"] for trial in summary["trials"]]
    assert labels == ["lower_bound", "upper_bound", "iter_01", "iter_02"]
