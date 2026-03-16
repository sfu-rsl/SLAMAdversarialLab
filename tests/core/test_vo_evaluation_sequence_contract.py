"""Tests for VO evaluation sequence-name contract."""

import json
from pathlib import Path

import pytest

from slamadverseriallab.cli import create_parser
from slamadverseriallab.config.parser import Config
from slamadverseriallab.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    OutputConfig,
    PerturbationConfig,
)
from slamadverseriallab.pipelines.vo_evaluation import VOEvaluationPipeline


class _DatasetStub:
    def __init__(self, path: Path):
        self.path = path

    def get_canonical_camera_name(self, camera: str = "left") -> str:
        if camera == "left":
            return "image_2"
        if camera == "right":
            return "image_3"
        raise ValueError(f"Unsupported camera role: {camera}")


def _make_config(tmp_path: Path, *, sequence: str | None) -> Config:
    dataset_dir = tmp_path / "dataset_dir_not_sequence"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        experiment=ExperimentConfig(name="vo_sequence_contract"),
        dataset=DatasetConfig(
            type="kitti",
            path=str(dataset_dir),
            sequence=sequence,
            load_stereo=False,
        ),
        perturbations=[],
        output=OutputConfig(
            base_dir=str(tmp_path / "results"),
            save_images=True,
            create_timestamp_dir=False,
        ),
    )


def _make_config_file(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment: {}\n", encoding="utf-8")
    return config_path


def _write_track_outputs(output_dir: Path, *, sequence: str, ages: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks_payload = {
        "tracks": {
            str(idx): {"age": int(age)}
            for idx, age in enumerate(ages)
        }
    }
    (output_dir / f"tracks_{sequence}.json").write_text(
        json.dumps(tracks_payload),
        encoding="utf-8",
    )

    values = [float(v) for v in ages]
    mean_val = sum(values) / len(values)
    sorted_values = sorted(values)
    median_val = (
        sorted_values[len(sorted_values) // 2]
        if len(sorted_values) % 2 == 1
        else (
            sorted_values[(len(sorted_values) // 2) - 1]
            + sorted_values[len(sorted_values) // 2]
        )
        / 2.0
    )
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)

    stats_payload = {
        "total_valid_tracks": len(values),
        "mean_track_length": mean_val,
        "median_track_length": median_val,
        "std_track_length": variance ** 0.5,
        "max_track_length": max(values),
        "min_track_length": min(values),
    }
    (output_dir / f"track_stats_{sequence}.json").write_text(
        json.dumps(stats_payload),
        encoding="utf-8",
    )


def test_vo_evaluation_requires_dataset_sequence(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, sequence=None)
    config_path = _make_config_file(tmp_path)

    create_calls = {"count": 0}

    def _fake_create_dataset(_cfg):
        create_calls["count"] += 1
        return _DatasetStub(Path(config.dataset.path))

    monkeypatch.setattr("slamadverseriallab.pipelines.vo_evaluation.load_config", lambda _p: config)
    monkeypatch.setattr("slamadverseriallab.datasets.create_dataset", _fake_create_dataset)

    with pytest.raises(ValueError, match="dataset.sequence is required for VO evaluation runs"):
        VOEvaluationPipeline(
            config_path=config_path,
            feature_configs=["ORB2"],
            sensor_type="mono",
        )

    assert create_calls["count"] == 0


def test_vo_evaluation_uses_config_sequence_not_dataset_path_name(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, sequence="04")
    config_path = _make_config_file(tmp_path)

    monkeypatch.setattr("slamadverseriallab.pipelines.vo_evaluation.load_config", lambda _p: config)
    monkeypatch.setattr(
        "slamadverseriallab.datasets.create_dataset",
        lambda _cfg: _DatasetStub(Path(config.dataset.path)),
    )

    pipeline = VOEvaluationPipeline(
        config_path=config_path,
        feature_configs=["ORB2"],
        sensor_type="mono",
    )

    assert pipeline.sequence_name == "04"
    assert pipeline.camera_settings == "KITTI04-12.yaml"


def test_vo_discovery_for_tum_uses_canonical_left_output_dir(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, sequence="freiburg1_desk")
    config.dataset.type = "tum"
    config_path = _make_config_file(tmp_path)

    monkeypatch.setattr("slamadverseriallab.pipelines.vo_evaluation.load_config", lambda _p: config)
    monkeypatch.setattr(
        "slamadverseriallab.datasets.create_dataset",
        lambda _cfg: _DatasetStub(Path(config.dataset.path)),
    )

    pipeline = VOEvaluationPipeline(
        config_path=config_path,
        feature_configs=["ORB2"],
        sensor_type="mono",
    )

    module_dir = pipeline.results_dir / "images" / "fog"
    image_dir = module_dir / "image_2"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "000000.png").write_bytes(b"x")

    discovered = pipeline._discover_perturbed_datasets()

    assert discovered == ["fog"]


def test_vo_discovery_ignores_metadata_only_module_directories(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, sequence="04")
    config_path = _make_config_file(tmp_path)

    monkeypatch.setattr("slamadverseriallab.pipelines.vo_evaluation.load_config", lambda _p: config)
    monkeypatch.setattr(
        "slamadverseriallab.datasets.create_dataset",
        lambda _cfg: _DatasetStub(Path(config.dataset.path)),
    )

    pipeline = VOEvaluationPipeline(
        config_path=config_path,
        feature_configs=["ORB2"],
        sensor_type="mono",
    )

    valid_module_dir = pipeline.results_dir / "images" / "fog"
    valid_image_dir = valid_module_dir / "image_2"
    valid_image_dir.mkdir(parents=True, exist_ok=True)
    (valid_image_dir / "000000.jpg").write_bytes(b"x")

    metadata_only_dir = pipeline.results_dir / "images" / "rain"
    metadata_only_dir.mkdir(parents=True, exist_ok=True)
    (metadata_only_dir / "associations.txt").write_text("dummy\n", encoding="utf-8")

    discovered = pipeline._discover_perturbed_datasets()

    assert discovered == ["fog"]


def test_vo_evaluation_rejects_non_positive_num_runs(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, sequence="04")
    config_path = _make_config_file(tmp_path)

    monkeypatch.setattr("slamadverseriallab.pipelines.vo_evaluation.load_config", lambda _p: config)
    monkeypatch.setattr(
        "slamadverseriallab.datasets.create_dataset",
        lambda _cfg: _DatasetStub(Path(config.dataset.path)),
    )

    with pytest.raises(ValueError, match="num_runs must be >= 1"):
        VOEvaluationPipeline(
            config_path=config_path,
            feature_configs=["ORB2"],
            sensor_type="mono",
            num_runs=0,
        )


def test_vo_evaluation_runs_multi_run_outputs_and_aggregates(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, sequence="04")
    config.perturbations = [
        PerturbationConfig(name="fog", type="fog", enabled=True, parameters={}),
    ]
    config_path = _make_config_file(tmp_path)

    monkeypatch.setattr("slamadverseriallab.pipelines.vo_evaluation.load_config", lambda _p: config)
    monkeypatch.setattr(
        "slamadverseriallab.datasets.create_dataset",
        lambda _cfg: _DatasetStub(Path(config.dataset.path)),
    )

    run_to_baseline_ages = {
        0: [10, 11, 12],
        1: [11, 12, 13],
    }
    run_to_fog_ages = {
        0: [6, 7, 8],
        1: [5, 6, 7],
    }

    def _fake_run_on_dataset(
        self,
        perturbed_images_path,
        output_dir: Path,
        feature_config: str,
    ):
        assert feature_config == "ORB2"
        run_id = int(output_dir.parent.name.split("_", 1)[1])
        if perturbed_images_path is None:
            ages = run_to_baseline_ages[run_id]
        else:
            ages = run_to_fog_ages[run_id]

        _write_track_outputs(output_dir, sequence=self.sequence_name, ages=ages)
        traj_path = output_dir / f"trajectory_{self.sequence_name}.txt"
        traj_path.write_text("0.0 0.0 0.0\n", encoding="utf-8")
        return traj_path

    monkeypatch.setattr(VOEvaluationPipeline, "_run_on_dataset", _fake_run_on_dataset)

    pipeline = VOEvaluationPipeline(
        config_path=config_path,
        feature_configs=["ORB2"],
        sensor_type="mono",
        num_runs=2,
    )
    monkeypatch.setattr(pipeline.runner, "cleanup", lambda: None)

    module_dir = pipeline.results_dir / "images" / "fog" / "image_2"
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "000000.png").write_bytes(b"x")

    results = pipeline.run()

    expected_keys = {
        "baseline_run_0",
        "baseline_run_1",
        "fog_run_0",
        "fog_run_1",
    }
    assert set(results["ORB2"].keys()) == expected_keys
    assert all(path is not None and path.exists() for path in results["ORB2"].values())

    comparison_root = pipeline.vo_results_dir / "ORB2" / "comparison"
    assert (comparison_root / "run_0" / "track_survival_comparison.png").exists()
    assert (comparison_root / "run_1" / "track_survival_comparison.png").exists()

    summary_path = pipeline.vo_results_dir / "ORB2" / "aggregated" / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_count"] == 2
    assert summary["baseline"]["num_runs"] == 2
    assert summary["perturbed_modules"]["fog"]["num_runs"] == 2
    assert summary["perturbed_modules"]["fog"]["metrics"]["mean_track_length"]["std"] >= 0.0

    aggregated_plot = pipeline.vo_results_dir / "ORB2" / "aggregated" / "aggregated_track_metrics.png"
    assert aggregated_plot.exists()


def test_vo_comparison_only_uses_existing_run_dirs_for_aggregation(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, sequence="04")
    config.perturbations = []
    config_path = _make_config_file(tmp_path)

    monkeypatch.setattr("slamadverseriallab.pipelines.vo_evaluation.load_config", lambda _p: config)
    monkeypatch.setattr(
        "slamadverseriallab.datasets.create_dataset",
        lambda _cfg: _DatasetStub(Path(config.dataset.path)),
    )

    pipeline = VOEvaluationPipeline(
        config_path=config_path,
        feature_configs=["ORB2"],
        sensor_type="mono",
        comparison_only=True,
    )
    monkeypatch.setattr(pipeline.runner, "cleanup", lambda: None)

    feature_dir = pipeline.vo_results_dir / "ORB2"
    _write_track_outputs(feature_dir / "run_0" / "baseline", sequence="04", ages=[10, 10, 10])
    _write_track_outputs(feature_dir / "run_0" / "fog", sequence="04", ages=[6, 6, 6])
    _write_track_outputs(feature_dir / "run_1" / "baseline", sequence="04", ages=[11, 11, 11])
    _write_track_outputs(feature_dir / "run_1" / "fog", sequence="04", ages=[5, 5, 5])

    results = pipeline.run()

    assert results["ORB2"] == {}

    comparison_root = feature_dir / "comparison"
    assert (comparison_root / "run_0" / "track_survival_comparison.png").exists()
    assert (comparison_root / "run_1" / "track_survival_comparison.png").exists()

    summary_path = feature_dir / "aggregated" / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_count"] == 2
    assert summary["module_order"] == ["fog"]
    assert summary["baseline"]["num_runs"] == 2
    assert summary["perturbed_modules"]["fog"]["num_runs"] == 2


def test_evaluate_vo_cli_accepts_num_runs_flag() -> None:
    parser = create_parser()
    args = parser.parse_args(["evaluate-vo", "config.yaml", "--num-runs", "3"])
    assert args.num_runs == 3
