"""Tests for evaluation sensor-mode inference via dataset camera-role contract."""

from pathlib import Path
from typing import Dict, List, Optional

import pytest

from slamadverseriallab.algorithms.types import SensorMode
from slamadverseriallab.config.parser import Config
from slamadverseriallab.config.schema import DatasetConfig, ExperimentConfig, OutputConfig
from slamadverseriallab.pipelines.evaluation import EvaluationPipeline


class _AlgorithmStub:
    """Minimal SLAM algorithm stub for EvaluationPipeline init tests."""

    def __init__(self, supported_datasets: Dict[str, List[str]]):
        self.name = "stub_algo"
        self.supported_datasets = supported_datasets

    def resolve_config_name(self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None) -> str:
        return "dummy_internal_config"


class _DatasetStub:
    """Minimal dataset stub exposing evaluation-time dataset APIs."""

    def __init__(
        self,
        active_roles: List[str],
        gt_path: Optional[Path] = None,
        camera_dirs: Optional[Dict[str, str]] = None,
        timestamps_by_frame: Optional[Dict[int, float]] = None,
    ):
        self._active_roles = list(active_roles)
        self._gt_path = gt_path
        self._camera_dirs = camera_dirs or {"left": "image_2", "right": "image_3"}
        self._timestamps_by_frame = timestamps_by_frame or {
            0: 0.0,
            1: 0.1,
            2: 0.2,
        }

    def get_active_camera_roles(self) -> List[str]:
        return list(self._active_roles)

    def resolve_camera_directories(self, _source_root: Path) -> Dict[str, str]:
        return {role: self._camera_dirs[role] for role in self._active_roles}

    def get_timestamps_file_path(self):
        return None

    def get_algorithm_timestamps(self) -> Dict[int, float]:
        return dict(self._timestamps_by_frame)

    def get_ground_truth_path(self):
        return self._gt_path

    def create_truncated_copy(self, _max_frames: int) -> Path:
        return Path("/tmp/dataset_truncated_stub")

    def __len__(self) -> int:
        return len(self._timestamps_by_frame)


def _make_config(tmp_path: Path, *, dataset_type: str, load_stereo: bool) -> Config:
    dataset_dir = tmp_path / f"{dataset_type}_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        experiment=ExperimentConfig(name=f"eval_{dataset_type}_sensor_mode"),
        dataset=DatasetConfig(
            type=dataset_type,
            path=str(dataset_dir),
            sequence="dummy_sequence",
            load_stereo=load_stereo,
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


def test_evaluation_uses_dataset_camera_roles_not_load_stereo_flag(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, dataset_type="kitti", load_stereo=False)
    config_path = _make_config_file(tmp_path)

    algorithm = _AlgorithmStub({"kitti": ["mono", "stereo"]})
    dataset = _DatasetStub(active_roles=["left", "right"])

    create_calls = {"count": 0}

    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.load_config", lambda _p: config)
    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.get_slam_algorithm", lambda _name: algorithm)

    def _fake_create_dataset(_cfg):
        create_calls["count"] += 1
        return dataset

    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.create_dataset", _fake_create_dataset)

    pipeline = EvaluationPipeline(config_path=config_path, slam_algorithm="stub")

    assert pipeline.active_camera_roles == ["left", "right"]
    assert pipeline.sensor_mode == SensorMode.STEREO
    assert create_calls["count"] == 1


def test_evaluation_prefers_rgbd_for_tum_when_dataset_is_mono(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, dataset_type="tum", load_stereo=True)
    config_path = _make_config_file(tmp_path)

    algorithm = _AlgorithmStub({"tum": ["mono", "rgbd"]})
    dataset = _DatasetStub(active_roles=["left"])

    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.load_config", lambda _p: config)
    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.get_slam_algorithm", lambda _name: algorithm)
    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.create_dataset", lambda _cfg: dataset)

    pipeline = EvaluationPipeline(config_path=config_path, slam_algorithm="stub")

    assert pipeline.active_camera_roles == ["left"]
    assert pipeline.sensor_mode == SensorMode.RGBD


def test_evaluation_fails_fast_when_dataset_contract_missing_left_camera(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, dataset_type="kitti", load_stereo=False)
    config_path = _make_config_file(tmp_path)

    algorithm = _AlgorithmStub({"kitti": ["mono", "stereo"]})
    dataset = _DatasetStub(active_roles=["right"])

    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.load_config", lambda _p: config)
    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.get_slam_algorithm", lambda _name: algorithm)
    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.create_dataset", lambda _cfg: dataset)

    with pytest.raises(ValueError, match="must expose an active 'left' camera role"):
        EvaluationPipeline(config_path=config_path, slam_algorithm="stub")


def test_evaluation_run_request_includes_dataset_resolved_camera_paths(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, dataset_type="euroc", load_stereo=True)
    config_path = _make_config_file(tmp_path)

    algorithm = _AlgorithmStub({"euroc": ["stereo"]})
    dataset = _DatasetStub(
        active_roles=["left", "right"],
        camera_dirs={"left": "mav0/cam0/data", "right": "mav0/cam1/data"},
    )

    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.load_config", lambda _p: config)
    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.get_slam_algorithm", lambda _name: algorithm)
    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.create_dataset", lambda _cfg: dataset)

    pipeline = EvaluationPipeline(config_path=config_path, slam_algorithm="stub")

    request = pipeline._create_run_request(
        dataset_path=Path(config.dataset.path),
        output_dir=tmp_path / "out",
    )

    assert request.extras["camera_dirs"] == {"left": "mav0/cam0/data", "right": "mav0/cam1/data"}
    assert request.extras["camera_paths"] == {
        "left": str((Path(config.dataset.path) / "mav0/cam0/data").resolve()),
        "right": str((Path(config.dataset.path) / "mav0/cam1/data").resolve()),
    }
    assert request.extras["timestamps_by_frame"] == {
        0: 0.0,
        1: 0.1,
        2: 0.2,
    }


def test_evaluation_run_request_fails_for_non_monotonic_timestamps(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, dataset_type="kitti", load_stereo=False)
    config_path = _make_config_file(tmp_path)

    algorithm = _AlgorithmStub({"kitti": ["mono"]})
    dataset = _DatasetStub(
        active_roles=["left"],
        timestamps_by_frame={0: 0.0, 1: 0.1, 2: 0.1},
    )

    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.load_config", lambda _p: config)
    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.get_slam_algorithm", lambda _name: algorithm)
    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.create_dataset", lambda _cfg: dataset)

    pipeline = EvaluationPipeline(config_path=config_path, slam_algorithm="stub")

    with pytest.raises(ValueError, match="strictly increasing"):
        pipeline._create_run_request(
            dataset_path=Path(config.dataset.path),
            output_dir=tmp_path / "out",
        )


def test_evaluation_requires_dataset_sequence_for_run_request(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path, dataset_type="kitti", load_stereo=False)
    config.dataset.sequence = None
    config_path = _make_config_file(tmp_path)

    algorithm = _AlgorithmStub({"kitti": ["mono", "stereo"]})
    dataset = _DatasetStub(active_roles=["left"])

    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.load_config", lambda _p: config)
    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.get_slam_algorithm", lambda _name: algorithm)
    monkeypatch.setattr("slamadverseriallab.pipelines.evaluation.create_dataset", lambda _cfg: dataset)

    pipeline = EvaluationPipeline(
        config_path=config_path,
        slam_algorithm="stub",
        slam_config_path="external-config.yaml",
    )

    with pytest.raises(ValueError, match="dataset.sequence is required"):
        pipeline._create_run_request(dataset_path=Path(config.dataset.path), output_dir=tmp_path / "out")
