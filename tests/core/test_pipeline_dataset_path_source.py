"""Tests for pipeline dataset path source-of-truth behavior."""

from pathlib import Path
from typing import List, Tuple

from slamadverseriallab.config.parser import Config
from slamadverseriallab.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    OutputConfig,
    PerturbationConfig,
)
from slamadverseriallab.core.pipeline import Pipeline


class _SetupDatasetStub:
    """Dataset stub used by Pipeline.setup tests."""

    def __init__(self, dataset_path: Path):
        self.path = dataset_path

    def get_metadata_files_with_dest(self) -> List[Tuple[Path, str, bool]]:
        return []


class _LoadModulesDatasetStub:
    """Dataset stub exposing path/length for ModuleSetupContext tests."""

    def __init__(self, dataset_path: Path, num_frames: int = 3):
        self.path = dataset_path
        self._num_frames = num_frames

    def __len__(self) -> int:
        return self._num_frames


class _CaptureModule:
    """Module stub that captures setup context."""

    def __init__(self, name: str = "capture_module"):
        self.name = name
        self.enabled = True
        self.setup_context = None

    def setup(self, context) -> None:
        self.setup_context = context

    def __repr__(self) -> str:
        return "CaptureModule(name='capture_module')"


def test_setup_does_not_mutate_config_dataset_path(tmp_path, monkeypatch) -> None:
    config = Config(
        experiment=ExperimentConfig(name="pipeline_setup_path_immutability"),
        dataset=DatasetConfig(type="kitti", path=None, sequence="04"),
        perturbations=[],
        output=OutputConfig(
            base_dir=str(tmp_path / "results"),
            save_images=False,
            create_timestamp_dir=False,
        ),
    )
    pipeline = Pipeline(config)
    dataset_stub = _SetupDatasetStub(dataset_path=tmp_path / "resolved_from_dataset")

    monkeypatch.setattr(pipeline, "load_dataset", lambda: dataset_stub)
    monkeypatch.setattr(pipeline, "load_modules", lambda: [])

    pipeline.setup()

    assert config.dataset.path is None
    assert pipeline.dataset is dataset_stub


def test_load_modules_uses_dataset_object_path_not_config_path(tmp_path, monkeypatch) -> None:
    stale_config_path = tmp_path / "stale_config_path"
    dataset_path = tmp_path / "dataset_source_of_truth"
    dataset_stub = _LoadModulesDatasetStub(dataset_path=dataset_path, num_frames=5)
    capture_module = _CaptureModule()

    config = Config(
        experiment=ExperimentConfig(name="pipeline_module_context_dataset_path"),
        dataset=DatasetConfig(type="kitti", path=str(stale_config_path)),
        perturbations=[PerturbationConfig(name="capture", type="none")],
        output=OutputConfig(
            base_dir=str(tmp_path / "results"),
            save_images=False,
            create_timestamp_dir=False,
        ),
    )

    pipeline = Pipeline(config)
    pipeline.dataset = dataset_stub

    monkeypatch.setattr(
        "slamadverseriallab.core.pipeline.create_module",
        lambda _pert_config: capture_module,
    )

    modules = pipeline.load_modules()

    assert modules == [capture_module]
    assert capture_module.setup_context is not None
    assert capture_module.setup_context.dataset is dataset_stub
    assert capture_module.setup_context.total_frames == 5
    assert capture_module.setup_context.dataset_path == dataset_path
    assert capture_module.setup_context.dataset_path != stale_config_path
