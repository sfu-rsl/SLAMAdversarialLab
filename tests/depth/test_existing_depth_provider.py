"""Tests for strict ExistingDepthProvider dataset/module contracts."""

from pathlib import Path

import pytest

from slamadverseriallab.depth.providers import ExistingDepthProvider, DepthProviderNotApplicable


class _DatasetStub:
    def __init__(self, depth_dirs: dict[str, Path], native_depth_dir: Path | None = None) -> None:
        self._depth_dirs = depth_dirs
        self._native_depth_dir = native_depth_dir

    def get_available_depth_dir(self, camera: str = "left") -> Path | None:
        return self._depth_dirs.get(camera)

    def get_depth_directory_path(self) -> Path | None:
        return self._native_depth_dir


class _RaisingDatasetStub(_DatasetStub):
    def get_available_depth_dir(self, camera: str = "left") -> Path | None:
        raise RuntimeError("boom")


class _ModuleStub:
    def __init__(self, camera_dirs: dict[str, Path]) -> None:
        self._camera_dirs = camera_dirs

    def get_camera_directory_path(self, source_path: Path, camera_role: str) -> Path:
        return self._camera_dirs[camera_role]


class _NoCameraResolverModule:
    pass


def _touch_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-png")


def test_existing_provider_requires_module_camera_directory_resolver(tmp_path: Path) -> None:
    depth_dir = tmp_path / "left_depth"
    _touch_png(depth_dir / "000000.png")

    dataset = _DatasetStub(depth_dirs={"left": depth_dir})
    provider = ExistingDepthProvider()

    with pytest.raises(DepthProviderNotApplicable, match="Could not resolve image directory"):
        provider.setup(
            module=_NoCameraResolverModule(),
            source_path=tmp_path,
            dataset=dataset,
            cameras=["left"],
        )


def test_existing_provider_rejects_missing_image_directory(tmp_path: Path) -> None:
    depth_dir = tmp_path / "left_depth"
    _touch_png(depth_dir / "000000.png")

    dataset = _DatasetStub(depth_dirs={"left": depth_dir})
    module = _ModuleStub(camera_dirs={"left": tmp_path / "does_not_exist"})
    provider = ExistingDepthProvider()

    with pytest.raises(DepthProviderNotApplicable, match="Reusable depth directory is invalid"):
        provider.setup(
            module=module,
            source_path=tmp_path,
            dataset=dataset,
            cameras=["left"],
        )


def test_existing_provider_marks_sensor_when_native_dir_matches(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "depth"

    _touch_png(rgb_dir / "000000.png")
    _touch_png(depth_dir / "000000.png")

    dataset = _DatasetStub(depth_dirs={"left": depth_dir}, native_depth_dir=depth_dir)
    module = _ModuleStub(camera_dirs={"left": rgb_dir})
    provider = ExistingDepthProvider()

    result = provider.setup(
        module=module,
        source_path=tmp_path,
        dataset=dataset,
        cameras=["left"],
    )

    assert result.backend == "existing"
    assert result.depth_source == "sensor"
    assert result.depth_dirs == {"left": depth_dir}


def test_existing_provider_rejects_depth_dirs_missing_frame_alignment(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "left_depth"

    _touch_png(rgb_dir / "000000.png")
    _touch_png(rgb_dir / "000001.png")
    _touch_png(depth_dir / "000000.png")

    dataset = _DatasetStub(depth_dirs={"left": depth_dir})
    module = _ModuleStub(camera_dirs={"left": rgb_dir})
    provider = ExistingDepthProvider()

    with pytest.raises(DepthProviderNotApplicable, match="Reusable depth directory is invalid"):
        provider.setup(
            module=module,
            source_path=tmp_path,
            dataset=dataset,
            cameras=["left"],
        )


def test_existing_provider_wraps_dataset_lookup_errors(tmp_path: Path) -> None:
    provider = ExistingDepthProvider()

    with pytest.raises(DepthProviderNotApplicable, match="Dataset failed depth-directory lookup"):
        provider.setup(
            module=_ModuleStub(camera_dirs={"left": tmp_path / "rgb"}),
            source_path=tmp_path,
            dataset=_RaisingDatasetStub(depth_dirs={}),
            cameras=["left"],
        )
