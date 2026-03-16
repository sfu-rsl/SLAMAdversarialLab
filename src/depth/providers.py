"""Depth backend providers for automatic depth source selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..utils import get_logger

logger = get_logger(__name__)


class DepthProviderNotApplicable(RuntimeError):
    """Raised when a depth provider cannot be used in current context."""


@dataclass
class ProviderResult:
    """Result returned by depth providers."""

    backend: str
    depth_source: str
    depth_dirs: Dict[str, Path]


class DepthProvider:
    """Base class for depth providers."""

    backend_name = "unknown"

    def setup(
        self,
        module: Any,
        source_path: Path,
        dataset: Any,
        cameras: List[str],
    ) -> ProviderResult:
        raise NotImplementedError


class ExistingDepthProvider(DepthProvider):
    """Reuse depth maps declared by dataset implementations."""

    backend_name = "existing"

    def setup(
        self,
        module: Any,
        source_path: Path,
        dataset: Any,
        cameras: List[str],
    ) -> ProviderResult:
        if dataset is None:
            raise DepthProviderNotApplicable("Dataset object is required for existing-depth lookup")

        depth_dirs: Dict[str, Path] = {}

        for camera in cameras:
            try:
                depth_dir = dataset.get_available_depth_dir(camera)
            except Exception as e:
                raise DepthProviderNotApplicable(
                    f"Dataset failed depth-directory lookup for camera '{camera}': {e}"
                ) from e
            if depth_dir is None:
                raise DepthProviderNotApplicable(
                    f"No reusable depth directory declared for camera '{camera}'"
                )

            try:
                camera_image_dir = module.get_camera_directory_path(source_path, camera)
            except Exception as e:
                raise DepthProviderNotApplicable(
                    f"Could not resolve image directory for camera '{camera}': {e}"
                ) from e

            depth_dir = Path(depth_dir)
            if not self._is_valid_depth_dir(camera_image_dir, camera, depth_dir):
                raise DepthProviderNotApplicable(
                    f"Reusable depth directory is invalid for camera '{camera}': {depth_dir}"
                )

            depth_dirs[camera] = depth_dir

        # Sensor depth needs dataset-native decoding rather than /256 decoding.
        depth_source = "existing"
        native_dir = dataset.get_depth_directory_path()
        if native_dir is not None:
            native_dir = Path(native_dir).resolve()
            if all(p.resolve() == native_dir for p in depth_dirs.values()):
                depth_source = "sensor"

        logger.info(
            "Reusing existing depth maps for all cameras: "
            + ", ".join(f"{cam}={path}" for cam, path in depth_dirs.items())
        )

        return ProviderResult(
            backend=self.backend_name,
            depth_source=depth_source,
            depth_dirs=depth_dirs,
        )

    @staticmethod
    def _collect_filenames(directory: Path) -> set[str]:
        names = {p.name for p in directory.glob("*.png")}
        if names:
            return names
        return {p.name for p in directory.glob("*.jpg")}

    def _is_valid_depth_dir(self, image_dir: Path, camera: str, depth_dir: Path) -> bool:
        if not depth_dir.exists() or not depth_dir.is_dir():
            return False

        depth_files = self._collect_filenames(depth_dir)
        if not depth_files:
            return False

        if not image_dir.exists():
            logger.warning(
                "Image directory %s for camera %s does not exist",
                image_dir,
                camera,
            )
            return False

        image_files = self._collect_filenames(image_dir)
        if not image_files:
            logger.warning(
                "Image directory %s for camera %s has no PNG/JPG files",
                image_dir,
                camera,
            )
            return False

        missing = image_files - depth_files
        if missing:
            logger.warning(
                "Depth directory %s is missing %d frame(s) for camera %s",
                depth_dir,
                len(missing),
                camera,
            )
            return False

        return True


class FoundationStereoProvider(DepthProvider):
    """Generate depth with FoundationStereo for stereo datasets."""

    backend_name = "foundation_stereo"

    def setup(
        self,
        module: Any,
        source_path: Path,
        dataset: Any,
        cameras: List[str],
    ) -> ProviderResult:
        if dataset is None:
            raise DepthProviderNotApplicable("FoundationStereo requires a stereo dataset")

        camera_roles = dataset.get_active_camera_roles()
        if "right" not in camera_roles:
            raise DepthProviderNotApplicable("FoundationStereo requires a stereo dataset")

        if not hasattr(module, "_setup_foundation_stereo_depth"):
            raise DepthProviderNotApplicable("Module does not implement FoundationStereo setup")

        module._setup_foundation_stereo_depth(source_path, dataset, cameras)
        depth_dirs = dict(module.depth_dirs or {})
        if not depth_dirs:
            raise RuntimeError("FoundationStereo setup did not produce depth directories")

        return ProviderResult(
            backend=self.backend_name,
            depth_source="foundation_stereo",
            depth_dirs=depth_dirs,
        )


class DA3Provider(DepthProvider):
    """Generate depth using Depth Anything 3."""

    backend_name = "da3"

    def setup(
        self,
        module: Any,
        source_path: Path,
        dataset: Any,
        cameras: List[str],
    ) -> ProviderResult:
        if not hasattr(module, "_setup_da3_depth"):
            raise DepthProviderNotApplicable("Module does not implement DA3 setup")

        module._setup_da3_depth(source_path, cameras)
        depth_dirs = dict(module.depth_dirs or {})
        if not depth_dirs:
            raise RuntimeError("DA3 setup did not produce depth directories")

        return ProviderResult(
            backend=self.backend_name,
            depth_source="da3",
            depth_dirs=depth_dirs,
        )


class DA2Provider(DepthProvider):
    """Generate depth using Depth Anything V2 (DA2)."""

    backend_name = "da2"

    def setup(
        self,
        module: Any,
        source_path: Path,
        dataset: Any,
        cameras: List[str],
    ) -> ProviderResult:
        if not hasattr(module, "_setup_dav2_depth"):
            raise DepthProviderNotApplicable("Module does not implement DA2 setup")

        module._setup_dav2_depth(source_path)
        depth_dirs = dict(module.depth_dirs or {})
        if not depth_dirs:
            raise RuntimeError("DA2 setup did not produce depth directories")

        return ProviderResult(
            backend=self.backend_name,
            depth_source="dav2",
            depth_dirs=depth_dirs,
        )
