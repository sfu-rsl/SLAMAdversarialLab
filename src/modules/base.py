"""Base class for perturbation modules."""

import sys
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field, replace
import numpy as np
import cv2
from enum import Enum

from ..utils import get_logger
from ..utils.paths import create_temp_dir
from ..config.schema import PerturbationConfig
from ..depth import resolve_depth_backend
from ..depth.da3_metric_action import DA3MetricAction
from ..depth.foundation_stereo_action import FoundationStereoAction
from ..robustness.param_spec import BoundaryParamSpec

logger = get_logger(__name__)

# Depth Anything V2 availability check (done once at module load)
_DEPTH_ANYTHING_AVAILABLE = False
_DEPTH_ANYTHING_IMPORT_ERROR = None

try:
    import torch

    # Clean up sys.modules to force reimport from correct location
    modules_to_remove = [k for k in sys.modules.keys() if k.startswith('depth_anything_v2')]
    for module_name in modules_to_remove:
        del sys.modules[module_name]

    metric_depth_path = str(Path(__file__).parent.parent.parent / "deps" / "depth-estimation" / "Depth-Anything-V2" / "metric_depth")
    if metric_depth_path in sys.path:
        sys.path.remove(metric_depth_path)
    sys.path.insert(0, metric_depth_path)

    from depth_anything_v2.dpt import DepthAnythingV2
    _DEPTH_ANYTHING_AVAILABLE = True
except ImportError as e:
    _DEPTH_ANYTHING_IMPORT_ERROR = str(e)
    logger.debug(f"Depth Anything V2 not available: {e}")


# Model configurations for Depth Anything V2
DEPTH_MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

DEPTH_MODEL_VARIANTS = {
    'vits': 'Small',
    'vitb': 'Base',
    'vitl': 'Large',
}


def check_depth_anything_available() -> None:
    """Check if Depth Anything V2 is available, raise ImportError if not."""
    if not _DEPTH_ANYTHING_AVAILABLE:
        raise ImportError(
            f"Depth Anything V2 is not available ({_DEPTH_ANYTHING_IMPORT_ERROR}). "
            "Please install required dependencies:\n"
            "  pip install torch opencv-python\n"
            "And ensure deps/depth-estimation/Depth-Anything-V2 exists with the model."
        )

# Module registry - populated by __init_subclass__
_MODULE_REGISTRY: Dict[str, 'ModuleRegistration'] = {}


class ModuleRegistration:
    """Registration info for a module."""
    def __init__(
        self,
        name: str,
        module_class: type,
        description: str = "",
        deprecated: bool = False,
        deprecation_message: str = "",
        replacement: str = ""
    ):
        self.name = name
        self.module_class = module_class
        self.description = description
        self.deprecated = deprecated
        self.deprecation_message = deprecation_message
        self.replacement = replacement


def get_module_registry() -> Dict[str, 'ModuleRegistration']:
    """Get the module registry."""
    return _MODULE_REGISTRY


class CompositionMode(Enum):
    """Mode for composing multiple perturbation modules."""
    SEQUENTIAL = "sequential"  # Apply modules one after another


@dataclass(frozen=True)
class ModuleSetupContext:
    """Explicit module setup context provided by the pipeline."""
    dataset: Optional[Any] = None
    dataset_path: Optional[Path] = None
    total_frames: Optional[int] = None
    input_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerturbationModule(ABC):
    """
    Abstract base class for all perturbation modules.

    Perturbation modules apply weather effects or other transformations
    to images during the SLAM evaluation pipeline.

    Class Attributes:
        module_name: Name to register this module under (e.g., "fog", "rain").
                    If not set, the module will not be registered.
        module_description: Short description of the module.
        deprecated: If True, module is deprecated and will raise error when used.
        deprecation_message: Message explaining why module is deprecated.
        replacement: Name of the replacement module.
        requires_full_sequence: If True, module requires all frames on disk before processing.
                         If False (default), module processes frames one at a time.
        PARAMETERS_CLASS: Dataclass defining the module's parameters. Must be set by
                         concrete module implementations. Used for auto-documentation.
        SEARCHABLE_PARAMS: Optional robustness-boundary declarations for searchable
                        parameters. Empty by default.
    """

    # Registration attributes (set by concrete modules)
    module_name: Optional[str] = None
    module_description: str = ""
    deprecated: bool = False
    deprecation_message: str = ""
    replacement: str = ""

    # Processing attributes
    requires_full_sequence = False  # Default: frame-by-frame processing
    PARAMETERS_CLASS = None   # Must be set by concrete modules
    SEARCHABLE_PARAMS: Dict[str, BoundaryParamSpec] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Auto-register modules and validate configuration.

        Modules are registered if they define `module_name`.
        """
        super().__init_subclass__(**kwargs)

        # Skip for internal classes
        if cls.__name__ in ('NullModule', 'CompositeModule'):
            return

        # Skip for base classes (names ending with 'Base' or 'ModuleBase')
        if cls.__name__.endswith('Base'):
            return

        # Skip check for classes with ABC in their bases (abstract classes)
        if ABC in cls.__bases__:
            return

        # Auto-register if module_name is defined
        if cls.module_name is not None:
            description = cls.module_description
            if not description and cls.__doc__:
                description = cls.__doc__.strip().split('\n')[0]

            # Register the module
            registration = ModuleRegistration(
                name=cls.module_name,
                module_class=cls,
                description=description,
                deprecated=cls.deprecated,
                deprecation_message=cls.deprecation_message,
                replacement=cls.replacement
            )

            if cls.module_name in _MODULE_REGISTRY:
                logger.debug(f"Module '{cls.module_name}' already registered, overwriting with {cls.__name__}")

            _MODULE_REGISTRY[cls.module_name] = registration
            logger.debug(f"Auto-registered module '{cls.module_name}' -> {cls.__name__}")

        # Warn if concrete module doesn't have PARAMETERS_CLASS
        if cls.module_name is not None and cls.PARAMETERS_CLASS is None:
            logger.warning(
                f"Module '{cls.__name__}' does not define PARAMETERS_CLASS. "
                f"Add a dataclass and set PARAMETERS_CLASS for auto-documentation."
            )

    def __init__(self, config: PerturbationConfig):
        """
        Initialize the perturbation module.

        Args:
            config: Perturbation configuration containing parameters
        """
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.parameters = config.parameters
        self._initialized = False
        self._dataset = None
        self._setup_context: Optional[ModuleSetupContext] = None

        logger.debug(f"Creating {self.__class__.__name__} module: {self.name}")

    @property
    def dataset(self):
        """Get the dataset object.

        Returns:
            Dataset object or None if not set.
        """
        return self._dataset

    @property
    def setup_context(self) -> Optional[ModuleSetupContext]:
        """Return the latest setup context."""
        return self._setup_context

    def _normalize_setup_context(self, context: ModuleSetupContext) -> ModuleSetupContext:
        """Normalize setup context values and path types."""
        if not isinstance(context, ModuleSetupContext):
            raise TypeError(
                f"setup() expects ModuleSetupContext, got {type(context).__name__}"
            )

        dataset_path = Path(context.dataset_path) if context.dataset_path is not None else None
        input_path = Path(context.input_path) if context.input_path is not None else None
        metadata = dict(context.metadata) if context.metadata else {}
        return ModuleSetupContext(
            dataset=context.dataset,
            dataset_path=dataset_path,
            total_frames=context.total_frames,
            input_path=input_path,
            metadata=metadata,
        )

    def _apply_setup_context(self, context: ModuleSetupContext) -> None:
        """Apply context to module state for shared helpers."""
        self._setup_context = context
        self._dataset = context.dataset

    def setup(self, context: ModuleSetupContext) -> None:
        """
        Perform one-time setup operations.

        This method is called once before processing begins with the full
        setup context required by the module.
        """
        if self._initialized:
            logger.warning(f"Module {self.name} already initialized")
            return

        normalized_context = self._normalize_setup_context(context)
        self._apply_setup_context(normalized_context)

        logger.info(f"Setting up module: {self.name}")
        self._setup(normalized_context)
        self._initialized = True

    def update_context(
        self,
        context: ModuleSetupContext,
        reason: str = "context_update"
    ) -> None:
        """Update module runtime context after setup (e.g., composite input override)."""
        normalized_context = self._normalize_setup_context(context)
        previous_context = self._setup_context
        self._apply_setup_context(normalized_context)

        if not self._initialized:
            self.setup(normalized_context)
            return

        logger.info(f"Updating context for module '{self.name}' ({reason})")
        self._on_context_updated(previous_context, normalized_context, reason)

    def _on_context_updated(
        self,
        previous_context: Optional[ModuleSetupContext],
        context: ModuleSetupContext,
        reason: str
    ) -> None:
        """Hook for modules that need to react when setup context changes."""
        return

    @abstractmethod
    def _setup(self, context: ModuleSetupContext) -> None:
        """
        Implementation-specific setup logic.

        Subclasses must implement this method to perform their
        specific initialization steps.
        """
        pass

    def apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        """
        Apply the perturbation to an image.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format
                  with values in [0, 255] as uint8
            depth: Optional depth map as numpy array (H, W) in meters
            frame_idx: Index of the current frame in the sequence
            camera: Camera role ("left" or "right")
            **kwargs: Additional context (e.g., rgb_filename for filename-based depth lookup)

        Returns:
            Perturbed image as numpy array with same shape and dtype as input

        Raises:
            ValueError: If image format is invalid
            RuntimeError: If module not initialized
        """
        if not self._initialized:
            raise RuntimeError(
                f"Module {self.name} not initialized. Call setup(context) first."
            )

        if not self.enabled:
            logger.debug(f"Module {self.name} is disabled, returning original image")
            return image

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image with shape (H, W, 3), got {image.shape}"
            )

        if image.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 image, got {image.dtype}"
            )

        if depth is not None:
            if depth.ndim != 2:
                raise ValueError(
                    f"Expected depth map with shape (H, W), got {depth.shape}"
                )
            if depth.shape[:2] != image.shape[:2]:
                raise ValueError(
                    f"Depth shape {depth.shape} doesn't match image shape {image.shape[:2]}"
                )

        from ..utils.profiling import get_profiler
        profiler = get_profiler()

        if profiler and profiler.enabled:
            with profiler.timer(f"{self.name}_apply",
                              metadata={"frame": frame_idx, "has_depth": depth is not None, "camera": camera}):
                return self._apply(image, depth, frame_idx, camera, **kwargs)
        else:
            return self._apply(image, depth, frame_idx, camera, **kwargs)

    @abstractmethod
    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str,
        **kwargs
    ) -> np.ndarray:
        """
        Implementation-specific perturbation logic.

        Subclasses must implement this method to apply their
        specific perturbation effect.

        Args:
            image: Validated input image
            depth: Optional validated depth map
            frame_idx: Frame index
            camera: Camera role ("left" or "right")
            **kwargs: Additional context (e.g., rgb_filename)

        Returns:
            Perturbed image
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up resources and perform shutdown operations.

        This method is called once after all processing is complete.
        Override this to free resources, close files, etc.
        """
        if not self._initialized:
            return

        logger.info(f"Cleaning up module: {self.name}")
        self._cleanup()
        self._initialized = False

    def _cleanup(self) -> None:
        """
        Implementation-specific cleanup logic.

        Subclasses can override this method to perform their
        specific cleanup steps. Default implementation does nothing.
        """
        pass

    # ========================================================================
    # Depth Estimation Support (for modules that need depth maps)
    # ========================================================================

    # Default depth configuration (can be overridden by subclass in _setup)
    depth_encoder: str = 'vitl'
    depth_model_type: str = 'vkitti'
    max_depth_range: float = 80.0
    depth_backend: str = "auto"
    da3_model_id: str = "depth-anything/DA3METRIC-LARGE"
    da3_fallback_hfov_deg: float = 70.0
    da3_save_npz: bool = True
    da3_device: str = "auto"
    da3_process_res: int = 504

    # Depth state attributes (initialized by _init_depth_state)
    dataset_path: Optional[Path] = None
    _input_path: Optional[Path] = None
    _depth_setup_complete: bool = False
    depth_dirs: Dict[str, Path] = None
    cameras: list = None
    camera_dirs: Dict[str, str] = None
    _depth_source: str = 'none'  # 'sensor', 'existing', 'foundation_stereo', 'da3', or 'dav2'
    _use_sensor_depth: bool = False  # True if using native sensor depth
    _depth_backend_requested: str = "auto"
    _depth_backend_selected: str = 'none'
    _depth_backend_candidates: List[Dict[str, str]] = []
    _foundation_stereo_action: Optional[FoundationStereoAction] = None
    _da3_metric_action: Optional[DA3MetricAction] = None

    def _get_source_path(self) -> Optional[Path]:
        """Get the path to read input images from (input_path override or dataset_path)."""
        if self._setup_context is not None:
            return self._setup_context.input_path or self._setup_context.dataset_path
        return self._input_path if self._input_path else self.dataset_path

    def _init_depth_state(self) -> None:
        """Initialize depth estimation state attributes.

        Call this in _setup() before setting depth configuration.
        """
        self.dataset_path = None
        self._input_path = None
        self._depth_setup_complete = False
        self.depth_dirs = {}
        self.cameras = ['left']  # Default, will be updated in _setup_depth_estimation
        self.camera_dirs = {}
        self._depth_source = 'none'  # Will be set in _setup_depth_estimation
        self._use_sensor_depth = False
        self._depth_backend_requested = "auto"
        self._depth_backend_selected = 'none'
        self._depth_backend_candidates = []
        self._foundation_stereo_action = None
        self._da3_metric_action = None

    def _configure_depth_runtime(
        self,
        *,
        encoder: Optional[str] = None,
        max_depth_range: Optional[float] = None,
        model_type: Optional[str] = None,
        depth_backend: Optional[str] = None,
        da3_model_id: Optional[str] = None,
        da3_fallback_hfov_deg: Optional[float] = None,
        da3_save_npz: Optional[bool] = None,
        da3_device: Optional[str] = None,
        da3_process_res: Optional[int] = None,
    ) -> None:
        """Configure depth-runtime model settings for this module."""
        if encoder is not None:
            self.depth_encoder = encoder
        if max_depth_range is not None:
            self.max_depth_range = max_depth_range
        if model_type is not None:
            self.depth_model_type = model_type
        if depth_backend is not None:
            requested_backend = str(depth_backend).strip().lower()
            supported_backends = {"auto", "existing", "foundation_stereo", "da3", "da2"}
            if requested_backend not in supported_backends:
                supported = ", ".join(sorted(supported_backends))
                raise ValueError(
                    f"Unsupported depth_backend '{depth_backend}'. Supported values: {supported}"
                )
            self.depth_backend = requested_backend
        if da3_model_id is not None:
            self.da3_model_id = da3_model_id
        if da3_fallback_hfov_deg is not None:
            self.da3_fallback_hfov_deg = float(da3_fallback_hfov_deg)
        if da3_save_npz is not None:
            self.da3_save_npz = bool(da3_save_npz)
        if da3_device is not None:
            self.da3_device = str(da3_device)
        if da3_process_res is not None:
            self.da3_process_res = int(da3_process_res)

    def _refresh_depth_runtime(self, context: ModuleSetupContext) -> None:
        """Reset depth runtime state and re-run backend resolution."""
        self._depth_setup_complete = False
        self.depth_dirs = {}
        self.camera_dirs = {}
        self._setup_depth_estimation(context)

    def _require_rgb_filename(
        self,
        kwargs: Dict[str, Any],
        *,
        module_label: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        """Return rgb filename from kwargs or raise a clear runtime error."""
        rgb_filename = kwargs.get("rgb_filename")
        if rgb_filename:
            return rgb_filename

        module_ref = module_label or self.__class__.__name__
        if reason:
            raise ValueError(
                f"rgb_filename is required for {module_ref} module to {reason}. "
                "Ensure the dataset provides rgb_filename in frame_data."
            )

        raise ValueError(
            f"rgb_filename is required for {module_ref} module. "
            "Ensure the dataset provides rgb_filename in frame_data."
        )

    def _load_depth_for_apply(
        self,
        camera: str,
        kwargs: Dict[str, Any],
        provided_depth: Optional[np.ndarray] = None,
        *,
        prefer_dataset: bool = False,
        module_label: Optional[str] = None,
        rgb_filename_reason: Optional[str] = None,
        require_rgb_filename_when_depth_provided: bool = True,
        missing_dataset_error: Optional[str] = None,
        missing_depth_error: Optional[Callable[[str, str], str]] = None,
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Resolve depth for module apply() from provided/depth-cache/dataset sources."""
        rgb_filename: Optional[str] = None
        if provided_depth is None or require_rgb_filename_when_depth_provided:
            rgb_filename = self._require_rgb_filename(
                kwargs,
                module_label=module_label,
                reason=rgb_filename_reason,
            )

        if provided_depth is not None:
            return provided_depth, rgb_filename

        if rgb_filename is None:
            rgb_filename = self._require_rgb_filename(
                kwargs,
                module_label=module_label,
                reason=rgb_filename_reason,
            )

        if prefer_dataset:
            if self.dataset is None:
                raise RuntimeError(
                    missing_dataset_error
                    or (
                        f"{self.__class__.__name__} '{self.name}' requires dataset to be set. "
                        "Ensure setup(context) includes the dataset instance."
                    )
                )

            depth_metric = self.dataset.load_depth_for_frame(
                rgb_filename=rgb_filename,
                camera=camera,
                use_estimated=True,
            )
            if depth_metric is None:
                if missing_depth_error is not None:
                    raise ValueError(missing_depth_error(rgb_filename, camera))
                raise ValueError(
                    f"No depth available for frame {rgb_filename} (camera={camera}). "
                    "Ensure depth maps are available for this sequence."
                )

            return depth_metric, rgb_filename

        return self._load_depth_from_disk(camera, rgb_filename), rgb_filename

    def _setup_depth_estimation(self, context: Optional[ModuleSetupContext] = None) -> None:
        """Setup depth estimation with fixed backend fallback order.

        Auto backend order:
        1. Existing depth declared by dataset
        2. FoundationStereo (stereo datasets with calibration)
        3. Depth Anything 3

        Explicit backend requests via depth_backend (e.g. "da2") are handled
        by the depth resolver and bypass the auto chain.

        This method should be called from module setup or context-update hooks.
        """
        if context is not None:
            normalized_context = self._normalize_setup_context(context)
            self._apply_setup_context(normalized_context)

        if self._depth_setup_complete:
            return

        source_path = self._get_source_path()
        if source_path is None:
            raise RuntimeError(
                f"Depth setup for module '{self.name}' requires setup context with "
                "either dataset_path or input_path."
            )

        if not source_path.exists():
            raise ValueError(f"Depth source path does not exist: {source_path}")

        if self.depth_dirs is None:
            self.depth_dirs = {}

        module_name = getattr(self, 'name', self.__class__.__name__)
        logger.info(f"Setting up depth estimation for '{module_name}'...")
        logger.info(f"Source path: {source_path}")

        self.cameras = self._detect_cameras(source_path)

        requested_backend = str(getattr(self, "depth_backend", "auto") or "auto").strip().lower()
        self._depth_backend_requested = requested_backend

        resolution = resolve_depth_backend(
            module=self,
            source_path=source_path,
            dataset=self.dataset,
            cameras=self.cameras,
            preferred_backend=requested_backend,
        )

        self.depth_dirs = resolution.depth_dirs
        self._depth_source = resolution.depth_source
        self._use_sensor_depth = self._depth_source == 'sensor'
        self._depth_backend_selected = resolution.backend
        self._depth_backend_candidates = resolution.candidates

        self._depth_setup_complete = True
        logger.info(
            f"Depth estimation setup complete for '{module_name}' "
            f"(source: {self._depth_source}, backend: {self._depth_backend_selected})"
        )

    def get_camera_directory_name(self, camera_role: str) -> str:
        """Return resolved camera directory name for a camera role."""
        if camera_role not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera role '{camera_role}'. Expected 'left' or 'right'."
            )

        if self.camera_dirs and camera_role in self.camera_dirs:
            return self.camera_dirs[camera_role]

        if self._dataset is None:
            raise RuntimeError(
                "Camera directory lookup requires dataset object in setup context."
            )

        source_path = self._get_source_path()
        if source_path is None:
            raise RuntimeError(
                "Camera directory lookup requires setup context with either dataset_path or input_path."
            )

        return self._dataset.resolve_camera_directory_name(source_path, camera_role)

    def get_camera_directory_path(self, source_path: Path, camera_role: str) -> Path:
        """Return resolved camera directory path for a camera role."""
        if camera_role not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera role '{camera_role}'. Expected 'left' or 'right'."
            )

        if self.camera_dirs and camera_role in self.camera_dirs:
            return source_path / self.camera_dirs[camera_role]

        if self._dataset is None:
            raise RuntimeError(
                "Camera directory lookup requires dataset object in setup context."
            )

        directory_name = self._dataset.resolve_camera_directory_name(source_path, camera_role)
        return source_path / directory_name

    def _detect_cameras(self, source_path: Path) -> List[str]:
        """Detect camera roles and resolve role->directory mapping."""
        if self._dataset is None:
            raise RuntimeError("Camera detection requires dataset object in setup context.")

        camera_roles = self._dataset.get_active_camera_roles()
        camera_dirs = self._dataset.resolve_camera_directories(source_path)

        if "left" not in camera_roles or "left" not in camera_dirs:
            raise RuntimeError(
                "Dataset must expose an active 'left' camera role during camera detection."
            )

        left_camera_dir = camera_dirs["left"]
        if "right" in camera_roles:
            if "right" not in camera_dirs:
                raise RuntimeError(
                    "Dataset reported right camera role but did not resolve a right camera directory."
                )

            right_camera_dir = camera_dirs["right"]
            logger.info(
                "Stereo mode detected - processing left/right cameras "
                f"(left_dir={left_camera_dir}, right_dir={right_camera_dir})"
            )
        else:
            logger.info(f"Mono mode - processing left camera (dir={left_camera_dir})")

        self.camera_dirs = {role: camera_dirs[role] for role in camera_roles if role in camera_dirs}
        return camera_roles

    def _setup_foundation_stereo_depth(self, source_path: Path, dataset, cameras: List[str]) -> None:
        """Setup depth using FoundationStereo for stereo datasets.

        This backend uses the local FoundationStereo dependency under
        deps/depth-estimation/FoundationStereo.
        """
        if dataset is None:
            raise RuntimeError("FoundationStereo requires a dataset object")

        camera_roles = dataset.get_active_camera_roles()
        if "right" not in camera_roles:
            raise RuntimeError("FoundationStereo requires stereo dataset mode")

        calib = dataset.get_camera_intrinsics("left")
        if calib is None:
            raise RuntimeError("Stereo calibration is required for FoundationStereo")

        fx = float(calib.fx)
        baseline = float(calib.baseline or 0.0)
        if fx <= 0 or baseline <= 0:
            raise RuntimeError(
                f"Invalid stereo calibration for FoundationStereo: fx={fx}, baseline={baseline}"
            )

        if "left" not in cameras or "right" not in cameras:
            raise RuntimeError(
                f"FoundationStereo requires left/right cameras but got: {cameras}"
            )

        left_role = "left"
        right_role = "right"
        left_dir = self.get_camera_directory_path(source_path, left_role)
        right_dir = self.get_camera_directory_path(source_path, right_role)

        if not left_dir.exists() or not right_dir.exists():
            raise RuntimeError(
                f"FoundationStereo requires left/right image directories. Missing: {left_dir}, {right_dir}"
            )

        self._depth_source = "foundation_stereo"
        self.depth_dirs = {}

        camera_jobs: List[Tuple[str, Path, Path]] = []
        if left_role in cameras:
            camera_jobs.append((left_role, left_dir, right_dir))
        if right_role in cameras:
            # Estimate right-view depth by swapping the stereo pair.
            camera_jobs.append((right_role, right_dir, left_dir))

        if not camera_jobs:
            raise RuntimeError(f"No stereo cameras found in requested camera list: {cameras}")

        max_frames_cfg = getattr(getattr(dataset, "config", None), "max_frames", None)
        max_frames: Optional[int] = None
        if isinstance(max_frames_cfg, int) and max_frames_cfg > 0:
            max_frames = max_frames_cfg

        for camera_role, left_images_dir, right_images_dir in camera_jobs:
            depth_dir = source_path / f"{camera_role}_foundation_stereo_depth"
            required_images = self._collect_image_files(left_images_dir)
            if max_frames is not None:
                required_images = required_images[:max_frames]
            if not required_images:
                raise RuntimeError(f"No images found for camera '{camera_role}' in {left_images_dir}")

            if self._depth_cache_complete_for_images(required_images, depth_dir):
                num_depth_files = len(required_images)
                logger.info(
                    f"Reusing FoundationStereo depth maps for {camera_role} "
                    f"from {depth_dir} ({num_depth_files} required files)"
                )
            else:
                logger.info(f"Generating FoundationStereo depth maps for {camera_role}...")
                self._run_foundation_stereo(
                    left_images_dir=left_images_dir,
                    right_images_dir=right_images_dir,
                    output_dir=depth_dir,
                    fx=fx,
                    baseline=baseline,
                    max_frames=max_frames,
                )
                if not self._depth_cache_complete_for_images(required_images, depth_dir):
                    raise RuntimeError(
                        f"FoundationStereo output is incomplete for {left_images_dir}. "
                        f"Expected depth files for {len(required_images)} frames in {depth_dir}."
                    )

            self.depth_dirs[camera_role] = depth_dir

    def _setup_da3_depth(self, source_path: Path, cameras: List[str]) -> None:
        """Setup for Depth Anything 3 estimated depth.

        DA3 inference runs through the internal DA3 metric action helper.
        """
        self._depth_source = "da3"
        self.depth_dirs = {}
        logger.info("Using Depth Anything 3 for depth estimation")

        for camera_role in cameras:
            image_dir = self.get_camera_directory_path(source_path, camera_role)
            depth_dir = source_path / f"{camera_role}_da3_depth"

            if not image_dir.exists():
                raise RuntimeError(f"Image directory not found for DA3: {image_dir}")

            if self._depth_cache_complete(image_dir, depth_dir):
                num_depth_files = len(list(depth_dir.glob("*.png")))
                logger.info(
                    f"Reusing DA3 depth maps for {camera_role} from {depth_dir} ({num_depth_files} files)"
                )
            else:
                logger.info(f"Generating DA3 depth maps for {camera_role}...")
                intrinsics = None
                if self._dataset is not None:
                    try:
                        intrinsics = self._dataset.get_camera_intrinsics(camera_role)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Failed to resolve dataset intrinsics for camera=%s: %s. "
                            "DA3 will use fallback focal estimation.",
                            camera_role,
                            e,
                        )
                self._run_da3_depth(
                    image_dir=image_dir,
                    output_dir=depth_dir,
                    camera_role=camera_role,
                    intrinsics=intrinsics,
                )

            self.depth_dirs[camera_role] = depth_dir

    @staticmethod
    def _collect_image_files(image_dir: Path) -> List[Path]:
        image_files = sorted(image_dir.glob("*.png"))
        if image_files:
            return image_files
        return sorted(image_dir.glob("*.jpg"))

    def _depth_cache_complete(self, image_dir: Path, depth_dir: Path) -> bool:
        """Check if depth cache directory has all expected filenames."""
        image_files = self._collect_image_files(image_dir)
        if not image_files:
            return False

        return self._depth_cache_complete_for_images(image_files, depth_dir)

    def _depth_cache_complete_for_images(self, image_files: List[Path], depth_dir: Path) -> bool:
        """Check if depth cache has depth files for the expected image filenames."""
        if not depth_dir.exists() or not depth_dir.is_dir():
            return False

        if not image_files:
            return False

        depth_names = {p.name for p in depth_dir.glob("*.png")}
        if not depth_names:
            depth_names = {p.name for p in depth_dir.glob("*.jpg")}
        if not depth_names:
            return False

        expected_names = {p.name for p in image_files}
        return expected_names.issubset(depth_names)

    def _get_foundation_stereo_action(self) -> FoundationStereoAction:
        """Get or create FoundationStereo action instance."""
        if self._foundation_stereo_action is not None:
            return self._foundation_stereo_action

        repo_dir = getattr(self, "foundation_stereo_repo_dir", None)
        checkpoint_path = getattr(self, "foundation_stereo_checkpoint_path", None)
        python_executable = getattr(self, "foundation_stereo_python", None)
        conda_env = getattr(self, "foundation_stereo_conda_env", "foundation_stereo")

        self._foundation_stereo_action = FoundationStereoAction(
            repo_dir=Path(repo_dir) if repo_dir else None,
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
            python_executable=python_executable,
            conda_env=conda_env,
        )
        return self._foundation_stereo_action

    def _get_da3_metric_action(self) -> DA3MetricAction:
        """Get or create DA3 metric action instance."""
        if self._da3_metric_action is not None:
            return self._da3_metric_action

        self._da3_metric_action = DA3MetricAction(
            model_id=self.da3_model_id,
            device=self.da3_device,
            process_res=self.da3_process_res,
            fallback_hfov_deg=self.da3_fallback_hfov_deg,
            save_npz=self.da3_save_npz,
        )
        return self._da3_metric_action

    def _run_foundation_stereo(
        self,
        left_images_dir: Path,
        right_images_dir: Path,
        output_dir: Path,
        fx: float,
        baseline: float,
        max_frames: Optional[int] = None,
    ) -> None:
        """Run FoundationStereo and write depth PNGs aligned to left image filenames."""
        output_dir.mkdir(parents=True, exist_ok=True)

        action = self._get_foundation_stereo_action()
        action.run_sequence(
            left_images_dir=left_images_dir,
            right_images_dir=right_images_dir,
            output_dir=output_dir,
            fx=fx,
            baseline=baseline,
            max_frames=max_frames,
        )

    def _run_da3_depth(
        self,
        image_dir: Path,
        output_dir: Path,
        camera_role: str,
        intrinsics: Optional[Any] = None,
    ) -> None:
        """Run DA3Metric and write depth PNG/NPZ outputs aligned to RGB filenames."""
        output_dir.mkdir(parents=True, exist_ok=True)

        action = self._get_da3_metric_action()
        action.run_sequence(
            image_dir=image_dir,
            output_dir=output_dir,
            camera_role=camera_role,
            intrinsics=intrinsics,
        )

        if not self._depth_cache_complete(image_dir, output_dir):
            raise RuntimeError(
                f"DA3 output is incomplete for {image_dir}. Expected depth files in {output_dir}."
            )

    def _setup_dav2_depth(self, source_path: Path) -> None:
        """Setup for Depth Anything V2 estimated depth.

        Generates depth maps using Depth Anything V2 model.

        Args:
            source_path: Path to dataset or input images
        """
        self._depth_source = 'dav2'
        logger.info("Using Depth Anything V2 for depth estimation")

        for camera_role in self.cameras:
            image_dir = self.get_camera_directory_path(source_path, camera_role)
            depth_dir = source_path / f"{camera_role}_depth"

            if not image_dir.exists():
                raise ValueError(f"Image directory not found: {image_dir}")

            if depth_dir.exists() and any(depth_dir.glob("*.png")):
                num_depth_files = len(list(depth_dir.glob("*.png")))
                logger.info(
                    f"Reusing existing depth maps for {camera_role} from {depth_dir} ({num_depth_files} files)"
                )
            else:
                logger.info(
                    f"Depth maps not found for {camera_role}, generating with Depth Anything V2..."
                )
                self._generate_depth_maps_for_camera(camera_role, image_dir, depth_dir)

            self.depth_dirs[camera_role] = depth_dir

    def _generate_depth_maps_for_camera(self, camera_name: str, image_dir: Path, depth_dir: Path) -> None:
        """Generate depth maps using Depth Anything V2 for a specific camera.

        Args:
            camera_name: Camera role name ("left" or "right")
            image_dir: Path to the image directory
            depth_dir: Path where depth maps should be saved
        """
        check_depth_anything_available()

        encoder = getattr(self, 'depth_encoder', 'vitl')
        model_type = getattr(self, 'depth_model_type', 'vkitti')
        max_depth = getattr(self, 'max_depth_range', 80.0)

        if encoder not in DEPTH_MODEL_CONFIGS:
            raise ValueError(f"Invalid encoder '{encoder}'. Choose from: {list(DEPTH_MODEL_CONFIGS.keys())}")

        metric_depth_path = Path(__file__).parent.parent.parent / "deps" / "depth-estimation" / "Depth-Anything-V2" / "metric_depth"
        checkpoint_path = metric_depth_path / "checkpoints" / f"depth_anything_v2_metric_{model_type}_{encoder}.pth"

        if not checkpoint_path.exists():
            variant = DEPTH_MODEL_VARIANTS[encoder]
            if model_type == "hypersim":
                download_url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-{variant}/resolve/main/depth_anything_v2_metric_hypersim_{encoder}.pth"
            else:
                download_url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-{variant}/resolve/main/depth_anything_v2_metric_vkitti_{encoder}.pth"
            raise FileNotFoundError(
                f"Metric depth checkpoint not found: {checkpoint_path}\n"
                f"Please download from: {download_url}\n"
                f"And place it in: {checkpoint_path.parent}"
            )

        logger.info(f"Loading Depth Anything V2 METRIC ({encoder}, {model_type}) for {camera_name}...")

        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
            logger.warning("Running Depth Anything V2 on CPU - this will be slow!")

        logger.info(f"Using device: {device}")

        model = DepthAnythingV2(**{**DEPTH_MODEL_CONFIGS[encoder], 'max_depth': max_depth})
        model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
        model = model.to(device).eval()

        depth_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(image_dir.glob("*.png"))
        if not image_files:
            image_files = sorted(image_dir.glob("*.jpg"))

        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        logger.info(f"Generating depth maps for {len(image_files)} images...")

        try:
            from tqdm import tqdm
            image_iterator = tqdm(image_files, desc=f"Depth estimation ({camera_name})")
        except ImportError:
            image_iterator = image_files

        for image_path in image_iterator:
            raw_image = cv2.imread(str(image_path))
            if raw_image is None:
                logger.warning(f"Failed to read image: {image_path}")
                continue

            with torch.no_grad():
                depth = model.infer_image(raw_image)

            depth_encoded = (depth * 256).astype(np.uint16)
            depth_path = depth_dir / image_path.name
            cv2.imwrite(str(depth_path), depth_encoded)

            logger.debug(f"Saved depth for {image_path.name}: range [{depth.min():.2f}, {depth.max():.2f}]m")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Depth generation complete for {camera_name}")

    def _load_depth_from_disk(self, camera: str, rgb_filename: str) -> np.ndarray:
        """Load depth map from disk with automatic source-aware decoding.

        Handles different depth sources and their encodings:
        - sensor: Uses dataset's native depth conversion (TUM=5000, 7-Scenes=1000)
        - existing/foundation_stereo/da3/dav2: Uses standard encoding (depth * 256)

        Args:
            camera: Camera role ("left" or "right")
            rgb_filename: RGB image filename (used to find corresponding depth)

        Returns:
            Depth map (H, W) in meters

        Raises:
            RuntimeError: If depth setup is not complete
            ValueError: If camera is not in available cameras
            FileNotFoundError: If depth map does not exist
            RuntimeError: If depth map cannot be read
        """
        if not self._depth_setup_complete:
            module_name = getattr(self, 'name', self.__class__.__name__)
            raise RuntimeError(
                f"Depth estimation not set up for '{module_name}'. "
                "Ensure setup(context) provided a valid source path and depth setup completed."
            )

        if self.depth_dirs is None or camera not in self.depth_dirs:
            raise ValueError(
                f"Camera '{camera}' not in available cameras. "
                f"Depth maps were only generated for: {list(self.depth_dirs.keys()) if self.depth_dirs else []}"
            )

        depth_dir = self.depth_dirs[camera]

        # Sensor depth must be loaded via dataset API to preserve dataset-specific
        # associations and native depth decoding semantics.
        if self._depth_source == 'sensor':
            if self._dataset is None:
                raise RuntimeError(
                    f"Sensor depth loading requires dataset context for frame '{rgb_filename}' "
                    f"(camera={camera})."
                )

            depth_metric = self._dataset.load_depth_for_frame(
                rgb_filename=rgb_filename,
                camera=camera,
                use_estimated=False,
            )
            if depth_metric is None:
                raise RuntimeError(
                    f"Sensor depth unavailable for frame '{rgb_filename}' (camera={camera}) in "
                    f"dataset {self._dataset.__class__.__name__}. Ensure association/depth metadata is valid."
                )

            logger.debug(
                f"Loaded sensor depth for {rgb_filename} ({camera}): "
                f"range [{depth_metric.min():.2f}, {depth_metric.max():.2f}]m"
            )
            return depth_metric

        depth_path = depth_dir / rgb_filename

        if not depth_path.exists():
            raise FileNotFoundError(
                f"Depth map not found: {depth_path}\n"
                f"This should have been generated during setup. "
                f"Please check if depth generation completed successfully."
            )

        depth_encoded = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth_encoded is None:
            raise RuntimeError(
                f"Failed to read depth map: {depth_path}. "
                f"File may be corrupted or unreadable."
            )

        # non-sensor sources use our standard encoding (depth * 256)
        depth_metric = depth_encoded.astype(np.float32) / 256.0

        logger.debug(
            f"Loaded {self._depth_source} depth for {rgb_filename} ({camera}): "
            f"range [{depth_metric.min():.2f}, {depth_metric.max():.2f}]m"
        )

        return depth_metric

    def _cleanup_depth(self) -> None:
        """Clean up depth estimation state."""
        self.depth_dirs = {}
        self.camera_dirs = {}
        self.cameras = ['left']
        self._depth_setup_complete = False
        self._depth_source = 'none'
        self._use_sensor_depth = False
        self._depth_backend_requested = "auto"
        self._depth_backend_selected = 'none'
        self._depth_backend_candidates = []
        self._foundation_stereo_action = None
        self._da3_metric_action = None

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the module.

        Returns:
            Dictionary containing module information
        """
        return {
            "name": self.name,
            "type": self.config.type,
            "enabled": self.enabled,
            "class": self.__class__.__name__,
            "initialized": self._initialized,
            "parameters": self.parameters,
            "depth_backend_requested": self._depth_backend_requested,
            "depth_backend_selected": self._depth_backend_selected,
            "depth_source": self._depth_source,
            "depth_backend_candidates": self._depth_backend_candidates,
        }

    def __repr__(self) -> str:
        """String representation of the module."""
        status = "initialized" if self._initialized else "not initialized"
        enabled = "enabled" if self.enabled else "disabled"
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"{enabled}, "
            f"{status})"
        )


class NullModule(PerturbationModule):
    """Return images unchanged."""

    def _setup(self, context: ModuleSetupContext) -> None:
        """No setup required for null module."""
        logger.debug(f"NullModule {self.name} setup complete")

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str,
        **kwargs
    ) -> np.ndarray:
        """
        Return the image unchanged.

        Args:
            image: Input image
            depth: Optional depth map (ignored)
            frame_idx: Frame index (ignored)
            **kwargs: Additional context (ignored)

        Returns:
            Unchanged input image
        """
        logger.debug(f"NullModule {self.name} passing through frame {frame_idx}")
        return image


class CompositeModule(PerturbationModule):
    """Chain perturbation modules in sequence."""

    def __init__(
        self,
        config: PerturbationConfig,
        modules: List[PerturbationModule],
        mode: Union[str, CompositionMode] = CompositionMode.SEQUENTIAL
    ):
        """
        Initialize composite module.

        Args:
            config: Composite module configuration
            modules: List of PerturbationModule instances to chain
            mode: Composition mode (sequential only)
        """
        super().__init__(config)
        self.modules = modules

        if isinstance(mode, CompositionMode):
            resolved_mode = mode.value
        else:
            resolved_mode = str(mode).strip().lower()

        if resolved_mode != CompositionMode.SEQUENTIAL.value:
            raise ValueError(
                f"Invalid composition mode '{mode}'. Only 'sequential' is supported."
            )
        self.mode = CompositionMode.SEQUENTIAL

        # State for batch processing
        self._temp_dirs: List[str] = []  # Track temp dirs for cleanup
        self._total_frames: int = 0
        self._dataset_path: Optional[str] = None
        self._batch_module_prepared: Dict[str, bool] = {}  # Track which batch modules are ready

        logger.info(
            f"Created CompositeModule '{self.name}' with {len(modules)} modules "
            f"in {self.mode.value} mode"
        )

    def _setup(self, context: ModuleSetupContext) -> None:
        """Setup all child modules."""
        self._total_frames = context.total_frames if context.total_frames is not None else 0
        self._dataset_path = str(context.dataset_path) if context.dataset_path else None

        for i, module in enumerate(self.modules):
            logger.debug(f"Setting up module {i+1}/{len(self.modules)}: {module.name}")
            module.setup(context)

        logger.info(f"All {len(self.modules)} modules initialized for '{self.name}'")

    def _on_context_updated(
        self,
        previous_context: Optional[ModuleSetupContext],
        context: ModuleSetupContext,
        reason: str
    ) -> None:
        """Propagate context updates to all child modules."""
        self._total_frames = context.total_frames if context.total_frames is not None else 0
        self._dataset_path = str(context.dataset_path) if context.dataset_path else None

        for module in self.modules:
            module.update_context(context, reason=f"composite_parent_{reason}")

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Apply modules sequentially.

        Args:
            image: Input image
            depth: Optional depth map
            frame_idx: Frame index
            camera: Camera identifier
            **kwargs: Additional context passed to child modules

        Returns:
            Image with perturbations applied, or None if dropped by a child module
        """
        return self._apply_sequential(image, depth, frame_idx, camera, **kwargs)

    def _apply_sequential(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str,
        **kwargs
    ) -> Optional[np.ndarray]:
        """Apply modules sequentially, handling batch processing modules."""
        result = image.copy()

        for i, module in enumerate(self.modules):
            if not module.enabled:
                logger.debug(f"Skipping disabled module: {module.name}")
                continue

            if module.requires_full_sequence and module.name not in self._batch_module_prepared:
                self._prepare_batch_module(i, module, depth, camera, **kwargs)

            result = module.apply(result, depth, frame_idx, camera, **kwargs)
            if result is None:
                logger.debug(
                    "Module %s dropped frame %d in composite '%s'; stopping chain",
                    module.name,
                    frame_idx,
                    self.name,
                )
                return None

            logger.debug(
                f"Applied module {i+1}/{len(self.modules)}: {module.name} "
                f"(frame {frame_idx})"
            )

        return result

    def _prepare_batch_module(
        self,
        batch_module_idx: int,
        batch_module: PerturbationModule,
        depth: Optional[np.ndarray],
        camera: str,
        **kwargs
    ) -> None:
        """
        Prepare a batch processing module by running all preceding modules
        on all frames and saving results to a temporary directory.

        For stereo datasets, processes both left and right camera roles.

        Args:
            batch_module_idx: Index of the batch module in self.modules
            batch_module: The batch processing module
            depth: Depth map (passed through)
            camera: Camera role identifier
            **kwargs: Additional context
        """
        import cv2
        from pathlib import Path

        logger.info(f"Preparing batch module '{batch_module.name}' - processing all frames first...")

        preceding_modules = [m for m in self.modules[:batch_module_idx] if m.enabled]

        if not preceding_modules:
            logger.info(f"No preceding modules - batch module will use original dataset")
            self._batch_module_prepared[batch_module.name] = True
            return

        temp_dir = str(create_temp_dir(prefix=f"composite_{batch_module.name}_"))
        self._temp_dirs.append(temp_dir)
        temp_path = Path(temp_dir)

        logger.info(f"Created temp directory for intermediate outputs: {temp_dir}")

        # Process all frames through preceding modules
        if not self._dataset_path:
            raise RuntimeError(
                f"CompositeModule '{self.name}' cannot prepare batch module '{batch_module.name}' "
                "without a dataset path in setup context."
            )

        dataset_path = Path(self._dataset_path)

        # Composite batch preparation must use explicit dataset context.
        dataset = self.setup_context.dataset if self.setup_context else None
        if dataset is None:
            raise RuntimeError(
                f"CompositeModule '{self.name}' cannot prepare batch module '{batch_module.name}' "
                "without a loaded dataset object in setup context. "
                "Path-based dataset inference is not supported."
            )

        camera_roles = dataset.get_active_camera_roles()
        camera_dir_by_role = dataset.resolve_camera_directories(dataset_path)

        if "left" not in camera_roles or "left" not in camera_dir_by_role:
            raise RuntimeError(
                "Dataset must expose an active 'left' camera role during composite batch preparation."
            )

        if "right" in camera_roles:
            if "right" not in camera_dir_by_role:
                raise RuntimeError(
                    "Stereo composite batch preparation requires a resolved right camera directory."
                )
            logger.info(
                f"Stereo dataset detected - processing camera roles: {camera_roles} "
                f"(dirs={camera_dir_by_role})"
            )
        else:
            logger.info(
                f"Mono dataset detected - processing camera role: left "
                f"(dir={camera_dir_by_role['left']})"
            )

        for camera_role in camera_roles:
            camera_dir_name = camera_dir_by_role[camera_role]
            camera_dir = temp_path / camera_dir_name
            camera_dir.mkdir(parents=True, exist_ok=True)

            original_camera_dir = dataset_path / camera_dir_name
            if not original_camera_dir.exists():
                raise RuntimeError(
                    f"Camera directory '{camera_dir_name}' not found under dataset path {dataset_path} "
                    f"while preparing batch module '{batch_module.name}'."
                )

            original_files = sorted(original_camera_dir.glob("*.png"))
            if not original_files:
                original_files = sorted(original_camera_dir.glob("*.jpg"))
            if not original_files:
                raise RuntimeError(
                    f"No image files found in camera directory {original_camera_dir} "
                    f"while preparing batch module '{batch_module.name}'."
                )

            logger.info(
                f"Processing {len(dataset)} frames for role={camera_role} (dir={camera_dir_name}) "
                f"through {len(preceding_modules)} preceding module(s)..."
            )

            for frame_idx, frame_data in enumerate(dataset):
                if camera_role == "right":
                    frame_image = frame_data.get("image_right")
                    frame_filename = frame_data.get("rgb_filename_right")
                else:
                    frame_image = frame_data.get("image")
                    frame_filename = frame_data.get("rgb_filename")

                if frame_image is None:
                    raise RuntimeError(
                        f"Missing image data for camera role '{camera_role}' at frame {frame_idx} "
                        f"while preparing batch module '{batch_module.name}'."
                    )

                frame_depth = frame_data.get("depth")

                result = frame_image.copy()
                apply_kwargs = dict(kwargs)
                if frame_filename:
                    apply_kwargs["rgb_filename"] = frame_filename

                for module in preceding_modules:
                    result = module.apply(result, frame_depth, frame_idx, camera_role, **apply_kwargs)

                if frame_idx < len(original_files):
                    filename = original_files[frame_idx].name
                elif frame_filename:
                    filename = frame_filename
                else:
                    filename = f"{frame_idx:06d}.png"

                output_path = camera_dir / filename
                cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

                if (frame_idx + 1) % 100 == 0:
                    logger.info(
                        f"Processed {frame_idx + 1}/{len(dataset)} frames for role={camera_role}"
                    )

            logger.info(f"Saved {len(dataset)} intermediate frames to {camera_dir}")

            import shutil

            depth_candidates = [
                dataset_path / f"{camera_role}_foundation_stereo_depth",
                dataset_path / f"{camera_role}_da3_depth",
                dataset_path / f"{camera_role}_depth",
            ]
            for depth_dir in depth_candidates:
                if not depth_dir.exists():
                    continue
                dest_depth_dir = temp_path / depth_dir.name
                if not dest_depth_dir.exists():
                    shutil.copytree(depth_dir, dest_depth_dir)
                    logger.info(f"Copied depth directory to {dest_depth_dir}")
                break

        original_calib = dataset_path / "calib.txt"
        if original_calib.exists():
            import shutil
            shutil.copy2(original_calib, temp_path / "calib.txt")
            logger.info("Copied calib.txt to temp directory")

        # Tell the batch module to use the temp directory
        batch_context = batch_module.setup_context
        if batch_context is None:
            raise RuntimeError(
                f"Batch module '{batch_module.name}' has no setup context. "
                "Composite setup must run before batch preparation."
            )
        batch_module.update_context(
            replace(batch_context, input_path=Path(temp_dir)),
            reason="composite_batch_input_override",
        )

        self._batch_module_prepared[batch_module.name] = True
        logger.info(f"Batch module '{batch_module.name}' prepared with input from {temp_dir}")

    def _cleanup(self) -> None:
        """Cleanup all child modules and temp directories."""
        import shutil

        # Cleanup child modules
        for module in reversed(self.modules):
            try:
                module.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up module {module.name}: {e}")

        # Cleanup temp directories
        for temp_dir in self._temp_dirs:
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Removed temp directory: {temp_dir}")
            except Exception as e:
                logger.error(f"Error removing temp directory {temp_dir}: {e}")

        self._temp_dirs.clear()
        self._batch_module_prepared.clear()

    def get_info(self) -> Dict[str, Any]:
        """Get information including child modules."""
        info = super().get_info()
        info["composition_mode"] = self.mode.value
        info["num_modules"] = len(self.modules)
        info["modules"] = [m.get_info() for m in self.modules]
        return info

    def add_module(self, module: PerturbationModule) -> None:
        """Add a module to the composite."""
        self.modules.append(module)
        if self._initialized:
            if self._setup_context is None:
                raise RuntimeError(
                    f"Composite '{self.name}' has no setup context while initialized."
                )
            module.setup(self._setup_context)

        logger.info(f"Added module {module.name} to composite {self.name}")

    def remove_module(self, module_name: str) -> bool:
        """Remove a module by name."""
        for i, module in enumerate(self.modules):
            if module.name == module_name:
                if self._initialized:
                    module.cleanup()
                self.modules.pop(i)

                logger.info(f"Removed module {module_name} from composite {self.name}")
                return True
        return False
