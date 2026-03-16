"""Physics-based fog using backend-resolved metric depth."""

import importlib
import sys
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
import cv2

from .fog_base import FogParametersBase, FogPresetsBase, FogModuleBase
from ...robustness.param_spec import BoundaryParamSpec
from ...utils import get_logger

logger = get_logger(__name__)


@dataclass
class FogDepthAnythingParameters(FogParametersBase):
    """Parameters for fog simulation with backend-resolved depth maps."""

    # Depth Anything V2 specific parameters
    encoder: str = field(default='vitl', metadata={'choices': ['vits', 'vitb', 'vitl', 'vitg']})
    """Model size: 'vits' (24.8M), 'vitb' (97.5M), 'vitl' (335.3M), 'vitg' (1.3B)"""

    max_depth_range: float = 80.0
    """Maximum depth range for metric depth model (meters) - used as max_depth parameter"""

    # Noise parameters for heterogeneous fog
    add_noise: bool = False
    """Add 3D noise for natural fog variation (inspired by fog_codes_public)"""

    noise_scale: float = 0.5
    """Scale of noise effect (k parameter from fog_codes_public)"""

    noise_steps: int = 10
    """Number of depth steps for 3D noise sampling"""

    # Camera intrinsics for 3D noise (optional)
    focal_length_px: Optional[float] = None
    """Focal length in pixels (for pinhole camera model)"""

    principal_point: Optional[Tuple[float, float]] = None
    """Principal point (cx, cy) for camera model"""

    noise_backend: str = field(default="auto", metadata={'choices': ['auto', 'simplex', 'perlin']})
    """Noise backend selection: 'auto', 'simplex', or 'perlin'."""


class FogModule(FogModuleBase):
    """Physics-based fog using backend-resolved metric depth."""

    module_name = "fog"
    module_description = "Physics-based fog with backend-resolved metric depth"

    PARAMETERS_CLASS = FogDepthAnythingParameters
    SEARCHABLE_PARAMS = {
        "visibility_m": BoundaryParamSpec(domain="integer"),
    }
    requires_full_sequence = True  # Needs all frames to generate depth maps

    def _setup(self, context) -> None:
        """Setup fog module parameters with explicit setup context."""
        params = self.parameters or {}

        self._init_depth_state()

        if 'preset' in params:
            preset_name = params['preset']
            preset_params = FogDepthAnythingPresets.get_preset(preset_name)
            visibility_m = preset_params.visibility_m
            atmospheric_light = preset_params.atmospheric_light
            min_depth_m = preset_params.min_depth_m
            max_depth_m = preset_params.max_depth_m
            max_depth_range = preset_params.max_depth_range
            logger.info(f"Using fog preset '{preset_name}'")
        else:
            visibility_m = 50.0
            atmospheric_light = (0.8, 0.8, 0.8)
            min_depth_m = 0.1
            max_depth_m = 100.0
            max_depth_range = 80.0

        # Override with custom parameters
        visibility_m = params.get('visibility_m', visibility_m)
        atmospheric_light = params.get('atmospheric_light', atmospheric_light)
        min_depth_m = params.get('min_depth_m', min_depth_m)
        max_depth_m = params.get('max_depth_m', max_depth_m)

        # Depth runtime parameters (set on base depth runtime attributes)
        self._configure_depth_runtime(
            encoder=params.get('encoder', 'vitl'),
            max_depth_range=params.get('max_depth_range', max_depth_range),
            model_type=params.get('depth_model_type', 'vkitti'),
            depth_backend=params.get("depth_backend", "auto"),
            da3_model_id=params.get("da3_model_id"),
            da3_fallback_hfov_deg=params.get("da3_fallback_hfov_deg"),
            da3_save_npz=params.get("da3_save_npz"),
            da3_device=params.get("da3_device"),
            da3_process_res=params.get("da3_process_res"),
        )

        # Noise parameters
        add_noise = params.get('add_noise', False)
        noise_scale = params.get('noise_scale', 0.5)
        noise_steps = params.get('noise_steps', 10)
        focal_length_px = params.get('focal_length_px', None)
        principal_point = params.get('principal_point', None)
        noise_backend = str(params.get('noise_backend', 'auto')).strip().lower()

        # Incremental fog
        incremental = params.get('incremental', False)
        start_visibility_m = params.get('start_visibility_m', None)
        end_visibility_m = params.get('end_visibility_m', None)
        if 'total_frames' in params:
            logger.warning(
                "Ignoring fog parameter 'total_frames'; using setup context total_frames instead."
            )
        total_frames = context.total_frames

        if isinstance(atmospheric_light, list):
            atmospheric_light = tuple(atmospheric_light)

        self.params = FogDepthAnythingParameters(
            visibility_m=visibility_m,
            atmospheric_light=atmospheric_light,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            encoder=self.depth_encoder,
            max_depth_range=self.max_depth_range,
            add_noise=add_noise,
            noise_scale=noise_scale,
            noise_steps=noise_steps,
            focal_length_px=focal_length_px,
            principal_point=principal_point,
            noise_backend=noise_backend,
            incremental=incremental,
            start_visibility_m=start_visibility_m,
            end_visibility_m=end_visibility_m,
            total_frames=total_frames
        )

        self._validate_incremental_config()

        if incremental:
            self.params.visibility_m = start_visibility_m

        self._setup_depth_estimation(context)

        # Exported in config for reproducibility.
        self.use_simplex = False
        self.noise_backend = "disabled"

        if self.params.add_noise:
            self._init_noise_generator()
        elif 'noise_backend' in params:
            logger.warning(
                "Ignoring fog parameter 'noise_backend' because add_noise is false."
            )

        logger.info(
            f"Setup complete for FogModule '{self.name}' with encoder={self.depth_encoder}, "
            f"visibility={self.params.visibility_m:.1f}m, beta={self.params.beta:.4f}, "
            f"add_noise={self.params.add_noise}, noise_backend={self.noise_backend}, "
            f"noise_backend_requested={self.params.noise_backend}"
        )

    def _on_context_updated(self, previous_context, context, reason: str) -> None:
        """Rebuild fog state when setup context changes."""
        if self.params.incremental:
            self.params.total_frames = context.total_frames

        self._refresh_depth_runtime(context)

    def _resolve_simplex_search_paths(self) -> Tuple[Path, Path]:
        """Return prioritized search paths for SimplexNoise."""
        repo_root = Path(__file__).resolve().parents[3]
        return (
            repo_root / "deps" / "perturbations" / "fog_codes_public",
            repo_root / "fog_codes_public",  # Legacy location
        )

    def _import_simplex_noise_module(self) -> Tuple[Any, str]:
        """Import SimplexNoise module from supported locations."""
        import_errors = []

        for search_path in self._resolve_simplex_search_paths():
            if not search_path.exists():
                continue
            search_path_str = str(search_path)
            if search_path_str not in sys.path:
                sys.path.insert(0, search_path_str)
            try:
                return importlib.import_module("SimplexNoise"), search_path_str
            except ImportError as exc:
                import_errors.append(f"{search_path}: {exc}")

        try:
            return importlib.import_module("SimplexNoise"), "pythonpath"
        except ImportError as exc:
            import_errors.append(f"pythonpath: {exc}")
            raise ImportError("; ".join(import_errors)) from exc

    def _init_noise_generator(self) -> None:
        """Initialize 3D noise generator for heterogeneous fog."""
        requested_backend = self.params.noise_backend.lower()

        if requested_backend == "perlin":
            self.use_simplex = False
            self.noise_backend = "perlin_forced"
            logger.info("Using forced 2D Perlin noise for heterogeneous fog")
            return

        if requested_backend not in {"auto", "simplex"}:
            raise ValueError(
                f"Unsupported fog noise_backend '{self.params.noise_backend}'. "
                "Expected one of: auto, simplex, perlin."
            )

        try:
            simplex_module, source = self._import_simplex_noise_module()
            self.simplex = simplex_module.SimplexNoise()
            self.use_simplex = True
            self.noise_backend = "simplex"
            logger.info(f"Using SimplexNoise for 3D heterogeneous fog (source={source})")
        except ImportError as exc:
            self.use_simplex = False
            if requested_backend == "simplex":
                raise RuntimeError(
                    "noise_backend='simplex' but SimplexNoise backend is unavailable. "
                    "Install fog_codes_public under deps/perturbations/fog_codes_public "
                    "or make SimplexNoise importable on PYTHONPATH."
                ) from exc
            self.noise_backend = "perlin_fallback"
            logger.warning(f"SimplexNoise not available ({exc}), using 2D Perlin noise instead")

    def _apply_3d_noise(
        self,
        depth: np.ndarray,
        beta: float,
        frame_idx: int
    ) -> np.ndarray:
        """Apply 3D heterogeneous noise to scattering coefficient.

        Args:
            depth: Depth map (H, W) in meters
            beta: Base scattering coefficient
            frame_idx: Frame index

        Returns:
            Beta map with 3D noise (H, W)
        """
        height, width = depth.shape

        if self.use_simplex:
            if not hasattr(self, '_simplex_initialized'):
                self.simplex.setup(depth)
                self._simplex_initialized = True
                logger.debug("Initialized SimplexNoise with depth template")

            if self.params.focal_length_px is not None:
                fu_inv = 1.0 / self.params.focal_length_px
                fv_inv = 1.0 / self.params.focal_length_px
            else:
                fu_inv = 1.0 / (width * 0.7)
                fv_inv = 1.0 / (width * 0.7)

            # Principal point
            if self.params.principal_point is not None:
                cx, cy = self.params.principal_point
            else:
                cx, cy = width / 2, height / 2

            # Image coordinates (centered at principal point)
            x_ = np.linspace(0, width, width, endpoint=False) - cx
            y_ = np.linspace(0, height, height, endpoint=False) - cy
            y_, x_ = np.meshgrid(y_, x_, indexing='ij')

            noise = np.zeros_like(depth)

            # Sample noise at multiple depths along the ray
            for i in range(self.params.noise_steps):
                Z = depth * i / self.params.noise_steps
                X = Z * x_ * fu_inv
                Y = Z * y_ * fv_inv

                noise += self.simplex.noise3d(X / 2000.0, Y / 2000.0, Z / 2000.0) / self.params.noise_steps

            beta_noise = beta * (1 + self.params.noise_scale * noise)

            logger.debug(
                f"Frame {frame_idx}: Applied 3D SimplexNoise with {self.params.noise_steps} steps"
            )

        else:
            # Fallback: Use 2D Perlin noise
            from ...utils.noise import generate_perlin_noise_2d

            noise = generate_perlin_noise_2d(
                shape=depth.shape,
                scale=100.0,
                octaves=2,
                persistence=0.5,
                seed=frame_idx
            )

            beta_noise = beta * (1 + self.params.noise_scale * (noise - 0.5) * 2)

            logger.debug(f"Frame {frame_idx}: Applied 2D Perlin noise")

        return np.clip(beta_noise, 0, None)

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        """Apply fog effect to image using pre-generated depth maps.

        Args:
            image: Input RGB image (H, W, 3) in range [0, 255]
            depth: Ignored - depth is loaded from disk instead
            frame_idx: Current frame index
            camera: Camera identifier (e.g., 'image_2', 'image_3')
            **kwargs: Additional context (rgb_filename required for depth lookup)

        Returns:
            Foggy image with same shape and dtype as input
        """
        # Ensure depth setup is complete
        if not self._depth_setup_complete:
            raise RuntimeError(
                f"FogModule '{self.name}' setup not complete. "
                "Depth backend resolution must complete during setup(context). "
                "This should happen automatically - please report this as a bug."
            )

        current_visibility, current_beta = self._get_current_fog_params(frame_idx)

        if camera not in self.cameras:
            raise ValueError(
                f"Camera '{camera}' not in available cameras: {self.cameras}. "
                f"Depth maps were only generated for: {list(self.depth_dirs.keys())}"
            )

        depth_loaded, rgb_filename = self._load_depth_for_apply(
            camera,
            kwargs,
            provided_depth=None,
            prefer_dataset=False,
            module_label="fog_depthanything",
        )

        logger.debug(f"Frame {frame_idx} ({rgb_filename}): Using camera {camera}")

        if depth_loaded.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Loaded depth shape {depth_loaded.shape[:2]} doesn't match "
                f"image shape {image.shape[:2]}"
            )

        image_float = image.astype(np.float32) / 255.0

        valid_depth = np.isfinite(depth_loaded) & (depth_loaded > 0.0)
        depth_clamped = np.full(depth_loaded.shape, self.params.max_depth_m, dtype=np.float32)
        if np.any(valid_depth):
            depth_clamped[valid_depth] = np.clip(
                depth_loaded[valid_depth],
                self.params.min_depth_m,
                self.params.max_depth_m
            )

        if self.params.add_noise:
            beta_map = np.full(depth_clamped.shape, current_beta, dtype=np.float32)
            beta_map_noisy = self._apply_3d_noise(depth_clamped, current_beta, frame_idx)
            beta_map[valid_depth] = beta_map_noisy[valid_depth]
            transmission = np.exp(-beta_map * depth_clamped)
        else:
            transmission = np.exp(-current_beta * depth_clamped)

        foggy_image = self._apply_koschmieder_model(image_float, transmission)

        # Log statistics
        mean_transmission = np.mean(transmission)
        invalid_ratio = 1.0 - float(np.mean(valid_depth))
        logger.debug(
            f"Frame {frame_idx}: Applied fog with mean transmission={mean_transmission:.3f}, "
            f"depth range=[{np.min(depth_clamped):.1f}, {np.max(depth_clamped):.1f}]m, "
            f"invalid_depth={invalid_ratio:.2%}"
        )

        return foggy_image

    def _cleanup(self) -> None:
        """Cleanup resources."""
        self._cleanup_depth()  # Depth state cleanup
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache during cleanup: {e}")

        logger.debug("FogModule cleanup complete")

    def get_config(self) -> Dict[str, Any]:
        """Get module configuration."""
        config = super().get_config()
        config['type'] = 'fog'
        config['encoder'] = self.depth_encoder
        config['add_noise'] = self.params.add_noise
        config['noise_backend_requested'] = self.params.noise_backend
        config['noise_backend'] = self.noise_backend
        config['max_depth_range'] = self.max_depth_range
        return config


class FogDepthAnythingPresets(FogPresetsBase):
    """Predefined fog intensity presets for Depth Anything module."""

    LIGHT = FogDepthAnythingParameters(
        visibility_m=FogPresetsBase.LIGHT_VISIBILITY,
        atmospheric_light=FogPresetsBase.LIGHT_ATMOSPHERIC_LIGHT,
        max_depth_range=80.0
    )

    MEDIUM = FogDepthAnythingParameters(
        visibility_m=FogPresetsBase.MEDIUM_VISIBILITY,
        atmospheric_light=FogPresetsBase.MEDIUM_ATMOSPHERIC_LIGHT,
        max_depth_range=80.0
    )

    HEAVY = FogDepthAnythingParameters(
        visibility_m=FogPresetsBase.HEAVY_VISIBILITY,
        atmospheric_light=FogPresetsBase.HEAVY_ATMOSPHERIC_LIGHT,
        max_depth_range=80.0
    )

    DENSE = FogDepthAnythingParameters(
        visibility_m=FogPresetsBase.DENSE_VISIBILITY,
        atmospheric_light=FogPresetsBase.DENSE_ATMOSPHERIC_LIGHT,
        max_depth_range=80.0
    )

    @classmethod
    def get_preset(cls, name: str) -> FogDepthAnythingParameters:
        """Get fog parameters by preset name.

        Args:
            name: Preset name (light, medium, heavy, dense)

        Returns:
            FogDepthAnythingParameters for the preset

        Raises:
            ValueError: If preset name is invalid
        """
        presets = {
            'light': cls.LIGHT,
            'medium': cls.MEDIUM,
            'heavy': cls.HEAVY,
            'dense': cls.DENSE
        }

        name_lower = name.lower()
        if name_lower not in presets:
            raise ValueError(
                f"Invalid fog preset '{name}'. "
                f"Choose from: {', '.join(presets.keys())}"
            )

        return presets[name_lower]
