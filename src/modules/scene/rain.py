"""Physics-based rain rendering using the external rain-rendering project."""

import numpy as np
import subprocess
import os
import cv2
from enum import Enum
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

from ..base import PerturbationModule
from ...robustness.param_spec import BoundaryParamSpec
from ...utils import get_logger

logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]


class DepthModel(str, Enum):
    """Depth Anything V2 model variants."""
    VITS = "vits"
    """Small model - fastest, lowest accuracy"""
    VITB = "vitb"
    """Base model - balanced speed and accuracy"""
    VITL = "vitl"
    """Large model - slowest, highest accuracy"""


@dataclass
class RainParameters:
    """Parameters for physics-based rain rendering."""

    intensity: int = 25
    """Rain intensity in mm/hr (1-200)"""

    depth_model: DepthModel = DepthModel.VITB
    """Depth model variant for depth estimation"""

    max_depth: int = 80
    """Maximum depth in meters for outdoor scenes"""

    rain_rendering_path: Optional[str] = None
    """Path to rain-rendering project (uses default if None)"""


class RainModule(PerturbationModule):
    """Physics-based rain rendering using the external rain-rendering tool."""

    module_name = "rain"
    module_description = "Physics-based rain rendering"

    PARAMETERS_CLASS = RainParameters
    SEARCHABLE_PARAMS = {
        "intensity": BoundaryParamSpec(domain="integer"),
    }
    requires_full_sequence = True

    def _setup(self, context) -> None:
        """Initialize parameters with explicit setup context."""
        params = self.parameters or {}
        self.intensity = params.get('intensity', 25)
        self.max_frames = context.total_frames
        self.rain_rendering_path = Path(params.get(
            'rain_rendering_path',
            str(REPO_ROOT / 'deps' / 'perturbations' / 'rain-rendering')
        ))

        self._init_depth_state()

        # Depth Anything V2 parameters
        self._configure_depth_runtime(
            encoder=params.get('depth_model', 'vitb'),
            max_depth_range=params.get('max_depth', 80),
            model_type='vkitti',  # Outdoor scenes
            depth_backend=params.get("depth_backend", "auto"),
            da3_model_id=params.get("da3_model_id"),
            da3_fallback_hfov_deg=params.get("da3_fallback_hfov_deg"),
            da3_save_npz=params.get("da3_save_npz"),
            da3_device=params.get("da3_device"),
            da3_process_res=params.get("da3_process_res"),
        )

        self._rain_rendered = False
        self._rainy_frames_dirs = {}  # Dict of {camera: output_dir}

        self._setup_depth_estimation(context)
        self._ensure_depth_aliases_for_rain()
        self._create_dataset_symlink()

        logger.info(f"RainModule '{self.name}' initialized")
        logger.info(f"Rain intensity: {self.intensity} mm/hr")
        logger.info(f"Depth model: {self.depth_encoder}")

    def _on_context_updated(self, previous_context, context, reason: str) -> None:
        """Rebuild rain prerequisites when input context changes."""
        self.max_frames = context.total_frames
        self._refresh_depth_runtime(context)
        self._ensure_depth_aliases_for_rain()
        self._create_dataset_symlink()
        self._rain_rendered = False
        self._rainy_frames_dirs = {}
        logger.info(f"Context update complete for RainModule '{self.name}' ({reason})")

    def _ensure_depth_aliases_for_rain(self) -> None:
        """Create role_depth aliases expected by rain-rendering config."""
        source_path = self._get_source_path()
        if source_path is None:
            raise RuntimeError("Rain depth alias setup requires dataset/input source path.")

        for camera_role in self.cameras:
            if camera_role not in self.depth_dirs:
                raise RuntimeError(
                    f"Depth directory for camera role '{camera_role}' not available after depth setup."
                )

            target_dir = self.depth_dirs[camera_role]
            alias_dir = source_path / f"{camera_role}_depth"
            target_resolved = target_dir.resolve()

            if alias_dir.is_symlink():
                if alias_dir.resolve() == target_resolved:
                    continue
                alias_dir.unlink()
            elif alias_dir.exists():
                if alias_dir.resolve() == target_resolved:
                    continue
                raise RuntimeError(
                    f"Rain depth alias path already exists and is not the active depth directory: {alias_dir}. "
                    "Remove or rename it before running rain module."
                )

            alias_dir.symlink_to(target_resolved, target_is_directory=True)
            logger.info("Created rain depth alias: %s -> %s", alias_dir, target_resolved)

    def _create_dataset_symlink(self) -> None:
        """Create symlink for rain-rendering to find the source images."""
        # rain-rendering expects: data/source/{dataset_name}
        symlink_path = self.rain_rendering_path / "data" / "source" / "slamadverseriallab"

        symlink_path.parent.mkdir(parents=True, exist_ok=True)

        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()

        source_path = self._get_source_path()
        target_path = source_path.resolve()
        symlink_path.symlink_to(target_path)

        logger.info(f"Created dataset symlink: {symlink_path} -> {target_path}")

    def _create_rain_config(self, camera_role: str) -> Path:
        """Create config file for rain-rendering for a specific camera.

        Args:
            camera_role: Camera role ("left" or "right")

        Returns:
            Path to the created config file
        """
        source_path = self._get_source_path()
        source_name = source_path.name
        camera_dir = self.get_camera_directory_name(camera_role)
        depth_dir = f"{camera_role}_depth"

        config_content = f'''"""Dynamic config for SLAMAdverserialLab KITTI rain rendering - {camera_role}."""
import os

def resolve_paths(params):
    dataset_root = "data/source/slamadverseriallab"
    sequence = "{source_name}"

    params.sequences = [sequence]

    params.images = {{sequence: os.path.join(dataset_root, "{camera_dir}")}}
    params.depth = {{sequence: os.path.join(dataset_root, "{depth_dir}")}}

    calib_file = os.path.join(dataset_root, "calib.txt")
    if os.path.exists(calib_file):
        params.calib = {{sequence: calib_file}}
    else:
        params.calib = {{sequence: None}}

    return params

def settings():
    settings = {{}}

    # Camera intrinsic parameters (KITTI-like defaults)
    settings["cam_hz"] = 10
    settings["cam_CCD_WH"] = [1242, 375]
    settings["cam_CCD_pixsize"] = 4.65
    settings["cam_WH"] = [1242, 375]
    settings["cam_focal"] = 6
    settings["cam_gain"] = 20
    settings["cam_f_number"] = 6.0
    settings["cam_focus_plane"] = 6.0
    settings["cam_exposure"] = 2

    # Camera extrinsic parameters
    settings["cam_pos"] = [1.5, 1.5, 0.3]
    settings["cam_lookat"] = [1.5, 1.5, -1.]
    settings["cam_up"] = [0., 1., 0.]

    # Sequence-wise settings
    settings["sequences"] = {{}}
    settings["sequences"]["{source_name}"] = {{}}
    settings["sequences"]["{source_name}"]["sim_mode"] = "normal"
    settings["sequences"]["{source_name}"]["sim_duration"] = 100  # Long enough for sequence

    return settings
'''

        # Write config file (camera-specific)
        config_path = self.rain_rendering_path / "config" / f"slamadverseriallab_{camera_role}.py"
        config_path.write_text(config_content)
        logger.info(f"Created rain-rendering config for {camera_role}: {config_path}")

        return config_path

    def _run_rain_rendering(self) -> None:
        """Run rain-rendering using Docker container to generate rainy images."""
        source_path = self._get_source_path()

        logger.info(f"Running rain-rendering with intensity {self.intensity} mm/hr...")
        logger.info(f"Source path: {source_path}")
        logger.info(f"Processing {len(self.cameras)} camera(s): {self.cameras}")
        logger.info("Using Docker container: rain-rendering:latest")

        user_id = os.getuid()
        group_id = os.getgid()

        source_abs = source_path.resolve()
        output_dir = self.rain_rendering_path / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_abs = output_dir.resolve()

        particles_dir = self.rain_rendering_path / "data" / "particles"
        particles_dir.mkdir(parents=True, exist_ok=True)
        particles_abs = particles_dir.resolve()

        simulation_py_abs = (self.rain_rendering_path / "tools" / "simulation.py").resolve()

        source_image_dir = self.get_camera_directory_path(source_path, self.cameras[0])
        source_frame_count = len(list(source_image_dir.glob("*.png")))
        expected_frame_count = min(source_frame_count, self.max_frames) if self.max_frames else source_frame_count
        logger.info(f"Expected frames: {expected_frame_count} (source has {source_frame_count}, max_frames={self.max_frames})")

        for camera_role in self.cameras:
            logger.info(f"--- Processing camera role: {camera_role} ---")

            output_base = self.rain_rendering_path / "data" / "output" / "slamadverseriallab" / source_path.name
            camera_rainy_dir = output_base / "rain" / f"{self.intensity}mm" / camera_role / "rainy_image"

            if camera_rainy_dir.exists() and self._input_path is None:
                existing_frame_count = len(list(camera_rainy_dir.glob("*.png")))
                logger.info(f"Found existing rain output for {camera_role}: {camera_rainy_dir}")
                logger.info(f"  Existing frames: {existing_frame_count}, Expected: {expected_frame_count}")

                if existing_frame_count == expected_frame_count:
                    logger.info(f"  Skipping rain rendering for {camera_role} (output already exists with {existing_frame_count} frames)")
                    self._rainy_frames_dirs[camera_role] = camera_rainy_dir
                    continue
                else:
                    logger.warning(f"  Frame count mismatch! Regenerating rain for {camera_role}...")
            else:
                logger.info(f"Generating rain output for {camera_role}")

            config_path = self._create_rain_config(camera_role)
            config_abs = config_path.resolve()

            cmd = [
                "docker", "run", "--rm",
                "--user", f"{user_id}:{group_id}",
                "-e", "HOME=/tmp",
                "-e", "MPLCONFIGDIR=/tmp/matplotlib",
                "-v", f"{source_abs}:/workspace/data/source/slamadverseriallab",
                "-v", f"{output_abs}:/workspace/data/output",
                "-v", f"{particles_abs}:/workspace/data/particles",
                "-v", f"{config_abs}:/workspace/config/slamadverseriallab.py",
                "-v", f"{simulation_py_abs}:/workspace/tools/simulation.py",
                "rain-rendering:latest",
                "python3", "main.py",
                "--dataset", "slamadverseriallab",
                "--intensity", str(self.intensity)
            ]

            if self.max_frames is not None:
                cmd.extend(["--frame_end", str(self.max_frames)])
                logger.info(f"Limiting to first {self.max_frames} frames")

            logger.info(f"Command: {' '.join(cmd)}")

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1  # Line buffered
                )

                # Stream output line by line
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        logger.info(f"[rain-rendering] {line}")

                process.wait()

                if process.returncode != 0:
                    logger.error(f"Rain-rendering failed for {camera_role} with return code {process.returncode}")
                    raise RuntimeError(f"Rain-rendering subprocess failed for {camera_role}")

                logger.info(f"Rain-rendering completed successfully for {camera_role}")

                output_base = self.rain_rendering_path / "data" / "output" / "slamadverseriallab" / source_path.name
                temp_rainy_dir = output_base / "rain" / f"{self.intensity}mm" / "rainy_image"

                if not temp_rainy_dir.exists():
                    raise RuntimeError(f"Expected output directory not found for {camera_role}: {temp_rainy_dir}")

                # Move to camera-specific directory to avoid overwriting
                camera_rainy_dir = output_base / "rain" / f"{self.intensity}mm" / camera_role / "rainy_image"
                camera_rainy_dir.parent.mkdir(parents=True, exist_ok=True)

                # Move the output directory
                import shutil
                if camera_rainy_dir.exists():
                    shutil.rmtree(camera_rainy_dir)
                shutil.move(str(temp_rainy_dir), str(camera_rainy_dir))

                self._rainy_frames_dirs[camera_role] = camera_rainy_dir
                logger.info(f"Rainy frames for {camera_role} available at: {camera_rainy_dir}")

                # Clean up config file
                config_path.unlink(missing_ok=True)
                logger.debug(f"Removed config file: {config_path}")

            except subprocess.TimeoutExpired:
                logger.error(f"Rain-rendering timed out after 3 hours for {camera_role}")
                raise RuntimeError(f"Rain-rendering subprocess timed out for {camera_role}")
            except Exception as e:
                logger.error(f"Error running rain-rendering for {camera_role}: {e}")
                raise

        logger.info(f"All cameras processed successfully!")

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        """Apply physics-based rain by loading pre-rendered frame.

        Args:
            image: Input RGB image (H, W, 3) in range [0, 255]
            depth: Depth map (unused, depth from files is used)
            frame_idx: Current frame index
            camera: Camera role ("left" or "right")
            **kwargs: Additional context (unused)

        Returns:
            Rain-affected image from rain-rendering output
        """
        # Ensure setup is complete
        if not self._depth_setup_complete:
            raise RuntimeError(
                f"RainModule '{self.name}' setup not complete. "
                "Depth backend resolution must complete during setup(context). "
                "This should happen automatically - please report this as a bug."
            )

        # Run rain-rendering on first frame
        if not self._rain_rendered:
            logger.info("First frame - running rain-rendering for all frames...")
            self._run_rain_rendering()
            self._rain_rendered = True

        source_path = self._get_source_path()
        image_files = sorted(self.get_camera_directory_path(source_path, self.cameras[0]).glob("*.png"))
        if frame_idx >= len(image_files):
            logger.warning(f"Frame index {frame_idx} out of range")
            return image.copy()

        frame_name = image_files[frame_idx].name

        camera_role = camera

        if camera_role not in self.cameras:
            raise ValueError(
                f"Camera '{camera_role}' not in available cameras: {self.cameras}. "
                f"Rain rendering was only performed for: {list(self._rainy_frames_dirs.keys())}"
            )

        logger.debug(f"Frame {frame_idx}: Using camera role {camera_role}")

        if camera_role not in self._rainy_frames_dirs:
            raise RuntimeError(
                f"No rainy output directory for {camera_role}. "
                f"Rain rendering may have failed. Available cameras: {list(self._rainy_frames_dirs.keys())}"
            )

        rainy_frame_path = self._rainy_frames_dirs[camera_role] / frame_name

        if not rainy_frame_path.exists():
            raise FileNotFoundError(
                f"Rainy frame not found: {rainy_frame_path}. "
                f"Rain rendering may have failed or produced incomplete output. "
                f"Check rain-rendering logs above for errors."
            )

        rainy_image = cv2.imread(str(rainy_frame_path))

        if rainy_image is None:
            raise RuntimeError(
                f"Failed to read rainy frame: {rainy_frame_path}. "
                f"File may be corrupted or unreadable. Check file permissions and integrity."
            )

        rainy_image = cv2.cvtColor(rainy_image, cv2.COLOR_BGR2RGB)

        logger.debug(f"Frame {frame_idx}: Loaded rainy frame from {camera_role}/{frame_name}")

        return rainy_image

    def _cleanup(self) -> None:
        """Cleanup temporary files."""
        logger.info("Cleaning up RainModule...")

        # Depth state cleanup
        self._cleanup_depth()

        # Remove dataset symlink
        symlink_path = self.rain_rendering_path / "data" / "source" / "slamadverseriallab"
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
            logger.info(f"Removed dataset symlink: {symlink_path}")

        # Remove any remaining camera-specific config files
        config_dir = self.rain_rendering_path / "config"
        if config_dir.exists():
            for config_file in config_dir.glob("slamadverseriallab_*.py"):
                config_file.unlink()
                logger.debug(f"Removed config file: {config_file}")

        logger.debug("RainModule cleanup complete")
