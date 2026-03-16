"""PySLAM runner."""

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import yaml

from ..utils.paths import create_temp_dir

if TYPE_CHECKING:
    from ..datasets.base import Dataset

logger = logging.getLogger(__name__)


class PySLAMRunner:
    """Run PySLAM via subprocess."""

    # Feature configs compatible with ORB vocabulary (no independent loop detector needed)
    ORB_COMPATIBLE_FEATURES = {"ORB", "ORB2", "ORB2_FREAK", "ORB2_BEBLID", "FAST_ORB", "SHI_TOMASI_ORB"}

    def __init__(self, pyslam_path: Optional[Path] = None, use_slam: bool = True):
        """Initialize PySLAM runner.

        Args:
            pyslam_path: Path to PySLAM installation. Defaults to deps/slam-frameworks/pyslam.
            use_slam: If True, use main_slam.py (full SLAM with loop closing).
                     If False, use main_vo.py (visual odometry only).
        """
        if pyslam_path is None:
            # Default: deps/slam-frameworks/pyslam relative to this file
            self.pyslam_path = Path(__file__).parent.parent.parent / "deps" / "slam-frameworks" / "pyslam"
        else:
            self.pyslam_path = Path(pyslam_path)

        if not self.pyslam_path.exists():
            raise FileNotFoundError(f"PySLAM not found at: {self.pyslam_path}")

        self.use_slam = use_slam
        self._temp_dirs: List[Path] = []

    def run(
        self,
        dataset: "Dataset",
        perturbed_images_path: Optional[Path],
        output_dir: Path,
        feature_config: str,
        camera_settings: str,
        max_frames: Optional[int] = None,
    ) -> Optional[Path]:
        """Run PySLAM on a dataset and return trajectory path.

        Args:
            dataset: Dataset object (knows its own structure)
            perturbed_images_path: Path to perturbed images, or None for baseline
            output_dir: Directory for PySLAM outputs
            feature_config: Feature tracker config name (e.g., 'ORB2', 'SIFT', 'SUPERPOINT')
            camera_settings: Camera settings filename (e.g., 'KITTI04-12.yaml', 'TUM1.yaml')
            max_frames: Maximum number of frames to process (None = all frames)

        Returns:
            Path to generated trajectory file, or None if failed
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if perturbed_images_path is None:
            # Baseline: use original dataset
            images_path = dataset.path
        else:
            images_path = perturbed_images_path

        # 1. Create dataset structure for PySLAM
        logger.info(f"Creating dataset structure for {dataset.dataset_type}...")
        temp_dataset = self._create_pyslam_structure(
            dataset=dataset,
            images_path=images_path,
            max_frames=max_frames,
        )

        # 2. Generate modified settings file with feature config
        logger.info(f"Generating settings file with feature config: {feature_config}")
        settings_path = self._generate_settings_file(
            base_settings=camera_settings,
            feature_config=feature_config,
            output_dir=output_dir,
        )

        # 3. Generate PySLAM config.yaml
        logger.info("Generating PySLAM config.yaml...")
        config_path = self._generate_config(
            dataset=dataset,
            dataset_path=temp_dataset,
            output_dir=output_dir,
            settings_file=settings_path,
            feature_config=feature_config,
            max_frames=max_frames,
        )

        # 4. Execute PySLAM via subprocess
        logger.info(f"Executing PySLAM with config: {config_path}")
        trajectory = self._execute(config_path, output_dir)

        return trajectory

    def _create_pyslam_structure(
        self,
        dataset: "Dataset",
        images_path: Path,
        max_frames: Optional[int] = None,
    ) -> Path:
        """Create dataset structure that PySLAM expects.

        Delegates to the dataset's create_pyslam_structure() method which knows
        how to arrange files in the format PySLAM expects for that dataset type.

        Args:
            dataset: Dataset object
            images_path: Path to images (original or perturbed)
            max_frames: Maximum number of frames (None = all)

        Returns:
            Path to temporary dataset root for PySLAM
        """
        dataset_type = dataset.dataset_type.lower()
        temp_root = create_temp_dir(prefix=f"pyslam_{dataset_type}_")
        self._temp_dirs.append(temp_root)

        return dataset.create_pyslam_structure(images_path, temp_root, max_frames)

    def _generate_settings_file(
        self,
        base_settings: str,
        feature_config: str,
        output_dir: Path,
    ) -> Path:
        """Create modified camera settings with desired feature config.

        Copies the base settings file from pyslam and sets FeatureTrackerConfig.name
        to the desired feature configuration.

        Args:
            base_settings: Base settings filename (e.g., 'KITTI04-12.yaml')
            feature_config: Feature tracker config name (e.g., 'ORB2', 'SIFT')
            output_dir: Output directory for modified settings

        Returns:
            Path to modified settings file
        """
        base_path = self.pyslam_path / "settings" / base_settings

        if not base_path.exists():
            raise FileNotFoundError(f"Camera settings not found: {base_path}")

        with open(base_path) as f:
            content = f.read()

        if re.search(r"FeatureTrackerConfig\.name:", content):
            # Replace existing value
            content = re.sub(
                r"FeatureTrackerConfig\.name:\s*\w+",
                f"FeatureTrackerConfig.name: {feature_config}",
                content,
            )
        else:
            content = re.sub(
                r"(FeatureTrackerConfig\.nFeatures:\s*\d+)",
                f"\\1\nFeatureTrackerConfig.name: {feature_config}",
                content,
            )

        # For non-ORB features, use independent loop detector with ORB2
        # This allows SUPERPOINT, SIFT, etc. to work with the ORB vocabulary
        if feature_config.upper() not in self.ORB_COMPATIBLE_FEATURES:
            if re.search(r"LoopDetectionConfig\.name:", content):
                content = re.sub(
                    r"LoopDetectionConfig\.name:\s*\w+",
                    "LoopDetectionConfig.name: DBOW3_INDEPENDENT",
                    content,
                )
            else:
                content += "\n# Use independent ORB2 loop detector for non-ORB features\n"
                content += "LoopDetectionConfig.name: DBOW3_INDEPENDENT\n"
            logger.info(f"Using DBOW3_INDEPENDENT loop detector for non-ORB feature: {feature_config}")

        modified_path = output_dir / f"settings_{feature_config}.yaml"
        with open(modified_path, "w") as f:
            f.write(content)

        logger.debug(f"Generated settings file: {modified_path}")
        return modified_path

    def _generate_config(
        self,
        dataset: "Dataset",
        dataset_path: Path,
        output_dir: Path,
        settings_file: Path,
        feature_config: str,
        max_frames: Optional[int] = None,
    ) -> Path:
        """Generate PySLAM config.yaml.

        Args:
            dataset: Dataset object
            dataset_path: Path to prepared dataset structure
            output_dir: Output directory for trajectories
            settings_file: Path to modified settings file
            feature_config: Feature tracker config name (e.g., 'ORB2', 'SUPERPOINT')
            max_frames: Maximum number of frames to process (None = all)

        Returns:
            Path to generated config.yaml
        """
        dataset_type = dataset.dataset_type.lower()
        dataset_type_upper = dataset_type.upper()
        sensor_type = "stereo" if "right" in dataset.get_active_camera_roles() else "mono"
        sequence_name = dataset.sequence_name

        dataset_config = {
            "type": dataset_type,
            "sensor_type": sensor_type,
            "base_path": str(dataset_path),
            "name": sequence_name,
            "settings": str(settings_file),
            "groundtruth_file": "auto",
        }

        if max_frames is not None:
            dataset_config["max_frames"] = max_frames

        config = {
            "DATASET": {"type": f"{dataset_type_upper}_DATASET"},
            f"{dataset_type_upper}_DATASET": dataset_config,
            "SAVE_TRAJECTORY": {
                "save_trajectory": True,
                "format_type": "tum",
                "output_folder": str(output_dir),
                "basename": "trajectory",
            },
            "SYSTEM_STATE": {
                "load_state": False,
                "folder_path": str(output_dir / "slam_state"),
            },
            "GLOBAL_PARAMETERS": {},
        }

        if dataset_type == "kitti":
            config[f"{dataset_type_upper}_DATASET"]["is_color"] = True
        elif dataset_type == "tum":
            config[f"{dataset_type_upper}_DATASET"]["associations"] = "associations.txt"

        config_path = output_dir / "pyslam_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.debug(f"Generated config: {config_path}")
        return config_path

    def _execute(self, config_path: Path, output_dir: Path) -> Optional[Path]:
        """Execute PySLAM via subprocess in headless mode.

        Args:
            config_path: Path to PySLAM config.yaml
            output_dir: Output directory for results

        Returns:
            Path to trajectory file, or None if failed
        """
        # Choose script based on use_slam setting
        script_name = "main_slam.py" if self.use_slam else "main_vo.py"

        cmd = [
            "conda",
            "run",
            "-n",
            "pyslam",
            "--cwd",
            str(self.pyslam_path),
            "--no-capture-output",
            "python",
            script_name,
            "-c",
            str(config_path),
            "--headless",
        ]

        # main_slam.py supports --no_output_date, main_vo.py does not
        if self.use_slam:
            cmd.append("--no_output_date")

        logger.info(f"Running PySLAM: conda run -n pyslam python {script_name} -c {config_path} --headless" +
                    (" --no_output_date" if self.use_slam else ""))

        import os
        env = os.environ.copy()

        # Custom OpenCV install directory (contains proper Python 3.11 bindings)
        opencv_install_path = self.pyslam_path / "thirdparty" / "opencv" / "install"
        opencv_lib_path = opencv_install_path / "lib"
        opencv_python_path = opencv_install_path / "lib" / "python3.11" / "site-packages"

        # Pangolin visualization library
        pangolin_lib_path = self.pyslam_path / "thirdparty" / "pangolin" / "build" / "src"

        conda_lib_path = Path(os.environ.get("CONDA_PREFIX", "")) / "lib"
        existing_ld_path = env.get("LD_LIBRARY_PATH", "")
        existing_python_path = env.get("PYTHONPATH", "")

        env["PYTHONPATH"] = f"{opencv_python_path}:{existing_python_path}"
        env["LD_LIBRARY_PATH"] = f"{opencv_lib_path}:{pangolin_lib_path}:{conda_lib_path}:{existing_ld_path}"

        try:
            result = subprocess.run(
                cmd,
                timeout=3600,  # 1 hour timeout
                env=env,
            )

            if result.returncode != 0:
                logger.error(f"PySLAM execution failed with return code {result.returncode}")
                return None

            track_stats_files = list(output_dir.glob("track_stats_*.json"))
            if track_stats_files:
                logger.info(f"PySLAM execution completed successfully")
                logger.info(f"Track stats saved: {track_stats_files[0]}")
                return track_stats_files[0]
            else:
                logger.warning(f"No track stats file found in {output_dir}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("PySLAM execution timed out after 1 hour")
            return None
        except Exception as e:
            logger.error(f"PySLAM execution failed: {e}")
            return None

    def cleanup(self) -> None:
        """Clean up temporary directories."""
        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp dir: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {temp_dir}: {e}")
        self._temp_dirs.clear()
