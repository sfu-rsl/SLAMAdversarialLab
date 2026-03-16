"""ORB-SLAM3 algorithm implementation for SLAMAdverserialLab."""

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..datasets.associations import (
    generate_tum_association_with_associate_py,
    resolve_tum_association_for_orbslam3,
)
from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


class ORBSLAM3Algorithm(SLAMAlgorithm):
    """ORB-SLAM3 SLAM algorithm via Docker container."""

    def __init__(self):
        self.docker_image = "orbslam3:latest"

    @property
    def name(self) -> str:
        return "orbslam3"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {
            "kitti": ["mono", "stereo"],
            "tum": ["mono", "rgbd"],
            "euroc": ["stereo"],
        }

    def resolve_config_name(self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None) -> Optional[str]:
        if dataset_type.lower() == "kitti":
            seq_num = int(sequence)
            if seq_num <= 2:
                return "KITTI00-02.yaml"
            elif seq_num == 3:
                return "KITTI03.yaml"
            else:
                return "KITTI04-12.yaml"
        elif dataset_type.lower() == "tum":
            seq_lower = str(sequence).lower()
            if "freiburg1" in seq_lower or "fr1" in seq_lower:
                return "TUM1.yaml"
            elif "freiburg2" in seq_lower or "fr2" in seq_lower:
                return "TUM2.yaml"
            elif "freiburg3" in seq_lower or "fr3" in seq_lower:
                return "TUM3.yaml"
        elif dataset_type.lower() == "euroc":
            # All EuRoC sequences use the same config
            return "EuRoC.yaml"
        return None

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """ORB-SLAM3 internal configs are container-relative, not host file paths."""
        return None

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate ORB-SLAM3 runtime dependencies."""
        if shutil.which("docker") is None:
            raise RuntimeError("ORB-SLAM3 requires Docker, but 'docker' was not found in PATH.")

        image_check = subprocess.run(
            ["docker", "image", "inspect", self.docker_image],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if image_check.returncode != 0:
            raise RuntimeError(
                f"ORB-SLAM3 Docker image not found: {self.docker_image}. "
                "Build or pull the image before running evaluation."
            )

        dataset_type = request.dataset_type.lower()

        if dataset_type == "kitti":
            required_roles = ["left"]
            if request.sensor_mode == SensorMode.STEREO:
                required_roles.append("right")
            self._require_camera_dirs(request, required_roles)
            self._require_camera_paths(request, required_roles)

        if dataset_type == "euroc":
            required_roles = ["left", "right"]
            self._require_camera_dirs(request, required_roles)
            self._require_camera_paths(request, required_roles)
            self._require_timestamps_by_frame(request)

        if dataset_type == "tum":
            association_file = self._find_association_file(
                request.dataset_path,
                generate_if_missing=False,
            )
            if association_file is None:
                raise RuntimeError(
                    "ORB-SLAM3 requires an existing TUM association file in dataset root "
                    f"(accepted names: associations.txt, association.txt, associate.txt, assoc.txt). "
                    f"None found in {request.dataset_path}."
                )

    def _require_camera_dirs(
        self,
        request: SLAMRunRequest,
        required_roles: List[str],
    ) -> Dict[str, str]:
        """Validate and return runtime camera directory names from request extras."""
        raw_camera_dirs = request.extras.get("camera_dirs")
        if not isinstance(raw_camera_dirs, dict) or not raw_camera_dirs:
            raise RuntimeError(
                "ORB-SLAM3 requires request.extras['camera_dirs'] with dataset-resolved camera directory names."
            )

        camera_dirs: Dict[str, str] = {}
        for role in required_roles:
            directory_name = raw_camera_dirs.get(role)
            if not isinstance(directory_name, str) or not directory_name.strip():
                raise RuntimeError(
                    f"ORB-SLAM3 missing required camera_dirs mapping for role '{role}'."
                )
            camera_dirs[role] = directory_name.strip()

        return camera_dirs

    def _require_camera_paths(
        self,
        request: SLAMRunRequest,
        required_roles: List[str],
    ) -> Dict[str, Path]:
        """Validate and return runtime camera paths from request extras."""
        raw_camera_paths = request.extras.get("camera_paths")
        if not isinstance(raw_camera_paths, dict) or not raw_camera_paths:
            raise RuntimeError(
                "ORB-SLAM3 requires request.extras['camera_paths'] with dataset-resolved camera directory paths."
            )

        camera_paths: Dict[str, Path] = {}
        for role in required_roles:
            path_value = raw_camera_paths.get(role)
            if not isinstance(path_value, str) or not path_value.strip():
                raise RuntimeError(
                    f"ORB-SLAM3 missing required camera_paths mapping for role '{role}'."
                )
            camera_path = Path(path_value).resolve()
            if not camera_path.exists() or not camera_path.is_dir():
                raise RuntimeError(
                    f"ORB-SLAM3 camera path for role '{role}' does not exist or is not a directory: {camera_path}"
                )
            camera_paths[role] = camera_path

        return camera_paths

    def _require_timestamps_by_frame(self, request: SLAMRunRequest) -> Dict[int, int]:
        """Validate and return EuRoC timestamps from request extras."""
        raw_timestamps = request.extras.get("timestamps_by_frame")
        if not isinstance(raw_timestamps, dict) or not raw_timestamps:
            raise RuntimeError(
                "ORB-SLAM3 requires request.extras['timestamps_by_frame'] for EuRoC execution."
            )

        normalized: Dict[int, int] = {}
        for frame_idx, timestamp in raw_timestamps.items():
            if not isinstance(frame_idx, int):
                raise RuntimeError(
                    f"ORB-SLAM3 invalid timestamps_by_frame key {frame_idx!r}; expected int frame index."
                )
            if frame_idx < 0:
                raise RuntimeError(
                    f"ORB-SLAM3 invalid timestamps_by_frame key {frame_idx}; frame indices must be >= 0."
                )

            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise RuntimeError(
                    f"ORB-SLAM3 invalid timestamp value for frame {frame_idx}: {timestamp!r}."
                )

            if isinstance(timestamp, float):
                if not timestamp.is_integer():
                    raise RuntimeError(
                        f"ORB-SLAM3 requires integer-like EuRoC timestamps; frame {frame_idx} has {timestamp!r}."
                    )
                timestamp_value = int(timestamp)
            else:
                timestamp_value = int(timestamp)

            normalized[frame_idx] = timestamp_value

        expected_indices = set(range(len(normalized)))
        actual_indices = set(normalized.keys())
        if actual_indices != expected_indices:
            missing = sorted(expected_indices - actual_indices)
            extra = sorted(actual_indices - expected_indices)
            raise RuntimeError(
                "ORB-SLAM3 timestamps_by_frame must cover contiguous frame indices 0..N-1 "
                f"(missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
                f"extra={extra[:5]}{'...' if len(extra) > 5 else ''})."
            )

        previous = None
        for frame_idx in range(len(normalized)):
            current = normalized[frame_idx]
            if previous is not None and current <= previous:
                raise RuntimeError(
                    "ORB-SLAM3 timestamps_by_frame must be strictly increasing by frame index: "
                    f"frame {frame_idx - 1}={previous}, frame {frame_idx}={current}."
                )
            previous = current

        return normalized

    def _stage_dataset(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Prepare dataset structure expected by ORB-SLAM3 executables."""
        dataset_path = request.dataset_path
        dataset_type = request.dataset_type.lower()
        is_stereo = request.sensor_mode == SensorMode.STEREO

        if dataset_type == "kitti":
            required_roles = ["left"]
            if is_stereo:
                required_roles.append("right")
            camera_dirs = self._require_camera_dirs(request, required_roles)
            camera_paths = self._require_camera_paths(request, required_roles)
            logger.info(
                "  Creating staged KITTI runtime structure from request contracts "
                "(no source dataset mutation)."
            )
            staged_path, kitti_image_mounts = self._create_kitti_structure(
                dataset_path=dataset_path,
                camera_paths=camera_paths,
                camera_dirs=camera_dirs,
                is_stereo=is_stereo,
            )
            ctx.staging_artifacts["kitti_temp_dir"] = staged_path
            ctx.staging_artifacts["kitti_image_mounts"] = kitti_image_mounts
            return staged_path

        if dataset_type == "euroc":
            camera_paths = self._require_camera_paths(request, ["left", "right"])
            camera_dirs = self._require_camera_dirs(request, ["left", "right"])
            timestamps_by_frame = self._require_timestamps_by_frame(request)
            logger.info("  Creating EuRoC runtime structure from request extras...")
            staged_path, euroc_image_mounts = self._create_euroc_structure(
                dataset_path=dataset_path,
                camera_paths=camera_paths,
                camera_dirs=camera_dirs,
                timestamps_by_frame=timestamps_by_frame,
            )
            ctx.staging_artifacts["euroc_temp_dir"] = staged_path
            ctx.staging_artifacts["euroc_image_mounts"] = euroc_image_mounts
            return staged_path

        if dataset_type == "tum":
            association_file = self._find_association_file(
                dataset_path,
                generate_if_missing=False,
            )
            if association_file is None:
                logger.error("  No association file found in TUM dataset")
                return None
            ctx.staging_artifacts["association_file"] = association_file
            return dataset_path

        return dataset_path

    def _cleanup_staged_dataset(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Clean temporary staging directories."""
        for artifact_key in ("kitti_temp_dir", "euroc_temp_dir"):
            temp_dir = ctx.staging_artifacts.get(artifact_key)
            if isinstance(temp_dir, Path) and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"  Cleaned up temp directory: {temp_dir}")

    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, object]]:
        """Build resolved runtime inputs used by ORB-SLAM3 execution."""
        return {
            "dataset_path": ctx.effective_dataset_path or request.dataset_path,
            "output_dir": request.output_dir,
            "dataset_type": request.dataset_type,
            "is_stereo": request.sensor_mode == SensorMode.STEREO,
            "sequence_name": ctx.sequence_name,
            "is_external": ctx.config_is_external,
            "kitti_image_mounts": ctx.staging_artifacts.get("kitti_image_mounts"),
            "euroc_image_mounts": ctx.staging_artifacts.get("euroc_image_mounts"),
        }

    def _build_execution_spec(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[ExecutionSpec]:
        inputs = ctx.execution_inputs
        dataset_path = inputs["dataset_path"]
        output_dir = inputs["output_dir"]
        dataset_type = inputs["dataset_type"]
        is_stereo = inputs["is_stereo"]
        is_external = inputs["is_external"]
        kitti_image_mounts = inputs["kitti_image_mounts"]
        euroc_image_mounts = inputs["euroc_image_mounts"]
        output_dir.mkdir(parents=True, exist_ok=True)

        if dataset_type == "kitti":
            if is_stereo:
                executable = "./Examples/Stereo/stereo_kitti"
                config_dir = "Examples/Stereo"
            else:
                executable = "./Examples/Monocular/mono_kitti"
                config_dir = "Examples/Monocular"
        elif dataset_type == "tum":
            if is_stereo:
                executable = "./Examples/Stereo/stereo_tum"
                config_dir = "Examples/Stereo"
            else:
                executable = "./Examples/RGB-D/rgbd_tum"
                config_dir = "Examples/RGB-D"
        elif dataset_type == "euroc":
            executable = "./Examples/Stereo/stereo_euroc"
            config_dir = "Examples/Stereo"
        else:
            logger.error(f"Unsupported dataset type: {dataset_type}")
            return None

        logger.info(f"  Using ORB-SLAM3 directory: {config_dir}")

        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{dataset_path.resolve()}:/dataset:ro",
            "-v", f"{output_dir.resolve()}:/output",
        ]

        # For KITTI, mount dataset-resolved camera paths to ORB-SLAM3 expected paths.
        if dataset_type == "kitti" and kitti_image_mounts:
            for host_path, container_path in kitti_image_mounts:
                docker_cmd.extend(["-v", f"{host_path}:{container_path}:ro"])
                logger.debug(f"  Mounting KITTI camera dir: {host_path} -> {container_path}")

        # For EuRoC perturbed datasets, mount the image directories at the expected paths
        if dataset_type == "euroc" and euroc_image_mounts:
            for host_path, container_path in euroc_image_mounts:
                docker_cmd.extend(["-v", f"{host_path}:{container_path}:ro"])
                logger.debug(f"  Mounting image dir: {host_path} -> {container_path}")

        # For TUM datasets, check if depth is a symlink and mount the target
        if dataset_type == "tum":
            depth_path = dataset_path / "depth"
            if depth_path.is_symlink():
                depth_target = depth_path.resolve()
                if depth_target.exists():
                    docker_cmd.extend(["-v", f"{depth_target}:/dataset/depth:ro"])
                    logger.info(f"  Mounting depth directory: {depth_target}")

        if is_external:
            slam_config_resolved = ctx.resolved_config_path
            if slam_config_resolved is None:
                logger.error("External config path is not resolved in runtime context")
                return None
            config_path_in_container = f"{config_dir}/{slam_config_resolved.name}"
            config_abs_path = f"/orbslam3/{config_path_in_container}"
            docker_cmd.extend(["-v", f"{slam_config_resolved}:{config_abs_path}:ro"])
            logger.info(f"  Using external config: {slam_config_resolved}")
        else:
            config_abs_path = f"{config_dir}/{ctx.internal_config_name}"
            logger.info(f"  Using internal config: {config_abs_path}")

        docker_cmd.append(self.docker_image)

        if dataset_type == "tum":
            association_file = ctx.staging_artifacts.get("association_file")
            if association_file:
                assoc_arg = f"/dataset/{association_file}"
                logger.info(f"  Using association file: {association_file}")
            else:
                logger.error("  No association file found in TUM dataset")
                return None

            bash_cmd = (
                f"xvfb-run -a {executable} Vocabulary/ORBvoc.txt {config_abs_path} /dataset {assoc_arg}; "
                f"cp CameraTrajectory.txt /output/ 2>/dev/null; "
                f"cp KeyFrameTrajectory.txt /output/ 2>/dev/null; "
                f"ls /output/*.txt 2>/dev/null || echo 'No trajectory files generated'"
            )
        elif dataset_type == "euroc":
            timestamps_file = "/dataset/orbslam3_timestamps.txt"
            logger.info("  Using staged EuRoC timestamps file: /dataset/orbslam3_timestamps.txt")

            bash_cmd = (
                f"xvfb-run -a {executable} Vocabulary/ORBvoc.txt {config_abs_path} /dataset {timestamps_file}; "
                f"cp CameraTrajectory.txt /output/ 2>/dev/null; "
                f"cp KeyFrameTrajectory.txt /output/ 2>/dev/null; "
                f"ls /output/*.txt 2>/dev/null || echo 'No trajectory files generated'"
            )
        else:
            # KITTI format
            bash_cmd = (
                f"xvfb-run -a {executable} Vocabulary/ORBvoc.txt {config_abs_path} /dataset; "
                f"cp CameraTrajectory.txt /output/ 2>/dev/null; "
                f"cp KeyFrameTrajectory.txt /output/ 2>/dev/null; "
                f"ls /output/*.txt 2>/dev/null || echo 'No trajectory files generated'"
            )

        docker_cmd.extend(["bash", "-c", bash_cmd])

        logger.info(f"  Executing: {executable}")

        return ExecutionSpec(
            cmd=docker_cmd,
            stream_output=False,
            log_prefix="ORB-SLAM3",
        )

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        spec = self._build_execution_spec(request, ctx)
        if spec is None:
            return False

        try:
            result_returncode = self._run_execution_spec(spec)

            if result_returncode != 0:
                logger.warning(f"  ORB-SLAM3 exited with return code {result_returncode}")
            return result_returncode == 0
        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return False

    def _find_raw_trajectory(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Find raw trajectory output from ORB-SLAM3."""
        output_dir = request.output_dir
        camera_traj_path = output_dir / "CameraTrajectory.txt"
        keyframe_traj_path = output_dir / "KeyFrameTrajectory.txt"

        if camera_traj_path.exists():
            logger.info(f"  Camera trajectory saved to: {camera_traj_path}")
            return camera_traj_path

        if keyframe_traj_path.exists():
            logger.warning(f"  Only keyframe trajectory available (possible tracking loss): {keyframe_traj_path}")
            return keyframe_traj_path

        logger.error(f"  No trajectory files found in {output_dir}")
        return None

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Path,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """ORB-SLAM3 trajectory files are already TUM-compatible."""
        return raw_trajectory

    def cleanup(self) -> None:
        """Stop and remove any ORB-SLAM3 Docker containers."""
        logger.debug("Cleaning up ORB-SLAM3 containers...")

        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "-q"],
                capture_output=True,
                text=True,
                timeout=10
            )

            all_containers = [cid for cid in result.stdout.strip().split('\n') if cid]
            if not all_containers:
                return

            orbslam_containers = []
            for cid in all_containers:
                inspect_result = subprocess.run(
                    ["docker", "inspect", "--format", "{{.Config.Image}}", cid],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "orbslam3" in inspect_result.stdout.lower():
                    orbslam_containers.append(cid)

            if orbslam_containers:
                logger.info(f"  Found {len(orbslam_containers)} ORB-SLAM3 container(s), removing...")
                for container_id in orbslam_containers:
                    subprocess.run(["docker", "stop", container_id], capture_output=True, timeout=10)
                    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=10)
                time.sleep(2)

        except subprocess.TimeoutExpired:
            logger.warning("  Docker cleanup timed out")
        except Exception as e:
            logger.warning(f"  Could not check Docker containers: {e}")

    def _find_association_file(self, dataset_path: Path, generate_if_missing: bool = False) -> Optional[str]:
        """Find the association file in a TUM dataset.

        Looks for common association file names used in TUM RGB-D datasets.
        Generation is disabled by default for strict fail-fast runtime behavior.

        Args:
            dataset_path: Path to the TUM dataset directory
            generate_if_missing: Forwarded to shared resolver; defaults to False.

        Returns:
            Filename of association file if found/generated, None otherwise
        """
        assoc_path = resolve_tum_association_for_orbslam3(
            dataset_path=dataset_path,
            generate_if_missing=generate_if_missing,
            log=logger,
        )
        return assoc_path.name if assoc_path else None

    def _generate_association_file(
        self, dataset_path: Path, rgb_file: Path, depth_file: Path, max_diff: float = 0.02
    ) -> Optional[Path]:
        """Generate association file using TUM's official associate.py script.

        Args:
            dataset_path: Path to dataset directory
            rgb_file: Path to rgb.txt
            depth_file: Path to depth.txt
            max_diff: Maximum timestamp difference for matching (seconds)

        Returns:
            Path to generated associations.txt, or None on failure
        """
        return generate_tum_association_with_associate_py(
            dataset_path=dataset_path,
            rgb_file=rgb_file,
            depth_file=depth_file,
            max_diff=max_diff,
            log=logger,
        )

    def _create_kitti_structure(
        self,
        dataset_path: Path,
        camera_paths: Dict[str, Path],
        camera_dirs: Dict[str, str],
        is_stereo: bool,
    ) -> tuple[Path, List[Tuple[str, str]]]:
        """Create a run-scoped KITTI staging root without mutating source dataset."""
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="kitti_struct_"))
        self._copy_top_level_files(dataset_path, temp_dir)
        (temp_dir / "image_0").mkdir(parents=True, exist_ok=True)

        image_mounts: List[Tuple[str, str]] = [
            (str(camera_paths["left"].resolve()), "/dataset/image_0"),
        ]
        if is_stereo:
            (temp_dir / "image_1").mkdir(parents=True, exist_ok=True)
            image_mounts.append((str(camera_paths["right"].resolve()), "/dataset/image_1"))

        logger.info(
            "  Created KITTI structure at %s (left_dir=%s%s)",
            temp_dir,
            camera_dirs["left"],
            f", right_dir={camera_dirs['right']}" if is_stereo else "",
        )
        return temp_dir, image_mounts

    def _copy_top_level_files(self, source_root: Path, destination_root: Path) -> None:
        """Copy top-level dataset files into a staging root."""
        destination_root.mkdir(parents=True, exist_ok=True)
        for entry in source_root.iterdir():
            if not entry.is_file():
                continue
            shutil.copy2(entry, destination_root / entry.name)

    def _create_euroc_structure(
        self,
        dataset_path: Path,
        camera_paths: Dict[str, Path],
        camera_dirs: Dict[str, str],
        timestamps_by_frame: Dict[int, int],
    ) -> tuple[Path, List[Tuple[str, str]]]:
        """Create EuRoC runtime structure using request camera/timestamp contracts."""
        import tempfile

        left_images_dir = camera_paths["left"]
        right_images_dir = camera_paths["right"]
        left_images = self._list_camera_images(left_images_dir)
        right_images = self._list_camera_images(right_images_dir)

        if len(left_images) != len(right_images):
            raise RuntimeError(
                "ORB-SLAM3 EuRoC staging requires synchronized stereo image counts "
                f"(left={len(left_images)}, right={len(right_images)})."
            )

        expected_frames = len(timestamps_by_frame)
        if len(left_images) != expected_frames:
            raise RuntimeError(
                "ORB-SLAM3 EuRoC timestamp mapping does not match image count "
                f"(timestamps={expected_frames}, images={len(left_images)})."
            )

        temp_dir = Path(tempfile.mkdtemp(prefix="euroc_struct_"))
        cam0_dir = temp_dir / "mav0" / "cam0" / "data"
        cam1_dir = temp_dir / "mav0" / "cam1" / "data"
        cam0_dir.mkdir(parents=True)
        cam1_dir.mkdir(parents=True)

        image_mounts = [
            (str(left_images_dir.resolve()), "/dataset/mav0/cam0/data"),
            (str(right_images_dir.resolve()), "/dataset/mav0/cam1/data"),
        ]

        self._write_orbslam_timestamps(
            timestamps_by_frame=timestamps_by_frame,
            output_path=temp_dir / "orbslam3_timestamps.txt",
        )
        self._generate_camera_data_csv(left_images, temp_dir / "mav0" / "cam0" / "data.csv")
        self._generate_camera_data_csv(right_images, temp_dir / "mav0" / "cam1" / "data.csv")

        cam0_sensor_src = self._copy_first_existing(
            [
                dataset_path / "cam0_sensor.yaml",
                dataset_path / "mav0" / "cam0" / "sensor.yaml",
                dataset_path / "sensor.yaml",
            ],
            temp_dir / "mav0" / "cam0" / "sensor.yaml",
        )
        cam1_sensor_src = self._copy_first_existing(
            [
                dataset_path / "cam1_sensor.yaml",
                dataset_path / "mav0" / "cam1" / "sensor.yaml",
                dataset_path / "sensor.yaml",
            ],
            temp_dir / "mav0" / "cam1" / "sensor.yaml",
        )

        logger.info(
            "  Created EuRoC structure at %s (left_dir=%s, right_dir=%s, left_mount=%s, right_mount=%s)",
            temp_dir,
            camera_dirs["left"],
            camera_dirs["right"],
            left_images_dir,
            right_images_dir,
        )
        if cam0_sensor_src is None or cam1_sensor_src is None:
            missing = []
            if cam0_sensor_src is None:
                missing.append("cam0")
            if cam1_sensor_src is None:
                missing.append("cam1")
            logger.warning(
                "  EuRoC sensor.yaml missing for %s; proceeding without one or more calibration files",
                ",".join(missing),
            )

        return temp_dir, image_mounts

    def _write_orbslam_timestamps(
        self,
        timestamps_by_frame: Dict[int, int],
        output_path: Path,
    ) -> None:
        """Write ORB-SLAM3 timestamps file from runtime frame-index mapping."""
        with open(output_path, "w", encoding="utf-8") as f:
            for frame_idx in range(len(timestamps_by_frame)):
                f.write(f"{timestamps_by_frame[frame_idx]}\n")
        logger.debug("  Generated ORB-SLAM3 timestamps file at %s", output_path)

    def _list_camera_images(self, image_dir: Path) -> List[Path]:
        """List camera images in deterministic order."""
        image_files = sorted(image_dir.glob("*.png"))
        if not image_files:
            image_files = sorted(image_dir.glob("*.jpg"))
        if not image_files:
            raise RuntimeError(f"ORB-SLAM3 found no PNG/JPG images in {image_dir}.")
        return image_files

    def _generate_camera_data_csv(self, image_files: List[Path], output_path: Path) -> None:
        """Generate camera data.csv from sorted image files."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("#timestamp [ns],filename\n")
            for img_file in image_files:
                f.write(f"{img_file.stem},{img_file.name}\n")

        logger.debug(f"  Generated camera data.csv at {output_path}")

    def _copy_first_existing(self, candidates: List[Path], destination: Path) -> Optional[Path]:
        """Copy first existing file from candidates to destination."""
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, destination)
                return candidate
        return None
