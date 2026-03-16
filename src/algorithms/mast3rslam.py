"""MASt3R-SLAM algorithm implementation."""

import logging
import os
import re
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


class MASt3RSLAMAlgorithm(SLAMAlgorithm):
    """MASt3R-SLAM via conda environment. Monocular dense SLAM with 3D reconstruction priors."""

    def __init__(self):
        self.mast3r_path = Path(__file__).parent.parent.parent / "deps" / "slam-algorithms" / "MASt3R-SLAM"
        self.conda_env = "mast3r-slam"

    @property
    def name(self) -> str:
        return "mast3rslam"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {
            "tum": ["mono"],
        }

    def resolve_config_name(self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None) -> Optional[str]:
        # eval_calib requires calibration which most perturbed datasets don't have
        return "base"

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        return self.mast3r_path / "config" / f"{ctx.internal_config_name}.yaml"

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate MASt3R-SLAM runtime dependencies."""
        if not self.mast3r_path.exists():
            raise RuntimeError(f"MASt3R-SLAM not found at {self.mast3r_path}")

        config_file = self._resolve_config_path(ctx)

        if config_file is None:
            raise RuntimeError("MASt3R-SLAM external config path is not resolved in runtime context")

        if not config_file.exists():
            raise RuntimeError(f"MASt3R-SLAM config file not found: {config_file}")

        if request.dataset_type.lower() == "tum":
            self._resolve_freiburg_id_from_sequence(ctx.sequence_name)

    def _stage_dataset(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Prepare dataset layout expected by MASt3R-SLAM."""
        return self._prepare_dataset(request, ctx.sequence_name)

    def _cleanup_staged_dataset(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Clean temporary symlinked TUM staging directories."""
        self._cleanup_temp_tum_link()

    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, object]]:
        """Build resolved runtime inputs used by MASt3R-SLAM execution."""
        prepared_path = ctx.effective_dataset_path
        config_file = self._resolve_config_path(ctx)
        if not prepared_path or config_file is None:
            return None
        return {
            "output_dir": request.output_dir,
            "is_stereo": request.sensor_mode == SensorMode.STEREO,
            "prepared_path": prepared_path,
            "config_file": config_file,
            "dataset_type": request.dataset_type,
            "sequence_name": ctx.sequence_name,
            "log_basenames": self._expected_log_basenames(request.dataset_type, ctx.sequence_name),
        }

    def _build_execution_spec(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[ExecutionSpec]:
        inputs = ctx.execution_inputs
        prepared_path = inputs["prepared_path"]
        config_file = inputs["config_file"]
        output_dir = inputs["output_dir"]
        log_basenames = inputs["log_basenames"]

        return ExecutionSpec(
            cmd=["mast3rslam"],
            custom_runner=lambda _spec, prepared_path=prepared_path, config_file=config_file, output_dir=output_dir, log_basenames=log_basenames: self._run_mast3rslam(
                prepared_path,
                config_file,
                output_dir,
                log_basenames,
            ),
            log_prefix="MASt3R-SLAM",
        )

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        inputs = ctx.execution_inputs
        output_dir = inputs["output_dir"]
        is_stereo = inputs["is_stereo"]
        config_file = inputs["config_file"]

        output_dir.mkdir(parents=True, exist_ok=True)

        if is_stereo:
            logger.warning("  MASt3R-SLAM only supports monocular mode. Stereo flag ignored.")

        logger.info(f"  Using config: {config_file}")

        spec = self._build_execution_spec(request, ctx)
        if spec is None:
            return False
        try:
            return self._run_execution_spec(spec) == 0
        except Exception as e:
            logger.error(f"Failed to run MASt3R-SLAM: {e}")
            return False

    def _prepare_dataset(
        self,
        request: SLAMRunRequest,
        sequence_name: str,
    ) -> Path:
        """Prepare dataset for MASt3R-SLAM.

        MASt3R-SLAM's TUM dataloader requires:
        - dataset path includes a 'tum' directory segment
        - dataset path includes 'freiburg{1,2,3}' marker
        - dataset root contains rgb.txt with entries like: "<timestamp> rgb/<filename>"
        - dataset root contains rgb/ images

        We enforce a deterministic, contract-driven staging path so we don't rely on
        source path probing or mutate input datasets.
        """
        logger.info("  Preparing dataset for MASt3R-SLAM...")
        dataset_type = request.dataset_type.lower()
        if dataset_type != "tum":
            raise ValueError(
                f"MASt3R-SLAM supports only TUM monocular datasets, got dataset_type='{request.dataset_type}'."
            )

        freiburg_id = self._resolve_freiburg_id_from_sequence(sequence_name)
        left_image_dir = self._require_left_camera_path(request)
        image_files = self._collect_sorted_image_files(left_image_dir)
        timestamps_by_frame = self._require_timestamps_by_frame(request)

        expected_frame_count = len(image_files)
        if len(timestamps_by_frame) != expected_frame_count:
            raise ValueError(
                "MASt3R-SLAM staging requires timestamp and image counts to match: "
                f"timestamps={len(timestamps_by_frame)}, images={expected_frame_count}."
            )

        stage_root = Path(tempfile.mkdtemp(prefix="tum_dataset_"))
        staged_dataset_path = (
            stage_root
            / "tum"
            / f"rgbd_dataset_{freiburg_id}_{self._sanitize_sequence_label(sequence_name)}"
        )
        staged_dataset_path.mkdir(parents=True, exist_ok=True)

        rgb_link = staged_dataset_path / "rgb"
        os.symlink(
            str(left_image_dir.resolve()),
            str(rgb_link),
            target_is_directory=True,
        )

        self._write_staged_rgb_txt(
            staged_dataset_path=staged_dataset_path,
            image_files=image_files,
            timestamps_by_frame=timestamps_by_frame,
        )

        logger.info("  Created MASt3R staged TUM dataset: %s", staged_dataset_path)
        self._temp_tum_link = staged_dataset_path
        return staged_dataset_path

    def _resolve_freiburg_id_from_sequence(self, sequence_name: str) -> str:
        """Resolve Freiburg variant from dataset.sequence for TUM runs.

        Accepted markers:
        - freiburg1 / fr1
        - freiburg2 / fr2
        - freiburg3 / fr3

        Raises:
            ValueError: if the sequence is missing or ambiguous.
        """
        sequence = (sequence_name or "").strip().lower()
        if not sequence:
            raise ValueError(
                "MASt3R-SLAM TUM calibration requires dataset.sequence to include Freiburg variant "
                "(freiburg1/freiburg2/freiburg3 or fr1/fr2/fr3)."
            )

        tokens = [t for t in re.split(r"[^a-z0-9]+", sequence) if t]
        variants: List[str] = []
        if "freiburg1" in tokens or "fr1" in tokens:
            variants.append("freiburg1")
        if "freiburg2" in tokens or "fr2" in tokens:
            variants.append("freiburg2")
        if "freiburg3" in tokens or "fr3" in tokens:
            variants.append("freiburg3")

        if len(variants) == 1:
            return variants[0]

        if not variants:
            raise ValueError(
                f"MASt3R-SLAM could not resolve Freiburg variant from dataset.sequence='{sequence_name}'. "
                "Use one of: freiburg1/freiburg2/freiburg3 (or fr1/fr2/fr3)."
            )

        raise ValueError(
            f"MASt3R-SLAM found ambiguous Freiburg markers in dataset.sequence='{sequence_name}': {variants}. "
            "Provide exactly one Freiburg variant (freiburg1/freiburg2/freiburg3)."
        )

    def _require_left_camera_path(self, request: SLAMRunRequest) -> Path:
        """Resolve and validate left camera path from runtime request extras."""
        camera_paths = request.extras.get("camera_paths")
        if not isinstance(camera_paths, dict):
            raise ValueError(
                "MASt3R-SLAM requires request.extras['camera_paths'] as a role->path mapping."
            )

        left_path_value = camera_paths.get("left")
        if not isinstance(left_path_value, str) or not left_path_value.strip():
            raise ValueError(
                "MASt3R-SLAM requires request.extras['camera_paths']['left']."
            )

        left_path = Path(left_path_value).resolve()
        if not left_path.exists() or not left_path.is_dir():
            raise ValueError(
                f"MASt3R-SLAM left camera directory does not exist: {left_path}"
            )
        return left_path

    def _collect_sorted_image_files(self, image_dir: Path) -> List[Path]:
        """Collect sorted image files from a camera directory."""
        image_files = sorted(
            list(image_dir.glob("*.png"))
            + list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.jpeg"))
        )
        if not image_files:
            raise ValueError(
                f"MASt3R-SLAM found no PNG/JPG/JPEG images under left camera directory: {image_dir}"
            )
        return image_files

    def _require_timestamps_by_frame(self, request: SLAMRunRequest) -> Dict[int, Union[int, float]]:
        """Resolve and validate frame-indexed timestamps from request extras."""
        raw_timestamps = request.extras.get("timestamps_by_frame")
        if not isinstance(raw_timestamps, dict) or not raw_timestamps:
            raise ValueError(
                "MASt3R-SLAM requires request.extras['timestamps_by_frame']."
            )

        normalized: Dict[int, Union[int, float]] = {}
        for frame_idx, timestamp in raw_timestamps.items():
            if not isinstance(frame_idx, int):
                raise ValueError(
                    f"MASt3R-SLAM invalid timestamps_by_frame key {frame_idx!r}; expected int frame index."
                )
            if frame_idx < 0:
                raise ValueError(
                    f"MASt3R-SLAM invalid timestamps_by_frame key {frame_idx}; expected non-negative frame index."
                )
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise ValueError(
                    f"MASt3R-SLAM invalid timestamp value for frame {frame_idx}: {timestamp!r}"
                )
            normalized[frame_idx] = timestamp

        expected_indices = set(range(len(normalized)))
        actual_indices = set(normalized.keys())
        if actual_indices != expected_indices:
            missing = sorted(expected_indices - actual_indices)
            extra = sorted(actual_indices - expected_indices)
            raise ValueError(
                "MASt3R-SLAM timestamps_by_frame must use contiguous frame indices 0..N-1 "
                f"(missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
                f"extra={extra[:5]}{'...' if len(extra) > 5 else ''})."
            )

        previous: Optional[float] = None
        for frame_idx in range(len(normalized)):
            current = float(normalized[frame_idx])
            if previous is not None and current <= previous:
                raise ValueError(
                    "MASt3R-SLAM timestamps_by_frame must be strictly increasing by frame index: "
                    f"frame {frame_idx - 1}={previous}, frame {frame_idx}={current}."
                )
            previous = current

        return normalized

    def _write_staged_rgb_txt(
        self,
        staged_dataset_path: Path,
        image_files: List[Path],
        timestamps_by_frame: Dict[int, Union[int, float]],
    ) -> None:
        """Write MASt3R-compatible TUM rgb.txt in staged dataset root."""
        rgb_txt = staged_dataset_path / "rgb.txt"
        with open(rgb_txt, "w", encoding="utf-8") as file_handle:
            for frame_idx, image_path in enumerate(image_files):
                timestamp = timestamps_by_frame[frame_idx]
                file_handle.write(f"{timestamp} rgb/{image_path.name}\n")

    @staticmethod
    def _sanitize_sequence_label(sequence_name: str) -> str:
        """Create a filesystem-safe label from a canonical sequence name."""
        cleaned = "".join(
            ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
            for ch in sequence_name.strip()
        )
        return cleaned or "sequence"

    def _expected_log_basenames(self, dataset_type: str, sequence_name: str) -> List[str]:
        """Return deterministic MASt3R log basenames for this run."""
        safe_sequence = self._sanitize_sequence_label(sequence_name)
        basenames = [safe_sequence]
        if dataset_type == "tum":
            freiburg_id = self._resolve_freiburg_id_from_sequence(sequence_name)
            basenames.insert(0, f"rgbd_dataset_{freiburg_id}_{safe_sequence}")
        return basenames

    def _run_mast3rslam(
        self,
        dataset_path: Path,
        config_file: Path,
        output_dir: Path,
        log_basenames: List[str],
    ) -> bool:
        """Execute MASt3R-SLAM."""
        logger.info("  Executing MASt3R-SLAM...")

        conda_init = "source ~/miniconda3/etc/profile.d/conda.sh"
        if not Path(os.path.expanduser("~/miniconda3")).exists():
            conda_init = "source ~/anaconda3/etc/profile.d/conda.sh"

        # MASt3R-SLAM saves to logs/{dataset_name}.txt
        cmd = (
            f"cd {self.mast3r_path} && "
            f"{conda_init} && "
            f"conda activate {self.conda_env} && "
            f"PYTHONPATH=\"{self.mast3r_path}/thirdparty/in3d:{self.mast3r_path}/thirdparty/mast3r:{self.mast3r_path}:${{PYTHONPATH:-}}\" "
            f"python main.py --dataset {dataset_path} --config {config_file} --no-viz"
        )

        try:
            process = self._spawn_streaming_process(
                ["bash", "-c", cmd],
            )

            self._stream_process_output(process, "MASt3R-SLAM")

            self._wait_for_process(process, timeout_seconds=7200)

            if process.returncode != 0:
                logger.error(f"MASt3R-SLAM failed with return code {process.returncode}")
                return False

            logger.info("  MASt3R-SLAM completed successfully")

            logs_dir = self.mast3r_path / "logs"
            traj_src: Optional[Path] = None

            for basename in log_basenames:
                candidate = logs_dir / f"{basename}.txt"
                if candidate.exists():
                    traj_src = candidate
                    logger.info(f"  Found trajectory: {traj_src.name}")
                    break
            if traj_src is None:
                expected_files = [f"{basename}.txt" for basename in log_basenames]
                available_files = sorted(path.name for path in logs_dir.glob("*.txt"))
                logger.error(
                    "  No matching MASt3R trajectory file found. "
                    f"Expected one of: {expected_files}. "
                    f"Available: {available_files if available_files else '[none]'}"
                )
                return False

            traj_dst = output_dir / "CameraTrajectory.txt"
            shutil.copy2(traj_src, traj_dst)
            logger.info(f"  Trajectory copied to {traj_dst}")

            return True

        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("MASt3R-SLAM timed out after 2 hours")
            return False
        except Exception as e:
            logger.error(f"Failed to run MASt3R-SLAM: {e}")
            return False

    def _find_raw_trajectory(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Find raw MASt3R-SLAM trajectory output."""
        camera_traj = request.output_dir / "CameraTrajectory.txt"
        if camera_traj.exists():
            return camera_traj
        return None

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Path,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """MASt3R-SLAM trajectory is already TUM-compatible."""
        return raw_trajectory

    def cleanup(self) -> None:
        """Clean up temporary files."""
        self._cleanup_temp_tum_link()

    def _cleanup_temp_tum_link(self) -> None:
        """Remove temporary TUM symlink staging directory when present."""
        if hasattr(self, "_temp_tum_link") and self._temp_tum_link:
            try:
                temp_dir = self._temp_tum_link.parent.parent
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._temp_tum_link = None
