"""Abstract base class for SLAM algorithms."""

from dataclasses import dataclass
import logging
import os
import signal
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .types import SLAMRunRequest, SLAMRunResult, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSpec:
    """Resolved process execution contract for a single SLAM run."""

    cmd: List[str]
    cwd: Optional[Path] = None
    env: Optional[Dict[str, str]] = None
    timeout_seconds: Optional[int] = None
    start_new_session: bool = False
    terminate_with_process_group: bool = False
    stream_output: bool = True
    log_prefix: str = "SLAM"
    custom_runner: Optional[Callable[["ExecutionSpec"], bool]] = None


class SLAMAlgorithm(ABC):
    """Abstract base class for SLAM algorithm implementations.

    Each SLAM algorithm (e.g., ORB-SLAM3, VINS) must implement this interface
    to be compatible with the evaluation pipeline.
    """

    def run(self, request: SLAMRunRequest) -> SLAMRunResult:
        """Run SLAM on a dataset and return a structured result."""
        try:
            self.validate_request(request)
        except ValueError as e:
            return SLAMRunResult(
                success=False,
                trajectory_path=None,
                message=str(e),
            )

        self._log_run_info(request)

        try:
            ctx = self._build_runtime_context(request)
        except Exception as e:
            return SLAMRunResult(
                success=False,
                trajectory_path=None,
                message=str(e),
            )

        try:
            self._preflight_checks(request, ctx)
        except Exception as e:
            return SLAMRunResult(
                success=False,
                trajectory_path=None,
                message=str(e),
            )

        execution_ok = False
        trajectory_path: Optional[Path] = None
        try:
            try:
                ctx.effective_dataset_path = self._stage_dataset(request, ctx)
            except Exception as e:
                logger.error(f"{self.name} dataset staging failed: {e}")
                return SLAMRunResult(
                    success=False,
                    trajectory_path=None,
                    message=str(e),
                )

            if ctx.effective_dataset_path is not None:
                try:
                    execution_inputs = self._build_execution_inputs(request, ctx)
                    if execution_inputs is not None:
                        ctx.execution_inputs = execution_inputs
                        logger.debug(
                            "  Execution inputs: %s",
                            self._format_execution_inputs_for_log(ctx.execution_inputs),
                        )
                        execution_ok = self._execute(request, ctx)
                    else:
                        logger.error(f"{self.name} execution input preparation failed and execution was skipped")
                except Exception as e:
                    logger.error(f"{self.name} execution failed: {e}")
            else:
                logger.error(f"{self.name} dataset staging failed and execution was skipped")

            try:
                trajectory_path = self._find_and_convert_trajectory(request, ctx)
            except Exception as e:
                logger.error(f"{self.name} trajectory collection failed: {e}")
                trajectory_path = None
        finally:
            try:
                self._cleanup_staged_dataset(request, ctx)
            except Exception as e:
                logger.warning(f"{self.name} dataset staging cleanup failed: {e}")

        if trajectory_path is None:
            if not execution_ok:
                message = f"{self.name} execution failed and did not produce a trajectory file"
            else:
                message = f"{self.name} did not produce a trajectory file"
            return SLAMRunResult(
                success=False,
                trajectory_path=None,
                message=message,
            )

        return SLAMRunResult(
            success=True,
            trajectory_path=trajectory_path,
            message="",
        )

    @abstractmethod
    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate runtime prerequisites before execution.

        Implementations should raise a descriptive exception when dependencies,
        models, binaries, configs, or other required runtime assets are missing.
        """
        pass

    def _stage_dataset(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Prepare dataset layout for the execution backend.

        Override in subclasses that require symlinks, temp directories, or
        generated metadata before SLAM execution. Return None when staging fails.
        """
        return request.dataset_path

    def _cleanup_staged_dataset(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Clean up temporary dataset staging artifacts for a single run."""
        pass

    @abstractmethod
    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, Any]]:
        """Resolve algorithm-specific execution inputs from request and runtime context."""
        pass

    @abstractmethod
    def _build_execution_spec(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[ExecutionSpec]:
        """Build process execution specification from prepared runtime inputs."""
        pass

    @abstractmethod
    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        """Implementation hook for SLAM execution.

        Implementations should run the SLAM backend (docker/conda/native) and
        return whether execution completed successfully.

        Trajectory discovery and conversion are handled by
        _find_and_convert_trajectory().
        """
        pass

    def _find_and_convert_trajectory(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Collect and normalize trajectory output after algorithm execution."""
        raw_trajectory = self._find_raw_trajectory(request, ctx)
        if raw_trajectory is None:
            return None
        return self._convert_raw_trajectory_to_tum(raw_trajectory, request, ctx)

    @abstractmethod
    def _find_raw_trajectory(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Any]:
        """Locate algorithm-specific raw trajectory output."""
        pass

    @abstractmethod
    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Any,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Convert located trajectory output to final TUM-compatible file."""
        pass

    def _build_runtime_context(self, request: SLAMRunRequest) -> SLAMRuntimeContext:
        """Build derived per-run context shared by lifecycle hooks."""
        config_is_external = self.is_external_config(request.slam_config)
        resolved_config_path = Path(request.slam_config).resolve() if config_is_external else None
        internal_config_name = None if config_is_external else request.slam_config
        sequence_name = (request.sequence_name or "").strip()
        if not sequence_name:
            raise ValueError(
                "SLAMRunRequest.sequence_name is required and must be non-empty. "
                "Provide dataset.sequence in evaluation config."
            )
        return SLAMRuntimeContext(
            request=request,
            config_is_external=config_is_external,
            resolved_config_path=resolved_config_path,
            internal_config_name=internal_config_name,
            sequence_name=sequence_name,
        )

    def _log_run_info(self, request: SLAMRunRequest) -> None:
        """Log standard run information.

        Args:
            request: Structured run request
        """
        is_external = self.is_external_config(request.slam_config)

        logger.info(f"Running {self.name} on {request.dataset_path}")
        logger.info(f"  Config: {request.slam_config}")
        logger.info(f"  Config type: {'external' if is_external else 'internal'}")
        logger.info(f"  Dataset type: {request.dataset_type}")
        logger.info(f"  Sensor mode: {request.sensor_mode.value}")

    def _format_execution_inputs_for_log(self, value: Any) -> Any:
        """Convert execution input values to log-friendly primitives."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: self._format_execution_inputs_for_log(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._format_execution_inputs_for_log(v) for v in value]
        return value

    def _format_execution_spec_for_log(self, spec: ExecutionSpec) -> Dict[str, Any]:
        """Convert execution spec fields to log-friendly primitives."""
        return {
            "cmd": [str(item) for item in spec.cmd],
            "cwd": str(spec.cwd) if spec.cwd is not None else None,
            "env_keys": sorted(spec.env.keys()) if spec.env else None,
            "timeout_seconds": spec.timeout_seconds,
            "start_new_session": spec.start_new_session,
            "terminate_with_process_group": spec.terminate_with_process_group,
            "stream_output": spec.stream_output,
            "log_prefix": spec.log_prefix,
            "custom_runner": spec.custom_runner is not None,
        }

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources (Docker containers, temporary files, etc.).

        This method is called after each SLAM run to ensure proper resource cleanup.
        It should be safe to call multiple times.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name (e.g., 'orbslam3', 'vins')."""
        pass

    @property
    @abstractmethod
    def supported_datasets(self) -> Dict[str, List[str]]:
        """Return supported dataset types and modes.

        Returns:
            Dict mapping dataset type to list of supported modes.
        """
        pass

    def supports(self, dataset_type: str, sensor_mode: Union[SensorMode, str]) -> bool:
        """Check if algorithm supports the given dataset type and mode."""
        mode = sensor_mode.value if isinstance(sensor_mode, SensorMode) else str(sensor_mode).lower()
        supported = self.supported_datasets.get(dataset_type.lower(), [])
        return mode in supported

    def validate_request(self, request: SLAMRunRequest) -> None:
        """Validate a run request against declared algorithm capabilities."""
        if request.dataset_type.lower() not in self.supported_datasets:
            available = ", ".join(sorted(self.supported_datasets.keys()))
            raise ValueError(
                f"Algorithm '{self.name}' does not support dataset '{request.dataset_type}'. "
                f"Available datasets: {available}"
            )

        if not self.supports(request.dataset_type, request.sensor_mode):
            available_modes = ", ".join(
                self.supported_datasets.get(request.dataset_type.lower(), [])
            )
            raise ValueError(
                f"Algorithm '{self.name}' does not support sensor mode "
                f"'{request.sensor_mode.value}' for dataset '{request.dataset_type}'. "
                f"Available modes: {available_modes}"
            )

    @abstractmethod
    def resolve_config_name(
        self,
        sequence: str,
        dataset_type: str,
        sensor_mode: Optional[SensorMode] = None
    ) -> Optional[str]:
        """Resolve internal config name from dataset sequence.

        Override in subclasses to provide algorithm-specific mapping.
        For example, S3PO-GS uses "04" directly, while ORB-SLAM3 uses "KITTI04-12.yaml".

        Args:
            sequence: Dataset sequence (e.g., "04", "freiburg1_desk")
            dataset_type: Dataset type (e.g., "kitti", "tum")
            sensor_mode: Optional sensor mode hint (mono/stereo/rgbd)

        Returns:
            Config name string, or None if cannot be inferred
        """
        pass

    @abstractmethod
    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Resolve algorithm-specific internal config path for a run."""
        pass

    def _resolve_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Resolve config path from runtime context without changing policy."""
        if ctx.config_is_external:
            return ctx.resolved_config_path
        return self._resolve_internal_config_path(ctx)

    def is_external_config(self, slam_config: str) -> bool:
        """Check if slam_config is an external file path or internal config name.

        Args:
            slam_config: Config string (path or name)

        Returns:
            True if slam_config is a path to an existing file, False otherwise
        """
        return Path(slam_config).is_file()

    def _kill_process_group(self, process) -> None:
        """Kill a subprocess and all its children via process group.

        This method safely terminates a process started with start_new_session=True.
        It first sends SIGTERM for graceful shutdown, then SIGKILL if needed.

        Args:
            process: subprocess.Popen instance (must have been started with start_new_session=True)
        """
        if process is None or process.poll() is not None:
            return  # Process already terminated

        try:
            # Send SIGTERM to entire process group for graceful shutdown
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if SIGTERM didn't work
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=2)
            except Exception:
                pass
        except (ProcessLookupError, PermissionError, OSError):
            # Process already dead or we don't have permission
            pass

    def _spawn_streaming_process(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        start_new_session: bool = False,
    ):
        """Start a subprocess configured for streamed combined stdout/stderr logs."""
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            start_new_session=start_new_session,
        )

    def _stream_process_output(
        self,
        process,
        log_prefix: str,
        stop_on_line: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        """Stream process output and optionally stop early when callback returns True.

        Returns:
            True when stopped early due to stop_on_line callback, False otherwise.
        """
        if process.stdout is None:
            return False

        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue

            logger.info(f"    [{log_prefix}] {line}")
            if stop_on_line and stop_on_line(line):
                return True

        return False

    def _wait_for_process(self, process, timeout_seconds: Optional[int] = None) -> int:
        """Wait for process completion and return exit code."""
        process.wait(timeout=timeout_seconds)
        if process.returncode is None:
            return -1
        return process.returncode

    def _terminate_process(self, process, use_process_group: bool) -> None:
        """Terminate a process using either process-group or single-process semantics."""
        if process is None:
            return

        if use_process_group:
            self._kill_process_group(process)
            return

        try:
            process.kill()
        except Exception:
            pass

    def _run_execution_spec(self, spec: ExecutionSpec) -> int:
        """Execute a pre-built execution specification and return process-style return code.

        Returns:
            Process return code (0 for success). Custom runners map True->0 and False->1.

        Raises:
            subprocess.TimeoutExpired: when stream-mode process exceeds timeout.
            Exception: propagated runner/process errors for caller-level handling.
        """
        logger.debug(
            "  Execution spec: %s",
            self._format_execution_spec_for_log(spec),
        )

        if spec.custom_runner is not None:
            return 0 if spec.custom_runner(spec) else 1

        if not spec.stream_output:
            result = subprocess.run(
                spec.cmd,
                check=False,
                cwd=str(spec.cwd) if spec.cwd is not None else None,
                env=spec.env,
            )
            return result.returncode

        process = self._spawn_streaming_process(
            spec.cmd,
            cwd=spec.cwd,
            env=spec.env,
            start_new_session=spec.start_new_session,
        )
        try:
            self._stream_process_output(process, spec.log_prefix)
            self._wait_for_process(process, timeout_seconds=spec.timeout_seconds)
            return process.returncode if process.returncode is not None else -1
        except subprocess.TimeoutExpired:
            self._terminate_process(process, spec.terminate_with_process_group)
            raise
        except Exception:
            self._terminate_process(process, spec.terminate_with_process_group)
            raise
