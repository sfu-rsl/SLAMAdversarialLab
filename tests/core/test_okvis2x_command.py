"""Tests for the OKVIS2-X EuRoC command construction, IMU staging, and the
trajectory CSV to TUM conversion (both the VI and stereo-depth-network modes).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from slamadversariallab.algorithms.okvis2x import (
    OKVIS2XAlgorithm,
    OKVIS2XNNAlgorithm,
)
from slamadversariallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRuntimeContext,
    SensorMode,
)


class _FakeSession:
    def gpu_launch_config(self) -> Dict[str, Any]:
        return {"env": {}, "mounts": [], "devices": []}


def _build_request(tmp_path: Path, config_name: str = "euroc_vi.yaml") -> SLAMRunRequest:
    dataset_path = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config=config_name,
        output_dir=output_dir,
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        sequence_name="V1_01_easy",
    )


def _build_context(
    request: SLAMRunRequest,
    staged_path: Path,
    config_name: str = "euroc_vi.yaml",
    image_mounts: Optional[List[Tuple[str, str]]] = None,
) -> SLAMRuntimeContext:
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name=config_name,
        sequence_name=request.sequence_name,
        effective_dataset_path=staged_path,
    )
    ctx.execution_inputs = {
        "dataset_path": staged_path,
        "output_dir": request.output_dir,
        "euroc_image_mounts": list(image_mounts or []),
    }
    ctx.runtime_stress = object()
    return ctx


def _build_spec(algo, ctx):
    algo._active_runtime_context = ctx
    ctx.runtime_stress_session = _FakeSession()
    try:
        return algo._build_execution_spec(ctx.request, ctx)
    finally:
        algo._active_runtime_context = None


def _bash_command(spec) -> str:
    # cmd = [runtime, run, ..., image, "bash", "-c", <bash_cmd>]
    assert spec.cmd[-2] == "-c"
    return spec.cmd[-1]


# ---------------------------------------------------------------------------
# Command construction: VI mode
# ---------------------------------------------------------------------------

def test_okvis2x_vi_command_shape(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    ctx = _build_context(request, staged)
    algo = OKVIS2XAlgorithm(container_runtime="podman")

    spec = _build_spec(algo, ctx)
    assert spec is not None
    bash_cmd = _bash_command(spec)

    # Verified CLI: config-yaml dataset-folder output-folder. Bare headless
    # (no X at all): the vendored VI config disables all display outputs.
    assert "./okvis_app_synchronous" in bash_cmd
    assert "xvfb" not in bash_cmd
    assert "Xvfb" not in bash_cmd
    assert "/okvis2x/sal_configs/euroc_vi.yaml" in bash_cmd
    assert "/dataset/mav0 /output" in bash_cmd
    # VI mode takes NO se2 (submapping) config.
    assert "se2_euroc.yaml" not in bash_cmd
    # OKVIS writes directly into /output via the CLI arg: no copy pairs.
    assert "cp " not in bash_cmd
    # Runtime-stress shape: log tee + exit-code file + status propagation.
    assert "tee /output/slam_output.log" in bash_cmd
    assert "slam_exit_code.txt" in bash_cmd


def test_okvis2x_vi_attaches_gpu_device(tmp_path: Path) -> None:
    """The USE_GPU image build initializes a CUDA context at startup in BOTH
    modes (verified: without the device, c10 aborts with 'CUDA driver version
    is insufficient'), so even the VI mode must attach the GPU."""
    request = _build_request(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    ctx = _build_context(request, staged)
    algo = OKVIS2XAlgorithm(container_runtime="podman")

    spec = _build_spec(algo, ctx)
    joined = " ".join(spec.cmd)
    assert "nvidia.com/gpu=all" in joined


# ---------------------------------------------------------------------------
# Command construction: NN (stereo depth network) mode
# ---------------------------------------------------------------------------

def test_okvis2xnn_command_shape_with_se2_and_gpu(tmp_path: Path) -> None:
    request = _build_request(tmp_path, config_name="euroc_nn.yaml")
    staged = tmp_path / "staged"
    staged.mkdir()
    ctx = _build_context(request, staged, config_name="euroc_nn.yaml")
    algo = OKVIS2XNNAlgorithm(container_runtime="podman")

    spec = _build_spec(algo, ctx)
    assert spec is not None
    bash_cmd = _bash_command(spec)

    # Verified CLI: okvis2-config se2-config dataset-folder output-folder.
    # The NN app calls cv::imshow unconditionally: an inline fixed-display
    # Xvfb is started (NOT xvfb-run, which kills the app in this image).
    assert "./okvis2x_app_snetwork_synchronous" in bash_cmd
    assert "Xvfb :99" in bash_cmd
    assert "DISPLAY=:99" in bash_cmd
    assert "xvfb-run" not in bash_cmd
    assert "/okvis2x/sal_configs/euroc_nn.yaml" in bash_cmd
    assert "/okvis2x/sal_configs/se2_euroc.yaml" in bash_cmd
    assert "/dataset/mav0 /output" in bash_cmd

    # Live CUDA workload: the GPU is attached (podman device syntax).
    joined = " ".join(spec.cmd)
    assert "nvidia.com/gpu=all" in joined


def test_okvis2x_metadata_and_mounts(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    image_mounts = [
        (str(tmp_path / "cam0"), "/dataset/mav0/cam0/data"),
        (str(tmp_path / "cam1"), "/dataset/mav0/cam1/data"),
    ]
    ctx = _build_context(request, staged, image_mounts=image_mounts)
    algo = OKVIS2XAlgorithm(container_runtime="podman")

    spec = _build_spec(algo, ctx)
    assert spec is not None
    assert spec.log_prefix == "OKVIS2X"
    assert spec.target_kind == "podman_container"
    assert spec.target_metadata is not None
    assert spec.target_metadata["container_name"].startswith("okvis2x-")
    assert spec.target_metadata["io_target_paths"]

    joined = " ".join(spec.cmd)
    assert f"{tmp_path / 'cam0'}:/dataset/mav0/cam0/data:ro" in joined
    assert f"{tmp_path / 'cam1'}:/dataset/mav0/cam1/data:ro" in joined
    assert f"{staged.resolve()}:/dataset:ro" in joined


def test_okvis2x_realtime_env_propagates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SAL_DEADLINE_FPS", "20")
    monkeypatch.setenv(
        "SAL_RUNTIME_PATH",
        # Any absolute path serves: this is injected, never read from disk. It
        # is a neutral one so the repo carries no author's home directory.
        "/opt/sal/src/runtime_stress",
    )
    request = _build_request(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    ctx = _build_context(request, staged)
    algo = OKVIS2XAlgorithm(container_runtime="podman")

    spec = _build_spec(algo, ctx)
    joined = " ".join(spec.cmd)
    assert "-e SAL_DEADLINE_FPS=20" in joined
    assert "-e SAL_DROP_LOG_PATH=/output/deadline_drops.json" in joined


# ---------------------------------------------------------------------------
# IMU staging (shared ORB-SLAM3-family helpers)
# ---------------------------------------------------------------------------

def _imu_request(tmp_path: Path, with_imu: bool) -> SLAMRunRequest:
    dataset_path = tmp_path / "euroc"
    if with_imu:
        imu_dir = dataset_path / "mav0" / "imu0"
        imu_dir.mkdir(parents=True)
        (imu_dir / "data.csv").write_text("#ts,wx,wy,wz,ax,ay,az\n1,0,0,0,0,0,9.8\n")
        (imu_dir / "sensor.yaml").write_text("sensor_type: imu\n")
    else:
        dataset_path.mkdir(parents=True)
    return SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="euroc_vi.yaml",
        output_dir=tmp_path / "out",
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        sequence_name="V1_01_easy",
    )


def test_okvis2x_stages_euroc_imu_stream(tmp_path: Path) -> None:
    request = _imu_request(tmp_path, with_imu=True)
    staged = tmp_path / "staged"
    staged.mkdir()
    algo = OKVIS2XAlgorithm(container_runtime="podman")

    algo._stage_euroc_imu(request, staged)

    assert (staged / "mav0" / "imu0" / "data.csv").exists()
    assert (staged / "mav0" / "imu0" / "sensor.yaml").exists()


def test_okvis2x_resolve_imu_returns_none_when_missing(tmp_path: Path) -> None:
    request = _imu_request(tmp_path, with_imu=False)
    algo = OKVIS2XAlgorithm(container_runtime="podman")
    assert algo._resolve_euroc_imu_csv(request) is None


def test_okvis2x_resolve_imu_falls_back_to_original_dataset(tmp_path: Path) -> None:
    """Perturbed EuRoC roots carry no IMU (SAL perturbs images only): the
    resolver recovers the original root from extras['frame_image_paths'] and
    uses its pristine mav0/imu0/data.csv."""
    # Original dataset with IMU + a left image.
    orig = tmp_path / "orig" / "V1_01_easy"
    (orig / "mav0" / "imu0").mkdir(parents=True)
    (orig / "mav0" / "imu0" / "data.csv").write_text("#ts,...\n1,0,0,0,0,0,9.8\n")
    img_dir = orig / "mav0" / "cam0" / "data"
    img_dir.mkdir(parents=True)
    (img_dir / "1403715273262142976.png").write_text("fake")
    # Perturbed root: cameras only, no imu0.
    perturbed = tmp_path / "perturbed" / "network_light"
    (perturbed / "mav0" / "cam0" / "data").mkdir(parents=True)

    request = SLAMRunRequest(
        dataset_path=perturbed,
        slam_config="euroc_vi.yaml",
        output_dir=tmp_path / "out",
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        sequence_name="V1_01_easy",
        extras={"frame_image_paths": [str(img_dir / "1403715273262142976.png")]},
    )
    algo = OKVIS2XAlgorithm(container_runtime="podman")
    resolved = algo._resolve_euroc_imu_csv(request)
    assert resolved is not None
    assert resolved == orig / "mav0" / "imu0" / "data.csv"


# ---------------------------------------------------------------------------
# Trajectory CSV -> TUM conversion
# ---------------------------------------------------------------------------

# REAL header captured from a V1_01_easy container run of the pinned image
# (okvis2-slam-final_trajectory.csv). The parser maps columns by name so
# ordering/extra columns are tolerated; gpsMode/SID/NrGps must not collide
# with the p*/q* matchers.
_OKVIS_STYLE_HEADER = (
    "timestamp, p_WS_W_x, p_WS_W_y, p_WS_W_z, "
    "q_WS_x, q_WS_y, q_WS_z, q_WS_w, "
    "v_WS_W_x, v_WS_W_y, v_WS_W_z, "
    "b_g_x, b_g_y, b_g_z, b_a_x, b_a_y, b_a_z, "
    "NrGps, SID, gpsMode"
)


def _write_csv(path: Path, header: str, rows: List[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def test_okvis_csv_parses_euroc_style_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "okvis2-slam-final_trajectory.csv"
    _write_csv(
        csv_path,
        _OKVIS_STYLE_HEADER,
        [
            "1403715273262142976, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 0,0,0, 0,0,0, 0,0,0",
            "1403715273312142976, 1.1, 2.1, 3.1, 0.1, 0.2, 0.3, 0.9, 0,0,0, 0,0,0, 0,0,0",
        ],
    )
    rows = OKVIS2XAlgorithm._parse_okvis_trajectory_csv(csv_path)
    assert len(rows) == 2
    ts, tx, ty, tz, qx, qy, qz, qw = rows[0]
    # Nanosecond timestamp passes through unchanged (EuRoC convention).
    assert ts == "1403715273262142976"
    assert (tx, ty, tz) == (1.0, 2.0, 3.0)
    assert (qx, qy, qz, qw) == (0.1, 0.2, 0.3, 0.9)


def test_okvis_csv_tolerates_shuffled_and_extra_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "okvis2-slam_trajectory.csv"
    _write_csv(
        csv_path,
        "#timestamp, q_WS_w [], q_WS_x [], q_WS_y [], q_WS_z [], "
        "extra_col, p_WS_W_x [m], p_WS_W_y [m], p_WS_W_z [m]",
        ["100, 0.9, 0.1, 0.2, 0.3, 42, 7.0, 8.0, 9.0"],
    )
    rows = OKVIS2XAlgorithm._parse_okvis_trajectory_csv(csv_path)
    assert rows == [("100", 7.0, 8.0, 9.0, 0.1, 0.2, 0.3, 0.9)]


def test_okvis_csv_unknown_header_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "okvis2-slam_trajectory.csv"
    _write_csv(csv_path, "#foo, bar, baz", ["1, 2, 3"])
    with pytest.raises(ValueError, match="could not locate column"):
        OKVIS2XAlgorithm._parse_okvis_trajectory_csv(csv_path)


def test_okvis_convert_writes_tum_file(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    csv_path = request.output_dir / "okvis2-slam-final_trajectory.csv"
    _write_csv(
        csv_path,
        _OKVIS_STYLE_HEADER,
        ["1403715273262142976, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 0,0,0, 0,0,0, 0,0,0"],
    )
    algo = OKVIS2XAlgorithm(container_runtime="podman")
    ctx = _build_context(request, request.dataset_path)

    out = algo._convert_raw_trajectory_to_tum(csv_path, request, ctx)
    assert out is not None and out.name == "CameraTrajectory.txt"
    line = out.read_text().strip()
    assert line == "1403715273262142976 1.0 2.0 3.0 0.1 0.2 0.3 0.9"


def test_okvis_find_raw_trajectory_prefers_final_ba(tmp_path: Path) -> None:
    """Preference order: final-BA (most optimized) > final (loop-closed) >
    online (causal). Mirrors the real V1_01_easy output file set."""
    request = _build_request(tmp_path)
    (request.output_dir / "okvis2-slam_trajectory.csv").write_text("#timestamp\n")
    (request.output_dir / "okvis2-slam-final_trajectory.csv").write_text("#timestamp\n")
    (request.output_dir / "okvis2-slam-final-ba_trajectory.csv").write_text("#timestamp\n")
    algo = OKVIS2XAlgorithm(container_runtime="podman")
    ctx = _build_context(request, request.dataset_path)

    found = algo._find_raw_trajectory(request, ctx)
    assert found is not None
    assert found.name == "okvis2-slam-final-ba_trajectory.csv"

    # Without the final-BA file, fall back to the loop-closed final.
    (request.output_dir / "okvis2-slam-final-ba_trajectory.csv").unlink()
    found = algo._find_raw_trajectory(request, ctx)
    assert found.name == "okvis2-slam-final_trajectory.csv"
