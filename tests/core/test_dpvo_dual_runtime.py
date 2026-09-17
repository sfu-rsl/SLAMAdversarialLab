"""Tests for the DPVO / DPV-SLAM podman-only wrapper (runtime selection,
container command shape, and the dpvslam loop-closure variant)."""

from pathlib import Path

import pytest

from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]

from slamadversariallab.algorithms.dpvo import DPVOAlgorithm, DPVSLAMAlgorithm
from slamadversariallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRuntimeContext,
    SensorMode,
)


def _build_request(tmp_path: Path) -> SLAMRunRequest:
    # "freiburg1" sequence name drives the TUM_CALIBRATIONS["freiburg1"]
    # lookup without env detection (DROID test convention).
    dataset_path = tmp_path / "freiburg1_desk"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    image_dir = dataset_path / "rgb"
    image_dir.mkdir()
    (image_dir / "0.png").write_bytes(b"")

    return SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="tum1",
        output_dir=output_dir,
        dataset_type="tum",
        sensor_mode=SensorMode.MONO,
        sequence_name="freiburg1_desk",
        extras={
            "camera_paths": {"left": str(image_dir)},
            "timestamps_by_frame": {0: 0.0, 1: 0.0333},
        },
    )


def _build_context(request: SLAMRunRequest) -> SLAMRuntimeContext:
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name=None,
        sequence_name=request.sequence_name,
        effective_dataset_path=request.dataset_path,
    )
    ctx.execution_inputs = {
        "dataset_path": request.dataset_path,
        "slam_config": "tum1",
        "output_dir": request.output_dir,
        "dataset_type": "tum",
        "camera_paths": request.extras["camera_paths"],
        "timestamps_by_frame": request.extras["timestamps_by_frame"],
    }
    return ctx


def _build_spec(algo, ctx):
    algo._active_runtime_context = ctx
    try:
        return algo._build_execution_spec(ctx.request, ctx)
    finally:
        algo._active_runtime_context = None


# ---------------------------------------------------------------------------
# Runtime selection (podman-only wrapper)
# ---------------------------------------------------------------------------

def test_dpvo_default_constructor_is_podman() -> None:
    """No conda path: a bare constructor must yield a working podman wrapper."""
    algo = DPVOAlgorithm()
    assert algo.container_runtime == "podman"
    assert algo.runtime_stress_target_kind == "podman_container"


def test_dpvo_accepts_docker_runtime() -> None:
    algo = DPVOAlgorithm(container_runtime="docker")
    assert algo.runtime_stress_target_kind == "docker_container"


def test_dpvo_invalid_runtime_raises() -> None:
    with pytest.raises(ValueError):
        DPVOAlgorithm(container_runtime="conda")


# ---------------------------------------------------------------------------
# Container command shape
# ---------------------------------------------------------------------------

def test_dpvo_podman_execution_spec_shape(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DPVOAlgorithm(container_runtime="podman")

    spec = _build_spec(algo, ctx)
    assert spec is not None
    assert spec.cmd[0] == "podman"
    assert spec.target_kind == "podman_container"
    assert spec.log_prefix == "DPVO"
    assert spec.target_metadata is not None
    assert spec.target_metadata["container_name"].startswith("dpvo-")
    assert spec.target_metadata["io_target_paths"]

    joined = " ".join(spec.cmd)
    # Core mounts: dataset ro, output rw, calib ro.
    assert ":/dataset:ro" in joined
    assert f"{request.output_dir.resolve()}:/output" in joined
    assert ":/calib/calib.txt:ro" in joined
    # GPU via CDI.
    assert "nvidia.com/gpu=all" in joined
    # Deadline-harness loop file override: dpvo/stream.py bind-mounted over
    # the image copy (requires the image's editable install to take effect).
    #
    # `apply_entrypoint_override` is a NO-OP when the host file is absent, and
    # deps/slam-algorithms/DPVO is a submodule. On a clone that has not run
    # `git submodule update --init`, the mount is correctly missing and this
    # assertion would fail for want of a checkout rather than a defect.
    if (_REPO_ROOT / "deps" / "slam-algorithms" / "DPVO" / "dpvo" / "stream.py").exists():
        assert "stream.py:/dpvo/dpvo/stream.py" in joined
    else:
        pytest.skip("needs the DPVO submodule: git submodule update --init "
                    "deps/slam-algorithms/DPVO")

    bash_cmd = spec.cmd[-1]
    assert "python demo.py" in bash_cmd
    assert "--imagedir /dataset" in bash_cmd
    assert "--calib /calib/calib.txt" in bash_cmd
    assert "--network dpvo.pth" in bash_cmd
    assert "--save_trajectory" in bash_cmd
    # Trajectory is copied from demo.py's cwd-relative save dir to /output.
    assert "cp saved_trajectories/salrun.txt /output/dpvo_traj_raw.txt" in bash_cmd
    # Plain dpvo must NOT enable loop closure.
    assert "LOOP_CLOSURE" not in bash_cmd


def test_dpvo_container_name_is_sanitized(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DPVOAlgorithm(container_runtime="podman")

    spec = _build_spec(algo, ctx)
    name = spec.target_metadata["container_name"]
    assert name == name.lower()
    assert " " not in name
    assert len(name) <= 120


# ---------------------------------------------------------------------------
# DPV-SLAM variant
# ---------------------------------------------------------------------------

def test_dpvslam_adds_loop_closure_opts(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DPVSLAMAlgorithm(container_runtime="podman")

    assert algo.name == "dpvslam"
    spec = _build_spec(algo, ctx)
    assert spec is not None
    assert spec.log_prefix == "DPV-SLAM"
    assert spec.target_metadata["container_name"].startswith("dpvslam-")

    bash_cmd = spec.cmd[-1]
    assert "--opts LOOP_CLOSURE True" in bash_cmd
    # The flag must come before the && cp (part of the demo invocation).
    assert bash_cmd.index("LOOP_CLOSURE") < bash_cmd.index("&& cp")


def test_dpvslam_shares_image_with_dpvo() -> None:
    assert DPVOAlgorithm().docker_image == DPVSLAMAlgorithm().docker_image == "dpvo:latest"
