"""Tests for DPVO trajectory conversion: counter stamps -> (drop-log remap)
-> real TUM seconds. DPVO's demo.py saves TUM-layout lines whose column 1 is
the sequential enumerate counter from dpvo/stream.py, so the conversion must
remap through the deadline drop log (identity when absent), mirroring
DROID-SLAM's reconstruction conversion."""

import json
from pathlib import Path

from slamadversariallab.algorithms.dpvo import DPVOAlgorithm, _RAW_TRAJECTORY_NAME
from slamadversariallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRuntimeContext,
    SensorMode,
)


def _write_raw_trajectory(output_dir: Path, counters) -> Path:
    """Write a DPVO-style raw trajectory: counter tx ty tz qx qy qz qw."""
    lines = [
        f"{c}.000000 {c}.1 {c}.2 {c}.3 0.0 0.0 0.0 1.0" for c in counters
    ]
    raw = output_dir / _RAW_TRAJECTORY_NAME
    raw.write_text("\n".join(lines) + "\n")
    return raw


def _build_request_and_ctx(tmp_path: Path, n_frames: int = 10,
                           dataset_type: str = "tum"):
    dataset_path = tmp_path / "freiburg1_desk"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    # Frame i has timestamp 100.0 + i/30 (TUM seconds).
    timestamps_by_frame = {i: 100.0 + i / 30.0 for i in range(n_frames)}

    request = SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="tum1",
        output_dir=output_dir,
        dataset_type=dataset_type,
        sensor_mode=SensorMode.MONO,
        sequence_name="freiburg1_desk",
        extras={"timestamps_by_frame": timestamps_by_frame},
    )
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name=None,
        sequence_name=request.sequence_name,
        effective_dataset_path=dataset_path,
    )
    ctx.execution_inputs = {
        "dataset_type": dataset_type,
        "timestamps_by_frame": timestamps_by_frame,
        "output_dir": output_dir,
    }
    return request, ctx


def test_no_drop_log_counters_map_to_sampled_timestamps(tmp_path: Path) -> None:
    """Without a drop log, counter i maps to the i-th stride-sampled frame."""
    request, ctx = _build_request_and_ctx(tmp_path, n_frames=10)
    # stride=1 for TUM -> sampled dataset frames are 0,1,2,3,4. The stride was
    # 2, which made the deadline twice as strict for this system than for any
    # other; see _resolve_stride. The stride>1 remap is still exercised by the
    # KITTI test below, so this change does not cost that path its coverage.
    raw = _write_raw_trajectory(request.output_dir, counters=[0, 1, 2, 3, 4])
    algo = DPVOAlgorithm(container_runtime="podman")

    out = algo._convert_raw_trajectory_to_tum(raw, request, ctx)
    assert out is not None and out.name == "CameraTrajectory.txt"

    stamps = [float(line.split()[0]) for line in out.read_text().splitlines()]
    expected = [100.0 + i / 30.0 for i in (0, 1, 2, 3, 4)]
    assert stamps == [round(e, 10) or e for e in expected] or stamps == expected


def test_drop_log_remaps_counters_through_survivors(tmp_path: Path) -> None:
    """With a drop log, counter i maps to survivors[i] in the sampled stream."""
    request, ctx = _build_request_and_ctx(tmp_path, n_frames=10)
    # Sampled stream has 5 positions (stride 1); positions 1 and 3 dropped.
    drop_log = {
        "survivors": [0, 2, 4],
        "dropped": [1, 3],
        "target_fps": 30.0,
        "total_items": 5,
        "warmup_frames": 0,
        "queue_size": 1,
        "drop_policy": "drop_oldest",
    }
    (request.output_dir / "deadline_drops.json").write_text(json.dumps(drop_log))
    # DPVO consumed 3 frames -> counters 0,1,2.
    raw = _write_raw_trajectory(request.output_dir, counters=[0, 1, 2])
    algo = DPVOAlgorithm(container_runtime="podman")

    out = algo._convert_raw_trajectory_to_tum(raw, request, ctx)
    assert out is not None

    stamps = [float(line.split()[0]) for line in out.read_text().splitlines()]
    # survivors [0,2,4] -> sampled positions -> dataset frames 0,2,4.
    expected = [100.0 + i / 30.0 for i in (0, 2, 4)]
    assert stamps == expected


def test_malformed_drop_log_fails_conversion(tmp_path: Path) -> None:
    """A present-but-corrupt drop log must fail the conversion loudly, NOT
    silently identity-map. A malformed log means the SLAM did run under a
    deadline but its survivor record is unusable, so an identity map would
    misalign every pose. The conversion returns None (visible no-traj) and the
    error + traceback are logged, rather than producing a wrong trajectory."""
    request, ctx = _build_request_and_ctx(tmp_path, n_frames=10)
    (request.output_dir / "deadline_drops.json").write_text("{not json")
    raw = _write_raw_trajectory(request.output_dir, counters=[0, 1])
    algo = DPVOAlgorithm(container_runtime="podman")

    out = algo._convert_raw_trajectory_to_tum(raw, request, ctx)
    assert out is None


def test_missing_drop_log_while_deadline_active_fails_conversion(
    tmp_path: Path, monkeypatch
) -> None:
    """Deadline active (SAL_DEADLINE_FPS set) but no drop log on disk: the
    counter->timestamp remap has no survivor record, so the conversion must
    fail loud (return None), never silently identity-map. Guards against a
    lost or failed drop log misaligning every pose."""
    request, ctx = _build_request_and_ctx(tmp_path, n_frames=10)
    # No deadline_drops.json is written.
    raw = _write_raw_trajectory(request.output_dir, counters=[0, 1])
    monkeypatch.setenv("SAL_DEADLINE_FPS", "20")
    algo = DPVOAlgorithm(container_runtime="podman")

    out = algo._convert_raw_trajectory_to_tum(raw, request, ctx)
    assert out is None


def test_missing_drop_log_without_deadline_is_identity(
    tmp_path: Path, monkeypatch
) -> None:
    """No deadline active AND no log: a legitimate unstressed run, where the
    identity map is correct. The guard must NOT fire here."""
    request, ctx = _build_request_and_ctx(tmp_path, n_frames=10)
    raw = _write_raw_trajectory(request.output_dir, counters=[0, 1])
    monkeypatch.delenv("SAL_DEADLINE_FPS", raising=False)
    algo = DPVOAlgorithm(container_runtime="podman")

    out = algo._convert_raw_trajectory_to_tum(raw, request, ctx)
    assert out is not None  # identity mapping is correct without a deadline


def test_out_of_range_counter_fails_conversion(tmp_path: Path) -> None:
    """A counter beyond the sampled stream must fail loudly, not corrupt."""
    request, ctx = _build_request_and_ctx(tmp_path, n_frames=4)
    # stride 1 over 4 frames -> only 4 sampled positions; counter 5 invalid.
    raw = _write_raw_trajectory(request.output_dir, counters=[0, 5])
    algo = DPVOAlgorithm(container_runtime="podman")

    out = algo._convert_raw_trajectory_to_tum(raw, request, ctx)
    assert out is None


def test_strided_remap_still_covered_by_a_stride_3_dataset(tmp_path: Path) -> None:
    """TUM is stride 1 now, so nothing else here exercises stride > 1.

    The remap arithmetic is unchanged and still live: KITTI resolves to stride
    3. Without this test, dropping TUM to stride 1 would have quietly removed
    the only coverage of the strided path, and a later regression in it would
    surface as wrong timestamps in a campaign rather than as a red test.
    """
    request, ctx = _build_request_and_ctx(tmp_path, n_frames=12,
                                          dataset_type="kitti")
    algo = DPVOAlgorithm(container_runtime="podman")
    assert algo._resolve_stride("kitti") == 3

    raw = _write_raw_trajectory(request.output_dir, counters=[0, 1, 2, 3])
    out = algo._convert_raw_trajectory_to_tum(raw, request, ctx)
    assert out is not None

    stamps = [float(line.split()[0]) for line in out.read_text().splitlines()]
    # counter i -> sampled position i -> dataset frame 3i.
    assert stamps == [100.0 + i / 30.0 for i in (0, 3, 6, 9)]
