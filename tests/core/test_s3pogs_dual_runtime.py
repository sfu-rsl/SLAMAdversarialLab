"""Tests for S3PO-GS dual-runtime (conda default + podman opt-in) wrapper."""

from pathlib import Path

import pytest

from slamadversariallab.algorithms.s3pogs import S3POGSAlgorithm
from slamadversariallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRuntimeContext,
    SensorMode,
)


def _s3pogs_with_fake_tree(tmp_path: Path) -> tuple[S3POGSAlgorithm, Path]:
    """Build an S3POGSAlgorithm pointed at an isolated fake source tree.

    Lets dual-runtime tests run without the real S3PO-GS submodule on
    disk, mirroring the GigaSLAM / MASt3R-SLAM dual-runtime test setup.
    """
    s3pogs_root = tmp_path / "S3PO-GS"
    cfg_dir = s3pogs_root / "configs" / "mono" / "KITTI"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # Minimal KITTI 04.yaml carrying the keys _build_container_execution_spec
    # rewrites + the inherit pointer slam.py's load_config will read.
    (cfg_dir / "04.yaml").write_text(
        'inherit_from: "configs/mono/KITTI/base_config.yaml"\n'
        "\n"
        "Dataset:\n"
        '  dataset_path: "datasets/KITTI/04/"\n'
        "  begin: 0\n"
        "  end: 200\n",
        encoding="utf-8",
    )
    (cfg_dir / "base_config.yaml").write_text(
        "Results:\n"
        '  save_dir: "results"\n'
        "Dataset:\n"
        "  type: 'KITTI'\n"
        "  sensor_type: 'monocular'\n",
        encoding="utf-8",
    )
    algo = S3POGSAlgorithm()
    algo.s3pogs_path = s3pogs_root
    return algo, s3pogs_root


def _build_request_and_context(
    tmp_path: Path,
    s3pogs_root: Path,
) -> tuple[SLAMRunRequest, SLAMRuntimeContext, Path, Path]:
    """Build an S3PO-GS request mirroring a real KITTI mono evaluation run.

    A fake staged KITTI sequence dir is created under
    ``s3pogs_root/datasets/KITTI/04`` (the same place
    ``_prepare_dataset`` would place it). The host left-camera dir
    lives elsewhere so ``_collect_symlink_target_mounts`` can verify
    the absolute symlink target is picked up for the podman bind-mount
    list.
    """
    real_image_dir = tmp_path / "real_left_camera"
    real_image_dir.mkdir(parents=True, exist_ok=True)
    (real_image_dir / "000000.png").write_bytes(b"")
    (real_image_dir / "000001.png").write_bytes(b"")

    staged_root = s3pogs_root / "datasets" / "KITTI" / "04"
    staged_root.mkdir(parents=True, exist_ok=True)
    rgb_link = staged_root / "rgb"
    if rgb_link.exists() or rgb_link.is_symlink():
        rgb_link.unlink()
    rgb_link.symlink_to(real_image_dir.resolve())
    (staged_root / "calib.txt").write_text("P0: 0\n", encoding="utf-8")
    (staged_root / "poses.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    request = SLAMRunRequest(
        dataset_path=staged_root,
        slam_config="04",
        output_dir=output_dir,
        dataset_type="kitti",
        sensor_mode=SensorMode.MONO,
        sequence_name="04",
        extras={
            "camera_paths": {"left": str(real_image_dir)},
            "timestamps_by_frame": {0: 0.0, 1: 0.1},
        },
    )
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name="04",
        sequence_name=request.sequence_name,
        effective_dataset_path=staged_root,
    )
    config_file = s3pogs_root / "configs" / "mono" / "KITTI" / "04.yaml"
    ctx.execution_inputs = {
        "prepared_path": staged_root,
        "output_dir": output_dir,
        "is_stereo": False,
        "config_file": config_file,
        "sequence_name": "04",
    }
    return request, ctx, staged_root, real_image_dir


def test_s3pogs_default_constructor_is_conda() -> None:
    algo = S3POGSAlgorithm()
    assert algo.container_runtime is None
    assert algo.runtime_stress_target_kind == "host_process_group"
    assert algo.docker_image == "s3pogs:latest"


def test_s3pogs_podman_runtime_selection() -> None:
    algo = S3POGSAlgorithm(container_runtime="podman")
    assert algo.container_runtime == "podman"
    assert algo.runtime_stress_target_kind == "podman_container"
    assert algo.docker_image == "s3pogs:latest"


def test_s3pogs_invalid_runtime_raises() -> None:
    with pytest.raises(ValueError, match="container_runtime must be None or 'podman'"):
        S3POGSAlgorithm(container_runtime="docker")


def test_s3pogs_conda_execution_spec_uses_custom_runner(tmp_path: Path) -> None:
    algo, s3pogs_root = _s3pogs_with_fake_tree(tmp_path)
    request, ctx, _staged_root, _real_image_dir = _build_request_and_context(tmp_path, s3pogs_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    assert spec.custom_runner is not None
    assert spec.cmd == ["s3pogs"]
    # Default for conda spec: target_kind is base-class default (None on
    # dataclass init, normalized to "host_process_group" on explicit
    # query).
    assert spec.target_kind in (None, "host_process_group")


def test_s3pogs_conda_does_not_generate_override_config(tmp_path: Path) -> None:
    """Conda path uses the upstream config directly (S3PO-GS reads
    Dataset.dataset_path as a relative path resolved against CWD =
    s3pogs_path; the staged sequence dir lands exactly there)."""
    algo, s3pogs_root = _s3pogs_with_fake_tree(tmp_path)
    request, ctx, _staged_root, _real_image_dir = _build_request_and_context(tmp_path, s3pogs_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    # The podman path generates output_dir/s3pogs_config.yaml; the
    # conda path must NOT (otherwise inherit_from / dataset_path
    # semantics shift unintentionally).
    generated_config = request.output_dir / "s3pogs_config.yaml"
    assert not generated_config.exists()


def test_s3pogs_podman_execution_spec_returns_cmd(tmp_path: Path) -> None:
    algo, s3pogs_root = _s3pogs_with_fake_tree(tmp_path)
    algo = S3POGSAlgorithm(container_runtime="podman")
    algo.s3pogs_path = s3pogs_root
    request, ctx, staged_root, _real_image_dir = _build_request_and_context(tmp_path, s3pogs_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    # The podman path uses a custom_runner so it can stop the container
    # on the "Total FPS" line — S3PO-GS's slam.py hangs in
    # ``backend_process.join()`` after the eval completes (mp.Process
    # cleanup bug). The runner kills the container as soon as the eval
    # is done so the trajectory (already written by the last
    # ``eval_ate`` call) is captured and the next run can start.
    assert spec.custom_runner is not None
    assert spec.target_kind == "podman_container"
    assert spec.target_metadata is not None
    assert spec.target_metadata["container_name"].startswith("s3pogs-04-")
    assert "io_target_paths" in spec.target_metadata

    assert spec.cmd[0] == "podman"
    assert spec.cmd[1] == "run"
    assert "--rm" in spec.cmd
    assert "--name" in spec.cmd
    # S3PO-GS uses torch.multiprocessing.set_start_method("spawn") and
    # spawns frontend/backend/gui worker processes that exchange torch
    # IPC tensors over mp.Queue (slam.py:70-74, 110-113). These tensors
    # back to /dev/shm; default 64 MB shm can SIGBUS the workers right
    # after spawn. Wrapper passes --shm-size=8g preemptively (parity
    # with DROID / Photo / MASt3R / Giga).
    assert "--shm-size=8g" in spec.cmd

    joined = " ".join(spec.cmd)
    # Staged dataset root → /dataset/KITTI/04 read-only. The container
    # path must preserve the KITTI/<NN> components because S3PO-GS's
    # save-subdir derivation (slam.py:274-277) reads path[-3]+"_"+path[-2]
    # to label the result directory.
    assert f"{staged_root.resolve()}:/dataset/KITTI/04:ro" in joined
    # Output dir → /output (the per-run YAML and the trajectory live here).
    assert f"{request.output_dir.resolve()}:/output" in joined
    # HuggingFace cache mounted so MASt3R weights persist across runs.
    assert "/root/.cache/huggingface" in joined
    # Torch hub cache mounted for parity with VGGT/DROID/Photo/MASt3R/Giga.
    assert "/root/.cache/torch/hub" in joined
    assert "s3pogs:latest" in spec.cmd

    # slam.py invocation tail.
    assert spec.cmd[-3] == "bash"
    assert spec.cmd[-2] == "-c"
    main_cmd = spec.cmd[-1]
    assert "cd /s3pogs" in main_cmd
    assert "python -u slam.py" in main_cmd
    assert "--config /output/s3pogs_config.yaml" in main_cmd


def test_s3pogs_podman_writes_container_paths_to_config(tmp_path: Path) -> None:
    """Podman path: the generated per-run config must contain the
    container-side dataset_path and save_dir so the file the container
    reads (via the /output bind-mount) refers to its own filesystem
    layout. The inherit_from points to the relative path of the
    upstream KITTI config inside the image."""
    algo, s3pogs_root = _s3pogs_with_fake_tree(tmp_path)
    algo = S3POGSAlgorithm(container_runtime="podman")
    algo.s3pogs_path = s3pogs_root
    request, ctx, _staged_root, _real_image_dir = _build_request_and_context(tmp_path, s3pogs_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    generated_config = request.output_dir / "s3pogs_config.yaml"
    assert generated_config.exists()
    text = generated_config.read_text(encoding="utf-8")
    assert 'dataset_path: "/dataset/KITTI/04/"' in text
    assert 'save_dir: "/output"' in text
    # inherit_from must be the relative path inside the image (where
    # CWD = /s3pogs and the upstream KITTI configs live at
    # configs/mono/KITTI/04.yaml).
    assert 'inherit_from: "configs/mono/KITTI/04.yaml"' in text


def test_s3pogs_podman_bind_mounts_symlink_targets(tmp_path: Path) -> None:
    """The staged KITTI dir's rgb/ entry is an absolute-path symlink
    into the real perturbed camera dir; the wrapper must bind-mount
    that target at its own host path so the symlink resolves inside
    the container."""
    algo, s3pogs_root = _s3pogs_with_fake_tree(tmp_path)
    algo = S3POGSAlgorithm(container_runtime="podman")
    algo.s3pogs_path = s3pogs_root
    request, ctx, _staged_root, real_image_dir = _build_request_and_context(tmp_path, s3pogs_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    joined = " ".join(spec.cmd)
    real_image_dir_str = str(real_image_dir.resolve())
    # The absolute-symlink target must appear as a same-path bind-mount.
    assert f"{real_image_dir_str}:{real_image_dir_str}:ro" in joined


def test_s3pogs_podman_target_metadata_container_name_is_sanitized(tmp_path: Path) -> None:
    algo, s3pogs_root = _s3pogs_with_fake_tree(tmp_path)
    algo = S3POGSAlgorithm(container_runtime="podman")
    algo.s3pogs_path = s3pogs_root
    request, ctx, _staged_root, _real_image_dir = _build_request_and_context(tmp_path, s3pogs_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    name = spec.target_metadata["container_name"]
    assert name == name.lower()
    assert " " not in name
    assert len(name) <= 120


def test_s3pogs_podman_io_target_paths_includes_dataset_and_output(tmp_path: Path) -> None:
    algo, s3pogs_root = _s3pogs_with_fake_tree(tmp_path)
    algo = S3POGSAlgorithm(container_runtime="podman")
    algo.s3pogs_path = s3pogs_root
    request, ctx, staged_root, _real_image_dir = _build_request_and_context(tmp_path, s3pogs_root)

    spec = algo._build_execution_spec(request, ctx)

    paths = spec.target_metadata["io_target_paths"]
    assert str(staged_root.resolve()) in paths
    assert str(request.output_dir.resolve()) in paths


def test_s3pogs_podman_custom_runner_threads_io_target_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: the custom_runner's inner ``_spawn_streaming_process`` must
    carry ``io_target_paths`` in its ``target_metadata``, not just the outer
    ExecutionSpec. The runtime-stress orchestrator attaches to the process the
    custom_runner spawns; if io_target_paths is dropped there, an IO scenario
    fails with "requires target metadata with io_target_paths" even though the
    outer spec looks correct.
    """
    algo, s3pogs_root = _s3pogs_with_fake_tree(tmp_path)
    algo = S3POGSAlgorithm(container_runtime="podman")
    algo.s3pogs_path = s3pogs_root
    request, ctx, _staged_root, _real_image_dir = _build_request_and_context(
        tmp_path, s3pogs_root
    )

    spec = algo._build_execution_spec(request, ctx)

    captured: dict = {}

    class _FakeProc:
        returncode = 0

        def wait(self, timeout=None):
            return 0

    def _fake_spawn(cmd, target_kind=None, target_metadata=None):
        captured["target_kind"] = target_kind
        captured["target_metadata"] = target_metadata
        return _FakeProc()

    monkeypatch.setattr(algo, "_spawn_streaming_process", _fake_spawn)
    # Pretend the "Total FPS" line appeared so the runner takes the
    # stop-early branch and returns without waiting on a real process.
    monkeypatch.setattr(algo, "_stream_process_output", lambda *a, **k: True)
    monkeypatch.setattr(algo, "_stop_container", lambda *a, **k: None)

    spec.custom_runner(spec)

    assert captured["target_kind"] == "podman_container"
    inner_md = captured["target_metadata"]
    assert inner_md is not None
    assert "io_target_paths" in inner_md, (
        "custom_runner dropped io_target_paths from the spawned process's "
        "target_metadata; IO runtime-stress would fail to attach"
    )
    assert str(request.output_dir.resolve()) in inner_md["io_target_paths"]
