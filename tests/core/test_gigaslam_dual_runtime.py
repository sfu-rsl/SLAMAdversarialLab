"""Tests for GigaSLAM dual-runtime (conda default + podman opt-in) wrapper."""

from pathlib import Path

import pytest

from slamadversariallab.algorithms.gigaslam import GigaSLAMAlgorithm
from slamadversariallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRuntimeContext,
    SensorMode,
)


def _gigaslam_with_fake_tree(tmp_path: Path) -> tuple[GigaSLAMAlgorithm, Path]:
    """Build a GigaSLAMAlgorithm pointed at an isolated fake source tree.

    Lets dual-runtime tests run without the real GigaSLAM submodule on
    disk, mirroring the MASt3R-SLAM dual-runtime test setup.
    """
    giga_root = tmp_path / "GigaSLAM"
    cfg_dir = giga_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # Minimal kitti_06.yaml carrying the keys _create_config rewrites.
    (cfg_dir / "kitti_06.yaml").write_text(
        "inherit_from: \"configs/base_config.yaml\"\n"
        "\n"
        "Dataset:\n"
        "  color_path: \"/path/to/KITTI/sequences/04/image_2\"\n"
        "  type: kitti\n",
        encoding="utf-8",
    )
    algo = GigaSLAMAlgorithm()
    algo.gigaslam_path = giga_root
    return algo, giga_root


def _build_request_and_context(
    tmp_path: Path,
    giga_root: Path,
) -> tuple[SLAMRunRequest, SLAMRuntimeContext, Path, Path]:
    """Build a GigaSLAM request mirroring a real KITTI mono evaluation run.

    The host left-camera dir is laid out as
    ``tmp_path/kitti/sequences/04/image_2/`` so that
    ``_build_container_execution_spec`` can derive the container-side
    sequence root + image leaf cleanly.
    """
    sequence_root = tmp_path / "kitti" / "sequences" / "04"
    image_dir = sequence_root / "image_2"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "000000.png").write_bytes(b"")
    (image_dir / "000001.png").write_bytes(b"")
    (sequence_root / "calib.txt").write_text("P0: 0\n", encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    request = SLAMRunRequest(
        dataset_path=sequence_root,
        slam_config="kitti_06",
        output_dir=output_dir,
        dataset_type="kitti",
        sensor_mode=SensorMode.MONO,
        sequence_name="04",
        extras={
            "camera_paths": {"left": str(image_dir)},
            "timestamps_by_frame": {0: 0.0, 1: 0.1},
        },
    )
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name="kitti_06",
        sequence_name=request.sequence_name,
        effective_dataset_path=sequence_root,
    )
    ctx.execution_inputs = {
        "dataset_path": sequence_root,
        "output_dir": output_dir,
        "is_stereo": False,
        "image_dir": image_dir,
    }
    return request, ctx, sequence_root, image_dir


def test_gigaslam_default_constructor_is_conda() -> None:
    algo = GigaSLAMAlgorithm()
    assert algo.container_runtime is None
    assert algo.runtime_stress_target_kind == "host_process_group"
    assert algo.docker_image == "gigaslam:latest"


def test_gigaslam_podman_runtime_selection() -> None:
    algo = GigaSLAMAlgorithm(container_runtime="podman")
    assert algo.container_runtime == "podman"
    assert algo.runtime_stress_target_kind == "podman_container"
    assert algo.docker_image == "gigaslam:latest"


def test_gigaslam_invalid_runtime_raises() -> None:
    with pytest.raises(ValueError, match="container_runtime must be None or 'podman'"):
        GigaSLAMAlgorithm(container_runtime="docker")


def test_gigaslam_conda_execution_spec_uses_custom_runner(tmp_path: Path) -> None:
    algo, giga_root = _gigaslam_with_fake_tree(tmp_path)
    request, ctx, _sequence_root, _image_dir = _build_request_and_context(tmp_path, giga_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    assert spec.custom_runner is not None
    assert spec.cmd == ["gigaslam"]
    # Default for conda spec: target_kind is the base-class default
    # (None on dataclass init, normalized to "host_process_group" on
    # explicit query).
    assert spec.target_kind in (None, "host_process_group")


def test_gigaslam_conda_execution_spec_writes_host_paths_to_config(tmp_path: Path) -> None:
    """Conda path: the generated per-run config must contain the HOST left
    camera path in color_path and the HOST output_dir in save_dir."""
    algo, giga_root = _gigaslam_with_fake_tree(tmp_path)
    request, ctx, _sequence_root, image_dir = _build_request_and_context(tmp_path, giga_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    generated_config = request.output_dir / "gigaslam_config.yaml"
    assert generated_config.exists()
    text = generated_config.read_text(encoding="utf-8")
    assert f'color_path: "{image_dir}"' in text
    assert f'save_dir: "{request.output_dir}"' in text


def test_gigaslam_podman_execution_spec_returns_cmd(tmp_path: Path) -> None:
    algo, giga_root = _gigaslam_with_fake_tree(tmp_path)
    algo = GigaSLAMAlgorithm(container_runtime="podman")
    algo.gigaslam_path = giga_root
    request, ctx, sequence_root, _image_dir = _build_request_and_context(tmp_path, giga_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    assert spec.custom_runner is None
    assert spec.target_kind == "podman_container"
    assert spec.target_metadata is not None
    assert spec.target_metadata["container_name"].startswith("gigaslam-04-")
    assert "io_target_paths" in spec.target_metadata

    assert spec.cmd[0] == "podman"
    assert spec.cmd[1] == "run"
    assert "--rm" in spec.cmd
    assert "--name" in spec.cmd
    # GigaSLAM uses torch.multiprocessing.set_start_method("spawn") and
    # spawns backend / gui worker processes that back IPC tensors in
    # /dev/shm; Podman's default 64 MB shm can SIGBUS the workers. The
    # wrapper passes --shm-size=8g preemptively (parity with DROID /
    # Photo / MASt3R).
    assert "--shm-size=8g" in spec.cmd

    joined = " ".join(spec.cmd)
    # Sequence root → /dataset/kitti/sequences/04 read-only. The container
    # path must preserve the sequences/<NN> / image_<X> components because
    # GigaSLAM derives the save-subdir name from color_path.split("/").
    assert f"{sequence_root.resolve()}:/dataset/kitti/sequences/04:ro" in joined
    # Output dir → /output (the per-run YAML and the trajectory live here).
    assert f"{request.output_dir.resolve()}:/output" in joined
    # HuggingFace cache mounted so UniDepthV2 weights persist across runs.
    assert "/root/.cache/huggingface" in joined
    # Torch hub cache mounted for parity with VGGT/DROID/Photo/MASt3R.
    assert "/root/.cache/torch/hub" in joined
    assert "gigaslam:latest" in spec.cmd

    # slam.py invocation tail.
    assert spec.cmd[-3] == "bash"
    assert spec.cmd[-2] == "-c"
    main_cmd = spec.cmd[-1]
    assert "cd /gigaslam" in main_cmd
    assert "python -u slam.py" in main_cmd
    assert "--config /output/gigaslam_config.yaml" in main_cmd
    assert "--eval" in main_cmd


def test_gigaslam_podman_writes_container_paths_to_config(tmp_path: Path) -> None:
    """Podman path: the generated per-run config must contain the
    CONTAINER-side paths so the file the container reads (via the
    /output bind-mount) refers to its own filesystem layout."""
    algo, giga_root = _gigaslam_with_fake_tree(tmp_path)
    algo = GigaSLAMAlgorithm(container_runtime="podman")
    algo.gigaslam_path = giga_root
    request, ctx, _sequence_root, _image_dir = _build_request_and_context(tmp_path, giga_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    generated_config = request.output_dir / "gigaslam_config.yaml"
    assert generated_config.exists()
    text = generated_config.read_text(encoding="utf-8")
    assert 'color_path: "/dataset/kitti/sequences/04/image_2"' in text
    assert 'save_dir: "/output"' in text


def test_gigaslam_podman_target_metadata_container_name_is_sanitized(tmp_path: Path) -> None:
    algo, giga_root = _gigaslam_with_fake_tree(tmp_path)
    algo = GigaSLAMAlgorithm(container_runtime="podman")
    algo.gigaslam_path = giga_root
    request, ctx, _sequence_root, _image_dir = _build_request_and_context(tmp_path, giga_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    name = spec.target_metadata["container_name"]
    assert name == name.lower()
    assert " " not in name
    assert len(name) <= 120


def test_gigaslam_podman_io_target_paths_includes_sequence_root_and_output(tmp_path: Path) -> None:
    algo, giga_root = _gigaslam_with_fake_tree(tmp_path)
    algo = GigaSLAMAlgorithm(container_runtime="podman")
    algo.gigaslam_path = giga_root
    request, ctx, sequence_root, _image_dir = _build_request_and_context(tmp_path, giga_root)

    spec = algo._build_execution_spec(request, ctx)

    paths = spec.target_metadata["io_target_paths"]
    assert str(sequence_root.resolve()) in paths
    assert str(request.output_dir.resolve()) in paths
