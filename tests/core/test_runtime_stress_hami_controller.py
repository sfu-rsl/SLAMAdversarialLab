"""Tests for the HAMi-core GPU isolation controller."""

import logging
from pathlib import Path

import pytest

from slamadversariallab.runtime_stress.hami_controller import (
    GpuHamiController,
    HAMI_LIB_CONTAINER_PATH,
    HAMI_SHARED_CACHE,
    hami_launch_env,
    hami_launch_mounts,
)
from slamadversariallab.runtime_stress.models import (
    GpuControl,
    MemoryControl,
    RuntimeStressControls,
)


class _Process:
    def poll(self):
        return None


def _make_controller(tmp_path, *, lib_name: str = "libvgpu.so") -> GpuHamiController:
    lib_path = tmp_path / lib_name
    lib_path.write_bytes(b"\x7fELF-stub")
    lock_dir = tmp_path / "vgpulock"
    return GpuHamiController(lib_path=str(lib_path), lock_dir=str(lock_dir))


def test_hami_controller_prepare_validates_libvgpu_exists(tmp_path) -> None:
    missing_lib = tmp_path / "missing-libvgpu.so"
    controller = GpuHamiController(
        lib_path=str(missing_lib),
        lock_dir=str(tmp_path / "vgpulock"),
    )
    with pytest.raises(RuntimeError, match="HAMi library not found"):
        controller.prepare(
            _Process(),
            target_kind="podman_container",
            target_metadata={"container_name": "c"},
        )


def test_hami_controller_prepare_creates_vgpulock_dir_if_missing(tmp_path) -> None:
    controller = _make_controller(tmp_path)
    lock_dir = tmp_path / "vgpulock"
    assert not lock_dir.exists()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "c"},
    )
    assert lock_dir.is_dir()


def test_hami_controller_prepare_clears_stale_cudevshr_cache(tmp_path, monkeypatch) -> None:
    stale_cache = tmp_path / "cudevshr.cache"
    stale_cache.write_text("stale")
    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.hami_controller.HAMI_SHARED_CACHE",
        str(stale_cache),
    )

    controller = _make_controller(tmp_path)
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "c"},
    )
    assert not stale_cache.exists()


def test_hami_controller_apply_no_ops_when_cache_file_missing(tmp_path) -> None:
    """If the wrapper didn't pre-create the bind-mounted cache file, apply()
    must safely no-op (legacy behavior) rather than crash. The launch-time
    env-injected cap stays in force for the whole run."""
    controller = _make_controller(tmp_path)
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "missing-cache-container"},
    )
    # No cache file exists for this container_name → controller's
    # _cache_host_path stays None and apply() must be a no-op.
    controller.apply(
        RuntimeStressControls(gpu=GpuControl(vram_limit_mb=4096, sm_limit_percent=50))
    )
    controller.release()  # also no-op
    # No exceptions; nothing to assert beyond "didn't crash"


def test_hami_controller_apply_writes_uint64_to_cache_file_at_correct_offsets(
    tmp_path, monkeypatch
) -> None:
    """apply() should pwrite the new caps into limit[0] (offset 1632) and
    sm_limit[0] (offset 1760) of the bind-mounted host cache file."""
    import struct
    from slamadversariallab.runtime_stress import hami_controller as hc

    # Redirect the per-container cache dir into tmp_path so we don't pollute
    # the real ~/.cache/sal/hami/.
    monkeypatch.setattr(hc, "HAMI_CACHE_HOST_DIR", tmp_path / "hami_cache")

    container_name = "uut-container"
    cache_path = hc.prepare_hami_cache_file(container_name)
    assert cache_path.exists()
    assert cache_path.stat().st_size == hc.SHARED_REGION_SIZE_BYTES

    # Simulate libvgpu.so init: write a non-zero initialized_flag and the
    # expected major/minor versions into the file.
    with cache_path.open("r+b") as f:
        f.seek(hc.INITIALIZED_FLAG_OFFSET)
        f.write(struct.pack("<i", 19920718))
        f.seek(hc.MAJOR_VERSION_OFFSET)
        f.write(struct.pack("<II", hc.EXPECTED_MAJOR_VERSION, hc.EXPECTED_MINOR_VERSION))

    controller = _make_controller(tmp_path)
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": container_name},
    )

    controller.apply(
        RuntimeStressControls(gpu=GpuControl(vram_limit_mb=2048, sm_limit_percent=75))
    )

    with cache_path.open("rb") as f:
        f.seek(hc.LIMIT_OFFSET)
        vram_bytes = struct.unpack("<Q", f.read(8))[0]
        f.seek(hc.SM_LIMIT_OFFSET)
        sm_value = struct.unpack("<Q", f.read(8))[0]

    assert vram_bytes == 2048 * 1024 * 1024  # 2 GB in bytes
    assert sm_value == 75


def test_hami_controller_apply_writes_zero_when_gpu_unset_for_phase(
    tmp_path, monkeypatch
) -> None:
    """A phase with no GPU controls should drop the cap (write 0)."""
    import struct
    from slamadversariallab.runtime_stress import hami_controller as hc

    monkeypatch.setattr(hc, "HAMI_CACHE_HOST_DIR", tmp_path / "hami_cache")

    container_name = "drop-cap-container"
    cache_path = hc.prepare_hami_cache_file(container_name)
    with cache_path.open("r+b") as f:
        f.seek(hc.INITIALIZED_FLAG_OFFSET)
        f.write(struct.pack("<i", 1))
        f.seek(hc.MAJOR_VERSION_OFFSET)
        f.write(struct.pack("<II", hc.EXPECTED_MAJOR_VERSION, hc.EXPECTED_MINOR_VERSION))
        # Pre-existing seeded cap to confirm apply() overwrites it.
        f.seek(hc.LIMIT_OFFSET)
        f.write(struct.pack("<Q", 4 * 1024**3))

    controller = _make_controller(tmp_path)
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": container_name},
    )
    controller.apply(RuntimeStressControls())  # no gpu controls

    with cache_path.open("rb") as f:
        f.seek(hc.LIMIT_OFFSET)
        assert struct.unpack("<Q", f.read(8))[0] == 0


def test_hami_controller_warns_on_version_mismatch(tmp_path, monkeypatch, caplog) -> None:
    """If the cache file's major/minor version doesn't match what the
    DWARF-derived offsets were computed against, apply() should still
    proceed (we have no fallback) but log a clear WARNING so the operator
    knows the offsets may be stale and the write may corrupt state."""
    import logging
    import struct
    from slamadversariallab.runtime_stress import hami_controller as hc

    monkeypatch.setattr(hc, "HAMI_CACHE_HOST_DIR", tmp_path / "hami_cache")

    container_name = "version-mismatch-container"
    cache_path = hc.prepare_hami_cache_file(container_name)
    with cache_path.open("r+b") as f:
        f.seek(hc.INITIALIZED_FLAG_OFFSET)
        f.write(struct.pack("<i", 1))
        # Write WRONG versions (off by one) to trigger the mismatch path.
        f.seek(hc.MAJOR_VERSION_OFFSET)
        f.write(struct.pack("<II",
                            hc.EXPECTED_MAJOR_VERSION + 1,
                            hc.EXPECTED_MINOR_VERSION + 1))

    controller = _make_controller(tmp_path)
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": container_name},
    )

    caplog.set_level(logging.WARNING)
    package_logger = logging.getLogger("slamadversariallab")
    prior_propagate = package_logger.propagate
    package_logger.propagate = True
    try:
        controller.apply(
            RuntimeStressControls(gpu=GpuControl(vram_limit_mb=1024))
        )
    finally:
        package_logger.propagate = prior_propagate

    assert any(
        "version mismatch" in rec.message.lower() for rec in caplog.records
    ), f"Expected version-mismatch warning; got: {[r.message for r in caplog.records]}"


def test_hami_controller_release_zeros_limit_and_sm_fields(
    tmp_path, monkeypatch
) -> None:
    import struct
    from slamadversariallab.runtime_stress import hami_controller as hc

    monkeypatch.setattr(hc, "HAMI_CACHE_HOST_DIR", tmp_path / "hami_cache")

    container_name = "release-container"
    cache_path = hc.prepare_hami_cache_file(container_name)
    # Pre-populate non-zero caps to confirm release zeros them.
    with cache_path.open("r+b") as f:
        f.seek(hc.LIMIT_OFFSET)
        f.write(struct.pack("<Q", 999_999_999))
        f.seek(hc.SM_LIMIT_OFFSET)
        f.write(struct.pack("<Q", 50))

    controller = _make_controller(tmp_path)
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": container_name},
    )
    controller.release()

    with cache_path.open("rb") as f:
        f.seek(hc.LIMIT_OFFSET)
        assert struct.unpack("<Q", f.read(8))[0] == 0
        f.seek(hc.SM_LIMIT_OFFSET)
        assert struct.unpack("<Q", f.read(8))[0] == 0


def test_hami_controller_cleanup_removes_cudevshr_cache(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "cudevshr.cache"
    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.hami_controller.HAMI_SHARED_CACHE",
        str(cache_path),
    )

    controller = _make_controller(tmp_path)
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "c"},
    )
    cache_path.write_text("residual")
    controller.cleanup()
    assert not cache_path.exists()


def test_hami_launch_env_builds_correct_vars() -> None:
    env = hami_launch_env(GpuControl(vram_limit_mb=4096, sm_limit_percent=50))
    assert env["LD_PRELOAD"] == HAMI_LIB_CONTAINER_PATH
    assert env["CUDA_DEVICE_MEMORY_LIMIT"] == "4096m"
    assert env["CUDA_DEVICE_SM_LIMIT"] == "50"


def test_hami_launch_env_handles_vram_only() -> None:
    env = hami_launch_env(GpuControl(vram_limit_mb=2048))
    assert env["LD_PRELOAD"] == HAMI_LIB_CONTAINER_PATH
    assert env["CUDA_DEVICE_MEMORY_LIMIT"] == "2048m"
    assert "CUDA_DEVICE_SM_LIMIT" not in env


def test_hami_launch_env_handles_sm_only() -> None:
    env = hami_launch_env(GpuControl(sm_limit_percent=30))
    assert env["LD_PRELOAD"] == HAMI_LIB_CONTAINER_PATH
    assert env["CUDA_DEVICE_SM_LIMIT"] == "30"
    assert "CUDA_DEVICE_MEMORY_LIMIT" not in env


def test_hami_launch_mounts_binds_libvgpu_read_only() -> None:
    mounts = hami_launch_mounts()
    assert len(mounts) == 1
    src, dst, mode = mounts[0]
    assert src.endswith("libvgpu.so")
    assert dst == HAMI_LIB_CONTAINER_PATH
    assert mode == "ro"


@pytest.mark.parametrize("target_kind", ["docker_container", "podman_container"])
def test_hami_controller_accepts_container_target_kinds(tmp_path, target_kind) -> None:
    controller = _make_controller(tmp_path)
    controller.prepare(
        _Process(),
        target_kind=target_kind,
        target_metadata={"container_name": "c"},
    )


def test_hami_controller_rejects_host_process_group_in_v1(tmp_path) -> None:
    controller = _make_controller(tmp_path)
    with pytest.raises(RuntimeError, match="host_process_group"):
        controller.prepare(
            _Process(),
            target_kind="host_process_group",
            target_metadata=None,
        )


def test_hami_controller_rejects_unknown_target(tmp_path) -> None:
    controller = _make_controller(tmp_path)
    with pytest.raises(RuntimeError, match="container targets"):
        controller.prepare(
            _Process(),
            target_kind="weird_target",
            target_metadata=None,
        )
