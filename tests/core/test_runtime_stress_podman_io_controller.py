"""Tests for the Podman block-IO runtime-stress controller."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List

import pytest

from slamadversariallab.runtime_stress import podman_controllers
from slamadversariallab.runtime_stress.podman_controllers import PodmanIoController
from slamadversariallab.runtime_stress.models import IoControl, RuntimeStressControls


class _Result:
    def __init__(self, returncode=0, stdout="", stderr="") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Process:
    def poll(self):
        return None


def _install_subprocess(monkeypatch, runner: Callable) -> List[List[str]]:
    """Wrap a runner so we record commands and return the runner's response."""
    commands: List[List[str]] = []

    def _fake_run(cmd, capture_output=False, text=False, check=False, timeout=None):
        commands.append(list(cmd))
        return runner(cmd)

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.podman_controllers.subprocess.run",
        _fake_run,
    )
    return commands


def _patch_cgroup_delegated(monkeypatch, *, delegated: bool = True) -> None:
    """Patch the host's cgroup.controllers file to look delegated (or not)."""
    fake = type("FakeControllersPath", (), {})()
    fake.exists = lambda: True  # type: ignore[assignment]
    fake.read_text = (lambda: "cpuset cpu io memory pids") if delegated else (lambda: "cpuset cpu memory pids")
    monkeypatch.setattr(
        podman_controllers, "CGROUP_V2_CONTROLLERS_PATH", fake, raising=False
    )


def _patch_device_resolution(
    monkeypatch,
    paths_to_devices: dict,
    *,
    schedulers: dict | None = None,
    pseudo_paths: set | None = None,
):
    """Stub out _resolve_target_devices to return canned _IoDevice instances."""
    schedulers = schedulers or {}
    pseudo_paths = pseudo_paths or set()

    def _fake_resolve(paths):
        seen = set()
        out = []
        for path in paths:
            device_path = paths_to_devices.get(path)
            if device_path is None or device_path in seen:
                continue
            seen.add(device_path)
            major, minor = {
                "/dev/sda": (8, 0),
                "/dev/sdb": (8, 16),
                "/dev/nvme0n1": (259, 0),
                "/dev/loop0": (7, 0),
            }.get(device_path, (8, 0))
            out.append(
                podman_controllers._IoDevice(
                    path=device_path,
                    major=major,
                    minor=minor,
                    scheduler=schedulers.get(device_path),
                    pseudo=device_path in pseudo_paths,
                )
            )
        return out

    monkeypatch.setattr(podman_controllers, "_resolve_target_devices", _fake_resolve)


# ----------------- prepare ----------------- #


def _patch_io_max_file(monkeypatch, tmp_path):
    """Give the controller a real, writable io.max to enforce into.

    `apply()` no longer trusts `podman update`: it writes io.max directly and
    reads it back, because podman exits 0 and records the setting while leaving
    the cgroup untouched (measured on this host under rootless podman). So these
    tests must supply a file, and the read-back has to see the device line.
    """
    io_max = tmp_path / "io.max"
    io_max.write_text("")

    real_write = type(io_max).write_text

    def _write_and_echo(self, data, *a, **kw):
        # The kernel keeps the line it accepted; mimic that so read-back passes.
        return real_write(self, data if data.endswith("\n") else data + "\n", *a, **kw)

    monkeypatch.setattr(
        podman_controllers.PodmanIoController,
        "_cgroup_io_max_path",
        lambda self: io_max,
    )
    return io_max


def test_prepare_resolves_devices_from_target_paths(monkeypatch, tmp_path) -> None:
    _patch_cgroup_delegated(monkeypatch, delegated=True)
    _patch_device_resolution(
        monkeypatch,
        {"/data/euroc": "/dev/sda", "/results": "/dev/sda"},
    )

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {}, "State": {"CgroupPath": "/sys/fs/cgroup/x"}}]')
        raise AssertionError(f"Unexpected command: {cmd}")

    _install_subprocess(monkeypatch, runner)

    controller = PodmanIoController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={
            "container_name": "test-container",
            "io_target_paths": ["/data/euroc", "/results"],
        },
    )

    summaries = controller.target_device_summaries()
    assert len(summaries) == 1
    assert summaries[0]["path"] == "/dev/sda"
    assert summaries[0]["major_minor"] == "8:0"


def test_prepare_raises_when_cgroup_io_not_delegated(monkeypatch) -> None:
    _patch_cgroup_delegated(monkeypatch, delegated=False)
    controller = PodmanIoController()
    with pytest.raises(RuntimeError, match="'io' controller delegated"):
        controller.prepare(
            _Process(),
            target_kind="podman_container",
            target_metadata={
                "container_name": "test-container",
                "io_target_paths": ["/results"],
            },
        )


def test_prepare_rejects_non_podman_target(monkeypatch) -> None:
    _patch_cgroup_delegated(monkeypatch)
    controller = PodmanIoController()
    with pytest.raises(RuntimeError, match="podman_container"):
        controller.prepare(
            _Process(),
            target_kind="docker_container",
            target_metadata={"container_name": "x", "io_target_paths": ["/x"]},
        )


def test_prepare_requires_io_target_paths(monkeypatch) -> None:
    _patch_cgroup_delegated(monkeypatch)
    controller = PodmanIoController()
    with pytest.raises(RuntimeError, match="io_target_paths"):
        controller.prepare(
            _Process(),
            target_kind="podman_container",
            target_metadata={"container_name": "test-container"},
        )


def test_target_devices_dedup_across_dataset_and_output_paths(monkeypatch) -> None:
    _patch_cgroup_delegated(monkeypatch)
    _patch_device_resolution(
        monkeypatch,
        {"/data/euroc": "/dev/sda", "/results": "/dev/sda"},
    )

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {}, "State": {"CgroupPath": ""}}]')
        raise AssertionError(f"Unexpected command: {cmd}")

    _install_subprocess(monkeypatch, runner)

    controller = PodmanIoController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={
            "container_name": "test-container",
            "io_target_paths": ["/data/euroc", "/results"],
        },
    )

    summaries = controller.target_device_summaries()
    assert [s["path"] for s in summaries] == ["/dev/sda"]


# ----------------- apply ----------------- #


def test_apply_emits_correct_podman_update_argv(monkeypatch, tmp_path) -> None:
    _patch_cgroup_delegated(monkeypatch)
    _patch_io_max_file(monkeypatch, tmp_path)
    _patch_device_resolution(monkeypatch, {"/results": "/dev/sda"})

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {}, "State": {"CgroupPath": "/scope"}}]')
        if cmd[:2] == ["podman", "update"]:
            return _Result(stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    commands = _install_subprocess(monkeypatch, runner)

    controller = PodmanIoController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "ctr", "io_target_paths": ["/results"]},
    )

    controller.apply(
        RuntimeStressControls(
            io=IoControl(read_bps=50_000_000, write_bps=10_000_000)
        )
    )

    update_calls = [cmd for cmd in commands if cmd[:2] == ["podman", "update"]]
    assert update_calls == [
        [
            "podman", "update",
            "--device-read-bps", "/dev/sda:50000000",
            "--device-write-bps", "/dev/sda:10000000",
            "ctr",
        ]
    ]


def test_apply_with_partial_controls_emits_only_set_flags(monkeypatch, tmp_path) -> None:
    _patch_cgroup_delegated(monkeypatch)
    _patch_io_max_file(monkeypatch, tmp_path)
    _patch_device_resolution(monkeypatch, {"/results": "/dev/sda"})

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {}, "State": {"CgroupPath": "/scope"}}]')
        if cmd[:2] == ["podman", "update"]:
            return _Result(stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    commands = _install_subprocess(monkeypatch, runner)

    controller = PodmanIoController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "ctr", "io_target_paths": ["/results"]},
    )
    controller.apply(RuntimeStressControls(io=IoControl(read_bps=42_000_000)))

    update_calls = [cmd for cmd in commands if cmd[:2] == ["podman", "update"]]
    assert update_calls == [
        ["podman", "update", "--device-read-bps", "/dev/sda:42000000", "ctr"]
    ]


def test_iops_warning_on_nvme_scheduler(monkeypatch, tmp_path, caplog) -> None:
    _patch_cgroup_delegated(monkeypatch)
    _patch_io_max_file(monkeypatch, tmp_path)
    _patch_device_resolution(
        monkeypatch,
        {"/results": "/dev/nvme0n1"},
        schedulers={"/dev/nvme0n1": "mq-deadline"},
    )

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {}, "State": {"CgroupPath": "/scope"}}]')
        if cmd[:2] == ["podman", "update"]:
            return _Result(stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    _install_subprocess(monkeypatch, runner)

    controller = PodmanIoController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "ctr", "io_target_paths": ["/results"]},
    )

    caplog.set_level(logging.WARNING)
    package_logger = logging.getLogger("slamadversariallab")
    prior_propagate = package_logger.propagate
    package_logger.propagate = True
    try:
        controller.apply(RuntimeStressControls(io=IoControl(read_iops=1000)))
    finally:
        package_logger.propagate = prior_propagate

    assert any("IOPS throttling" in rec.message for rec in caplog.records)


# ----------------- release ----------------- #


def test_release_no_op_when_never_throttled(monkeypatch) -> None:
    _patch_cgroup_delegated(monkeypatch)
    _patch_device_resolution(monkeypatch, {"/results": "/dev/sda"})

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {}, "State": {"CgroupPath": "/scope"}}]')
        raise AssertionError(f"Unexpected command: {cmd}")

    commands = _install_subprocess(monkeypatch, runner)

    controller = PodmanIoController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "ctr", "io_target_paths": ["/results"]},
    )
    controller.release()

    assert all(cmd[:2] != ["podman", "update"] for cmd in commands)


def test_release_writes_max_to_io_max_after_throttle(monkeypatch, tmp_path) -> None:
    _patch_cgroup_delegated(monkeypatch)
    _patch_device_resolution(monkeypatch, {"/results": "/dev/sda"})

    cgroup_dir = tmp_path / "scope" / "container"
    cgroup_dir.mkdir(parents=True)
    io_max = cgroup_dir / "io.max"
    io_max.write_text("8:0 rbps=1048576 wbps=max riops=max wiops=max")

    monkeypatch.setattr(
        podman_controllers, "Path", Path
    )  # ensure Path remains the real one
    # Redirect /sys/fs/cgroup to tmp_path by patching the helper
    real_method = PodmanIoController._cgroup_io_max_path

    def _fake_io_max_path(self):  # type: ignore[no-redef]
        return io_max

    monkeypatch.setattr(PodmanIoController, "_cgroup_io_max_path", _fake_io_max_path)

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {}, "State": {"CgroupPath": "/scope"}}]')
        if cmd[:2] == ["podman", "update"]:
            return _Result(stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    _install_subprocess(monkeypatch, runner)

    controller = PodmanIoController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "ctr", "io_target_paths": ["/results"]},
    )
    controller.apply(RuntimeStressControls(io=IoControl(read_bps=1_048_576)))
    controller.release()

    assert io_max.read_text() == "8:0 rbps=max wbps=max riops=max wiops=max"


def test_release_restores_originals_when_present(monkeypatch, tmp_path) -> None:
    _patch_cgroup_delegated(monkeypatch)
    _patch_device_resolution(monkeypatch, {"/results": "/dev/sda"})

    cgroup_dir = tmp_path / "scope" / "container"
    cgroup_dir.mkdir(parents=True)
    io_max = cgroup_dir / "io.max"
    io_max.write_text("8:0 rbps=1048576 wbps=max riops=max wiops=max")

    def _fake_io_max_path(self):  # type: ignore[no-redef]
        return io_max

    monkeypatch.setattr(PodmanIoController, "_cgroup_io_max_path", _fake_io_max_path)

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(
                stdout='[{"HostConfig": {"BlkioDeviceReadBps": [{"Path": "/dev/sda", "Rate": 5000000}]}, "State": {"CgroupPath": "/scope"}}]'
            )
        if cmd[:2] == ["podman", "update"]:
            return _Result(stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    _install_subprocess(monkeypatch, runner)

    controller = PodmanIoController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "ctr", "io_target_paths": ["/results"]},
    )
    controller.apply(RuntimeStressControls(io=IoControl(read_bps=1_048_576)))
    controller.release()

    assert io_max.read_text() == "8:0 rbps=5000000 wbps=max riops=max wiops=max"


# ----------------- cleanup ----------------- #


def test_cleanup_is_noop(monkeypatch) -> None:
    _patch_cgroup_delegated(monkeypatch)
    _patch_device_resolution(monkeypatch, {"/results": "/dev/sda"})

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {}, "State": {"CgroupPath": "/scope"}}]')
        raise AssertionError(f"Unexpected command: {cmd}")

    commands = _install_subprocess(monkeypatch, runner)

    controller = PodmanIoController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "ctr", "io_target_paths": ["/results"]},
    )
    controller.cleanup()

    assert all(cmd[:2] != ["podman", "update"] for cmd in commands)
    assert controller.target_device_summaries() == []


def test_apply_writes_io_max_directly_because_podman_update_does_not(monkeypatch, tmp_path) -> None:
    """The silent no-op this guard exists for, measured on this host.

    `podman update --device-read-bps /dev/sda:100000000` exits 0 and records
    `BlkioDeviceReadBps: [{/dev/sda 100000000}]` in inspect, while the container's
    io.max stays EMPTY. Every IO cell measured that way was unthrottled no matter
    what its config said. A direct write takes effect immediately, which is why
    release() has always used one. Apply must too.
    """
    _patch_cgroup_delegated(monkeypatch, delegated=True)
    io_max = _patch_io_max_file(monkeypatch, tmp_path)
    _patch_device_resolution(monkeypatch, {"/results": "/dev/sda"})

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {}, "State": {"CgroupPath": "/x"}}]')
        if cmd[:2] == ["podman", "update"]:
            return _Result(stdout="ok\n")          # podman "succeeds" and does nothing
        raise AssertionError(f"Unexpected command: {cmd}")

    _install_subprocess(monkeypatch, runner)
    c = PodmanIoController()
    c.prepare(_Process(), target_kind="podman_container",
              target_metadata={"container_name": "x", "io_target_paths": ["/results"]})
    c.apply(RuntimeStressControls(io=IoControl(read_bps=100_000_000)))

    written = io_max.read_text()
    assert "8:0" in written, written
    assert "rbps=100000000" in written, written


def test_apply_raises_when_io_max_cannot_be_resolved(monkeypatch, tmp_path) -> None:
    """No cgroup to write means the cap cannot be enforced, so the run must stop.

    Continuing would score an UNTHROTTLED run as IO-limited, which is worse than
    failing: it produces a number that looks like a measurement of a slow disk.
    """
    _patch_cgroup_delegated(monkeypatch, delegated=True)
    _patch_device_resolution(monkeypatch, {"/results": "/dev/sda"})
    monkeypatch.setattr(podman_controllers.PodmanIoController,
                        "_cgroup_io_max_path", lambda self: None)

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {}, "State": {"CgroupPath": "/x"}}]')
        if cmd[:2] == ["podman", "update"]:
            return _Result(stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    _install_subprocess(monkeypatch, runner)
    c = PodmanIoController()
    c.prepare(_Process(), target_kind="podman_container",
              target_metadata={"container_name": "x", "io_target_paths": ["/results"]})
    with pytest.raises(RuntimeError, match="cannot be enforced|Refusing"):
        c.apply(RuntimeStressControls(io=IoControl(read_bps=100_000_000)))


def test_apply_raises_when_the_device_line_does_not_stick(monkeypatch, tmp_path) -> None:
    """The kernel accepts a write but keeps only limits it can enforce.

    If our device's line is absent on read-back the cap did not take, and that
    must fail the run rather than pass silently.
    """
    _patch_cgroup_delegated(monkeypatch, delegated=True)
    _patch_device_resolution(monkeypatch, {"/results": "/dev/sda"})
    io_max = tmp_path / "io.max"
    io_max.write_text("")
    # Swallow the write: the file stays empty, as an unenforceable device would.
    monkeypatch.setattr(type(io_max), "write_text", lambda self, *a, **k: None)
    monkeypatch.setattr(podman_controllers.PodmanIoController,
                        "_cgroup_io_max_path", lambda self: io_max)

    def runner(cmd):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {}, "State": {"CgroupPath": "/x"}}]')
        if cmd[:2] == ["podman", "update"]:
            return _Result(stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    _install_subprocess(monkeypatch, runner)
    c = PodmanIoController()
    c.prepare(_Process(), target_kind="podman_container",
              target_metadata={"container_name": "x", "io_target_paths": ["/results"]})
    with pytest.raises(RuntimeError, match="did not take effect|no entry"):
        c.apply(RuntimeStressControls(io=IoControl(read_bps=100_000_000)))
