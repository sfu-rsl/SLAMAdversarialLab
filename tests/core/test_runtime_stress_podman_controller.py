"""Tests for Podman-backed runtime-stress controllers."""

import pytest

from slamadversariallab.runtime_stress.podman_controllers import (
    PodmanCpuController,
    PodmanMemoryController,
)
from slamadversariallab.runtime_stress.models import CpuControl, MemoryControl, RuntimeStressControls


class _Result:
    def __init__(self, returncode=0, stdout="", stderr="") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Process:
    def poll(self):
        return None


def test_podman_cpu_controller_prepare_apply_and_release(monkeypatch) -> None:
    commands = []

    # Stateful fake: an apply is reflected in the next inspect, like real
    # podman, so the controller's post-apply verification sees the new cap.
    state = {"nano": 250000000}

    def _fake_run(cmd, capture_output, text, check, timeout):
        commands.append(cmd)
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {"NanoCpus": %d}}]' % state["nano"])
        if cmd[:2] == ["podman", "update"]:
            if "--cpus" in cmd:
                state["nano"] = int(round(float(cmd[cmd.index("--cpus") + 1]) * 1_000_000_000))
            return _Result(stdout="orbslam3-test\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.podman_controllers.subprocess.run",
        _fake_run,
    )

    controller = PodmanCpuController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "orbslam3-test"},
    )
    controller.apply(RuntimeStressControls(cpu=CpuControl(max_cores=0.5)))
    controller.release()

    update_commands = [cmd for cmd in commands if cmd[:2] == ["podman", "update"]]
    assert update_commands == [
        ["podman", "update", "--cpus", "0.25", "orbslam3-test"],
        ["podman", "update", "--cpus", "0.5", "orbslam3-test"],
        ["podman", "update", "--cpus", "0.25", "orbslam3-test"],
    ]


def test_podman_cpu_controller_rejects_non_podman_target() -> None:
    controller = PodmanCpuController()
    with pytest.raises(RuntimeError, match="podman_container"):
        controller.prepare(
            _Process(),
            target_kind="docker_container",
            target_metadata={"container_name": "x"},
        )


def test_podman_memory_controller_prepare_apply_and_release(monkeypatch) -> None:
    commands = []

    # Stateful fake: an apply is reflected in the next inspect (post-apply
    # verification sees the new cap).
    state = {"mem": 1073741824, "swap": 2147483648}

    def _fake_run(cmd, capture_output, text, check, timeout):
        commands.append(cmd)
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {"Memory": %d, "MemorySwap": %d}}]'
                           % (state["mem"], state["swap"]))
        if cmd[:2] == ["podman", "update"]:
            if "--memory" in cmd:
                v = cmd[cmd.index("--memory") + 1]
                state["mem"] = int(v[:-1]) * 1024 * 1024 if v.endswith("m") else int(v)
            return _Result(stdout="orbslam3-test\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.podman_controllers.subprocess.run",
        _fake_run,
    )

    controller = PodmanMemoryController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "orbslam3-test"},
    )
    controller.apply(RuntimeStressControls(memory=MemoryControl(max_mb=512)))
    controller.release()

    update_commands = [cmd for cmd in commands if cmd[:2] == ["podman", "update"]]
    assert update_commands == [
        ["podman", "update", "--memory", "1073741824", "orbslam3-test"],
        ["podman", "update", "--memory", "512m", "orbslam3-test"],
        ["podman", "update", "--memory", "1073741824", "orbslam3-test"],
    ]


def test_podman_memory_controller_release_restores_unlimited(monkeypatch) -> None:
    commands = []

    def _fake_run(cmd, capture_output, text, check, timeout):
        commands.append(cmd)
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {"Memory": 0, "MemorySwap": 0}}]')
        if cmd[:2] == ["podman", "update"]:
            return _Result(stdout="orbslam3-test\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.podman_controllers.subprocess.run",
        _fake_run,
    )

    controller = PodmanMemoryController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "orbslam3-test"},
    )
    controller.release()

    # Originally-unlimited containers get an effectively-infinite sentinel
    # because `podman update --memory 0` is a silent no-op once a real cap
    # has been applied (Podman 5.8.2). 2**53 == 9007199254740992 bytes.
    sentinel = str(1 << 53)
    update_commands = [cmd for cmd in commands if cmd[:2] == ["podman", "update"]]
    assert update_commands == [
        ["podman", "update", "--memory", sentinel, "orbslam3-test"],
        ["podman", "update", "--memory", sentinel, "orbslam3-test"],
    ]


def test_podman_cpu_controller_release_unsets_with_sentinel_when_originally_unlimited(
    monkeypatch,
) -> None:
    """Mirror of the memory test above; covers the CPU equivalent path."""
    commands = []

    def _fake_run(cmd, capture_output, text, check, timeout):
        commands.append(cmd)
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {"NanoCpus": 0}}]')
        if cmd[:2] == ["podman", "update"]:
            return _Result(stdout="orbslam3-test\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.podman_controllers.subprocess.run",
        _fake_run,
    )

    controller = PodmanCpuController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "orbslam3-test"},
    )
    controller.release()

    # Originally-unlimited container; release uses the sentinel (1M cores)
    # because `podman update --cpus 0` is a no-op after a real cap.
    update_commands = [cmd for cmd in commands if cmd[:2] == ["podman", "update"]]
    assert update_commands == [
        ["podman", "update", "--cpus", "1000000", "orbslam3-test"],
        ["podman", "update", "--cpus", "1000000", "orbslam3-test"],
    ]


def test_podman_memory_controller_tolerates_kernel_set_memory_limit_failure(monkeypatch) -> None:
    inspect_calls = {"n": 0}

    def _fake_run(cmd, capture_output, text, check, timeout):
        if cmd[:2] == ["podman", "inspect"]:
            inspect_calls["n"] += 1
            return _Result(stdout='[{"HostConfig": {"Memory": 1073741824, "MemorySwap": 0}}]')
        if cmd[:2] == ["podman", "update"]:
            if "--memory" in cmd and "256m" in cmd:
                return _Result(returncode=125, stderr="Error: unable to set memory limit lower than current usage")
            return _Result(stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.podman_controllers.subprocess.run",
        _fake_run,
    )

    controller = PodmanMemoryController()
    controller.prepare(
        _Process(),
        target_kind="podman_container",
        target_metadata={"container_name": "orbslam3-test"},
    )
    # Should not raise even though the kernel rejects the limit
    controller.apply(RuntimeStressControls(memory=MemoryControl(max_mb=256)))


def test_podman_cpu_controller_raises_on_silent_no_op(monkeypatch) -> None:
    """`podman update --cpus` reports success but the cap does not change
    (the podman/17880 silent-no-op footgun). The controller must fail loud so
    an unstressed cell is never scored as CPU-stressed."""
    def _fake_run(cmd, capture_output, text, check, timeout):
        if cmd[:2] == ["podman", "inspect"]:
            # NanoCpus never changes, even after a "successful" update.
            return _Result(stdout='[{"HostConfig": {"NanoCpus": 0}}]')
        if cmd[:2] == ["podman", "update"]:
            return _Result(returncode=0, stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.podman_controllers.subprocess.run", _fake_run)
    c = PodmanCpuController()
    c.prepare(_Process(), target_kind="podman_container",
              target_metadata={"container_name": "x"})
    with pytest.raises(RuntimeError, match="did not take effect"):
        c.apply(RuntimeStressControls(cpu=CpuControl(max_cores=1.0)))


def test_podman_memory_controller_raises_on_silent_no_op(monkeypatch) -> None:
    """`podman update --memory` reports success but Memory does not change
    (issue 17880 was reported against memory limits). Fail loud."""
    def _fake_run(cmd, capture_output, text, check, timeout):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {"Memory": 0, "MemorySwap": 0}}]')
        if cmd[:2] == ["podman", "update"]:
            return _Result(returncode=0, stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.podman_controllers.subprocess.run", _fake_run)
    c = PodmanMemoryController()
    c.prepare(_Process(), target_kind="podman_container",
              target_metadata={"container_name": "x"})
    with pytest.raises(RuntimeError, match="did not take effect"):
        c.apply(RuntimeStressControls(memory=MemoryControl(max_mb=512)))


def test_podman_cpu_release_raises_when_the_cap_outlives_its_phase(monkeypatch) -> None:
    """The release goes through the SAME silently-no-op-prone `podman update`.

    An unverified release fails in the more misleading direction than an
    unverified apply: the cap survives into the next phase, so a system that was
    never actually let go gets scored as having failed to recover. That is a
    false failure manufactured by the harness, and E5's recovery verdicts rest
    on it not happening.

    The container starts unlimited, takes a real cap, and then the closing
    release "succeeds" while the cap stays put. `prepare()` fences with its own
    release first, so only the SECOND unlimited request is the one under test.
    """
    state = {"nano": 0, "sentinel_seen": 0}
    sentinel = 1_000_000

    def _fake_run(cmd, capture_output, text, check, timeout):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {"NanoCpus": %d}}]' % state["nano"])
        if cmd[:2] == ["podman", "update"]:
            if "--cpus" in cmd:
                requested = float(cmd[cmd.index("--cpus") + 1])
                if requested >= sentinel:
                    state["sentinel_seen"] += 1
                    # The prepare-time fence lands; the closing release does not.
                    if state["sentinel_seen"] >= 2:
                        return _Result(returncode=0, stdout="ok\n")
                state["nano"] = int(round(requested * 1_000_000_000))
            return _Result(returncode=0, stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.podman_controllers.subprocess.run", _fake_run)
    c = PodmanCpuController()
    c.prepare(_Process(), target_kind="podman_container",
              target_metadata={"container_name": "x"})
    c.apply(RuntimeStressControls(cpu=CpuControl(max_cores=1.0)))
    with pytest.raises(RuntimeError, match="outlived its phase"):
        c.release()


def test_podman_cpu_release_accepts_the_unlimited_sentinel(monkeypatch) -> None:
    """A real release must not be mistaken for a failed one."""
    state = {"nano": 0}

    def _fake_run(cmd, capture_output, text, check, timeout):
        if cmd[:2] == ["podman", "inspect"]:
            return _Result(stdout='[{"HostConfig": {"NanoCpus": %d}}]' % state["nano"])
        if cmd[:2] == ["podman", "update"]:
            if "--cpus" in cmd:
                state["nano"] = int(round(float(cmd[cmd.index("--cpus") + 1]) * 1_000_000_000))
            return _Result(returncode=0, stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.podman_controllers.subprocess.run", _fake_run)
    c = PodmanCpuController()
    c.prepare(_Process(), target_kind="podman_container",
              target_metadata={"container_name": "x"})
    c.apply(RuntimeStressControls(cpu=CpuControl(max_cores=1.0)))
    c.release()   # must not raise


def test_podman_cpu_release_is_silent_when_the_container_has_exited(monkeypatch) -> None:
    """No cap can outlive a container that is gone, and the run is over anyway.

    Attach must succeed first, so the container disappears only after prepare.
    """
    state = {"nano": 0, "gone": False}

    def _fake_run(cmd, capture_output, text, check, timeout):
        if cmd[:2] == ["podman", "inspect"]:
            if state["gone"]:
                return _Result(returncode=1, stdout="")
            return _Result(stdout='[{"HostConfig": {"NanoCpus": %d}}]' % state["nano"])
        if cmd[:2] == ["podman", "update"]:
            return _Result(returncode=0, stdout="ok\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.podman_controllers.subprocess.run", _fake_run)
    c = PodmanCpuController()
    c.prepare(_Process(), target_kind="podman_container",
              target_metadata={"container_name": "x"})
    state["gone"] = True
    c.release()   # must not raise
