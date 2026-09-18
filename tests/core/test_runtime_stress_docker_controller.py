"""Tests for Docker-backed runtime-stress controllers."""

from slamadversariallab.runtime_stress.controllers import DockerCpuController, DockerMemoryController
from slamadversariallab.runtime_stress.models import CpuControl, MemoryControl, RuntimeStressControls


def test_docker_cpu_controller_prepare_apply_and_release(monkeypatch) -> None:
    commands = []

    class _Result:
        def __init__(self, returncode=0, stdout="", stderr="") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(cmd, capture_output, text, check, timeout):
        commands.append(cmd)
        if cmd[:2] == ["docker", "inspect"]:
            return _Result(stdout='[{"HostConfig": {"NanoCpus": 250000000}}]')
        if cmd[:2] == ["docker", "update"]:
            return _Result(stdout="orbslam3-test\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.controllers.subprocess.run",
        _fake_run,
    )

    class _Process:
        def poll(self):
            return None

    controller = DockerCpuController()
    controller.prepare(
        _Process(),
        target_kind="docker_container",
        target_metadata={"container_name": "orbslam3-test"},
    )
    controller.apply(RuntimeStressControls(cpu=CpuControl(max_cores=0.5)))
    controller.release()

    update_commands = [cmd for cmd in commands if cmd[:2] == ["docker", "update"]]
    assert update_commands == [
        ["docker", "update", "--cpus", "0.25", "orbslam3-test"],
        ["docker", "update", "--cpus", "0.5", "orbslam3-test"],
        ["docker", "update", "--cpus", "0.25", "orbslam3-test"],
    ]


def test_docker_memory_controller_prepare_apply_and_release(monkeypatch) -> None:
    commands = []

    class _Result:
        def __init__(self, returncode=0, stdout="", stderr="") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(cmd, capture_output, text, check, timeout):
        commands.append(cmd)
        if cmd[:2] == ["docker", "inspect"]:
            return _Result(stdout='[{"HostConfig": {"Memory": 1073741824, "MemorySwap": 2147483648}}]')
        if cmd[:2] == ["docker", "update"]:
            return _Result(stdout="orbslam3-test\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.controllers.subprocess.run",
        _fake_run,
    )

    class _Process:
        def poll(self):
            return None

    controller = DockerMemoryController()
    controller.prepare(
        _Process(),
        target_kind="docker_container",
        target_metadata={"container_name": "orbslam3-test"},
    )
    controller.apply(RuntimeStressControls(memory=MemoryControl(max_mb=512)))
    controller.release()

    update_commands = [cmd for cmd in commands if cmd[:2] == ["docker", "update"]]
    assert update_commands == [
        ["docker", "update", "--memory-swap", "2147483648", "--memory", "1073741824", "orbslam3-test"],
        ["docker", "update", "--memory", "512m", "--memory-swap", "512m", "orbslam3-test"],
        ["docker", "update", "--memory-swap", "2147483648", "--memory", "1073741824", "orbslam3-test"],
    ]


def test_docker_memory_controller_release_restores_unlimited(monkeypatch) -> None:
    commands = []

    class _Result:
        def __init__(self, returncode=0, stdout="", stderr="") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(cmd, capture_output, text, check, timeout):
        commands.append(cmd)
        if cmd[:2] == ["docker", "inspect"]:
            return _Result(stdout='[{"HostConfig": {"Memory": 0, "MemorySwap": 0}}]')
        if cmd[:2] == ["docker", "update"]:
            return _Result(stdout="orbslam3-test\n")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.controllers.subprocess.run",
        _fake_run,
    )

    class _Process:
        def poll(self):
            return None

    controller = DockerMemoryController()
    controller.prepare(
        _Process(),
        target_kind="docker_container",
        target_metadata={"container_name": "orbslam3-test"},
    )
    controller.release()

    update_commands = [cmd for cmd in commands if cmd[:2] == ["docker", "update"]]
    assert update_commands == [
        ["docker", "update", "--memory-swap", "-1", "--memory", "0", "orbslam3-test"],
        ["docker", "update", "--memory-swap", "-1", "--memory", "0", "orbslam3-test"],
    ]
