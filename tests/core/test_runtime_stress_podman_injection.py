"""Tests for the SAL realtime-harness Podman command injection helper.

Covers the four cases that any container-runtime SLAM wrapper relies
on: harness-off (no-op), harness-on with full env, harness-on without
warmup, and harness-on without SAL_RUNTIME_PATH.

The DROID-specific command-level test in test_droidslam_realtime_command.py
exercises the same logic through the wrapper; these tests pin the
helper's contract independent of any SLAM.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slamadversariallab.runtime_stress.podman_injection import (
    SAL_RUNTIME_CONTAINER_PATH,
    apply_realtime_to_podman_cmd,
)


def _starting_cmd():
    """Plausible Podman command midway through construction."""
    return [
        "podman", "run", "--rm",
        "--name", "fake-slam",
        "-v", "/host/dataset:/dataset:ro",
        "-v", "/host/output:/output",
    ]


def test_harness_off_is_no_op(monkeypatch):
    monkeypatch.delenv("SAL_DEADLINE_FPS", raising=False)
    cmd = _starting_cmd()
    before = list(cmd)
    apply_realtime_to_podman_cmd(cmd, container_output_path="/output")
    assert cmd == before  # no mutation


def test_harness_on_full_env_appends_all_pieces(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "runtime_stress"
    runtime_dir.mkdir()
    monkeypatch.setenv("SAL_DEADLINE_FPS", "10")
    monkeypatch.setenv("SAL_DEADLINE_WARMUP_FRAMES", "5")
    monkeypatch.setenv("SAL_DEADLINE_QUEUE_SIZE", "4")
    monkeypatch.setenv("SAL_DEADLINE_DROP_POLICY", "drop_newest")
    monkeypatch.setenv("SAL_RUNTIME_PATH", str(runtime_dir))

    cmd = _starting_cmd()
    apply_realtime_to_podman_cmd(cmd, container_output_path="/output")
    cmdline = " ".join(cmd)

    assert (
        f"-v {runtime_dir.resolve()}:{SAL_RUNTIME_CONTAINER_PATH}:ro"
        in cmdline
    )
    assert "-e SAL_DEADLINE_FPS=10" in cmdline
    assert "-e SAL_DEADLINE_WARMUP_FRAMES=5" in cmdline
    assert "-e SAL_DEADLINE_QUEUE_SIZE=4" in cmdline
    assert "-e SAL_DEADLINE_DROP_POLICY=drop_newest" in cmdline
    assert f"-e SAL_RUNTIME_PATH={SAL_RUNTIME_CONTAINER_PATH}" in cmdline
    assert "-e SAL_DROP_LOG_PATH=/output/deadline_drops.json" in cmdline
    assert "-e SAL_PROGRESS_PATH=/output/deadline_progress.json" in cmdline


def test_harness_on_without_queue_size_omits_queue_flag(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "runtime_stress"
    runtime_dir.mkdir()
    monkeypatch.setenv("SAL_DEADLINE_FPS", "10")
    monkeypatch.delenv("SAL_DEADLINE_QUEUE_SIZE", raising=False)
    monkeypatch.setenv("SAL_RUNTIME_PATH", str(runtime_dir))

    cmd = _starting_cmd()
    apply_realtime_to_podman_cmd(cmd, container_output_path="/output")
    cmdline = " ".join(cmd)

    assert "-e SAL_DEADLINE_FPS=10" in cmdline
    assert "SAL_DEADLINE_QUEUE_SIZE" not in cmdline


def test_harness_on_without_warmup_omits_warmup_flag(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "runtime_stress"
    runtime_dir.mkdir()
    monkeypatch.setenv("SAL_DEADLINE_FPS", "10")
    monkeypatch.delenv("SAL_DEADLINE_WARMUP_FRAMES", raising=False)
    monkeypatch.setenv("SAL_RUNTIME_PATH", str(runtime_dir))

    cmd = _starting_cmd()
    apply_realtime_to_podman_cmd(cmd, container_output_path="/output")
    cmdline = " ".join(cmd)

    assert "-e SAL_DEADLINE_FPS=10" in cmdline
    assert "SAL_DEADLINE_WARMUP_FRAMES" not in cmdline


def test_harness_on_without_runtime_path_omits_bind_mount(monkeypatch):
    """If SAL_RUNTIME_PATH isn't set the env vars still propagate but
    no bind-mount is emitted (the SLAM would import-fail at runtime --
    but that's the pipeline's responsibility, not the helper's)."""
    monkeypatch.setenv("SAL_DEADLINE_FPS", "10")
    monkeypatch.delenv("SAL_RUNTIME_PATH", raising=False)
    monkeypatch.delenv("SAL_DEADLINE_WARMUP_FRAMES", raising=False)

    cmd = _starting_cmd()
    apply_realtime_to_podman_cmd(cmd, container_output_path="/output")
    cmdline = " ".join(cmd)

    assert ":/sal_runtime:ro" not in cmdline
    assert "-e SAL_DEADLINE_FPS=10" in cmdline
    assert f"-e SAL_RUNTIME_PATH={SAL_RUNTIME_CONTAINER_PATH}" in cmdline
    assert "-e SAL_DROP_LOG_PATH=/output/deadline_drops.json" in cmdline


def test_drop_log_path_uses_container_output_path(monkeypatch):
    monkeypatch.setenv("SAL_DEADLINE_FPS", "10")
    cmd = _starting_cmd()
    apply_realtime_to_podman_cmd(cmd, container_output_path="/workspace/out")
    cmdline = " ".join(cmd)
    assert "-e SAL_DROP_LOG_PATH=/workspace/out/deadline_drops.json" in cmdline


def test_helper_exported_at_package_root():
    from slamadversariallab import runtime_stress
    assert hasattr(runtime_stress, "apply_realtime_to_podman_cmd")
    assert hasattr(runtime_stress, "SAL_RUNTIME_CONTAINER_PATH")
    assert hasattr(runtime_stress, "apply_entrypoint_override")


# ---------------------------------------------------------------------------
# apply_entrypoint_override
# ---------------------------------------------------------------------------

def test_entrypoint_override_no_op_when_host_missing(tmp_path):
    from slamadversariallab.runtime_stress.podman_injection import (
        apply_entrypoint_override,
    )

    cmd = _starting_cmd()
    before = list(cmd)
    apply_entrypoint_override(
        cmd,
        host_path=tmp_path / "no_such_file.py",
        container_path="/droid-slam/demo.py",
    )
    assert cmd == before


def test_entrypoint_override_appends_bind_mount(tmp_path):
    from slamadversariallab.runtime_stress.podman_injection import (
        apply_entrypoint_override,
    )

    host = tmp_path / "demo.py"
    host.write_text("# fake demo\n")

    cmd = _starting_cmd()
    apply_entrypoint_override(
        cmd, host_path=host, container_path="/droid-slam/demo.py"
    )
    cmdline = " ".join(cmd)
    assert f"-v {host.resolve()}:/droid-slam/demo.py:ro" in cmdline


def test_entrypoint_override_unrelated_to_realtime_env(tmp_path, monkeypatch):
    """The entry-point bind-mount applies regardless of whether the SAL
    deadline harness is active -- it's a development convenience for
    any host-side edit, not just SAL hooks."""
    from slamadversariallab.runtime_stress.podman_injection import (
        apply_entrypoint_override,
    )

    monkeypatch.delenv("SAL_DEADLINE_FPS", raising=False)
    host = tmp_path / "main.py"
    host.write_text("# fake main\n")

    cmd = _starting_cmd()
    apply_entrypoint_override(cmd, host_path=host, container_path="/vggt-slam/main.py")
    cmdline = " ".join(cmd)
    assert f"-v {host.resolve()}:/vggt-slam/main.py:ro" in cmdline
