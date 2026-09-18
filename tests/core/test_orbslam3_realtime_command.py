"""Verify the ORB-SLAM3 container command injects the SAL realtime deadline
env vars for the monocular KITTI binary (the only one plumbed to honor them),
and does NOT inject them for the other binaries (stereo/tum/euroc) even when
the harness env is set.

This mirrors tests/core/test_droidslam_realtime_command.py. The gating is the
key contract: mono_kitti.cc is the only ORB entry point that reads
SAL_DEADLINE_FPS, so a "deadline set but silently ignored" run for the other
modes would produce misleading results.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from slamadversariallab.algorithms.orbslam3 import ORBSLAM3Algorithm

# Injected into SAL_RUNTIME_PATH and asserted back out of the command line, so
# any absolute path serves and this one is deliberately not a real location.
_RUNTIME_DIR = "/opt/sal/src/runtime_stress"


@pytest.fixture
def algo():
    a = ORBSLAM3Algorithm(container_runtime="podman")
    # Isolate from the host-stress (HAMi) plumbing; this test only covers the
    # deadline env injection.
    a._runtime_stress_launch_extras = MagicMock(return_value={})
    return a


def _make_ctx(tmp_path, dataset_type, is_stereo):
    dataset = tmp_path / "dataset"
    out = tmp_path / "out"
    dataset.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    ctx = MagicMock()
    ctx.execution_inputs = {
        "dataset_path": dataset,
        "output_dir": out,
        "dataset_type": dataset_type,
        "is_stereo": is_stereo,
        "is_external": False,
        "kitti_image_mounts": [],
        "euroc_image_mounts": None,
    }
    ctx.sequence_name = "04"
    ctx.internal_config_name = "KITTI04-12.yaml"
    ctx.runtime_stress = None
    ctx.staging_artifacts = {"association_file": "associations.txt"}
    return ctx


def _cmdline(algo, ctx):
    spec = algo._build_execution_spec(MagicMock(), ctx)
    assert spec is not None
    return " ".join(spec.cmd)


def _set_deadline_env(monkeypatch, *, warmup="5"):
    monkeypatch.setenv("SAL_DEADLINE_FPS", "10")
    monkeypatch.setenv("SAL_RUNTIME_PATH", _RUNTIME_DIR)
    if warmup is None:
        monkeypatch.delenv("SAL_DEADLINE_WARMUP_FRAMES", raising=False)
    else:
        monkeypatch.setenv("SAL_DEADLINE_WARMUP_FRAMES", warmup)


def _clear_deadline_env(monkeypatch):
    for var in (
        "SAL_DEADLINE_FPS",
        "SAL_RUNTIME_PATH",
        "SAL_DROP_LOG_PATH",
        "SAL_DEADLINE_WARMUP_FRAMES",
        "SAL_DEADLINE_QUEUE_SIZE",
        "SAL_DEADLINE_DROP_POLICY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_mono_kitti_no_env_no_sal_flags(algo, tmp_path, monkeypatch):
    """Without SAL_DEADLINE_FPS, the mono KITTI command has no SAL flags/mount."""
    _clear_deadline_env(monkeypatch)
    ctx = _make_ctx(tmp_path, "kitti", is_stereo=False)
    cmdline = _cmdline(algo, ctx)

    assert "SAL_DEADLINE_FPS" not in cmdline
    assert "SAL_RUNTIME_PATH" not in cmdline
    assert "SAL_DROP_LOG_PATH" not in cmdline
    assert "/sal_runtime" not in cmdline


def test_mono_kitti_env_propagates(algo, tmp_path, monkeypatch):
    """With the harness env set, the mono KITTI command carries the SAL env
    vars and the (harmless) runtime_stress bind mount."""
    _set_deadline_env(monkeypatch, warmup="5")
    ctx = _make_ctx(tmp_path, "kitti", is_stereo=False)
    cmdline = _cmdline(algo, ctx)

    assert "-e SAL_DEADLINE_FPS=10" in cmdline
    assert "-e SAL_DEADLINE_WARMUP_FRAMES=5" in cmdline
    assert "-e SAL_RUNTIME_PATH=/sal_runtime" in cmdline
    assert "-e SAL_DROP_LOG_PATH=/output/deadline_drops.json" in cmdline
    assert "/runtime_stress:/sal_runtime:ro" in cmdline


def test_mono_kitti_warmup_omitted_when_unset(algo, tmp_path, monkeypatch):
    _set_deadline_env(monkeypatch, warmup=None)
    ctx = _make_ctx(tmp_path, "kitti", is_stereo=False)
    cmdline = _cmdline(algo, ctx)

    assert "-e SAL_DEADLINE_FPS=10" in cmdline
    assert "SAL_DEADLINE_WARMUP_FRAMES" not in cmdline


def test_stereo_kitti_env_not_injected(algo, tmp_path, monkeypatch):
    """Gating: stereo_kitti is not plumbed, so even with the harness env set the
    command must carry no SAL deadline flags."""
    _set_deadline_env(monkeypatch)
    ctx = _make_ctx(tmp_path, "kitti", is_stereo=True)
    cmdline = _cmdline(algo, ctx)

    assert "stereo_kitti" in cmdline  # sanity: we built the stereo command
    assert "SAL_DEADLINE_FPS" not in cmdline
    assert "/sal_runtime" not in cmdline


def test_tum_env_not_injected(algo, tmp_path, monkeypatch):
    """Gating: the TUM binary is not plumbed; no SAL deadline flags."""
    _set_deadline_env(monkeypatch)
    ctx = _make_ctx(tmp_path, "tum", is_stereo=False)
    cmdline = _cmdline(algo, ctx)

    assert "SAL_DEADLINE_FPS" not in cmdline
    assert "/sal_runtime" not in cmdline
