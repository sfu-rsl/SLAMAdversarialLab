"""Verify the DROID-SLAM Podman command propagates SAL realtime env vars
and bind-mounts the runtime_stress directory + the host's demo.py.

This test would have caught the two bugs we hit during smoke testing:
1. The first run produced no drop log because Podman's '-e SAL_*' flags
   weren't being added (env-var injection wasn't wired).
2. The second run still didn't activate the harness because the
   container had a stale copy of demo.py baked in at image-build time.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]

from slamadversariallab.algorithms.droidslam import DROIDSLAMAlgorithm


@pytest.fixture
def algo():
    return DROIDSLAMAlgorithm(container_runtime="podman")


@pytest.fixture
def fake_inputs(tmp_path):
    img_dir = tmp_path / "img"
    out_dir = tmp_path / "out"
    img_dir.mkdir()
    out_dir.mkdir()
    calib = tmp_path / "calib.txt"
    calib.write_text("0 0 0 0\n")
    ctx = MagicMock()
    ctx.execution_inputs = {}
    return ctx, img_dir, calib, out_dir


def test_no_realtime_env_no_sal_flags(algo, fake_inputs, monkeypatch):
    """Without SAL_DEADLINE_FPS set, the command must not include any
    SAL_* env-var flags or the runtime_stress bind mount."""
    monkeypatch.delenv("SAL_DEADLINE_FPS", raising=False)
    monkeypatch.delenv("SAL_RUNTIME_PATH", raising=False)
    monkeypatch.delenv("SAL_DROP_LOG_PATH", raising=False)
    monkeypatch.delenv("SAL_DEADLINE_WARMUP_FRAMES", raising=False)

    ctx, img_dir, calib, out_dir = fake_inputs
    spec = algo._build_container_execution_spec(ctx, img_dir, calib, out_dir, "tum")

    cmdline = " ".join(spec.cmd)
    assert "SAL_DEADLINE_FPS" not in cmdline
    assert "SAL_RUNTIME_PATH" not in cmdline
    assert "SAL_DROP_LOG_PATH" not in cmdline
    assert "SAL_DEADLINE_WARMUP_FRAMES" not in cmdline
    assert "/sal_runtime" not in cmdline


def test_realtime_env_propagates_into_podman_command(algo, fake_inputs, monkeypatch):
    monkeypatch.setenv("SAL_DEADLINE_FPS", "10")
    monkeypatch.setenv("SAL_DEADLINE_WARMUP_FRAMES", "5")
    monkeypatch.setenv(
        "SAL_RUNTIME_PATH",
        # Any absolute path serves: this is injected, never read from disk. It
        # is a neutral one so the repo carries no author's home directory.
        "/opt/sal/src/runtime_stress",
    )

    ctx, img_dir, calib, out_dir = fake_inputs
    spec = algo._build_container_execution_spec(ctx, img_dir, calib, out_dir, "tum")

    cmdline = " ".join(spec.cmd)
    # The four SAL env vars must all be passed into the container.
    assert "-e SAL_DEADLINE_FPS=10" in cmdline
    assert "-e SAL_DEADLINE_WARMUP_FRAMES=5" in cmdline
    assert "-e SAL_RUNTIME_PATH=/sal_runtime" in cmdline
    assert "-e SAL_DROP_LOG_PATH=/output/deadline_drops.json" in cmdline
    # The runtime_stress dir must be bind-mounted at /sal_runtime.
    assert "/runtime_stress:/sal_runtime:ro" in cmdline


def test_warmup_omitted_when_unset(algo, fake_inputs, monkeypatch):
    """If only SAL_DEADLINE_FPS is set (no warmup), command should still
    work and just not include the warmup env-var line."""
    monkeypatch.setenv("SAL_DEADLINE_FPS", "10")
    monkeypatch.delenv("SAL_DEADLINE_WARMUP_FRAMES", raising=False)
    monkeypatch.setenv("SAL_RUNTIME_PATH", "/some/path")

    ctx, img_dir, calib, out_dir = fake_inputs
    spec = algo._build_container_execution_spec(ctx, img_dir, calib, out_dir, "tum")
    cmdline = " ".join(spec.cmd)

    assert "-e SAL_DEADLINE_FPS=10" in cmdline
    assert "SAL_DEADLINE_WARMUP_FRAMES" not in cmdline


def test_demo_py_bind_mount_always_present(algo, fake_inputs, monkeypatch):
    """The host's demo.py must be bind-mounted over the container's copy
    so changes to the SAL deadline hook reach the SLAM without rebuilding
    the image."""
    monkeypatch.delenv("SAL_DEADLINE_FPS", raising=False)
    ctx, img_dir, calib, out_dir = fake_inputs
    spec = algo._build_container_execution_spec(ctx, img_dir, calib, out_dir, "tum")
    cmdline = " ".join(spec.cmd)

    # The mount only appears when the host file exists: DROID-SLAM is a
    # submodule, and `apply_entrypoint_override` no-ops without it. Skipping
    # keeps a fresh clone honest instead of failing on an absent checkout.
    if not (_REPO_ROOT / "deps" / "slam-algorithms" / "DROID-SLAM" / "demo.py").exists():
        pytest.skip("needs the DROID-SLAM submodule: git submodule update --init "
                    "deps/slam-algorithms/DROID-SLAM")
    assert "demo.py:/droid-slam/demo.py:ro" in cmdline
