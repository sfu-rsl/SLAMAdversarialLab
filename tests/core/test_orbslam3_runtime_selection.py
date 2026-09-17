"""Tests for ORB-SLAM3 algorithm container_runtime parameterization."""

import pytest

from slamadversariallab.algorithms.orbslam3 import ORBSLAM3Algorithm


def test_orbslam3_default_runtime_is_docker() -> None:
    algo = ORBSLAM3Algorithm()
    assert algo.container_runtime == "docker"
    assert algo.runtime_stress_target_kind == "docker_container"


def test_orbslam3_target_kind_reflects_runtime() -> None:
    algo = ORBSLAM3Algorithm(container_runtime="podman")
    assert algo.container_runtime == "podman"
    assert algo.runtime_stress_target_kind == "podman_container"


def test_orbslam3_rejects_unknown_runtime() -> None:
    with pytest.raises(ValueError, match="container_runtime"):
        ORBSLAM3Algorithm(container_runtime="lxc")


def test_orbslam3_preflight_uses_podman_binary(monkeypatch) -> None:
    """Preflight should look up podman (not docker) when runtime is podman."""
    looked_up = []

    def _fake_which(name):
        looked_up.append(name)
        return None  # simulate not found so preflight raises

    monkeypatch.setattr(
        "slamadversariallab.algorithms.orbslam3.shutil.which",
        _fake_which,
    )

    algo = ORBSLAM3Algorithm(container_runtime="podman")
    with pytest.raises(RuntimeError, match="podman"):
        algo._preflight_checks(request=None, ctx=None)

    assert "podman" in looked_up
    assert "docker" not in looked_up
