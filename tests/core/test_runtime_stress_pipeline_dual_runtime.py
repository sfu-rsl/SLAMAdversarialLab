"""Tests for the pipeline's container-runtime reconciliation helper."""

import pytest

from slamadversariallab.algorithms.orbslam3 import ORBSLAM3Algorithm
from slamadversariallab.algorithms.vggtslam import VGGTSLAMAlgorithm
from slamadversariallab.pipelines.runtime_stress_evaluation import _reconcile_algorithm_runtime


def test_reconcile_vggtslam_conda_to_podman_reconstructs() -> None:
    algo = VGGTSLAMAlgorithm()
    assert algo.container_runtime is None

    reconciled = _reconcile_algorithm_runtime(algo, "podman")

    assert isinstance(reconciled, VGGTSLAMAlgorithm)
    assert reconciled is not algo
    assert reconciled.container_runtime == "podman"
    assert reconciled.runtime_stress_target_kind == "podman_container"


def test_reconcile_vggtslam_conda_with_docker_default_stays_conda() -> None:
    algo = VGGTSLAMAlgorithm()

    reconciled = _reconcile_algorithm_runtime(algo, "docker")

    # Schema default "docker" should not drag a host-process wrapper into a
    # container runtime it doesn't support.
    assert reconciled is algo
    assert reconciled.container_runtime is None


def test_reconcile_vggtslam_podman_already_matches_stays() -> None:
    algo = VGGTSLAMAlgorithm(container_runtime="podman")

    reconciled = _reconcile_algorithm_runtime(algo, "podman")

    assert reconciled is algo


def test_reconcile_orbslam3_docker_to_podman_reconstructs() -> None:
    algo = ORBSLAM3Algorithm(container_runtime="docker")

    reconciled = _reconcile_algorithm_runtime(algo, "podman")

    assert isinstance(reconciled, ORBSLAM3Algorithm)
    assert reconciled is not algo
    assert reconciled.container_runtime == "podman"


def test_reconcile_orbslam3_docker_matches_docker() -> None:
    algo = ORBSLAM3Algorithm(container_runtime="docker")

    reconciled = _reconcile_algorithm_runtime(algo, "docker")

    assert reconciled is algo


def test_reconcile_wrapper_without_container_runtime_attr_is_passthrough() -> None:
    class _NoRuntimeWrapper:
        name = "noop"

    algo = _NoRuntimeWrapper()
    reconciled = _reconcile_algorithm_runtime(algo, "podman")

    assert reconciled is algo


def test_reconcile_vggtslam_conda_to_unsupported_runtime_raises() -> None:
    """VGGT-SLAM only supports None or 'podman'; reconstruction to anything else raises."""
    algo = VGGTSLAMAlgorithm()

    with pytest.raises(ValueError, match="container_runtime must be None or 'podman'"):
        _reconcile_algorithm_runtime(algo, "kubernetes")
