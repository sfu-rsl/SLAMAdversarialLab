"""Tests for Nitro-SLAM runtime selection and declared capabilities."""

import pytest

from slamadversariallab.algorithms.base import SLAMAlgorithm
from slamadversariallab.algorithms.nitroslam import NitroSLAMAlgorithm


def test_nitroslam_default_constructor_is_docker() -> None:
    algo = NitroSLAMAlgorithm()
    assert algo.container_runtime == "docker"
    assert algo.runtime_stress_target_kind == "docker_container"
    assert algo.docker_image == "nitroslam:latest"


def test_nitroslam_podman_runtime_selection() -> None:
    algo = NitroSLAMAlgorithm(container_runtime="podman")
    assert algo.container_runtime == "podman"
    assert algo.runtime_stress_target_kind == "podman_container"


def test_nitroslam_invalid_runtime_rejected() -> None:
    with pytest.raises(ValueError):
        NitroSLAMAlgorithm(container_runtime="singularity")


def test_nitroslam_is_slam_algorithm() -> None:
    assert isinstance(NitroSLAMAlgorithm(), SLAMAlgorithm)


def test_nitroslam_name_and_supported_datasets() -> None:
    algo = NitroSLAMAlgorithm()
    assert algo.name == "nitroslam"
    # EuRoC stereo-inertial only: the one path that activates the GPU modules.
    assert algo.supported_datasets == {"euroc": ["stereo"]}
    assert algo.supports("euroc", "stereo") is True
    assert algo.supports("kitti", "mono") is False
    assert algo.supports("tum", "rgbd") is False


def test_nitroslam_resolve_config_name() -> None:
    algo = NitroSLAMAlgorithm()
    assert algo.resolve_config_name("V1_01_easy", "euroc") == "EuRoC.yaml"
    assert algo.resolve_config_name("MH_01_easy", "EuRoC") == "EuRoC.yaml"
    assert algo.resolve_config_name("00", "kitti") is None


def test_nitroslam_registered_in_registry() -> None:
    from slamadversariallab.algorithms.registry import (
        get_slam_algorithm,
        list_slam_algorithms,
    )

    assert "nitroslam" in list_slam_algorithms()
    assert isinstance(get_slam_algorithm("nitroslam"), NitroSLAMAlgorithm)
