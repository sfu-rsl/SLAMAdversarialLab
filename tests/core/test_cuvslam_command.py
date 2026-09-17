"""cuVSLAM wrapper: the command shape the runtime-stress framework depends on.

Every assertion here corresponds to something that, if wrong, fails SILENTLY --
the run completes, produces a trajectory, and is scored as a valid stressed cell
when it was nothing of the kind. That is the failure mode this repo keeps
hitting, so the contract is pinned rather than trusted.
"""
from __future__ import annotations

import pytest

from slamadversariallab.algorithms.cuvslam import CuVSLAMAlgorithm
from slamadversariallab.algorithms.types import SensorMode


def test_default_runtime_is_podman_and_ctor_takes_no_arguments():
    """The registry instantiates with NO arguments, and
    `_reconcile_algorithm_runtime` reconstructs with the keyword
    `container_runtime`. Rename it and the YAML's runtime is silently ignored.
    """
    algo = CuVSLAMAlgorithm()
    assert algo.container_runtime == "podman"
    assert CuVSLAMAlgorithm(container_runtime="docker").container_runtime == "docker"


def test_an_unknown_runtime_is_refused():
    with pytest.raises(ValueError, match="container_runtime"):
        CuVSLAMAlgorithm(container_runtime="containerd")


def test_target_kind_follows_the_runtime():
    """Runtime stress refuses any target kind outside its whitelist, so this
    must track the runtime rather than being hardcoded."""
    assert CuVSLAMAlgorithm().runtime_stress_target_kind == "podman_container"
    assert (CuVSLAMAlgorithm(container_runtime="docker").runtime_stress_target_kind
            == "docker_container")


def test_only_euroc_stereo_is_claimed():
    """cuVSLAM has no mono-inertial mode and its Mono mode is scale-free, so
    the inertial path is what makes this system metric. Claiming a mode the
    driver does not select would let the pipeline route data it cannot use.
    """
    assert CuVSLAMAlgorithm().supported_datasets == {"euroc": ["stereo"]}


def test_config_name_is_an_identifier_not_a_file():
    """Calibration comes from the dataset's own sensor.yaml at run time, so
    there is no per-sequence config file to resolve. `_resolve_internal_config_path`
    must therefore return None rather than a path that does not exist.
    """
    algo = CuVSLAMAlgorithm()
    name = algo.resolve_config_name("V1_01_easy", "euroc", SensorMode.STEREO)
    assert name and "euroc" in name
    assert algo.resolve_config_name("07", "kitti") is None
    assert algo._resolve_internal_config_path(object()) is None


def test_the_driver_path_points_at_a_real_file():
    """The driver is bind-mounted over its in-image copy. `apply_entrypoint_override`
    is silently a NO-OP when the host file is missing, which would leave the
    container running the stale baked-in driver -- the exact trap that makes a
    deadline config complete with zero drops and look like a clean run.
    """
    assert CuVSLAMAlgorithm().cuvslam_driver_path.exists()


def test_driver_wraps_only_camera_frames_not_the_merged_record_list():
    """The single most consequential line in the driver.

    `prepare_frame_metadata_euroc` returns ONE timestamp-sorted list mixing
    stereo and IMU records, and on V1_01_easy that is 10:1 in IMU's favour.
    Wrapping the merged list would set total_items an order of magnitude too
    high and count IMU samples as dropped frames.
    """
    src = (CuVSLAMAlgorithm().cuvslam_driver_path).read_text()
    assert 'camera_frames = [r for r in records if r["type"] != "imu"]' in src
    assert "DeadlineIterator(\n            camera_frames," in src


def test_driver_pins_async_sba_off():
    """cuVSLAM documents no determinism guarantee and asynchronous bundle
    adjustment makes the result depend on thread scheduling -- the very variable
    this framework manipulates. A stressor must not be able to change the answer
    through scheduling alone.
    """
    assert "async_sba=False" in CuVSLAMAlgorithm().cuvslam_driver_path.read_text()


def test_driver_does_not_rectify_and_says_so():
    """EuRoC ships raw radial-tangential images and cuVSLAM undistorts
    internally only when rectified_stereo_camera is False. Setting it True on
    unrectified input is a silent accuracy bug, not an error.
    """
    assert "rectified_stereo_camera=False" in CuVSLAMAlgorithm().cuvslam_driver_path.read_text()


def test_driver_records_lost_frames_instead_of_skipping_them():
    """The upstream example does a bare `continue` when world_from_rig is None.
    Pose count is a health signal here, so a delivered-but-untracked frame has
    to be visible in the console rather than inferred from a short trajectory.
    """
    src = CuVSLAMAlgorithm().cuvslam_driver_path.read_text()
    assert "tracking failed at frame" in src
    assert "lost=" in src
