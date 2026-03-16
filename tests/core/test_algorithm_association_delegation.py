"""Tests that algorithm adapters delegate TUM association handling to shared service."""

from pathlib import Path

import pytest

from slamadverseriallab.algorithms.orbslam3 import ORBSLAM3Algorithm
from slamadverseriallab.algorithms.photoslam import PhotoSLAMAlgorithm
from slamadverseriallab.algorithms.types import SLAMRunRequest, SensorMode


def test_orbslam3_find_association_delegates_to_shared_service(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected = tmp_path / "associations.txt"
    expected.write_text("a\n", encoding="utf-8")
    captured = {}

    def _fake_resolver(dataset_path: Path, generate_if_missing: bool, log):
        captured["dataset_path"] = dataset_path
        captured["generate_if_missing"] = generate_if_missing
        captured["log"] = log
        return expected

    monkeypatch.setattr(
        "slamadverseriallab.algorithms.orbslam3.resolve_tum_association_for_orbslam3",
        _fake_resolver,
    )

    algo = ORBSLAM3Algorithm()
    result = algo._find_association_file(tmp_path)

    assert result == "associations.txt"
    assert captured["dataset_path"] == tmp_path
    assert captured["generate_if_missing"] is False
    assert captured["log"] is not None


def test_orbslam3_generate_association_delegates_to_shared_service(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rgb_file = tmp_path / "rgb.txt"
    depth_file = tmp_path / "depth.txt"
    expected = tmp_path / "associations.txt"
    expected.write_text("a\n", encoding="utf-8")
    captured = {}

    def _fake_generator(dataset_path: Path, rgb_file: Path, depth_file: Path, max_diff: float, log):
        captured["dataset_path"] = dataset_path
        captured["rgb_file"] = rgb_file
        captured["depth_file"] = depth_file
        captured["max_diff"] = max_diff
        captured["log"] = log
        return expected

    monkeypatch.setattr(
        "slamadverseriallab.algorithms.orbslam3.generate_tum_association_with_associate_py",
        _fake_generator,
    )

    algo = ORBSLAM3Algorithm()
    result = algo._generate_association_file(tmp_path, rgb_file, depth_file, max_diff=0.03)

    assert result == expected
    assert captured["dataset_path"] == tmp_path
    assert captured["rgb_file"] == rgb_file
    assert captured["depth_file"] == depth_file
    assert captured["max_diff"] == 0.03
    assert captured["log"] is not None


def test_photoslam_finds_existing_tum_association_file_only(tmp_path: Path) -> None:
    assoc_file = tmp_path / "associations.txt"
    assoc_file.write_text("0.0 rgb/0.png 0.0 depth/0.png\n", encoding="utf-8")

    algo = PhotoSLAMAlgorithm()
    resolved = algo._find_existing_tum_association_file(tmp_path)

    assert resolved == assoc_file


def test_photoslam_euroc_timestamps_missing_contract_raises(tmp_path: Path) -> None:
    request = SLAMRunRequest(
        dataset_path=tmp_path,
        slam_config="dummy_config",
        output_dir=tmp_path / "output",
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        sequence_name="V1_01_easy",
        extras={},
    )

    algo = PhotoSLAMAlgorithm()
    with pytest.raises(RuntimeError, match="timestamps_by_frame"):
        algo._require_euroc_timestamps_by_frame(request)
