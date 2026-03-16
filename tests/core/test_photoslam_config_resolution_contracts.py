"""Tests for Photo-SLAM strict config resolution contracts."""

from pathlib import Path

from slamadverseriallab.algorithms.photoslam import PhotoSLAMAlgorithm
from slamadverseriallab.algorithms.types import SLAMRunRequest, SLAMRuntimeContext, SensorMode


def _build_algo_with_cfg_root(tmp_path: Path) -> tuple[PhotoSLAMAlgorithm, Path]:
    photoslam_root = tmp_path / "Photo-SLAM"
    (photoslam_root / "cfg" / "ORB_SLAM3" / "Monocular" / "TUM").mkdir(parents=True, exist_ok=True)
    (photoslam_root / "cfg" / "gaussian_mapper" / "Monocular" / "TUM").mkdir(
        parents=True, exist_ok=True
    )
    algo = PhotoSLAMAlgorithm()
    algo.photoslam_path = photoslam_root
    return algo, photoslam_root


def test_photoslam_orb_config_resolves_exact_yaml_name(tmp_path: Path) -> None:
    algo, root = _build_algo_with_cfg_root(tmp_path)
    expected = root / "cfg" / "ORB_SLAM3" / "Monocular" / "TUM" / "tum_freiburg1_desk.yaml"
    expected.write_text("dummy: true\n", encoding="utf-8")

    resolved = algo._get_orb_config("tum_freiburg1_desk", "Monocular", "TUM", is_external=False)

    assert resolved == expected


def test_photoslam_orb_config_missing_requested_does_not_fallback(tmp_path: Path) -> None:
    algo, root = _build_algo_with_cfg_root(tmp_path)
    other = root / "cfg" / "ORB_SLAM3" / "Monocular" / "TUM" / "some_other_config.yaml"
    other.write_text("dummy: true\n", encoding="utf-8")

    resolved = algo._get_orb_config("requested_config", "Monocular", "TUM", is_external=False)

    assert resolved is None


def test_photoslam_gaussian_config_missing_requested_does_not_fallback(tmp_path: Path) -> None:
    algo, root = _build_algo_with_cfg_root(tmp_path)
    # These names previously participated in fallback selection and must no longer be used.
    (root / "cfg" / "gaussian_mapper" / "Monocular" / "TUM" / "tum_monocular.yaml").write_text(
        "dummy: true\n", encoding="utf-8"
    )
    (root / "cfg" / "gaussian_mapper" / "Monocular" / "TUM" / "first_available.yaml").write_text(
        "dummy: true\n", encoding="utf-8"
    )

    resolved = algo._get_gaussian_config("requested_config", "Monocular", "TUM")

    assert resolved is None


def test_photoslam_execution_spec_fails_when_requested_configs_are_missing(tmp_path: Path) -> None:
    algo, root = _build_algo_with_cfg_root(tmp_path)
    (root / "cfg" / "ORB_SLAM3" / "Monocular" / "TUM" / "fallback_candidate.yaml").write_text(
        "dummy: true\n", encoding="utf-8"
    )
    (root / "cfg" / "gaussian_mapper" / "Monocular" / "TUM" / "fallback_candidate.yaml").write_text(
        "dummy: true\n", encoding="utf-8"
    )

    dataset_path = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    request = SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="requested_config",
        output_dir=output_dir,
        dataset_type="tum",
        sensor_mode=SensorMode.MONO,
        sequence_name="freiburg1_desk",
        extras={},
    )
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name="requested_config",
        sequence_name=request.sequence_name,
    )
    ctx.execution_inputs = {
        "dataset_path": dataset_path,
        "slam_config": "requested_config",
        "output_dir": output_dir,
        "dataset_type": "tum",
        "is_stereo": False,
        "is_external": False,
        "staged_association_file": None,
        "staged_timestamps_file": None,
    }

    spec = algo._build_execution_spec(request, ctx)

    assert spec is None
