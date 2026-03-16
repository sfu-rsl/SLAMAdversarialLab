"""Tests for robustness-boundary config parsing and validation."""

from types import SimpleNamespace

import pytest

from slamadverseriallab.config.parser import Config, parse_robustness_boundary
from slamadverseriallab.config.schema import (
    DatasetConfig,
    ExperimentConfig,
    OutputConfig,
    RobustnessBoundaryConfig,
)
from slamadverseriallab.robustness.param_spec import BoundaryParamSpec


class _FogBoundaryStub:
    SEARCHABLE_PARAMS = {"visibility_m": BoundaryParamSpec(domain="continuous")}


class _RainBoundaryStub:
    SEARCHABLE_PARAMS = {"intensity": BoundaryParamSpec(domain="integer")}


def _set_registry(monkeypatch: pytest.MonkeyPatch, entries):
    monkeypatch.setattr(
        "slamadverseriallab.modules.base.get_module_registry",
        lambda: entries,
    )


def test_parse_robustness_boundary_accepts_continuous_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_registry(
        monkeypatch,
        {"fog": SimpleNamespace(module_class=_FogBoundaryStub)},
    )
    config_dict = {
        "robustness_boundary": {
            "enabled": True,
            "module": "fog",
            "parameter": "visibility_m",
            "lower_bound": 15.0,
            "upper_bound": 120.0,
        }
    }

    rb = parse_robustness_boundary(config_dict)

    assert rb is not None
    assert rb.enabled is True
    assert rb.module == "fog"
    assert rb.parameter == "visibility_m"
    assert rb.lower_bound == 15.0
    assert rb.upper_bound == 120.0
    assert rb.fail_on_tracking_failure is True


def test_parse_robustness_boundary_accepts_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_registry(
        monkeypatch,
        {"fog": SimpleNamespace(module_class=_FogBoundaryStub)},
    )
    config_dict = {
        "robustness_boundary": {
            "enabled": True,
            "name": "fog_visibility_boundary",
            "module": "fog",
            "parameter": "visibility_m",
            "lower_bound": 15.0,
            "upper_bound": 120.0,
        }
    }

    rb = parse_robustness_boundary(config_dict)

    assert rb is not None
    assert rb.name == "fog_visibility_boundary"


def test_parse_robustness_boundary_rejects_integer_decimal_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_registry(
        monkeypatch,
        {"rain": SimpleNamespace(module_class=_RainBoundaryStub)},
    )
    config_dict = {
        "robustness_boundary": {
            "enabled": True,
            "module": "rain",
            "parameter": "intensity",
            "lower_bound": 1.5,
            "upper_bound": 10,
        }
    }

    with pytest.raises(ValueError, match="Invalid robustness_boundary.lower_bound"):
        parse_robustness_boundary(config_dict)


def test_parse_robustness_boundary_rejects_unsupported_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_registry(
        monkeypatch,
        {"fog": SimpleNamespace(module_class=_FogBoundaryStub)},
    )
    config_dict = {
        "robustness_boundary": {
            "enabled": True,
            "module": "fog",
            "parameter": "intensity",
            "lower_bound": 1,
            "upper_bound": 10,
        }
    }

    with pytest.raises(ValueError, match="is not supported for robustness boundary"):
        parse_robustness_boundary(config_dict)


def test_config_to_dict_includes_lower_and_upper_bound() -> None:
    rb = RobustnessBoundaryConfig(
        enabled=True,
        name="fog_boundary_test",
        target_perturbation="night_fog_heavy",
        module="fog",
        parameter="visibility_m",
        lower_bound=20.0,
        upper_bound=100.0,
    )
    cfg = Config(
        experiment=ExperimentConfig(name="rb_to_dict"),
        dataset=DatasetConfig(type="mock", path="mock"),
        perturbations=[],
        output=OutputConfig(base_dir="./results", save_images=False, create_timestamp_dir=False),
        robustness_boundary=rb,
    )

    as_dict = cfg.to_dict()

    assert "robustness_boundary" in as_dict
    assert as_dict["robustness_boundary"]["name"] == "fog_boundary_test"
    assert as_dict["robustness_boundary"]["target_perturbation"] == "night_fog_heavy"
    assert as_dict["robustness_boundary"]["lower_bound"] == 20.0
    assert as_dict["robustness_boundary"]["upper_bound"] == 100.0


def test_config_to_dict_preserves_dataset_mode_flags() -> None:
    cfg = Config(
        experiment=ExperimentConfig(name="rb_dataset_flags"),
        dataset=DatasetConfig(
            type="kitti",
            path="/tmp/kitti",
            skip_depth=True,
            load_stereo=True,
        ),
        perturbations=[],
        output=OutputConfig(base_dir="./results", save_images=False, create_timestamp_dir=False),
    )

    as_dict = cfg.to_dict()

    assert as_dict["dataset"]["skip_depth"] is True
    assert as_dict["dataset"]["load_stereo"] is True


def test_parse_robustness_boundary_accepts_frame_drop_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FrameDropBoundaryStub:
        SEARCHABLE_PARAMS = {"drop_rate": BoundaryParamSpec(domain="integer")}

    _set_registry(
        monkeypatch,
        {"frame_drop": SimpleNamespace(module_class=_FrameDropBoundaryStub)},
    )
    config_dict = {
        "robustness_boundary": {
            "enabled": True,
            "module": "frame_drop",
            "parameter": "drop_rate",
            "lower_bound": 10,
            "upper_bound": 50,
        }
    }

    rb = parse_robustness_boundary(config_dict)

    assert rb is not None
    assert rb.module == "frame_drop"
    assert rb.parameter == "drop_rate"
    assert rb.lower_bound == 10
    assert rb.upper_bound == 50


def test_parse_robustness_boundary_accepts_target_perturbation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_registry(
        monkeypatch,
        {"fog": SimpleNamespace(module_class=_FogBoundaryStub)},
    )
    config_dict = {
        "robustness_boundary": {
            "enabled": True,
            "target_perturbation": "night_fog_heavy",
            "module": "fog",
            "parameter": "visibility_m",
            "lower_bound": 10,
            "upper_bound": 200,
        }
    }

    rb = parse_robustness_boundary(config_dict)

    assert rb is not None
    assert rb.target_perturbation == "night_fog_heavy"


def test_parse_robustness_boundary_accepts_disable_tracking_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_registry(
        monkeypatch,
        {"fog": SimpleNamespace(module_class=_FogBoundaryStub)},
    )
    config_dict = {
        "robustness_boundary": {
            "enabled": True,
            "module": "fog",
            "parameter": "visibility_m",
            "lower_bound": 10,
            "upper_bound": 200,
            "fail_on_tracking_failure": False,
        }
    }

    rb = parse_robustness_boundary(config_dict)

    assert rb is not None
    assert rb.fail_on_tracking_failure is False
