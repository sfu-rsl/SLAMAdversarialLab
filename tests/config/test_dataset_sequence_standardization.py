"""Tests for standardized dataset.sequence config behavior."""

from pathlib import Path

import pytest

from slamadverseriallab.config.parser import parse_dataset, load_config


def test_parse_dataset_rejects_legacy_sequences_key() -> None:
    config_dict = {
        "dataset": {
            "type": "kitti",
            "path": "./datasets/kitti/sequences/04",
            "sequences": ["04"],
        }
    }

    with pytest.raises(ValueError, match="dataset\\.sequences is no longer supported"):
        parse_dataset(config_dict)


def test_parse_dataset_accepts_singular_sequence_key() -> None:
    config_dict = {
        "dataset": {
            "type": "kitti",
            "sequence": "04",
        }
    }

    dataset = parse_dataset(config_dict)
    assert dataset.sequence == "04"


def test_config_to_dict_emits_sequence_not_sequences(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: sequence_standardization",
                "dataset:",
                "  type: tum",
                "  sequence: freiburg1_desk",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    data = config.to_dict()

    assert data["dataset"]["sequence"] == "freiburg1_desk"
    assert "sequences" not in data["dataset"]
