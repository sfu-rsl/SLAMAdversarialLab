"""Tests for module registry parameter documentation output."""

import yaml

from slamadverseriallab.modules import get_module_documentation, list_modules


def test_module_registry_exposes_descriptions_for_representative_fields() -> None:
    modules = list_modules(detailed=True)

    fog_params = modules["fog"]["parameters"]
    assert fog_params["visibility_m"]["description"] == "Visibility distance in meters"
    assert fog_params["encoder"]["choices"] == ["vits", "vitb", "vitl", "vitg"]

    cracked_lens_params = modules["cracked_lens"]["parameters"]
    assert cracked_lens_params["impact_force"]["description"]

    lens_patch_params = modules["lens_patch"]["parameters"]
    assert lens_patch_params["position"]["choices"]

    frame_drop_params = modules["frame_drop"]["parameters"]
    assert frame_drop_params["mode"]["choices"] == ["random", "periodic"]


def test_get_module_documentation_includes_yaml_example() -> None:
    documentation = get_module_documentation("fog")

    assert "Example YAML:" in documentation
    assert "type: fog" in documentation
    assert "visibility_m:" in documentation
    assert "Description: Visibility distance in meters" in documentation


def test_get_module_documentation_yaml_format_is_valid_yaml() -> None:
    yaml_text = get_module_documentation("fog", output_format="yaml")

    parsed = yaml.safe_load(yaml_text)
    perturbation = parsed["perturbations"][0]

    assert perturbation["type"] == "fog"
    assert perturbation["enabled"] is True
    assert perturbation["parameters"]["visibility_m"] == 50.0
    assert perturbation["parameters"]["encoder"] == "vitl"
    assert "start_visibility_m" not in perturbation["parameters"]
    assert "# Optional parameters:" in yaml_text
