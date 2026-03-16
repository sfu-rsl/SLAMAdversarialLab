"""Tests for list-modules CLI output modes."""

from types import SimpleNamespace

import yaml

from slamadverseriallab.cli import create_parser, list_modules_command


def test_list_modules_parser_accepts_yaml_format() -> None:
    parser = create_parser()
    args = parser.parse_args(["list-modules", "--module", "fog", "--format", "yaml"])

    assert args.command == "list-modules"
    assert args.module == "fog"
    assert args.format == "yaml"


def test_list_modules_command_requires_module_for_non_text_format(capsys) -> None:
    args = SimpleNamespace(detailed=False, module=None, all=False, format="yaml")

    rc = list_modules_command(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert "--format requires --module" in captured.err


def test_list_modules_command_yaml_output(capsys) -> None:
    args = SimpleNamespace(detailed=False, module="fog", all=False, format="yaml")

    rc = list_modules_command(args)

    captured = capsys.readouterr()
    parsed = yaml.safe_load(captured.out)

    assert rc == 0
    assert parsed["perturbations"][0]["type"] == "fog"
    assert parsed["perturbations"][0]["parameters"]["visibility_m"] == 50.0
