"""Tests for CLI routing of robustness-boundary evaluation mode."""

from types import SimpleNamespace

from slamadverseriallab.cli import create_parser, evaluate_command


def test_evaluate_mode_includes_robustness_boundary_choice() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "config.yaml",
            "--slam",
            "orbslam3",
            "--mode",
            "robustness-boundary",
        ]
    )
    assert args.mode == "robustness-boundary"


def test_evaluate_command_routes_to_robustness_boundary_pipeline(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment:\n  name: test\n", encoding="utf-8")

    called = {}

    class _StubBoundaryPipeline:
        def __init__(
            self,
            config_path,
            slam_algorithm,
            slam_config_path=None,
            num_runs=1,
            paper_mode=False,
        ):
            called["config_path"] = config_path
            called["slam_algorithm"] = slam_algorithm
            called["slam_config_path"] = slam_config_path
            called["num_runs"] = num_runs
            called["paper_mode"] = paper_mode

        def run(self):
            return {"summary_path": tmp_path / "boundary_summary.json"}

    monkeypatch.setattr(
        "slamadverseriallab.pipelines.robustness_boundary.RobustnessBoundaryPipeline",
        _StubBoundaryPipeline,
    )

    args = SimpleNamespace(
        config=str(config_path),
        slam=["orbslam3"],
        mode="robustness-boundary",
        verbose=False,
        slam_config_path=None,
        num_runs=1,
        paper_mode=False,
    )

    rc = evaluate_command(args)

    assert rc == 0
    assert called["slam_algorithm"] == "orbslam3"
