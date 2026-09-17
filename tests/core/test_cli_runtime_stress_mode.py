"""Tests for CLI routing of runtime-stress evaluation mode."""

from types import SimpleNamespace

from slamadversariallab.cli import create_parser, evaluate_command


def test_evaluate_mode_includes_runtime_stress_choice() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "config.yaml",
            "--slam",
            "gigaslam",
            "--mode",
            "runtime-stress",
        ]
    )
    assert args.mode == "runtime-stress"


def test_evaluate_command_routes_to_runtime_stress_pipeline(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment:\n  name: test\n", encoding="utf-8")

    called = {}

    class _StubRuntimeStressPipeline:
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
            return {"baseline_run_0": tmp_path / "traj.txt"}

    monkeypatch.setattr(
        "slamadversariallab.pipelines.runtime_stress_evaluation.RuntimeStressEvaluationPipeline",
        _StubRuntimeStressPipeline,
    )

    args = SimpleNamespace(
        config=str(config_path),
        slam=["gigaslam"],
        mode="runtime-stress",
        verbose=False,
        slam_config_path=None,
        num_runs=1,
        paper_mode=False,
    )

    rc = evaluate_command(args)

    assert rc == 0
    assert called["slam_algorithm"] == "gigaslam"
