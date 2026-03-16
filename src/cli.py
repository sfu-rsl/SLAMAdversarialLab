"""Command-line interface for SLAMAdverserialLab."""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

from .__version__ import __version__
from .utils import get_logger
from .config import load_config

logger = get_logger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging based on verbosity.

    Args:
        verbose: If True, set log level to DEBUG
    """
    level = "DEBUG" if verbose else "INFO"

    # Update root logger level
    active_roots = {__name__.split(".")[0], "slamadverseriallab"}

    for root_name in active_roots:
        root_logger = logging.getLogger(root_name)
        if not root_logger.handlers:
            continue
        root_logger.setLevel(getattr(logging, level))

        # Update all handlers
        for handler in root_logger.handlers:
            handler.setLevel(getattr(logging, level))


def run_command(args: argparse.Namespace) -> int:
    """
    Execute the run command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Setup logging
    setup_logging(args.verbose)

    config_path = Path(args.config)
    logger.info(f"Loading configuration from {config_path}")

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Print configuration summary
    print("\n" + "=" * 60)
    print("SLAMAdverserialLab Configuration Summary")
    print("=" * 60)
    print(f"Experiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description or 'N/A'}")
    print(f"Version: {config.experiment.version}")
    print(f"Random Seed: {config.experiment.seed or 'Not set'}")
    print("-" * 60)
    print(f"Dataset Type: {config.dataset.type}")
    # Display path or sequence
    if config.dataset.path:
        print(f"Dataset Path: {config.dataset.path}")
    elif config.dataset.sequence:
        print(f"Dataset Sequence: {config.dataset.sequence}")
    else:
        print(f"Dataset Path: Not specified")

    # Extract sequence information
    if config.dataset.sequence:
        # Sequence is explicitly specified
        sequences_display = config.dataset.sequence
    elif config.dataset.path:
        # Fall back to path-derived sequence label for path-based configs
        sequences_display = Path(config.dataset.path).name
    else:
        sequences_display = 'All'

    print(f"Sequences: {sequences_display}")
    print(f"Max Frames: {config.dataset.max_frames or 'Unlimited'}")
    print("-" * 60)
    print(f"Perturbations ({len(config.perturbations)}):")
    for pert in config.perturbations:
        status = "Enabled" if pert.enabled else "Disabled"
        print(f"  - {pert.name} ({pert.type}): {status}")
        if args.verbose and pert.parameters:
            for key, value in pert.parameters.items():
                print(f"      {key}: {value}")
    print("-" * 60)
    print(f"Output Directory: {config.output.base_dir}")
    print(f"Save Images: {config.output.save_images}")
    print(f"Timestamp Subdirectory: {config.output.create_timestamp_dir}")
    print("=" * 60)

    if args.dry_run:
        logger.info("Dry run mode - configuration validated successfully")
        print("\n[Dry Run] Configuration validated successfully. No processing performed.")
        return 0

    # Execute pipeline
    from .core import Pipeline

    print("\nInitializing pipeline...")
    pipeline = Pipeline(config)

    try:
        print("Setting up pipeline components...")
        pipeline.setup()

        print("\nExecuting pipeline...")
        results = pipeline.run()

        print("\n" + "=" * 60)
        print("Pipeline Execution Complete")
        print("=" * 60)
        print(f"Status: {results.get('status', 'unknown')}")
        print(f"Frames Processed: {results.get('frames_processed', 0)}")
        print(f"Elapsed Time: {results.get('elapsed_time', 0):.2f} seconds")

        if results.get('errors'):
            print(f"Errors: {len(results['errors'])}")
            for err in results['errors']:
                print(f"  - {err}")

        return 0
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\n[Error] Pipeline execution failed: {e}")
        return 1


def evaluate_command(args: argparse.Namespace) -> int:
    """
    Execute the evaluate command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Setup logging
    setup_logging(args.verbose)

    config_path = Path(args.config)
    slam_config_path = args.slam_config_path  # Optional external config path

    mode = args.mode
    is_robustness_boundary = mode == "robustness-boundary"
    if not is_robustness_boundary:
        if mode == "full":
            skip_slam = False
            compute_metrics = True
        elif mode == "slam-only":
            skip_slam = False
            compute_metrics = False
        elif mode == "metrics-only":
            skip_slam = True
            compute_metrics = True
        else:
            logger.error(f"Unknown mode: {mode}")
            return 1

    # args.slam is now a list (nargs='+')
    slam_algorithms = args.slam

    logger.info(f"Loading experiment config from {config_path}")
    logger.info(f"SLAM algorithm(s): {', '.join(slam_algorithms)}")
    logger.info(f"Mode: {mode}")
    if slam_config_path:
        logger.info(f"SLAM config path: {slam_config_path}")
    else:
        logger.info("SLAM config: will infer from dataset.sequence")

    num_runs = getattr(args, 'num_runs', 1)
    if num_runs > 1:
        logger.info(f"Multi-run mode: {num_runs} SLAM executions on same perturbed data")

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    if slam_config_path:
        external_path = Path(slam_config_path)
        if not external_path.exists():
            logger.error(f"External SLAM config file not found: {external_path}")
            return 1

    paper_mode = getattr(args, 'paper_mode', False)
    all_trajectories = {}
    failed_algorithms = []
    boundary_summaries = {}

    for idx, slam_algorithm in enumerate(slam_algorithms):
        print("\n" + "=" * 60)
        print(f"EVALUATING ALGORITHM [{idx + 1}/{len(slam_algorithms)}]: {slam_algorithm}")
        print("=" * 60)

        try:
            if is_robustness_boundary:
                from .pipelines.robustness_boundary import RobustnessBoundaryPipeline

                pipeline = RobustnessBoundaryPipeline(
                    config_path=config_path,
                    slam_algorithm=slam_algorithm,
                    slam_config_path=slam_config_path,
                    num_runs=num_runs,
                    paper_mode=paper_mode,
                )
                result = pipeline.run()
                boundary_summaries[slam_algorithm] = result["summary_path"]
            else:
                from .pipelines.evaluation import EvaluationPipeline

                pipeline = EvaluationPipeline(
                    config_path=config_path,
                    slam_algorithm=slam_algorithm,
                    slam_config_path=slam_config_path,
                    compute_metrics=compute_metrics,
                    skip_slam=skip_slam,
                    num_runs=num_runs,
                    paper_mode=paper_mode
                )

                trajectories = pipeline.run()

                # Store results with algorithm prefix
                for name, path in trajectories.items():
                    all_trajectories[f"{slam_algorithm}/{name}"] = path

        except FileNotFoundError as e:
            logger.error(f"[{slam_algorithm}] {str(e)}")
            failed_algorithms.append(slam_algorithm)
        except Exception as e:
            logger.error(f"[{slam_algorithm}] Evaluation failed: {e}", exc_info=args.verbose)
            failed_algorithms.append(slam_algorithm)

    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Algorithms evaluated: {len(slam_algorithms) - len(failed_algorithms)}/{len(slam_algorithms)}")
    if failed_algorithms:
        print(f"Failed: {', '.join(failed_algorithms)}")
    if is_robustness_boundary:
        print(f"Boundary summaries generated: {len(boundary_summaries)}")
        for algorithm, summary_path in sorted(boundary_summaries.items()):
            print(f"  {algorithm}: {summary_path}")
    else:
        print(f"Total trajectories generated: {len(all_trajectories)}")
        for name, path in sorted(all_trajectories.items()):
            print(f"  {name}: {path}")
    print("=" * 60)

    return 1 if failed_algorithms else 0


def evaluate_vo_command(args: argparse.Namespace) -> int:
    """
    Execute the evaluate-vo command using PySLAM Visual SLAM framework.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Setup logging
    setup_logging(args.verbose)

    config_path = Path(args.config)
    camera_settings = args.camera_settings
    sensor_type = args.sensor_type
    skip_run = args.skip_run
    comparison_only = args.comparison_only
    num_runs = args.num_runs

    # Parse features (comma-separated)
    feature_configs = [f.strip() for f in args.features.split(",")]

    logger.info(f"Loading experiment config from {config_path}")
    if camera_settings:
        logger.info(f"Camera settings: {camera_settings}")
    else:
        logger.info("Camera settings: (will be inferred from dataset)")
    logger.info(f"Feature configs: {feature_configs}")
    logger.info(f"Sensor type: {sensor_type}")
    logger.info(f"Runs: {num_runs}")

    if comparison_only:
        logger.info("Comparison only: True (skipping PySLAM, generating comparison plots)")
    elif skip_run:
        logger.info("Skip run: True (will only print what would run)")

    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    try:
        from .pipelines.vo_evaluation import VOEvaluationPipeline

        pipeline = VOEvaluationPipeline(
            config_path=config_path,
            feature_configs=feature_configs,
            sensor_type=sensor_type,
            camera_settings=camera_settings,
            skip_run=skip_run,
            comparison_only=comparison_only,
            num_runs=num_runs,
        )

        results = pipeline.run()

        print("\n" + "=" * 60)
        print("VISUAL ODOMETRY EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Feature configs evaluated: {len(feature_configs)}")
        for feature_config, feature_results in results.items():
            success_count = sum(1 for p in feature_results.values() if p is not None)
            total_count = len(feature_results)
            print(f"  {feature_config}: {success_count}/{total_count} successful")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"VO evaluation failed: {e}", exc_info=args.verbose)
        return 1


def list_algorithms_command(args: argparse.Namespace) -> int:
    """
    Execute the list-algorithms command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    from .algorithms.registry import list_slam_algorithms_detailed, get_algorithm_documentation

    algorithms = list_slam_algorithms_detailed()

    if args.algorithm:
        # Show documentation for specific algorithm
        print(get_algorithm_documentation(args.algorithm))
        return 0

    # Display algorithm list
    print("\nAvailable SLAM Algorithms")
    print("=" * 60)

    if not algorithms:
        print("No algorithms registered.")
        return 0

    if args.detailed:
        # Detailed view
        for name, info in sorted(algorithms.items()):
            print(f"\n{name}")
            print("-" * 40)
            print(f"  Description: {info['description']}")
            print(f"  Class: {info['class_name']}")
            print("  Supported Datasets:")
            if info['supported_datasets']:
                for dataset_type, modes in info['supported_datasets'].items():
                    modes_str = ', '.join(modes)
                    print(f"    - {dataset_type}: {modes_str}")
            else:
                print("    (none or error loading)")
    else:
        # Table view
        print("\nAlgorithm        Datasets (modes)")
        print("-" * 60)
        for name in sorted(algorithms.keys()):
            info = algorithms[name]
            # Format supported datasets
            datasets_parts = []
            for dataset_type, modes in info.get('supported_datasets', {}).items():
                modes_str = '/'.join(modes)
                datasets_parts.append(f"{dataset_type}({modes_str})")
            datasets_str = ', '.join(datasets_parts) if datasets_parts else 'N/A'

            # Truncate if too long
            max_len = 42
            if len(datasets_str) > max_len:
                datasets_str = datasets_str[:max_len - 3] + "..."

            print(f"{name:16} {datasets_str}")

    print()
    return 0


def list_modules_command(args: argparse.Namespace) -> int:
    """
    Execute the list-modules command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    from .modules import list_modules, get_module_documentation

    if getattr(args, 'format', 'text') != "text" and not args.module:
        print("Error: --format requires --module", file=sys.stderr)
        return 1

    include_deprecated = getattr(args, 'all', False)
    modules = list_modules(detailed=args.detailed, include_deprecated=include_deprecated)

    if args.module:
        # Show documentation for specific module
        print(get_module_documentation(args.module, output_format=args.format))
        return 0

    # Display module list
    print("\nAvailable Perturbation Modules")
    print("=" * 60)

    if not modules:
        print("No modules registered.")
        return 0

    if args.detailed:
        # Detailed view
        for name, info in sorted(modules.items()):
            deprecated_marker = " [DEPRECATED]" if info.get('deprecated') else ""
            print(f"\n{name}{deprecated_marker}")
            print("-" * 40)
            print(f"  Description: {info['description']}")
            print(f"  Loaded: {'Yes' if info['loaded'] else 'No (lazy)'}")

            if info.get('deprecated'):
                if info.get('deprecation_message'):
                    print(f"  Deprecation: {info['deprecation_message']}")
                if info.get('replacement'):
                    print(f"  Replacement: {info['replacement']}")

            if info.get('parameters'):
                print("  Parameters:")
                params = info['parameters']
                if isinstance(params, dict):
                    for param_name, param_info in params.items():
                        if isinstance(param_info, dict):
                            # Format: name (type) = default
                            ptype = param_info.get('type', '?')
                            pdefault = param_info.get('default', '?')
                            pdesc = param_info.get('description', '')
                            pchoices = param_info.get('choices', None)

                            # Format default value nicely
                            if pdefault is None:
                                default_str = 'None'
                            elif isinstance(pdefault, str):
                                default_str = f'"{pdefault}"'
                            else:
                                default_str = str(pdefault)

                            print(f"    - {param_name}: {ptype} = {default_str}")
                            if pdesc:
                                print(f"        {pdesc}")
                            if pchoices:
                                print(f"        Choices: {pchoices}")
                        else:
                            print(f"    - {param_name}: {param_info}")
    else:
        # Simple list
        print("\nModule Name        Description")
        print("-" * 60)
        # Re-fetch with details for descriptions
        detailed_modules = list_modules(detailed=True, include_deprecated=include_deprecated)
        for name in sorted(modules.keys()):
            info = detailed_modules.get(name, {})
            desc = info.get('description', 'No description')
            deprecated = info.get('deprecated', False)
            # Truncate description if too long
            max_desc_len = 32 if deprecated else 40
            if len(desc) > max_desc_len:
                desc = desc[:max_desc_len - 3] + "..."
            deprecated_marker = " [DEPRECATED]" if deprecated else ""
            print(f"{name:18} {desc}{deprecated_marker}")

    # Show hint about deprecated modules if not showing all
    if not include_deprecated:
        all_modules = list_modules(detailed=True, include_deprecated=True)
        deprecated_count = sum(1 for m in all_modules.values() if m.get('deprecated'))
        if deprecated_count > 0:
            print(f"\n({deprecated_count} deprecated module(s) hidden. Use --all to show them.)")

    print()
    return 0


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="slamadverseriallab",
        description="SLAMAdverserialLab - A research framework for stress-testing SLAM systems using weather perturbations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate perturbed data
  slamadverseriallab run configs/experiment.yaml

  slamadverseriallab run configs/experiment.yaml --dry-run

  # Evaluate with S3PO-GS (run SLAM + compute metrics, default mode)
  slamadverseriallab evaluate configs/experiment.yaml --slam s3pogs

  # Run SLAM only, skip metrics computation
  slamadverseriallab evaluate configs/experiment.yaml --slam s3pogs --mode slam-only

  slamadverseriallab evaluate configs/experiment.yaml --slam s3pogs --mode metrics-only

  # Evaluate with external config file
  slamadverseriallab evaluate configs/experiment.yaml \\
      --slam orbslam3 \\
      --slam-config-path /path/to/custom_config.yaml

  # Run with verbose logging
  slamadverseriallab run configs/experiment.yaml --verbose

  # Show version
  slamadverseriallab --version
"""
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"SLAMAdverserialLab v{__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run a SLAMAdverserialLab experiment",
        description="Execute an experiment using a configuration file"
    )
    run_parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file"
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running the experiment"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run SLAM evaluation on original and perturbed datasets",
        description="Execute SLAM algorithm on datasets generated by 'run' command"
    )
    evaluate_parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file (same as used for 'run' command)"
    )
    from .algorithms.registry import list_slam_algorithms
    evaluate_parser.add_argument(
        "--slam",
        type=str,
        nargs='+',
        required=True,
        choices=list_slam_algorithms(),
        metavar='ALGORITHM',
        help="SLAM algorithm(s) to use for evaluation (can specify multiple)"
    )
    evaluate_parser.add_argument(
        "--slam-config-path",
        type=str,
        required=False,
        default=None,
        help="Path to external SLAM config file. If not provided, config is inferred from dataset.sequence."
    )
    evaluate_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    evaluate_parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "slam-only", "metrics-only", "robustness-boundary"],
        help="Evaluation mode: 'full' (run SLAM + compute metrics), 'slam-only' (run SLAM, no metrics), 'metrics-only' (skip SLAM, compute metrics from existing trajectories), 'robustness-boundary' (search pass/fail boundary for one perturbation parameter). Default: full"
    )
    evaluate_parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of SLAM runs to execute on the same perturbed data (default: 1)"
    )
    evaluate_parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="Generate paper-ready plots: no legends, consistent severity colors, output to slam_results_for_paper/"
    )

    # List modules command
    list_parser = subparsers.add_parser(
        "list-modules",
        help="List available perturbation modules",
        description="Display information about registered perturbation modules"
    )
    list_parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed information including parameters"
    )
    list_parser.add_argument(
        "--module", "-m",
        type=str,
        help="Show detailed documentation for a specific module"
    )
    list_parser.add_argument(
        "--format",
        choices=["text", "yaml"],
        default="text",
        help="Output format for --module documentation"
    )
    list_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Include deprecated modules in the listing"
    )

    # List algorithms command
    list_algo_parser = subparsers.add_parser(
        "list-algorithms",
        help="List available SLAM algorithms and their supported datasets",
        description="Display information about registered SLAM algorithms and what datasets/cameras they support"
    )
    list_algo_parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed information including descriptions"
    )
    list_algo_parser.add_argument(
        "--algorithm", "-a",
        type=str,
        help="Show detailed documentation for a specific algorithm"
    )

    # Evaluate-vo command (PySLAM Visual Odometry framework)
    evaluate_vo_parser = subparsers.add_parser(
        "evaluate-vo",
        help="Evaluate using PySLAM Visual SLAM framework with various feature extractors",
        description="Run PySLAM on perturbed datasets generated by 'run' command to evaluate feature extractor robustness"
    )
    evaluate_vo_parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file (same as used for 'run' command)"
    )
    evaluate_vo_parser.add_argument(
        "--camera-settings",
        type=str,
        default=None,
        help="PySLAM camera settings file (e.g., 'KITTI04-12.yaml', 'TUM1.yaml'). "
             "If not specified, will be inferred from dataset type and sequence."
    )
    evaluate_vo_parser.add_argument(
        "--features",
        type=str,
        default="ORB2",
        help="Feature tracker config(s), comma-separated (e.g., 'ORB2,SIFT,SUPERPOINT'). Default: ORB2"
    )
    evaluate_vo_parser.add_argument(
        "--sensor-type",
        type=str,
        default="stereo",
        choices=["mono", "stereo", "rgbd"],
        help="Sensor type for SLAM (default: stereo)"
    )
    evaluate_vo_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    evaluate_vo_parser.add_argument(
        "--skip-run",
        action="store_true",
        default=False,
        help="Skip PySLAM execution, just print what would run"
    )
    evaluate_vo_parser.add_argument(
        "--comparison-only",
        action="store_true",
        default=False,
        help="Skip PySLAM execution and only generate comparison plots from existing track data"
    )
    evaluate_vo_parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of VO runs to execute on the same perturbed data (default: 1)"
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate command handler
    if args.command == "run":
        return run_command(args)
    elif args.command == "evaluate":
        return evaluate_command(args)
    elif args.command == "evaluate-vo":
        return evaluate_vo_command(args)
    elif args.command == "list-modules":
        return list_modules_command(args)
    elif args.command == "list-algorithms":
        return list_algorithms_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
