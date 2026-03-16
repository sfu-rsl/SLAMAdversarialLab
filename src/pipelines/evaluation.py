"""SLAM evaluation pipeline for running SLAM on original and perturbed datasets."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.parser import load_config
from ..algorithms.base import SLAMAlgorithm
from ..algorithms.types import SensorMode, SLAMRunRequest
from ..algorithms.registry import get_slam_algorithm
from ..metrics.trajectory import MetricsEvaluator, detect_trajectory_format, plot_trajectories, plot_metric_comparison
from ..datasets.factory import create_dataset
from ..utils.paths import create_temp_dir, set_temp_dir_root, cleanup_temp_root

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Evaluate a SLAM backend against baseline and perturbed datasets."""

    def __init__(
        self,
        config_path: Path,
        slam_algorithm: str,
        slam_config_path: Optional[str] = None,
        compute_metrics: bool = False,
        skip_slam: bool = False,
        num_runs: int = 1,
        paper_mode: bool = False
    ):
        """Initialize evaluation pipeline.

        Args:
            config_path: Path to experiment YAML config (same as used for 'run')
            slam_algorithm: Name of SLAM algorithm to use ('orbslam3', 's3pogs', etc.)
            slam_config_path: Optional path to external SLAM config file.
                If not provided, config is inferred from dataset.sequence.
            compute_metrics: If True, compute APE/RPE metrics using evo (requires ground truth)
            skip_slam: If True, skip SLAM execution and use existing trajectories
            num_runs: Number of SLAM runs to execute on the same perturbed data
            paper_mode: If True, generate paper-ready plots (no legends, severity colors)
                and output to slam_results_for_paper/ instead of slam_results/

        Raises:
            ValueError: If config is invalid or algorithm not found
            FileNotFoundError: If config files don't exist
        """
        self.config_path = Path(config_path)
        self.compute_metrics = compute_metrics
        self.skip_slam = skip_slam
        self.num_runs = num_runs
        self.paper_mode = paper_mode

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        logger.info(f"Loading experiment config from {self.config_path}")
        self.config = load_config(str(self.config_path))

        logger.info(f"Initializing SLAM algorithm: {slam_algorithm}")
        self.algorithm: SLAMAlgorithm = get_slam_algorithm(slam_algorithm)

        if slam_config_path:
            # External config file provided
            self.slam_config = slam_config_path
            logger.info(f"Using external SLAM config: {slam_config_path}")
        else:
            # Infer from dataset.sequence using algorithm's resolve_config_name()
            if not self.config.dataset.sequence:
                raise ValueError(
                    f"No --slam-config-path provided and no dataset.sequence in config. "
                    f"Either provide --slam-config-path or add 'sequence' to your dataset config."
                )
            resolved_config = self.algorithm.resolve_config_name(
                self.config.dataset.sequence,
                self.config.dataset.type
            )
            if not resolved_config:
                raise ValueError(
                    f"Algorithm '{slam_algorithm}' cannot infer config from sequence "
                    f"'{self.config.dataset.sequence}' (dataset type: {self.config.dataset.type}). "
                    f"Please provide --slam-config-path explicitly."
                )
            self.slam_config = resolved_config
            logger.info(f"Inferred SLAM config from sequence: {resolved_config}")

        # Extract experiment info from config
        self.experiment_name = self.config.experiment.name

        self.dataset_type = self.config.dataset.type
        self.dataset = create_dataset(self.config.dataset)
        dataset_path_value = getattr(self.dataset, "path", None) or self.config.dataset.path
        if not dataset_path_value:
            raise ValueError("Dataset path not specified and could not be resolved from sequence")
        self.dataset_path = Path(dataset_path_value).resolve()
        self.active_camera_roles = self.dataset.get_active_camera_roles()
        self.sensor_mode = self._infer_sensor_mode()

        base_dir = Path(self.config.output.base_dir).resolve()
        self.results_dir = base_dir / self.experiment_name
        self.slam_results_dir = self.results_dir / "slam_results" / self.algorithm.name
        self.trajectories_dir = self.slam_results_dir / "trajectories"
        self.metrics_dir = self.slam_results_dir / "metrics"

        # Paper mode: output plots to separate directory
        if self.paper_mode:
            self.paper_output_dir = self.results_dir / "slam_results_for_paper" / self.algorithm.name
            self.paper_plots_dir = self.paper_output_dir / "trajectory_plots"
        else:
            self.paper_output_dir = None
            self.paper_plots_dir = None

        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Original dataset: {self.dataset_path}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Trajectories: {self.trajectories_dir}")
        logger.info(f"Metrics: {self.metrics_dir}")
        logger.info(f"Active camera roles: {self.active_camera_roles}")
        logger.info(f"Sensor mode: {self.sensor_mode.value}")
        if self.paper_mode:
            logger.info(f"Paper plots output: {self.paper_output_dir}")
        if self.num_runs > 1:
            logger.info(f"Multi-run SLAM: will execute {self.num_runs} runs")

    def run(self) -> Dict[str, Path]:
        """Run the full evaluation pipeline.

        Returns:
            Dictionary mapping trajectory names to output paths.

        Raises:
            FileNotFoundError: If perturbed data doesn't exist
            RuntimeError: If SLAM execution fails
        """
        logger.info("=" * 60)
        logger.info("SLAM EVALUATION PIPELINE")
        logger.info("=" * 60)

        all_trajectories = {}

        if not self.skip_slam:
            logger.info(f"\n[BASELINE] Running SLAM on ORIGINAL dataset {self.num_runs} time(s)...")
            for run_id in range(self.num_runs):
                logger.info(f"\n  Baseline Run {run_id + 1}/{self.num_runs}")
                baseline_traj = self._run_baseline_slam(run_id)
                all_trajectories[f"baseline_run_{run_id}"] = baseline_traj
        else:
            logger.info("\n[BASELINE] Loading existing baseline trajectories...")
            for run_id in range(self.num_runs):
                baseline_traj = self.trajectories_dir / f"run_{run_id}" / "baseline.txt"

                if baseline_traj.exists():
                    all_trajectories[f"baseline_run_{run_id}"] = baseline_traj
                    logger.info(f"  Baseline run {run_id}: {baseline_traj}")
                else:
                    logger.warning(f"  No baseline trajectory found for run {run_id}")

        for run_id in range(self.num_runs):
            if self.num_runs > 1:
                logger.info(f"\n{'='*60}")
                logger.info(f"Perturbed Run {run_id + 1}/{self.num_runs}")
                logger.info(f"{'='*60}")

            run_trajectories = self._run_perturbed_slam(run_id)

            for name, path in run_trajectories.items():
                key = f"{name}_run_{run_id}"
                all_trajectories[key] = path

        if self.compute_metrics:
            self._compute_metrics(all_trajectories)

        self._print_summary(all_trajectories)

        return all_trajectories

    def _run_baseline_slam(self, run_id: int) -> Path:
        """Run SLAM on the original (unperturbed) dataset.

        Args:
            run_id: Run number for this baseline execution

        Returns:
            Path to baseline trajectory file

        Raises:
            RuntimeError: If baseline SLAM execution fails
        """
        import shutil

        self.algorithm.cleanup()

        temp_output_dir = create_temp_dir(prefix="slam_baseline_")

        max_frames = self.config.dataset.max_frames
        temp_dataset_dir = None
        dataset_to_use = self.dataset_path

        if max_frames:
            temp_dataset_dir = self.dataset.create_truncated_copy(max_frames)
            dataset_to_use = temp_dataset_dir
            logger.info(f"  Using truncated baseline dataset ({max_frames} frames): {temp_dataset_dir}")

        try:
            baseline_request = self._create_run_request(dataset_to_use, temp_output_dir)
            baseline_result = self.algorithm.run(baseline_request)
            baseline_traj = baseline_result.trajectory_path if baseline_result.success else None
            if not baseline_result.success and baseline_result.message:
                logger.error(f"  {baseline_result.message}")

            if baseline_traj and baseline_traj.exists():
                run_traj_dir = self.trajectories_dir / f"run_{run_id}"
                run_traj_dir.mkdir(parents=True, exist_ok=True)
                final_path = run_traj_dir / "baseline.txt"
                shutil.copy2(baseline_traj, final_path)

                logger.info(f"  Baseline trajectory: {final_path}")
                self.algorithm.cleanup()
                return final_path
            else:
                logger.error("  Failed to generate baseline trajectory")
                raise RuntimeError("Baseline SLAM execution failed")
        finally:
            # Clean up temp directories
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            if temp_dataset_dir:
                shutil.rmtree(temp_dataset_dir, ignore_errors=True)

    def _run_perturbed_slam(self, run_id: int) -> Dict[str, Path]:
        """Run SLAM on perturbed datasets for a single run.

        Args:
            run_id: Run identifier (0, 1, 2, ...)

        Returns:
            Dictionary mapping perturbed module names to trajectory file paths
            (does NOT include baseline)
        """
        import shutil

        # New structure: trajectories/run_X/{module}.txt
        run_traj_dir = self.trajectories_dir / f"run_{run_id}"

        trajectories: Dict[str, Path] = {}

        if self.skip_slam:
            logger.info("\n[SKIP-SLAM] Loading existing perturbed trajectories...")
            trajectories = self._discover_existing_trajectories(run_traj_dir)
            if trajectories:
                logger.info(f"  Found {len(trajectories)} existing trajectory file(s):")
                for name, path in trajectories.items():
                    logger.info(f"    - {name}: {path}")
            else:
                logger.warning(f"  No perturbed trajectories found for run_{run_id}")
                return {}
        else:
            logger.info("\n[1/2] Discovering perturbed datasets...")
            perturbed_modules = self._discover_perturbed_datasets(self.results_dir)

            if not perturbed_modules:
                logger.warning(
                    f"No perturbed data found in {self.results_dir}. "
                    f"Run 'slamadverseriallab run {self.config_path}' to generate perturbed data. "
                    f"Continuing with baseline only."
                )

            logger.info(f"  Found {len(perturbed_modules)} perturbed module(s):")
            for module_name in perturbed_modules:
                logger.info(f"    - {module_name}")

            run_traj_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Trajectories will be saved to: {run_traj_dir}")

            logger.info(f"\n[PERTURBED] Running SLAM on perturbed datasets...")

            for i, module_name in enumerate(perturbed_modules, start=1):
                logger.info(f"\n  [{i}/{len(perturbed_modules)}] Processing module: {module_name}")

                perturbed_dataset_path = self.results_dir / "images" / module_name

                temp_output_dir = create_temp_dir(prefix=f"slam_{module_name}_")

                perturbed_request = self._create_run_request(perturbed_dataset_path, temp_output_dir)
                perturbed_result = self.algorithm.run(perturbed_request)
                perturbed_traj = perturbed_result.trajectory_path if perturbed_result.success else None
                if not perturbed_result.success and perturbed_result.message:
                    logger.warning(f"    {perturbed_result.message}")

                if perturbed_traj and perturbed_traj.exists():
                    final_path = run_traj_dir / f"{module_name}.txt"
                    shutil.copy2(perturbed_traj, final_path)
                    shutil.rmtree(temp_output_dir, ignore_errors=True)
                    trajectories[module_name] = final_path
                    logger.info(f"    Trajectory saved: {final_path}")
                else:
                    shutil.rmtree(temp_output_dir, ignore_errors=True)
                    logger.warning(f"    Failed to generate trajectory for {module_name}")

                # Clean up after each run
                self.algorithm.cleanup()

        return trajectories

    def _compute_metrics(self, all_trajectories: Dict[str, Path]) -> Dict[str, Any]:
        """Compute metrics comparing baseline to perturbed trajectories.

        Args:
            all_trajectories: Dictionary with 'baseline' and perturbed trajectory paths

        Returns:
            Dictionary of metrics results per perturbed trajectory

        Raises:
            RuntimeError: If metrics computation fails
        """
        logger.info("\n" + "=" * 60)
        logger.info("COMPUTING METRICS")
        logger.info("=" * 60)

        baseline_keys = [k for k in all_trajectories.keys() if k.startswith('baseline')]
        if not baseline_keys:
            logger.error("  Cannot compute metrics: baseline trajectory not available")
            raise RuntimeError("No baseline trajectory generated")

        perturbed_keys = [k for k in all_trajectories.keys() if not k.startswith('baseline')]
        if not perturbed_keys:
            logger.warning("  No perturbed trajectories available - skipping metrics computation")
            return {}

        try:
            gt_path = self._get_ground_truth_path()

            timestamps_path = self.dataset.get_timestamps_file_path()

            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            max_frames = self.config.dataset.max_frames
            evaluator = MetricsEvaluator(
                output_dir=self.metrics_dir,
                dataset_type=self.dataset_type,
                max_frames=max_frames,
                timestamps_path=timestamps_path
            )

            metrics_results = {}
            logger.info(f"\n  Using PAIRED comparison (baseline_run_X vs perturbed_run_X)")

            for traj_name, perturbed_path in all_trajectories.items():
                if traj_name.startswith('baseline'):
                    continue

                run_id = int(traj_name.split('_run_')[-1])
                baseline_key = f"baseline_run_{run_id}"

                if baseline_key not in all_trajectories:
                    logger.warning(f"  Skipping {traj_name}: no matching baseline {baseline_key}")
                    continue

                baseline_path = all_trajectories[baseline_key]

                logger.info(f"\n  Computing metrics for: {traj_name} vs {baseline_key}")
                result = evaluator.compare_trajectories(
                    baseline_path=baseline_path,
                    perturbed_path=perturbed_path,
                    ground_truth_path=gt_path,
                    baseline_name=baseline_key,
                    perturbed_name=traj_name
                )
                metrics_results[traj_name] = result

            logger.info(f"\n  Metrics saved to: {self.metrics_dir}")

            # Aggregate metrics across runs
            logger.info("\nAggregating metrics across runs...")
            aggregated_summary = self._aggregate_metrics(metrics_results, all_trajectories)

            summary_path = self.metrics_dir / "summary.json"
            with open(summary_path, 'w') as f:
                import json
                json.dump(aggregated_summary, f, indent=2)
            logger.info(f"Aggregated metrics summary saved to: {summary_path}")

            # Generate aggregated plots
            try:
                logger.info("\nGenerating aggregated plots...")
                self._generate_aggregated_plots(aggregated_summary, self.metrics_dir)
            except Exception as e:
                logger.warning(f"Failed to generate aggregated plots: {e}")

            # Generate multi-trajectory comparison plots (all modules together)
            try:
                logger.info("\nGenerating multi-trajectory comparison plots...")
                self._generate_multi_trajectory_comparison(all_trajectories, self.metrics_dir)
            except Exception as e:
                logger.warning(f"Failed to generate multi-trajectory comparison plots: {e}")

            # Generate trajectory visualization plots using evo_traj
            try:
                logger.info("\nGenerating trajectory visualization plots (evo_traj)...")
                self._generate_trajectory_plots(all_trajectories, gt_path, evaluator)
            except Exception as e:
                logger.warning(f"Failed to generate trajectory plots: {e}")

            # Print metrics summary
            logger.info("\n" + "=" * 60)
            logger.info("METRICS SUMMARY")
            logger.info("=" * 60)
            self._print_aggregated_metrics(aggregated_summary)

            return metrics_results

        except FileNotFoundError as e:
            logger.error(f"  Metrics computation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"  Metrics computation failed: {e}")
            raise RuntimeError(f"Failed to compute metrics: {e}")

    def _aggregate_metrics(self, metrics_results: Dict[str, Any], all_trajectories: Dict[str, Path]) -> Dict[str, Any]:
        """Aggregate metrics across multiple runs per module.

        Args:
            metrics_results: Individual metrics for each trajectory
            all_trajectories: All trajectory paths

        Returns:
            Aggregated metrics with mean, std, min, max per module
        """
        import numpy as np

        module_groups = {}
        for traj_name in metrics_results.keys():
            if '_run_' in traj_name:
                module_name = traj_name.rsplit('_run_', 1)[0]
            else:
                module_name = traj_name

            if module_name not in module_groups:
                module_groups[module_name] = []
            module_groups[module_name].append(traj_name)

        baseline_metrics_by_run = {}

        for traj_name, result in metrics_results.items():
            if result.get('baseline'):
                run_id = int(traj_name.split('_run_')[-1])

                if run_id not in baseline_metrics_by_run:
                    baseline_metrics_by_run[run_id] = result['baseline']

        baseline_ape_rmses = []
        baseline_rpe_rmses = []
        baseline_tracking = []

        for run_id in sorted(baseline_metrics_by_run.keys()):
            baseline = baseline_metrics_by_run[run_id]
            if baseline.get('ape') and baseline['ape'].get('rmse'):
                baseline_ape_rmses.append(baseline['ape']['rmse'])
            if baseline.get('rpe') and baseline['rpe'].get('rmse'):
                baseline_rpe_rmses.append(baseline['rpe']['rmse'])
            baseline_tracking.append(baseline.get('tracking_completeness', 100.0))

        baseline_metrics = {
            'num_runs': len(baseline_ape_rmses),
            'ape_rmse': self._compute_stats(baseline_ape_rmses) if baseline_ape_rmses else None,
            'rpe_rmse': self._compute_stats(baseline_rpe_rmses) if baseline_rpe_rmses else None,
            'tracking_completeness': self._compute_stats(baseline_tracking) if baseline_tracking else None,
        }

        config_module_order = [p.name for p in self.config.perturbations if p.enabled]

        # Aggregate metrics per module
        aggregated = {
            'baseline': baseline_metrics,
            'perturbed_modules': {},
            'run_count': self.num_runs,
            'module_order': config_module_order  # Preserve config order for plotting
        }

        ordered_modules = [m for m in config_module_order if m in module_groups]
        for m in sorted(module_groups.keys()):
            if m not in ordered_modules:
                ordered_modules.append(m)

        for module_name in ordered_modules:
            traj_names = module_groups[module_name]
            ape_rmses = []
            rpe_rmses = []
            tracking_completeness = []
            tracking_losses = []

            for traj_name in traj_names:
                result = metrics_results[traj_name]
                perturbed = result['perturbed']
                degradation = result['degradation']

                if perturbed.get('ape') and perturbed['ape'].get('rmse') is not None:
                    ape_rmses.append(perturbed['ape']['rmse'])
                if perturbed.get('rpe') and perturbed['rpe'].get('rmse') is not None:
                    rpe_rmses.append(perturbed['rpe']['rmse'])

                tracking_completeness.append(degradation['perturbed_tracking_completeness'])
                tracking_losses.append(degradation['tracking_loss'])

            module_stats = {
                'num_runs': len(traj_names),
                'trajectory_names': traj_names,
                'ape_rmse': self._compute_stats(ape_rmses) if ape_rmses else None,
                'rpe_rmse': self._compute_stats(rpe_rmses) if rpe_rmses else None,
                'tracking_completeness': self._compute_stats(tracking_completeness),
                'tracking_loss': self._compute_stats(tracking_losses),
            }

            aggregated['perturbed_modules'][module_name] = module_stats

        return aggregated

    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute statistical measures for a list of values.

        Args:
            values: List of numeric values

        Returns:
            Dictionary with mean, std, min, max, median
        """
        import numpy as np

        if not values:
            return None

        arr = np.array(values)

        if len(arr) > 1:
            std_val = float(np.std(arr, ddof=1))
        else:
            std_val = float('nan')  # Standard deviation undefined for single value

        return {
            'mean': float(np.mean(arr)),
            'std': std_val,
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'median': float(np.median(arr)),
            'values': [float(v) for v in values]  # Keep individual values for reference
        }

    def _generate_aggregated_plots(self, aggregated: Dict[str, Any], metrics_dir: Path) -> None:
        """Generate plot showing baseline as reference line and perturbed modules as points with error bars.

        Args:
            aggregated: Aggregated metrics dictionary
            metrics_dir: Directory to save plots
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np

        # Extract baseline metrics
        baseline = aggregated.get('baseline', {})
        baseline_ape = baseline.get('ape_rmse', {}).get('mean', 0)
        baseline_rpe = baseline.get('rpe_rmse', {}).get('mean', 0)
        baseline_ape_std = baseline.get('ape_rmse', {}).get('std', 0)
        baseline_rpe_std = baseline.get('rpe_rmse', {}).get('std', 0)

        # Extract data for perturbed modules only (baseline will be a line)
        modules = []
        ape_means = []
        ape_stds = []
        rpe_means = []
        rpe_stds = []

        module_order = aggregated.get('module_order', sorted(aggregated['perturbed_modules'].keys()))
        module_order = [m for m in module_order if m in aggregated['perturbed_modules']]

        for module_name in module_order:
            stats = aggregated['perturbed_modules'][module_name]
            modules.append(module_name)

            if stats['ape_rmse']:
                ape_means.append(stats['ape_rmse']['mean'])
                ape_stds.append(stats['ape_rmse']['std'])
            else:
                ape_means.append(0)
                ape_stds.append(0)

            if stats['rpe_rmse']:
                rpe_means.append(stats['rpe_rmse']['mean'])
                rpe_stds.append(stats['rpe_rmse']['std'])
            else:
                rpe_means.append(0)
                rpe_stds.append(0)

        if not modules:
            logger.warning("No perturbed modules to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')

        x = np.arange(len(modules))

        ax1.axhline(y=baseline_ape, color='green', linestyle='--', linewidth=2, zorder=1)
        ax1.fill_between([-0.5, len(modules)-0.5],
                        baseline_ape - baseline_ape_std,
                        baseline_ape + baseline_ape_std,
                        color='green', alpha=0.2,
                        label=f'Baseline ({baseline_ape:.4f} ± {baseline_ape_std:.4f} m)', zorder=0)
        ax1.errorbar(x, ape_means, yerr=ape_stds, fmt='o', markersize=8,
                     capsize=5, capthick=2, color='steelblue', ecolor='steelblue',
                     label='Perturbed (mean ± std)', zorder=2)

        ax1.set_xlabel('Perturbation Module', fontsize=12)
        ax1.set_ylabel('APE RMSE (m)', fontsize=12)
        ax1.set_title(f'APE RMSE ({aggregated["run_count"]} runs per module)',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modules, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3, zorder=0)
        ax1.legend(loc='upper left')

        # Plot RPE RMSE - baseline as band (mean ± std), perturbed as points with error bars
        ax2.axhline(y=baseline_rpe, color='green', linestyle='--', linewidth=2, zorder=1)
        ax2.fill_between([-0.5, len(modules)-0.5],
                        baseline_rpe - baseline_rpe_std,
                        baseline_rpe + baseline_rpe_std,
                        color='green', alpha=0.2,
                        label=f'Baseline ({baseline_rpe:.4f} ± {baseline_rpe_std:.4f} m)', zorder=0)
        ax2.errorbar(x, rpe_means, yerr=rpe_stds, fmt='o', markersize=8,
                     capsize=5, capthick=2, color='coral', ecolor='coral',
                     label='Perturbed (mean ± std)', zorder=2)

        ax2.set_xlabel('Perturbation Module', fontsize=12)
        ax2.set_ylabel('RPE RMSE (m)', fontsize=12)
        ax2.set_title(f'RPE RMSE ({aggregated["run_count"]} runs per module)',
                      fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(modules, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3, zorder=0)
        ax2.legend(loc='upper left')

        plot_path = metrics_dir / "aggregated_metrics.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Aggregated plot saved to: {plot_path}")

    def _print_aggregated_metrics(self, aggregated: Dict[str, Any]) -> None:
        """Print aggregated metrics summary.

        Args:
            aggregated: Aggregated metrics dictionary
        """
        logger.info(f"\nBaseline Performance:")
        if aggregated['baseline']:
            baseline = aggregated['baseline']
            logger.info(f"  (Aggregated across {baseline['num_runs']} runs)")
            if baseline.get('ape_rmse'):
                ape = baseline['ape_rmse']
                logger.info(f"  APE RMSE: {ape['mean']:.4f} m (±{ape['std']:.4f} m)")
                logger.info(f"    Range: [{ape['min']:.4f} m, {ape['max']:.4f} m]")
            if baseline.get('rpe_rmse'):
                rpe = baseline['rpe_rmse']
                logger.info(f"  RPE RMSE: {rpe['mean']:.4f} m (±{rpe['std']:.4f} m)")
                logger.info(f"    Range: [{rpe['min']:.4f} m, {rpe['max']:.4f} m]")
            if baseline.get('tracking_completeness'):
                tc = baseline['tracking_completeness']
                logger.info(f"  Tracking: {tc['mean']:.1f}% (±{tc['std']:.1f}%)")

        logger.info(f"\nPerturbed Modules (aggregated across {aggregated['run_count']} run(s)):")
        module_order = aggregated.get('module_order', sorted(aggregated['perturbed_modules'].keys()))
        module_order = [m for m in module_order if m in aggregated['perturbed_modules']]

        for module_name in module_order:
            stats = aggregated['perturbed_modules'][module_name]
            logger.info(f"\n  {module_name}:")

            # APE RMSE
            if stats['ape_rmse']:
                ape = stats['ape_rmse']
                logger.info(f"    APE RMSE: {ape['mean']:.4f} m (±{ape['std']:.4f} m)")
                logger.info(f"      Range: [{ape['min']:.4f} m, {ape['max']:.4f} m]")

            # RPE RMSE
            if stats['rpe_rmse']:
                rpe = stats['rpe_rmse']
                logger.info(f"    RPE RMSE: {rpe['mean']:.4f} m (±{rpe['std']:.4f} m)")
                logger.info(f"      Range: [{rpe['min']:.4f} m, {rpe['max']:.4f} m]")

            # Tracking
            if stats['tracking_completeness']:
                track = stats['tracking_completeness']
                loss = stats['tracking_loss']
                logger.info(f"    Tracking: {track['mean']:.1f}% (±{track['std']:.1f}%)")
                logger.info(f"      Frames lost: {loss['mean']:.1f} (±{loss['std']:.1f})")

    def _print_summary(self, all_trajectories: Dict[str, Path]) -> None:
        """Print evaluation summary.

        Args:
            all_trajectories: Dictionary of all trajectory paths
        """
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Results directory: {self.slam_results_dir}")
        logger.info(f"Total trajectories: {len(all_trajectories)}")

        baseline_keys = [k for k in all_trajectories.keys() if k.startswith('baseline_run_')]
        has_baseline = len(baseline_keys) > 0
        perturbed_count = len(all_trajectories) - len(baseline_keys)

        if has_baseline:
            logger.info(f"  Baseline: {len(baseline_keys)}")
            logger.info(f"  Perturbed: {perturbed_count}")

        logger.info("\nTrajectory files:")
        for name in sorted(all_trajectories.keys()):
            path = all_trajectories[name]
            logger.info(f"  {name}: {path}")

        if not self.compute_metrics:
            logger.info("\nNext steps:")
            logger.info("  Run with --mode full (default) to compute APE/RPE metrics automatically")

    def _discover_perturbed_datasets(self, results_dir: Path) -> List[str]:
        """Discover perturbed datasets from previous 'run' command.

        Args:
            results_dir: Path to results directory (e.g., run_0/ or experiment root)

        Returns:
            List of module names that have perturbed data, ordered by config definition
        """
        images_dir = results_dir / "images"

        if not images_dir.exists():
            return []

        config_order = [p.name for p in self.config.perturbations if p.enabled]

        image_dir_name = "image_2"

        available_modules = set()
        for module_dir in images_dir.iterdir():
            if not module_dir.is_dir():
                continue
            image_dir = module_dir / image_dir_name
            if image_dir.exists() and any(image_dir.glob("*.png")):
                available_modules.add(module_dir.name)

        # Return in config order, filtering to only available modules
        ordered_modules = [name for name in config_order if name in available_modules]

        # Preserve any discovered outputs not present in the current config.
        for name in sorted(available_modules - set(ordered_modules)):
            ordered_modules.append(name)

        return ordered_modules

    def _infer_sensor_mode(self) -> SensorMode:
        """Infer sensor mode from dataset contract and algorithm capabilities."""
        dataset_type = self.dataset_type.lower()
        supported = [m.lower() for m in self.algorithm.supported_datasets.get(dataset_type, [])]

        if "left" not in self.active_camera_roles:
            raise ValueError(
                f"Dataset '{self.dataset_type}' must expose an active 'left' camera role. "
                f"Got: {self.active_camera_roles}"
            )

        if "right" in self.active_camera_roles:
            return SensorMode.STEREO

        if dataset_type == "tum" and "rgbd" in supported:
            return SensorMode.RGBD

        return SensorMode.MONO

    def _resolve_algorithm_timestamps(self) -> Dict[int, Any]:
        """Resolve and validate frame-indexed timestamps for SLAM wrappers."""
        timestamps_by_frame = self.dataset.get_algorithm_timestamps()
        if not isinstance(timestamps_by_frame, dict) or not timestamps_by_frame:
            raise ValueError(
                "Dataset must provide non-empty timestamps via get_algorithm_timestamps()."
            )

        normalized: Dict[int, Any] = {}
        for frame_idx, timestamp in timestamps_by_frame.items():
            if not isinstance(frame_idx, int):
                raise ValueError(
                    f"Invalid timestamp mapping key {frame_idx!r}; expected int frame indices."
                )
            if frame_idx < 0:
                raise ValueError(
                    f"Invalid timestamp mapping key {frame_idx}; frame indices must be >= 0."
                )
            if not isinstance(timestamp, (int, float)):
                raise ValueError(
                    f"Invalid timestamp value for frame {frame_idx}: {timestamp!r}"
                )
            normalized[frame_idx] = timestamp

        expected_count = len(self.dataset)
        expected_indices = set(range(expected_count))
        missing = sorted(expected_indices - set(normalized.keys()))
        extra = sorted(set(normalized.keys()) - expected_indices)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing={missing[:5]}{'...' if len(missing) > 5 else ''}")
            if extra:
                details.append(f"extra={extra[:5]}{'...' if len(extra) > 5 else ''}")
            raise ValueError(
                "Dataset timestamp mapping must exactly match loaded frame indices "
                f"0..{expected_count - 1}: " + ", ".join(details)
            )

        previous = None
        for idx in range(expected_count):
            current = normalized[idx]
            if previous is not None and float(current) <= float(previous):
                raise ValueError(
                    "Dataset timestamps must be strictly increasing by frame index; "
                    f"frame {idx - 1}={previous}, frame {idx}={current}"
                )
            previous = current

        return normalized

    def _create_run_request(self, dataset_path: Path, output_dir: Path) -> SLAMRunRequest:
        """Create a structured request for SLAM algorithm execution."""
        sequence_name = (self.config.dataset.sequence or "").strip()
        if not sequence_name:
            raise ValueError(
                "dataset.sequence is required for SLAM evaluation runs. "
                "Set dataset.sequence explicitly in the evaluation config."
            )

        dataset_path = Path(dataset_path).resolve()
        camera_dirs = self.dataset.resolve_camera_directories(dataset_path)
        camera_paths = {
            role: str((dataset_path / directory_name).resolve())
            for role, directory_name in camera_dirs.items()
        }
        timestamps_by_frame = self._resolve_algorithm_timestamps()

        return SLAMRunRequest(
            dataset_path=dataset_path,
            slam_config=self.slam_config,
            output_dir=output_dir,
            dataset_type=self.dataset_type,
            sensor_mode=self.sensor_mode,
            sequence_name=sequence_name,
            extras={
                "camera_dirs": camera_dirs,
                "camera_paths": camera_paths,
                "timestamps_by_frame": timestamps_by_frame,
            },
        )

    def _get_ground_truth_path(self) -> Path:
        """Get ground truth trajectory path using the dataset's method.

        Returns:
            Path to ground truth trajectory file

        Raises:
            FileNotFoundError: If ground truth file doesn't exist
        """
        gt_path = self.dataset.get_ground_truth_path()

        if gt_path is None:
            raise FileNotFoundError(
                f"Ground truth file not found for {self.dataset_type} dataset at {self.dataset_path}.\n"
                f"Please ensure ground truth data is available."
            )

        logger.info(f"Ground truth path: {gt_path}")
        return gt_path

    def _discover_existing_trajectories(self, run_traj_dir: Path) -> Dict[str, Path]:
        """Discover existing trajectory files from previous SLAM runs.

        Args:
            run_traj_dir: Directory containing trajectories for a run (trajectories/run_X/)

        Returns:
            Dictionary mapping module names to trajectory file paths

        Raises:
            FileNotFoundError: If directory doesn't exist or has no perturbed trajectories
        """
        trajectories: Dict[str, Path] = {}

        if not run_traj_dir.exists():
            raise FileNotFoundError(
                f"Trajectory directory not found: {run_traj_dir}\n"
                f"Run with --mode full or --mode slam-only first to generate trajectories."
            )

        # New structure: trajectories/run_X/{module}.txt (flat .txt files)
        for traj_file in run_traj_dir.glob("*.txt"):
            # Skip baseline (handled separately)
            if traj_file.stem == "baseline":
                continue

            module_name = traj_file.stem
            trajectories[module_name] = traj_file
            logger.debug(f"  Found trajectory: {module_name} -> {traj_file}")

        if not trajectories:
            raise FileNotFoundError(
                f"No perturbed trajectory files found in {run_traj_dir}\n"
                f"Run with --mode full or --mode slam-only first to generate trajectories."
            )

        return trajectories

    def _generate_trajectory_plots(
        self,
        all_trajectories: Dict[str, Path],
        ground_truth_path: Path,
        evaluator
    ) -> None:
        """Generate trajectory visualization plots using evo_traj.

        Groups trajectories by run and creates comparison plots showing all modules together.
        Plots are saved in metrics/comparison/run_X/

        Args:
            all_trajectories: Dictionary of all trajectory paths (baseline + perturbed)
            ground_truth_path: Path to ground truth trajectory
            evaluator: MetricsEvaluator instance
        """
        from collections import defaultdict

        baseline_keys = [k for k in all_trajectories.keys() if k.startswith('baseline')]
        if not baseline_keys:
            logger.warning("  No baseline trajectory found for plotting")
            return

        runs_grouped = defaultdict(list)
        baseline_by_run = {}

        for traj_name in all_trajectories.keys():
            if traj_name.startswith('baseline'):
                if '_run_' in traj_name:
                    run_id = traj_name.split('_run_')[-1]
                    run_key = f"run_{run_id}"
                    baseline_by_run[run_key] = all_trajectories[traj_name]
                else:
                    baseline_by_run['run_0'] = all_trajectories[traj_name]
                continue

            # Extract run_id from trajectory name
            if '_run_' in traj_name:
                run_id = traj_name.split('_run_')[-1]
                run_key = f"run_{run_id}"
                runs_grouped[run_key].append(traj_name)
            else:
                # Single run case
                runs_grouped['run_0'].append(traj_name)

        if not runs_grouped:
            logger.warning("  No perturbed trajectories found for plotting")
            return

        if self.paper_mode and self.paper_plots_dir:
            metrics_dir = self.paper_plots_dir
        else:
            metrics_dir = self.slam_results_dir / "trajectory_plots"

        for run_key in sorted(runs_grouped.keys()):
            run_trajectories = runs_grouped[run_key]

            baseline_path = baseline_by_run.get(run_key)
            if not baseline_path:
                logger.warning(f"  No baseline trajectory found for {run_key}, skipping plot")
                continue

            logger.info(f"  Generating trajectory plots for {run_key} ({len(run_trajectories)} modules + baseline)")

            run_comparison_dir = metrics_dir / "comparison" / run_key
            run_comparison_dir.mkdir(parents=True, exist_ok=True)

            trajectories_to_plot = {
                'baseline': baseline_path
            }

            for traj_name in sorted(run_trajectories):
                if '_run_' in traj_name:
                    clean_name = traj_name.rsplit('_run_', 1)[0]
                else:
                    clean_name = traj_name
                trajectories_to_plot[clean_name] = all_trajectories[traj_name]

            # Generate plots for this run
            self._generate_module_trajectory_plot(
                trajectories=trajectories_to_plot,
                ground_truth_path=ground_truth_path,
                output_dir=run_comparison_dir,
                plot_name="trajectory_comparison",
                plot_modes=['xy', 'xz', 'xyz'],
                max_frames=self.config.dataset.max_frames,
                paper_mode=self.paper_mode
            )

    def _generate_module_trajectory_plot(
        self,
        trajectories: Dict[str, Path],
        ground_truth_path: Path,
        output_dir: Path,
        plot_name: str = "trajectory_comparison",
        plot_modes: list = None,
        max_frames: int = None,
        paper_mode: bool = False
    ) -> None:
        """Generate trajectory comparison plot showing multiple trajectories.

        Creates one plot per view mode (xy, xz, xyz) showing all trajectories overlaid
        with ground truth reference.

        Args:
            trajectories: Dictionary with all trajectory paths to compare (baseline + modules)
            ground_truth_path: Path to ground truth trajectory
            output_dir: Directory to save plots (e.g., metrics/comparison/run_0/)
            plot_name: Name prefix for output plots
            plot_modes: List of plot modes (default: ['xy', 'xz', 'xyz'])
            max_frames: If set, truncate ground truth to this many poses for plotting
            paper_mode: If True, use severity-based colors and hide legend/title
        """
        if plot_modes is None:
            plot_modes = ['xy', 'xz', 'xyz']

        first_traj_path = next(iter(trajectories.values()))
        try:
            traj_format = detect_trajectory_format(first_traj_path)
        except RuntimeError as e:
            logger.warning(f"    {e}")
            return

        if traj_format == "tum":
            logger.info(f"    Detected TUM format (mono SLAM - KeyFrameTrajectory)")
        else:
            logger.info(f"    Detected KITTI format (stereo SLAM - CameraTrajectory)")

        converted_gt_path = None
        gt_to_use = ground_truth_path

        if traj_format == "tum" and self.dataset_type == "kitti":
            logger.info(f"    KITTI mono mode detected - converting ground truth to TUM format")
            sequence_num = ground_truth_path.stem
            dataset_root = ground_truth_path.parent.parent
            timestamps_path = dataset_root / "sequences" / sequence_num / "times.txt"

            if timestamps_path.exists():
                from ..metrics.trajectory import convert_kitti_to_tum, trajectory_uses_frame_indices

                use_frame_indices = trajectory_uses_frame_indices(first_traj_path)
                if use_frame_indices:
                    logger.info(f"    Detected frame indices in trajectory - using matching timestamps for GT")

                converted_gt_path = ground_truth_path.parent / f".{ground_truth_path.stem}_tum_plot.txt"
                convert_kitti_to_tum(
                    ground_truth_path, timestamps_path, converted_gt_path,
                    use_frame_indices=use_frame_indices
                )
                gt_to_use = converted_gt_path
                logger.debug(f"    Ground truth converted: {converted_gt_path}")
            else:
                logger.warning(f"    Timestamps not found at {timestamps_path}, skipping ground truth in plot")
                gt_to_use = None
        elif self.dataset_type == "euroc":
            logger.info(f"    EuRoC mode detected - converting ground truth to TUM format")
            from ..metrics.trajectory import convert_euroc_to_tum
            converted_gt_path = ground_truth_path.parent / f".{ground_truth_path.stem}_tum_plot.txt"
            convert_euroc_to_tum(ground_truth_path, converted_gt_path)
            gt_to_use = converted_gt_path
            logger.debug(f"    Ground truth converted: {converted_gt_path}")

        if max_frames and gt_to_use:
            logger.info(f"    Using full ground truth for plotting (alignment requires spatial coverage)")

        try:
            for plot_mode in plot_modes:
                plot_path = output_dir / f"{plot_name}_{plot_mode}"

                success = plot_trajectories(
                    trajectories=trajectories,
                    output_path=plot_path,
                    reference_path=gt_to_use,
                    format_type=traj_format,
                    plot_mode=plot_mode,
                    paper_mode=paper_mode
                )

                if success:
                    logger.info(f"    Saved {plot_mode} plot: {plot_path}.png")
                else:
                    logger.warning(f"    Failed to generate {plot_mode} plot")

        finally:
            # Clean up temporary files
            if converted_gt_path and converted_gt_path.exists():
                converted_gt_path.unlink()

    def _generate_multi_trajectory_comparison(self, all_trajectories: Dict[str, Path], metrics_dir: Path) -> None:
        """Generate comparison plots grouped by run (baseline vs all modules per run).

        Args:
            all_trajectories: Dictionary of all trajectory paths (baseline + perturbed)
            metrics_dir: Directory where metrics are saved
        """
        import json
        from collections import defaultdict

        comparison_dir = metrics_dir / "comparison"

        runs_grouped = defaultdict(list)

        for traj_name in all_trajectories.keys():
            if traj_name == 'baseline':
                # Baseline goes into all runs
                continue

            if '_run_' in traj_name:
                run_id = traj_name.split('_run_')[-1]
                run_key = f"run_{run_id}"
                runs_grouped[run_key].append(traj_name)
            else:
                # Single run case (no _run_ suffix)
                runs_grouped['run_0'].append(traj_name)

        # If no runs found, treat everything as one comparison
        if not runs_grouped:
            logger.warning("  No perturbed trajectories found for comparison")
            return

        # Generate one comparison plot per run
        for run_key in sorted(runs_grouped.keys()):
            run_trajectories = runs_grouped[run_key]

            run_comparison_dir = comparison_dir / run_key
            run_comparison_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"  Generating comparison plots for {run_key} ({len(run_trajectories)} modules)")

            ape_metrics = {}
            rpe_metrics = {}

            run_metrics_dir = metrics_dir / run_key

            baseline_ape_path = run_metrics_dir / "baseline" / "ape.json"
            baseline_rpe_path = run_metrics_dir / "baseline" / "rpe.json"

            if baseline_ape_path.exists():
                with open(baseline_ape_path) as f:
                    ape_metrics['baseline'] = json.load(f)
            if baseline_rpe_path.exists():
                with open(baseline_rpe_path) as f:
                    rpe_metrics['baseline'] = json.load(f)

            for traj_name in sorted(run_trajectories):
                if '_run_' in traj_name:
                    perturbation_name = traj_name.rsplit('_run_', 1)[0]
                else:
                    perturbation_name = traj_name

                ape_path = run_metrics_dir / perturbation_name / "ape.json"
                rpe_path = run_metrics_dir / perturbation_name / "rpe.json"

                if ape_path.exists():
                    with open(ape_path) as f:
                        ape_metrics[perturbation_name] = json.load(f)
                else:
                    logger.debug(f"    APE json not found for {traj_name}: {ape_path}")

                if rpe_path.exists():
                    with open(rpe_path) as f:
                        rpe_metrics[perturbation_name] = json.load(f)
                else:
                    logger.debug(f"    RPE json not found for {traj_name}: {rpe_path}")

            if len(ape_metrics) < 2:
                logger.warning(f"    Need at least 2 trajectories for {run_key}, found {len(ape_metrics)}")
                continue

            # Generate APE comparison plot for this run using Python API
            ape_comparison_path = run_comparison_dir / "ape_comparison"
            success = plot_metric_comparison(ape_metrics, ape_comparison_path, metric_name="APE")
            if success:
                logger.info(f"    APE comparison saved to {ape_comparison_path}.png")
            else:
                logger.warning(f"    Failed to generate APE comparison for {run_key}")

            # Generate RPE comparison plot for this run
            if len(rpe_metrics) >= 2:
                rpe_comparison_path = run_comparison_dir / "rpe_comparison"
                success = plot_metric_comparison(rpe_metrics, rpe_comparison_path, metric_name="RPE")
                if success:
                    logger.info(f"    RPE comparison saved to {rpe_comparison_path}.png")
                else:
                    logger.warning(f"    Failed to generate RPE comparison for {run_key}")
