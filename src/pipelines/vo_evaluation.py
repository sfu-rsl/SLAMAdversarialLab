"""Visual odometry evaluation pipeline using PySLAM."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config.parser import load_config
from ..algorithms.pyslam_runner import PySLAMRunner

logger = logging.getLogger(__name__)


def infer_camera_settings(dataset_type: str, sequence_name: str, sensor_type: str) -> str:
    """Infer PySLAM camera settings file from dataset type and sequence.

    Maps dataset type and sequence name to the appropriate PySLAM settings file
    in deps/slam-frameworks/pyslam/settings/.

    Args:
        dataset_type: Dataset type ('kitti', 'tum', 'euroc')
        sequence_name: Sequence name (e.g., '04' for KITTI, 'rgbd_dataset_freiburg1_desk' for TUM)
        sensor_type: Sensor type ('mono', 'stereo', 'rgbd')

    Returns:
        PySLAM settings filename (e.g., 'KITTI04-12.yaml', 'TUM1.yaml')

    Raises:
        ValueError: If dataset type is not supported
    """
    dataset_type = dataset_type.lower()

    if dataset_type == "kitti":
        try:
            seq_num = int(sequence_name)
        except ValueError:
            logger.warning(f"Could not parse KITTI sequence '{sequence_name}', using KITTI04-12.yaml")
            return "KITTI04-12.yaml"

        if seq_num <= 2:
            return "KITTI00-02.yaml"
        elif seq_num == 3:
            return "KITTI03.yaml"
        else:
            return "KITTI04-12.yaml"

    elif dataset_type == "tum":
        sequence_lower = sequence_name.lower()
        if "freiburg1" in sequence_lower:
            return "TUM1.yaml"
        elif "freiburg2" in sequence_lower:
            return "TUM2.yaml"
        elif "freiburg3" in sequence_lower:
            return "TUM3.yaml"
        else:
            logger.warning(f"Could not determine TUM camera from '{sequence_name}', using TUM1.yaml")
            return "TUM1.yaml"

    elif dataset_type == "euroc":
        if sensor_type == "stereo":
            return "EuRoC_stereo.yaml"
        else:
            return "EuRoC_mono.yaml"

    else:
        raise ValueError(
            f"Unsupported dataset type '{dataset_type}' for camera settings inference. "
            f"Supported types: kitti, tum, euroc. "
            f"Please specify --camera-settings manually."
        )


class VOEvaluationPipeline:
    """Run PySLAM feature-tracker evaluations on baseline and perturbed datasets."""

    def __init__(
        self,
        config_path: Path,
        feature_configs: List[str],
        sensor_type: str,
        camera_settings: Optional[str] = None,
        skip_run: bool = False,
        comparison_only: bool = False,
        num_runs: int = 1,
    ):
        """Initialize VO evaluation pipeline.

        Args:
            config_path: Path to experiment YAML config (same as used for 'run')
            feature_configs: List of feature tracker configs (e.g., ['ORB2', 'SIFT'])
            sensor_type: Sensor type ('mono', 'stereo', 'rgbd')
            camera_settings: PySLAM camera settings file (e.g., 'KITTI04-12.yaml').
                           If None, will be inferred from dataset type and sequence.
            skip_run: If True, skip execution and just print what would run
            comparison_only: If True, skip PySLAM execution and only generate comparison plots
            num_runs: Number of VO runs to execute on the same perturbed data

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        self.config_path = Path(config_path)
        self.feature_configs = feature_configs
        self.sensor_type = sensor_type
        self.skip_run = skip_run
        self.comparison_only = comparison_only
        self.num_runs = int(num_runs)

        if self.num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {self.num_runs}")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        logger.info(f"Loading experiment config from {self.config_path}")
        self.config = load_config(str(self.config_path))

        self.experiment_name = self.config.experiment.name
        self.dataset_type = self.config.dataset.type

        # Require explicit dataset.sequence for deterministic sequence handling
        sequence_name = (self.config.dataset.sequence or "").strip()
        if not sequence_name:
            raise ValueError(
                "dataset.sequence is required for VO evaluation runs. "
                "Set dataset.sequence explicitly in the evaluation config."
            )
        self.sequence_name = sequence_name

        from ..datasets import create_dataset
        self.dataset = create_dataset(self.config.dataset)
        self.dataset_path = Path(self.dataset.path).resolve()

        if camera_settings is None:
            self.camera_settings = infer_camera_settings(
                self.dataset_type, self.sequence_name, self.sensor_type
            )
            logger.info(f"Inferred camera settings: {self.camera_settings}")
        else:
            self.camera_settings = camera_settings

        base_dir = Path(self.config.output.base_dir).resolve()
        self.results_dir = base_dir / self.experiment_name
        self.vo_results_dir = self.results_dir / "vo_results"

        self.runner = PySLAMRunner()

        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Dataset: {self.dataset_path} (type: {self.dataset_type})")
        logger.info(f"Sequence: {self.sequence_name}")
        logger.info(f"Camera settings: {self.camera_settings}")
        logger.info(f"Feature configs: {self.feature_configs}")
        logger.info(f"Sensor type: {self.sensor_type}")
        logger.info(f"Runs: {self.num_runs}")
        logger.info(f"Results directory: {self.vo_results_dir}")

    def run(self) -> Dict[str, Dict[str, Optional[Path]]]:
        """Run PySLAM on baseline and perturbed datasets.

        Returns:
            Dictionary mapping feature configs to trajectory paths:
            {
                'ORB2': {
                    'baseline': Path(...),
                    'fog': Path(...),
                    'rain': Path(...)
                },
                'SIFT': {...}
            }
        """
        logger.info("=" * 60)
        if self.comparison_only:
            logger.info("GENERATING COMPARISON PLOTS ONLY")
        else:
            logger.info("VISUAL ODOMETRY EVALUATION PIPELINE (PySLAM)")
        logger.info("=" * 60)

        perturbed_modules = self._discover_perturbed_datasets()
        if not perturbed_modules:
            logger.warning(
                f"No perturbed data found in {self.results_dir}. "
                f"Run 'slamadverseriallab run {self.config_path}' first."
            )

        logger.info(f"\nFound {len(perturbed_modules)} perturbed module(s):")
        for module in perturbed_modules:
            logger.info(f"  - {module}")

        all_results: Dict[str, Dict[str, Optional[Path]]] = {}

        for feature_config in self.feature_configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running with feature config: {feature_config}")
            logger.info("=" * 60)

            feature_results: Dict[str, Optional[Path]] = {}

            feature_output_dir = self.vo_results_dir / feature_config

            if self.comparison_only:
                run_ids = self._discover_existing_run_ids(feature_config)
                if not run_ids:
                    logger.warning(
                        "No run_* directories found for feature=%s under %s",
                        feature_config,
                        feature_output_dir,
                    )
                    all_results[feature_config] = {}
                    continue

                logger.info(
                    "Generating comparison/aggregated VO outputs from existing runs: %s",
                    ", ".join(f"run_{run_id}" for run_id in run_ids),
                )
                self._generate_comparison_plots(feature_config, run_ids)
                self._write_aggregated_track_outputs(
                    feature_config=feature_config,
                    run_ids=run_ids,
                    perturbed_modules=perturbed_modules,
                )
                all_results[feature_config] = {}
                continue

            run_ids = list(range(self.num_runs))
            for run_id in run_ids:
                logger.info(f"\n{'-'*60}")
                logger.info(f"Run {run_id + 1}/{self.num_runs} (run_{run_id})")
                logger.info(f"{'-'*60}")

                run_output_dir = feature_output_dir / f"run_{run_id}"

                logger.info("\n[BASELINE] Running on original dataset...")
                baseline_output = run_output_dir / "baseline"
                baseline_key = f"baseline_run_{run_id}"

                if self.skip_run:
                    logger.info(f"  [SKIP] Would run PySLAM on {self.dataset_path}")
                    logger.info(f"  [SKIP] Output to: {baseline_output}")
                    feature_results[baseline_key] = None
                else:
                    baseline_traj = self._run_on_dataset(
                        perturbed_images_path=None,  # None means baseline
                        output_dir=baseline_output,
                        feature_config=feature_config,
                    )
                    feature_results[baseline_key] = baseline_traj

                for i, module_name in enumerate(perturbed_modules, start=1):
                    logger.info(
                        f"\n[{i}/{len(perturbed_modules)}] Running on perturbed: {module_name}"
                    )

                    perturbed_images_path = self.results_dir / "images" / module_name
                    module_output = run_output_dir / module_name
                    module_key = f"{module_name}_run_{run_id}"

                    if self.skip_run:
                        logger.info(f"  [SKIP] Would run PySLAM on {perturbed_images_path}")
                        logger.info(f"  [SKIP] Output to: {module_output}")
                        feature_results[module_key] = None
                    else:
                        perturbed_traj = self._run_on_dataset(
                            perturbed_images_path=perturbed_images_path,
                            output_dir=module_output,
                            feature_config=feature_config,
                        )
                        feature_results[module_key] = perturbed_traj

            all_results[feature_config] = feature_results

            if not self.skip_run:
                logger.info(f"\nGenerating per-run comparison plots for {feature_config}...")
                self._generate_comparison_plots(feature_config, run_ids)
                self._write_aggregated_track_outputs(
                    feature_config=feature_config,
                    run_ids=run_ids,
                    perturbed_modules=perturbed_modules,
                )

            self.runner.cleanup()

        self._print_summary(all_results)

        return all_results

    def _run_on_dataset(
        self,
        perturbed_images_path: Optional[Path],
        output_dir: Path,
        feature_config: str,
    ) -> Optional[Path]:
        """Run PySLAM on a single dataset.

        Args:
            perturbed_images_path: Path to perturbed images, or None for baseline
            output_dir: Output directory for this run
            feature_config: Feature tracker config name

        Returns:
            Path to trajectory file, or None if failed
        """
        try:
            trajectory = self.runner.run(
                dataset=self.dataset,
                perturbed_images_path=perturbed_images_path,
                output_dir=output_dir,
                feature_config=feature_config,
                camera_settings=self.camera_settings,
                max_frames=self.config.dataset.max_frames,
            )

            if trajectory:
                logger.info(f"  Trajectory saved to: {trajectory}")
            else:
                name = "baseline" if perturbed_images_path is None else perturbed_images_path.name
                logger.warning(f"  Failed to generate trajectory for {name}")

            return trajectory

        except Exception as e:
            logger.error(f"  Error running PySLAM: {e}")
            return None

    def _discover_existing_run_ids(self, feature_config: str) -> List[int]:
        """Discover existing run ids for a feature output directory.

        Args:
            feature_config: Feature tracker config name.

        Returns:
            Sorted run ids from directories named ``run_<N>``.
        """
        feature_output_dir = self.vo_results_dir / feature_config
        if not feature_output_dir.exists():
            return []

        run_ids: List[int] = []
        for candidate in feature_output_dir.iterdir():
            if not candidate.is_dir():
                continue
            name = candidate.name
            if not name.startswith("run_"):
                continue
            suffix = name.split("run_", 1)[-1]
            if not suffix.isdigit():
                continue
            run_ids.append(int(suffix))
        return sorted(run_ids)

    def _discover_perturbed_datasets(self) -> List[str]:
        """Return module names that have canonical left-camera outputs."""
        images_dir = self.results_dir / "images"

        if not images_dir.exists():
            return []

        left_dir_name = self.dataset.get_canonical_camera_name("left")
        perturbed_modules: List[str] = []

        for module_dir in sorted(images_dir.iterdir()):
            if not module_dir.is_dir():
                continue

            left_image_dir = module_dir / left_dir_name
            if not left_image_dir.exists() or not left_image_dir.is_dir():
                continue

            has_images = any(left_image_dir.glob("*.png")) or any(left_image_dir.glob("*.jpg"))
            if has_images:
                perturbed_modules.append(module_dir.name)

        return perturbed_modules

    def _print_summary(self, all_results: Dict[str, Dict[str, Optional[Path]]]) -> None:
        """Print evaluation summary.

        Args:
            all_results: Results dictionary from run()
        """
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Results directory: {self.vo_results_dir}")

        for feature_config, results in all_results.items():
            logger.info(f"\n{feature_config}:")
            success_count = sum(1 for path in results.values() if path is not None)
            total_count = len(results)
            logger.info(f"  Success: {success_count}/{total_count}")

            for name, path in sorted(results.items()):
                status = "OK" if path else "FAILED"
                logger.info(f"    {name}: {status}")

        logger.info("\nPNG plots saved by PySLAM in output directories.")

    def _generate_comparison_plots(self, feature_config: str, run_ids: List[int]) -> None:
        """Generate run-scoped comparison plots for all perturbations.

        Args:
            feature_config: Feature tracker config name (e.g., 'ORB2')
            run_ids: Run ids to process.
        """
        feature_output_dir = self.vo_results_dir / feature_config

        for run_id in run_ids:
            run_output_dir = feature_output_dir / f"run_{run_id}"
            if not run_output_dir.exists():
                logger.warning(
                    "Run directory missing for comparison plot generation: %s",
                    run_output_dir,
                )
                continue

            comparison_dir = feature_output_dir / "comparison" / f"run_{run_id}"
            comparison_dir.mkdir(parents=True, exist_ok=True)

            perturbation_data: Dict[str, List[int]] = {}

            candidate_dirs = ["baseline"] + sorted(
                p.name
                for p in run_output_dir.iterdir()
                if p.is_dir() and p.name != "baseline"
            )

            for perturbation_name in candidate_dirs:
                ages = self._load_track_ages_from_output_dir(
                    run_output_dir / perturbation_name
                )
                if ages is None:
                    logger.warning(
                        "Track data not found for feature=%s run=%s perturbation=%s",
                        feature_config,
                        run_id,
                        perturbation_name,
                    )
                    continue
                perturbation_data[perturbation_name] = ages

            if not perturbation_data:
                logger.warning(
                    "No track data found for comparison plots (feature=%s, run_%d)",
                    feature_config,
                    run_id,
                )
                continue

            self._plot_track_survival_comparison(
                perturbation_data,
                output_path=comparison_dir / "track_survival_comparison.png",
            )

    def _load_track_ages_from_output_dir(self, output_dir: Path) -> Optional[List[int]]:
        """Load per-track ages from a PySLAM output directory."""
        explicit_path = output_dir / f"tracks_{self.sequence_name}.json"
        tracks_path = explicit_path if explicit_path.exists() else None

        if tracks_path is None:
            matches = sorted(output_dir.glob("tracks_*.json"))
            if matches:
                tracks_path = matches[0]

        if tracks_path is None or not tracks_path.exists():
            return None

        try:
            data = json.loads(tracks_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to read track ages from %s: %s", tracks_path, e)
            return None

        tracks = data.get("tracks")
        if not isinstance(tracks, dict):
            return None

        ages: List[int] = []
        for payload in tracks.values():
            if not isinstance(payload, dict):
                continue
            age = payload.get("age")
            if isinstance(age, (int, float)):
                ages.append(int(age))

        if not ages:
            return None
        return ages

    def _plot_track_survival_comparison(
        self,
        perturbation_data: Dict[str, List[int]],
        output_path: Path,
    ) -> None:
        """Plot track survival curves for all perturbations on one graph.

        Args:
            perturbation_data: Dict mapping perturbation names to list of track ages
            output_path: Path to save the plot
        """
        plt.figure(figsize=(12, 8))

        colors = plt.cm.tab10.colors  # 10 distinct colors

        sorted_names = sorted(
            perturbation_data.keys(),
            key=lambda x: (0 if x == "baseline" else 1, x),
        )

        all_ages = [np.array(ages) for ages in perturbation_data.values()]
        max_possible = max(ages.max() for ages in all_ages)

        x_limit = 1
        for frame in range(1, min(max_possible + 1, 200)):
            has_survival = False
            for ages in all_ages:
                survival_pct = (ages >= frame).sum() / len(ages) * 100
                if survival_pct >= 1.0:
                    has_survival = True
                    break
            if has_survival:
                x_limit = frame
            else:
                break

        x_limit = min(x_limit + 2, max_possible)
        logger.debug(f"Using x-axis limit: {x_limit} frames")

        survival_frames = np.arange(1, x_limit + 1)

        for idx, name in enumerate(sorted_names):
            ages = np.array(perturbation_data[name])

            survival_percentages = np.array(
                [(ages >= frame).sum() / len(ages) * 100 for frame in survival_frames]
            )

            color = colors[idx % len(colors)]
            plt.plot(
                survival_frames,
                survival_percentages,
                label=name,
                color=color,
                linewidth=2,
            )

        plt.xlabel("Track Age (frames)")
        plt.ylabel("% of Tracks Surviving")
        plt.title(f"Track Survival Comparison - {self.sequence_name}")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.xlim(1, x_limit)
        plt.ylim(0, 100)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Saved comparison plot: {output_path}")

    def _write_aggregated_track_outputs(
        self,
        feature_config: str,
        run_ids: List[int],
        perturbed_modules: List[str],
    ) -> None:
        """Write aggregated track statistics and plots for one feature config."""
        aggregated = self._aggregate_track_stats(feature_config, run_ids, perturbed_modules)
        feature_output_dir = self.vo_results_dir / feature_config
        aggregated_dir = feature_output_dir / "aggregated"
        aggregated_dir.mkdir(parents=True, exist_ok=True)

        summary_path = aggregated_dir / "summary.json"
        summary_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
        logger.info("Saved aggregated VO track summary: %s", summary_path)

        self._generate_aggregated_track_plot(
            aggregated=aggregated,
            output_path=aggregated_dir / "aggregated_track_metrics.png",
        )

    def _aggregate_track_stats(
        self,
        feature_config: str,
        run_ids: List[int],
        perturbed_modules: List[str],
    ) -> Dict[str, Any]:
        """Aggregate run-scoped track metrics for baseline and perturbations."""
        feature_output_dir = self.vo_results_dir / feature_config

        baseline_records: List[Dict[str, Any]] = []
        module_records: Dict[str, List[Dict[str, Any]]] = {m: [] for m in perturbed_modules}

        for run_id in run_ids:
            run_dir = feature_output_dir / f"run_{run_id}"
            if not run_dir.exists():
                continue
            for candidate in run_dir.iterdir():
                if not candidate.is_dir() or candidate.name == "baseline":
                    continue
                module_records.setdefault(candidate.name, [])

        for run_id in run_ids:
            run_dir = feature_output_dir / f"run_{run_id}"
            if not run_dir.exists():
                continue

            baseline_metrics = self._load_track_metrics_from_output_dir(run_dir / "baseline")
            if baseline_metrics is not None:
                baseline_records.append({"run_id": run_id, "metrics": baseline_metrics})

            for module_name in module_records:
                metrics = self._load_track_metrics_from_output_dir(run_dir / module_name)
                if metrics is None:
                    continue
                module_records[module_name].append({"run_id": run_id, "metrics": metrics})

        configured_order = [p.name for p in self.config.perturbations if p.enabled]
        module_order = [m for m in configured_order if m in module_records]
        for module_name in sorted(module_records.keys()):
            if module_name not in module_order:
                module_order.append(module_name)

        summary: Dict[str, Any] = {
            "feature_config": feature_config,
            "run_count": len(run_ids),
            "run_ids": run_ids,
            "module_order": module_order,
            "baseline": {
                "num_runs": len(baseline_records),
                "run_ids": [record["run_id"] for record in baseline_records],
                "metrics": self._aggregate_metric_dicts(
                    [record["metrics"] for record in baseline_records]
                ),
            },
            "perturbed_modules": {},
        }

        for module_name in module_order:
            records = module_records.get(module_name, [])
            summary["perturbed_modules"][module_name] = {
                "num_runs": len(records),
                "run_ids": [record["run_id"] for record in records],
                "metrics": self._aggregate_metric_dicts(
                    [record["metrics"] for record in records]
                ),
            }

        return summary

    def _load_track_metrics_from_output_dir(self, output_dir: Path) -> Optional[Dict[str, float]]:
        """Load run-level track metrics from PySLAM output directory.

        Prefers ``track_stats_*.json``. Falls back to deriving metrics from
        per-track ages in ``tracks_*.json``.
        """
        if not output_dir.exists() or not output_dir.is_dir():
            return None

        stats_path: Optional[Path] = None
        explicit_stats = output_dir / f"track_stats_{self.sequence_name}.json"
        if explicit_stats.exists():
            stats_path = explicit_stats
        else:
            matches = sorted(output_dir.glob("track_stats_*.json"))
            if matches:
                stats_path = matches[0]

        if stats_path is not None:
            try:
                payload = json.loads(stats_path.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to read track stats from %s: %s", stats_path, e)
                payload = {}

            metrics: Dict[str, float] = {}
            scalar_keys = [
                "total_map_points",
                "total_valid_tracks",
                "mean_track_length",
                "median_track_length",
                "std_track_length",
                "max_track_length",
                "min_track_length",
                "num_total_frames",
                "num_processed_frames",
            ]
            for key in scalar_keys:
                value = payload.get(key)
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)

            features_per_frame = payload.get("features_per_frame")
            if isinstance(features_per_frame, list) and features_per_frame:
                values = [float(v) for v in features_per_frame if isinstance(v, (int, float))]
                if values:
                    metrics["features_per_frame_mean"] = float(np.mean(values))

            matches_per_frame = payload.get("matches_per_frame")
            if isinstance(matches_per_frame, list) and matches_per_frame:
                values = [float(v) for v in matches_per_frame if isinstance(v, (int, float))]
                if values:
                    metrics["matches_per_frame_mean"] = float(np.mean(values))

            if metrics:
                return metrics

        ages = self._load_track_ages_from_output_dir(output_dir)
        if ages is None:
            return None

        ages_arr = np.asarray(ages, dtype=np.float32)
        return {
            "total_valid_tracks": float(len(ages)),
            "mean_track_length": float(np.mean(ages_arr)),
            "median_track_length": float(np.median(ages_arr)),
            "std_track_length": float(np.std(ages_arr)),
            "max_track_length": float(np.max(ages_arr)),
            "min_track_length": float(np.min(ages_arr)),
        }

    def _aggregate_metric_dicts(self, metric_dicts: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Aggregate a list of run-level metric dictionaries."""
        if not metric_dicts:
            return {}

        all_keys = set()
        for metrics in metric_dicts:
            all_keys.update(metrics.keys())

        aggregated: Dict[str, Dict[str, float]] = {}
        for key in sorted(all_keys):
            values = []
            for metrics in metric_dicts:
                value = metrics.get(key)
                if isinstance(value, (int, float)) and np.isfinite(value):
                    values.append(float(value))
            if not values:
                continue
            aggregated[key] = self._compute_stats(values)

        return aggregated

    def _compute_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute mean/std/min/max/median for a numeric sequence."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return {}

        std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        return {
            "mean": float(np.mean(arr)),
            "std": std_val,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "values": [float(v) for v in arr.tolist()],
        }

    def _generate_aggregated_track_plot(self, aggregated: Dict[str, Any], output_path: Path) -> None:
        """Generate aggregated run comparison plot with error bars."""
        modules = [
            module
            for module in aggregated.get("module_order", [])
            if module in aggregated.get("perturbed_modules", {})
        ]
        if not modules:
            logger.warning("No modules available for aggregated VO plotting")
            return

        baseline = aggregated.get("baseline", {}).get("metrics", {})
        baseline_mean_len = baseline.get("mean_track_length", {}).get("mean")
        baseline_std_len = baseline.get("mean_track_length", {}).get("std")
        baseline_mean_tracks = baseline.get("total_valid_tracks", {}).get("mean")
        baseline_std_tracks = baseline.get("total_valid_tracks", {}).get("std")

        mean_lengths = []
        std_lengths = []
        track_counts = []
        std_counts = []
        selected_modules = []

        for module in modules:
            metrics = aggregated["perturbed_modules"][module].get("metrics", {})
            mean_len = metrics.get("mean_track_length", {}).get("mean")
            std_len = metrics.get("mean_track_length", {}).get("std")
            total_tracks = metrics.get("total_valid_tracks", {}).get("mean")
            std_tracks = metrics.get("total_valid_tracks", {}).get("std")
            if mean_len is None or total_tracks is None:
                continue

            selected_modules.append(module)
            mean_lengths.append(float(mean_len))
            std_lengths.append(float(std_len) if isinstance(std_len, (int, float)) else 0.0)
            track_counts.append(float(total_tracks))
            std_counts.append(float(std_tracks) if isinstance(std_tracks, (int, float)) else 0.0)

        if not selected_modules:
            logger.warning("No complete metrics available for aggregated VO plotting")
            return

        # Avoid matplotlib warnings from NaN std in single-run mode.
        std_lengths = [0.0 if not np.isfinite(v) else float(v) for v in std_lengths]
        std_counts = [0.0 if not np.isfinite(v) else float(v) for v in std_counts]

        x = np.arange(len(selected_modules))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")

        if isinstance(baseline_mean_len, (int, float)):
            base_std = (
                float(baseline_std_len)
                if isinstance(baseline_std_len, (int, float)) and np.isfinite(baseline_std_len)
                else 0.0
            )
            ax1.axhline(y=float(baseline_mean_len), color="green", linestyle="--", linewidth=2)
            ax1.fill_between(
                [-0.5, len(selected_modules) - 0.5],
                float(baseline_mean_len) - base_std,
                float(baseline_mean_len) + base_std,
                color="green",
                alpha=0.2,
                label=f"Baseline ({float(baseline_mean_len):.2f} ± {base_std:.2f})",
            )

        ax1.errorbar(
            x,
            mean_lengths,
            yerr=std_lengths,
            fmt="o",
            markersize=8,
            capsize=5,
            capthick=2,
            color="steelblue",
            ecolor="steelblue",
            label="Perturbed (mean ± std)",
        )
        ax1.set_xlabel("Perturbation Module")
        ax1.set_ylabel("Mean Track Length")
        ax1.set_title(f"Mean Track Length ({aggregated.get('run_count', 0)} runs)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(selected_modules, rotation=45, ha="right")
        ax1.grid(axis="y", alpha=0.3)
        ax1.legend(loc="upper left")

        if isinstance(baseline_mean_tracks, (int, float)):
            base_std = (
                float(baseline_std_tracks)
                if isinstance(baseline_std_tracks, (int, float)) and np.isfinite(baseline_std_tracks)
                else 0.0
            )
            ax2.axhline(y=float(baseline_mean_tracks), color="green", linestyle="--", linewidth=2)
            ax2.fill_between(
                [-0.5, len(selected_modules) - 0.5],
                float(baseline_mean_tracks) - base_std,
                float(baseline_mean_tracks) + base_std,
                color="green",
                alpha=0.2,
                label=f"Baseline ({float(baseline_mean_tracks):.1f} ± {base_std:.1f})",
            )

        ax2.errorbar(
            x,
            track_counts,
            yerr=std_counts,
            fmt="o",
            markersize=8,
            capsize=5,
            capthick=2,
            color="coral",
            ecolor="coral",
            label="Perturbed (mean ± std)",
        )
        ax2.set_xlabel("Perturbation Module")
        ax2.set_ylabel("Total Valid Tracks")
        ax2.set_title(f"Total Valid Tracks ({aggregated.get('run_count', 0)} runs)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(selected_modules, rotation=45, ha="right")
        ax2.grid(axis="y", alpha=0.3)
        ax2.legend(loc="upper left")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved aggregated VO plot: %s", output_path)
