"""Runtime-stress evaluation pipeline."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..runtime_stress.deadline_remap import DROP_LOG_FILENAME, PROGRESS_FILENAME
from ..runtime_stress.models import RuntimeStressRequest, compile_runtime_stress_request
from .evaluation import EvaluationPipeline

logger = logging.getLogger(__name__)

_RECONCILE_MISSING = object()

# Path to the directory containing deadline_iterator.py. Injected into
# the SLAM subprocess as SAL_RUNTIME_PATH so the SLAM's own Python entry
# point can `sys.path.insert(0, …)` and `from deadline_iterator import
# DeadlineIterator` without our framework being installed in the SLAM's
# conda env.
_DEADLINE_ITERATOR_DIR = Path(__file__).resolve().parent.parent / "runtime_stress"


@contextlib.contextmanager
def _realtime_env(realtime, drop_log_path: Path):
    """Temporarily set the env vars a SLAM needs to honor a realtime deadline.

    The SLAM (e.g. DROID-SLAM's demo.py) reads SAL_DEADLINE_FPS to opt
    into the deadline harness, SAL_DEADLINE_WARMUP_FRAMES to absorb
    init costs, SAL_RUNTIME_PATH to locate deadline_iterator.py, and
    SAL_DROP_LOG_PATH to write the drop log on exit. After the SLAM
    run finishes, the framework reads the log and clears the env vars.
    """
    keys = (
        "SAL_DEADLINE_FPS",
        "SAL_DEADLINE_WARMUP_FRAMES",
        "SAL_DEADLINE_QUEUE_SIZE",
        "SAL_DEADLINE_DROP_POLICY",
        "SAL_RUNTIME_PATH",
        "SAL_DROP_LOG_PATH",
        "SAL_PROGRESS_PATH",
    )
    old_values: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in keys}

    # Clear stale logs from a prior run that reused this output dir, for
    # realtime AND non-realtime runs alike. A leftover deadline_drops.json
    # from an earlier deadline run would otherwise be picked up by
    # trajectory conversion (droidslam loads it unconditionally) and remap
    # poses through a survivor list that does not belong to this run; a
    # leftover progress file would let the orchestrator jump phases.
    progress_path = drop_log_path.parent / PROGRESS_FILENAME
    for stale in (drop_log_path, progress_path):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    if realtime is not None:
        os.environ["SAL_DEADLINE_FPS"] = str(realtime.target_fps)
        os.environ["SAL_DEADLINE_WARMUP_FRAMES"] = str(realtime.warmup_frames)
        os.environ["SAL_DEADLINE_QUEUE_SIZE"] = str(realtime.queue_size)
        os.environ["SAL_DEADLINE_DROP_POLICY"] = str(realtime.drop_policy)
        os.environ["SAL_RUNTIME_PATH"] = str(_DEADLINE_ITERATOR_DIR)
        os.environ["SAL_DROP_LOG_PATH"] = str(drop_log_path)
        # Host-side path for the live frame-progress file. For the Podman
        # path this is overridden inside the container to /output/... by
        # apply_realtime_to_podman_cmd; both resolve to the same bind-mounted
        # file. The orchestrator reads this host path to advance
        # frame-anchored control phases.
        os.environ["SAL_PROGRESS_PATH"] = str(progress_path)
    else:
        for k in keys:
            os.environ.pop(k, None)

    try:
        yield
    finally:
        for k, v in old_values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _merge_deadline_into_stress_summary(
    output_dir: Path,
    deadline_summary: Dict[str, Any],
) -> None:
    """Merge a ``deadline`` block into the orchestrator's stress_summary.json.

    The orchestrator writes stress_summary.json synchronously during
    ``algorithm.run`` (in ``RuntimeStressOrchestrator.finalize``), so by the
    time the pipeline computes the deadline summary the file is already on
    disk. Reading, merging, and writing back keeps both pieces in one file
    so downstream aggregation/plotting reads a single source of truth per
    scenario instead of opening a sidecar.

    No-op if stress_summary.json doesn't exist (e.g. the orchestrator
    failed to write it).
    """
    summary_path = output_dir / "stress_summary.json"
    if not summary_path.exists():
        logger.warning(
            "  stress_summary.json missing in %s; deadline telemetry not merged",
            output_dir,
        )
        return
    try:
        with open(summary_path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("  Failed to read %s for merge: %s", summary_path, exc)
        return

    payload["deadline"] = deadline_summary
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summarize_drop_log(drop_log_path: Path) -> Optional[Dict[str, Any]]:
    """Read the SLAM's drop log and compute summary statistics.

    Returns None if no log was written (e.g. the SLAM crashed before
    completing iteration, or the patch wasn't applied to this SLAM).
    """
    if not drop_log_path.exists():
        return None
    try:
        with open(drop_log_path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    survivors = payload.get("survivors") or []
    dropped = payload.get("dropped") or []
    total_items = payload.get("total_items") or (len(survivors) + len(dropped))
    target_fps = payload.get("target_fps")

    longest_dropped_streak = 0
    current = 0
    prev = None
    for idx in dropped:
        if prev is not None and idx == prev + 1:
            current += 1
        else:
            current = 1
        longest_dropped_streak = max(longest_dropped_streak, current)
        prev = idx

    longest_dropped_gap = 0
    if len(survivors) >= 2:
        for a, b in zip(survivors[:-1], survivors[1:]):
            # Number of dropped frames between two consecutive survivors:
            # if survivors are 4 and 9, frames 5..8 (4 of them) were dropped.
            longest_dropped_gap = max(longest_dropped_gap, b - a - 1)

    return {
        "target_fps": target_fps,
        "warmup_frames": payload.get("warmup_frames", 0),
        "queue_size": payload.get("queue_size", 1),
        "drop_policy": payload.get("drop_policy", "drop_oldest"),
        "total_items": total_items,
        "drop_count": len(dropped),
        "drop_rate": (len(dropped) / total_items) if total_items > 0 else 0.0,
        "longest_dropped_streak": longest_dropped_streak,
        "longest_dropped_gap": longest_dropped_gap,
    }


def _evict_paths_from_page_cache(paths: List[Path]) -> None:
    """Evict each path's files from the page cache (shared impl in page_cache).

    Used between the baseline run and each IO-throttled scenario so that
    the cgroup IO controller's cap actually bites — otherwise the
    baseline pre-warms the page cache and reads service from RAM,
    making the cap a no-op.
    """
    from ..runtime_stress.page_cache import evict

    total_files = 0
    total_bytes = 0
    for path in paths:
        if not path.exists():
            continue
        n, b = evict(path)
        total_files += n
        total_bytes += b
    logger.info(
        "  Evicted page cache for %d files (%.1f MB) before scenario",
        total_files,
        total_bytes / 1e6,
    )


def _reconcile_algorithm_runtime(algorithm, yaml_runtime: str):
    """Return the (possibly reconstructed) algorithm to match the YAML runtime.

    Reconstruction rules:
    * Wrapper has no ``container_runtime`` field → leave as-is.
    * Wrapper has ``container_runtime=None`` and YAML is the schema default
      ``"docker"`` → treat as host-process (conda) stress; no reconstruction.
    * Any other mismatch → reconstruct via ``type(algorithm)(container_runtime=yaml_runtime)``.
    """
    algorithm_runtime = getattr(algorithm, "container_runtime", _RECONCILE_MISSING)

    if algorithm_runtime is _RECONCILE_MISSING:
        return algorithm
    if algorithm_runtime is None and yaml_runtime == "docker":
        return algorithm
    if algorithm_runtime == yaml_runtime:
        return algorithm

    logger.info(
        "Reconstructing %s algorithm with container_runtime=%r to match runtime_stress config",
        algorithm.name,
        yaml_runtime,
    )
    return type(algorithm)(container_runtime=yaml_runtime)


class RuntimeStressEvaluationPipeline(EvaluationPipeline):
    """Run baseline and runtime-stress scenarios for one SLAM backend."""

    def __init__(
        self,
        config_path: Path,
        slam_algorithm: str,
        slam_config_path: Optional[str] = None,
        num_runs: int = 1,
        paper_mode: bool = False,
    ):
        super().__init__(
            config_path=config_path,
            slam_algorithm=slam_algorithm,
            slam_config_path=slam_config_path,
            compute_metrics=True,
            skip_slam=False,
            num_runs=num_runs,
            paper_mode=paper_mode,
        )

        runtime_stress = getattr(self.config, "runtime_stress", None)
        if runtime_stress is None or not runtime_stress.enabled:
            raise ValueError(
                "runtime_stress.enabled=true is required for --mode runtime-stress"
            )

        self.algorithm = _reconcile_algorithm_runtime(
            self.algorithm, runtime_stress.container_runtime
        )

        if self.algorithm.runtime_stress_target_kind not in {
            "host_process_group",
            "docker_container",
            "podman_container",
        }:
            raise ValueError(
                f"Algorithm '{self.algorithm.name}' does not support runtime-stress mode "
                f"(target kind: {self.algorithm.runtime_stress_target_kind})."
            )

        self.runtime_stress = runtime_stress
        self.compiled_scenarios = [
            compile_runtime_stress_request(runtime_stress, scenario)
            for scenario in runtime_stress.scenarios
            if scenario.enabled
        ]
        self.stress_root_dir = self.results_dir / "slam_results" / self.algorithm.name / "runtime_stress"
        self.slam_results_dir = self.stress_root_dir
        self.metrics_dir = self.stress_root_dir / "metrics"
        self.trajectories_dir = self.stress_root_dir / "trajectories"
        self._run_durations: Dict[str, float] = {}

    def _metric_evaluator_kwargs(self) -> Dict[str, Any]:
        """Exclude deadline-warmup frames from every metric of a deadline condition.

        Warmup frames are delivered unpaced and undropped by design, so they
        measure the unstressed system. Scoring them lets a run that survived
        nothing but warmup report a good ATE (e.g. a warmup-only fragment
        scoring 0.05 m at 84% drops). When any enabled scenario uses the
        realtime deadline, all trajectories of this condition, baselines
        included, are scored on the post-warmup frames only, so stressed and
        clean cells compare the same frame set.
        """
        warmups = [
            s.realtime.warmup_frames for s in self.compiled_scenarios
            if s.realtime is not None
        ]
        warmup = max(warmups) if warmups else 0
        if warmup <= 0:
            return {}

        from ..runtime_stress.warmup_cutoff import compute_warmup_cutoff_ts

        # The SLAM processes the stride-sampled stream, and warmup_frames counts
        # those sampled frames. Convert to the raw dataset timebase used by the
        # cutoff and the completeness denominator. Stride is 1 for every-frame
        # SLAMs; DROID-SLAM and DPVO use 2 on TUM.
        try:
            stride = max(1, int(self.algorithm.frame_stride(self.dataset_type)))
        except Exception:
            stride = 1

        timestamps_path = None
        try:
            timestamps_path = self.dataset.get_timestamps_file_path()
        except Exception:
            pass
        cutoff = compute_warmup_cutoff_ts(
            self.dataset_type, self.dataset_path, warmup, timestamps_path, stride
        )
        if cutoff is None:
            return {}
        logger.info(
            "Deadline condition: metrics exclude the %d warmup frames "
            "(stride %d, cutoff timestamp %s)", warmup, stride, cutoff
        )
        return {
            "deadline_cutoff_ts": cutoff,
            "deadline_warmup_frames": warmup,
            "deadline_stride": stride,
        }

    def run(self) -> Dict[str, Path]:
        logger.info("=" * 60)
        logger.info("RUNTIME-STRESS EVALUATION PIPELINE")
        logger.info("=" * 60)

        all_trajectories: Dict[str, Path] = {}

        for run_id in range(self.num_runs):
            logger.info("\n[BASELINE] Running baseline without runtime stress...")
            baseline_traj = self._run_variant(
                label="baseline",
                run_id=run_id,
                runtime_stress=None,
                fail_hard=True,
            )
            all_trajectories[f"baseline_run_{run_id}"] = baseline_traj

            for scenario in self.compiled_scenarios:
                logger.info(
                    "\n[RUNTIME-STRESS] Running scenario '%s'...",
                    scenario.scenario_name,
                )
                if self.runtime_stress.evict_page_cache and any(
                    p.controls.io is not None for p in scenario.phases
                ):
                    _evict_paths_from_page_cache([self.dataset_path])
                try:
                    traj = self._run_variant(
                        label=scenario.scenario_name,
                        run_id=run_id,
                        runtime_stress=scenario,
                        fail_hard=False,
                    )
                except RuntimeError as exc:
                    logger.warning("Scenario '%s' failed: %s", scenario.scenario_name, exc)
                    continue

                all_trajectories[f"{scenario.scenario_name}_run_{run_id}"] = traj

        if self.compute_metrics:
            self._compute_metrics(all_trajectories)

        # Timing summary (always produced, independent of trajectory metrics)
        timing = self._compute_timing_summary()
        self._save_timing_summary(timing)
        try:
            self._generate_timing_plot(timing)
        except Exception as e:
            logger.warning("Failed to generate timing plot: %s", e)
        try:
            self._generate_deadline_drop_plot()
        except Exception as e:
            logger.warning("Failed to generate deadline-drop plot: %s", e)
        try:
            self._generate_stress_timeline_plots()
        except Exception as e:
            logger.warning("Failed to generate stress-timeline plots: %s", e)
        self._print_timing_summary(timing)

        self._print_summary(all_trajectories)
        return all_trajectories

    def _run_variant(
        self,
        label: str,
        run_id: int,
        runtime_stress: Optional[RuntimeStressRequest],
        fail_hard: bool,
    ) -> Path:
        self.algorithm.cleanup()

        output_dir = self.stress_root_dir / label / f"run_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_dataset_dir = None
        dataset_to_use = self.dataset_path
        max_frames = self.config.dataset.max_frames
        if max_frames:
            temp_dataset_dir = self.dataset.create_truncated_copy(max_frames)
            dataset_to_use = temp_dataset_dir
            logger.info(
                "  Using truncated dataset (%s frames): %s",
                max_frames,
                temp_dataset_dir,
            )

        try:
            request = self._create_run_request(
                dataset_to_use,
                output_dir,
                runtime_stress=runtime_stress,
            )

            realtime = runtime_stress.realtime if runtime_stress is not None else None
            drop_log_path = output_dir / DROP_LOG_FILENAME

            with _realtime_env(realtime, drop_log_path):
                if realtime is not None:
                    logger.info(
                        "  Realtime deadline active: target_fps=%.2f, "
                        "warmup_frames=%d, drop log -> %s",
                        realtime.target_fps,
                        realtime.warmup_frames,
                        drop_log_path,
                    )
                t0 = time.monotonic()
                result = self.algorithm.run(request)
                duration_s = round(time.monotonic() - t0, 3)

            duration_key = f"{label}_run_{run_id}"
            self._run_durations[duration_key] = duration_s
            logger.info("  %s duration: %.3f s", label, duration_s)

            if runtime_stress is not None:
                # A controller failure invalidates the cell even when the SLAM
                # left a plausible trajectory behind (e.g. a prepare failure
                # whose termination raced a fast run to completion): the
                # trajectory is from an UNstressed or partially-stressed run
                # and must never be scored under this scenario's label.
                summary_path = output_dir / "stress_summary.json"
                try:
                    with open(summary_path) as f:
                        controller_error = json.load(f).get("controller_error")
                except (OSError, json.JSONDecodeError):
                    controller_error = None
                if controller_error:
                    raise RuntimeError(
                        f"{label}: controller error invalidates this run "
                        f"(results discarded): {controller_error}"
                    )

                # A run with no console log cannot be health-verified, and an
                # unverified run is not a scorable cell: pose output alone never
                # counts as success, since a SLAM can lose tracking and keep
                # emitting poses. Missing or empty means the wrapper stopped
                # teeing (preflight P18) -- fail here rather than let the gap
                # reach aggregation, where it would look like a healthy cell.
                log_path = output_dir / "slam_output.log"
                try:
                    log_bytes = log_path.stat().st_size
                except OSError:
                    log_bytes = -1
                if log_bytes <= 0:
                    raise RuntimeError(
                        f"{label}: no usable slam_output.log "
                        f"({'missing' if log_bytes < 0 else 'empty'}) — the run "
                        "cannot be health-verified, so it is invalid"
                    )

            if realtime is not None:
                summary = _summarize_drop_log(drop_log_path)
                if summary is not None:
                    _merge_deadline_into_stress_summary(output_dir, summary)
                    logger.info(
                        "  Realtime drop summary: %d/%d frames dropped (%.1f%%), "
                        "longest streak=%d, longest gap=%d",
                        summary["drop_count"],
                        summary["total_items"],
                        summary["drop_rate"] * 100.0,
                        summary["longest_dropped_streak"],
                        summary["longest_dropped_gap"],
                    )
                else:
                    logger.warning(
                        "  Realtime deadline was active but no drop log was written "
                        "(SLAM may not yet honor SAL_DEADLINE_FPS, or crashed before exit)."
                    )

            if not result.success or result.trajectory_path is None:
                message = result.message or f"{label} SLAM execution failed"
                if fail_hard:
                    raise RuntimeError(message)
                raise RuntimeError(message)

            trajectory_path = result.trajectory_path
            if not trajectory_path.exists():
                raise RuntimeError(f"Missing trajectory output for {label}: {trajectory_path}")

            self.algorithm.cleanup()
            return trajectory_path
        finally:
            if temp_dataset_dir:
                shutil.rmtree(temp_dataset_dir, ignore_errors=True)

    def _comparison_count_label(self) -> str:
        """Return summary label for runtime-stress comparisons."""
        return "Scenarios"

    def _comparison_collection_label(self) -> str:
        """Return aggregated label for runtime-stress scenarios."""
        return "Runtime Stress Scenarios"

    def _comparison_axis_label(self) -> str:
        """Return x-axis label for runtime-stress plots."""
        return "Runtime Stress Scenario"

    def _comparison_series_label(self) -> str:
        """Return legend label for runtime-stress scenario series."""
        return "Runtime stress (mean ± std)"

    def _comparison_order_from_config(self) -> list[str]:
        """Return scenario order from the runtime-stress config."""
        return [scenario.scenario_name for scenario in self.compiled_scenarios]

    # ------------------------------------------------------------------
    # Timing summary
    # ------------------------------------------------------------------

    def _compute_timing_summary(self) -> Dict[str, Any]:
        """Aggregate wall-clock durations across runs per scenario."""
        baseline_durations: List[float] = []
        scenario_durations: Dict[str, List[float]] = {}

        for key, dur in self._run_durations.items():
            if key.startswith("baseline"):
                baseline_durations.append(dur)
            else:
                scenario_name = key.rsplit("_run_", 1)[0]
                scenario_durations.setdefault(scenario_name, []).append(dur)

        summary: Dict[str, Any] = {
            "baseline": self._compute_stats(baseline_durations) if baseline_durations else None,
            "scenarios": {},
            "scenario_order": self._comparison_order_from_config(),
            "run_count": self.num_runs,
        }
        for name in summary["scenario_order"]:
            vals = scenario_durations.get(name, [])
            summary["scenarios"][name] = self._compute_stats(vals) if vals else None

        return summary

    def _save_timing_summary(self, timing: Dict[str, Any]) -> Path:
        """Write timing_summary.json to the metrics directory."""
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        path = self.metrics_dir / "timing_summary.json"
        path.write_text(json.dumps(timing, indent=2), encoding="utf-8")
        logger.info("Timing summary saved to: %s", path)
        return path

    def _generate_timing_plot(self, timing: Dict[str, Any]) -> None:
        """Generate a bar chart comparing wall-clock duration across scenarios."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        scenarios = [s for s in timing["scenario_order"] if timing["scenarios"].get(s)]
        if not scenarios:
            return

        means = [timing["scenarios"][s]["mean"] for s in scenarios]
        stds = [timing["scenarios"][s]["std"] for s in scenarios]

        x = np.arange(len(scenarios))
        _, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.2), 5))

        # Baseline reference band
        if timing["baseline"] is not None:
            bl_mean = timing["baseline"]["mean"]
            bl_std = timing["baseline"]["std"]
            ax.axhline(y=bl_mean, color="green", linestyle="--", linewidth=2, zorder=1)
            ax.fill_between(
                [-0.5, len(scenarios) - 0.5],
                bl_mean - bl_std,
                bl_mean + bl_std,
                color="green",
                alpha=0.2,
                label=f"Baseline ({bl_mean:.1f} ± {bl_std:.1f} s)",
                zorder=0,
            )

        ax.errorbar(
            x,
            means,
            yerr=stds,
            fmt="o",
            markersize=8,
            capsize=5,
            capthick=2,
            color="steelblue",
            ecolor="steelblue",
            label=self._comparison_series_label(),
            zorder=2,
        )

        ax.set_xlabel(self._comparison_axis_label(), fontsize=12)
        ax.set_ylabel("Wall-clock Duration (s)", fontsize=12)
        ax.set_title(
            f"SLAM Execution Duration ({timing['run_count']} run(s) per scenario)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(loc="upper left")

        plot_path = self.metrics_dir / "timing_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Timing plot saved to: %s", plot_path)

    def _generate_deadline_drop_plot(self) -> None:
        """Plot drop_rate across scenarios that ran under a deadline.

        The x-axis is whichever deadline knob actually varied across the
        scenarios: ``target_fps`` for an fps sweep, or ``queue_size`` for
        a fixed-fps queue-depth sweep. When both vary (or neither does),
        falls back to the scenario name.

        No-op when no scenario emitted a ``deadline`` block (i.e., the
        experiment didn't use the realtime harness).
        """
        points: List[Dict[str, float]] = []
        for scenario in self.compiled_scenarios:
            rates: List[float] = []
            fps_values: List[float] = []
            queue_values: List[float] = []
            for run_id in range(self.num_runs):
                summary_path = (
                    self.stress_root_dir
                    / scenario.scenario_name
                    / f"run_{run_id}"
                    / "stress_summary.json"
                )
                if not summary_path.exists():
                    continue
                try:
                    summary = json.loads(summary_path.read_text())
                except (OSError, ValueError):
                    continue
                dl = summary.get("deadline")
                if not dl or "drop_rate" not in dl or "target_fps" not in dl:
                    continue
                rates.append(float(dl["drop_rate"]))
                fps_values.append(float(dl["target_fps"]))
                queue_values.append(float(dl.get("queue_size", 1)))
            if not rates:
                continue
            points.append(
                {
                    "scenario": scenario.scenario_name,
                    "target_fps": sum(fps_values) / len(fps_values),
                    "queue_size": sum(queue_values) / len(queue_values),
                    "drop_rate": sum(rates) / len(rates),
                }
            )

        if not points:
            return

        # Choose the x-axis dimension: prefer whichever of target_fps /
        # queue_size has more than one distinct value across scenarios.
        fps_varies = len({round(p["target_fps"], 6) for p in points}) > 1
        queue_varies = len({round(p["queue_size"], 6) for p in points}) > 1
        if queue_varies and not fps_varies:
            x_key, x_label = "queue_size", "Queue Depth (frames)"
            fps = points[0]["target_fps"]
            title = f"Deadline Drop Rate vs Queue Depth (target {fps:.0f} fps)"
        elif fps_varies and not queue_varies:
            x_key, x_label = "target_fps", "Target FPS"
            title = "Deadline Drop Rate vs Target FPS"
        else:
            x_key, x_label, title = (
                "scenario",
                "Scenario",
                "Deadline Drop Rate by Scenario",
            )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if x_key == "scenario":
            points_sorted = points
            xs = list(range(len(points_sorted)))
            xticklabels = [p["scenario"] for p in points_sorted]
        else:
            points_sorted = sorted(points, key=lambda p: p[x_key])
            xs = [p[x_key] for p in points_sorted]
            xticklabels = None

        drops = [p["drop_rate"] * 100.0 for p in points_sorted]

        _, ax = plt.subplots(figsize=(max(7, len(points_sorted) * 1.0), 5))
        ax.plot(xs, drops, marker="o", linewidth=2, color="steelblue")

        for x, y in zip(xs, drops):
            ax.annotate(
                f"{y:.1f}%",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Drop Rate (%)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        if xticklabels is not None:
            ax.set_xticks(xs)
            ax.set_xticklabels(xticklabels, rotation=45, ha="right")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=-2)

        plot_path = self.metrics_dir / "deadline_drops.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Deadline-drop plot saved to: %s", plot_path)

    def _generate_stress_timeline_plots(self) -> None:
        """Draw a resource-consumption timeline for every run with telemetry.

        One chart per run: elapsed time on the x axis, consumption on the y,
        phases shaded. Baseline runs carry no trace and are skipped.
        """
        from src.runtime_stress.trace_plot import plot_run, walk_runs

        written = 0
        for run_dir in sorted(walk_runs(self.stress_root_dir)):
            try:
                if plot_run(run_dir, quiet=True):
                    written += 1
            except Exception as exc:  # one bad run must not lose the rest
                logger.warning("Stress-timeline plot failed for %s: %s", run_dir, exc)
        if written:
            logger.info("Stress-timeline charts saved: %d (stress_timeline.png per run)", written)

    def _print_timing_summary(self, timing: Dict[str, Any]) -> None:
        """Print timing summary to log."""
        logger.info("\n" + "=" * 60)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 60)

        if timing["baseline"] is not None:
            bl = timing["baseline"]
            logger.info("Baseline: %.3f s (±%.3f s)", bl["mean"], bl["std"])

        for name in timing["scenario_order"]:
            stats = timing["scenarios"].get(name)
            if stats is None:
                logger.info("  %s: no data", name)
                continue
            logger.info(
                "  %s: %.3f s (±%.3f s)  [min=%.3f, max=%.3f]",
                name,
                stats["mean"],
                stats["std"],
                stats["min"],
                stats["max"],
            )
