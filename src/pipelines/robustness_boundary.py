"""Robustness-boundary search pipeline built on existing run/evaluate flows."""

from __future__ import annotations

import copy
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.parser import Config, load_config, save_config
from ..config.schema import PerturbationConfig
from ..core.pipeline import Pipeline
from ..modules.base import get_module_registry
from ..robustness.param_spec import (
    BoundaryParamSpec,
    apply_canonicalize,
    format_trial_value,
    is_interval_small_enough,
    is_param_active,
    midpoint,
    parse_domain_value,
)
from .evaluation import EvaluationPipeline

logger = logging.getLogger(__name__)


@dataclass
class BoundaryTrialResult:
    """Result for one robustness-boundary trial evaluation."""

    label: str
    search_value: float | int
    parameter_value: Any
    passed: bool
    failed: bool
    tracking_failure: bool
    ate_rmse: Optional[float]
    reason: str
    trial_config_path: Path
    trial_output_dir: Path
    metrics_summary_path: Optional[Path]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "search_value": self.search_value,
            "parameter_value": self.parameter_value,
            "passed": self.passed,
            "failed": self.failed,
            "tracking_failure": self.tracking_failure,
            "ate_rmse": self.ate_rmse,
            "reason": self.reason,
            "trial_config_path": str(self.trial_config_path),
            "trial_output_dir": str(self.trial_output_dir),
            "metrics_summary_path": str(self.metrics_summary_path) if self.metrics_summary_path else None,
            "error": self.error,
        }


class RobustnessBoundaryPipeline:
    """Run bracketed robustness-boundary search for one SLAM algorithm."""

    def __init__(
        self,
        config_path: Path,
        slam_algorithm: str,
        slam_config_path: Optional[str] = None,
        num_runs: int = 1,
        paper_mode: bool = False,
    ):
        self.config_path = Path(config_path)
        self.slam_algorithm = slam_algorithm
        self.slam_config_path = slam_config_path
        self.num_runs = num_runs
        self.paper_mode = paper_mode

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.config: Config = load_config(self.config_path)
        self.boundary_cfg = getattr(self.config, "robustness_boundary", None)
        if self.boundary_cfg is None or not self.boundary_cfg.enabled:
            raise ValueError(
                "robustness_boundary.enabled=true is required for --mode robustness-boundary"
            )

        if not self.config.output.save_images:
            raise ValueError(
                "robustness-boundary requires output.save_images=true to evaluate perturbed trials."
            )

        # None means boundary target is a top-level perturbation; otherwise this points
        # to a nested module index inside a selected composite perturbation.
        self._target_nested_module_index: Optional[int] = None

        self.target_perturbation = self._resolve_target_perturbation()
        self.param_spec = self._resolve_param_spec()
        active_params = self._get_target_parameter_dict(self.target_perturbation)
        if not is_param_active(self.param_spec, active_params):
            raise ValueError(
                f"robustness_boundary.parameter '{self.boundary_cfg.parameter}' is inactive for "
                f"module '{self.boundary_cfg.module}' under current parameters."
            )

        self.lower_bound_value = parse_domain_value(self.param_spec, self.boundary_cfg.lower_bound)
        self.upper_bound_value = parse_domain_value(self.param_spec, self.boundary_cfg.upper_bound)

        boundary_name = (getattr(self.boundary_cfg, "name", "") or "").strip()
        base_dir = Path(self.config.output.base_dir).resolve()
        boundary_root = (
            base_dir
            / self.config.experiment.name
            / "robustness_boundary"
            / self.slam_algorithm
        )
        if boundary_name:
            self.boundary_dir = boundary_root / boundary_name
        else:
            self.boundary_dir = (
                boundary_root
                / self.boundary_cfg.module
                / self.boundary_cfg.parameter
            )
        self.trials_dir = self.boundary_dir / "trials"
        self.configs_dir = self.boundary_dir / "configs"
        self.boundary_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Run endpoint checks and optional bisection to estimate boundary."""
        logger.info("=" * 60)
        logger.info("ROBUSTNESS-BOUNDARY SEARCH")
        logger.info("=" * 60)
        logger.info(
            "Algorithm=%s, target=%s.%s, bounds=(%s, %s)",
            self.slam_algorithm,
            self.boundary_cfg.module,
            self.boundary_cfg.parameter,
            self.boundary_cfg.lower_bound,
            self.boundary_cfg.upper_bound,
        )

        trials: List[BoundaryTrialResult] = []
        lower_trial = self._run_trial(self.lower_bound_value, "lower_bound")
        upper_trial = self._run_trial(self.upper_bound_value, "upper_bound")
        trials.extend([lower_trial, upper_trial])

        pass_trial: Optional[BoundaryTrialResult] = None
        fail_trial: Optional[BoundaryTrialResult] = None
        iteration_count = 0
        termination_reason = "not_bracketed"

        if lower_trial.passed != upper_trial.passed:
            if lower_trial.passed:
                pass_trial, fail_trial = lower_trial, upper_trial
            else:
                pass_trial, fail_trial = upper_trial, lower_trial

            while iteration_count < self.boundary_cfg.max_iters:
                if is_interval_small_enough(
                    self.param_spec,
                    pass_trial.search_value,
                    fail_trial.search_value,
                    self.boundary_cfg.tolerance,
                ):
                    termination_reason = "tolerance"
                    break

                mid_value = midpoint(
                    self.param_spec,
                    pass_trial.search_value,
                    fail_trial.search_value,
                )

                # Integer/bitrate domains may collapse to one side before tolerance rule.
                if mid_value in (pass_trial.search_value, fail_trial.search_value):
                    termination_reason = "domain_resolution"
                    break

                iteration_count += 1
                mid_trial = self._run_trial(mid_value, f"iter_{iteration_count:02d}")
                trials.append(mid_trial)

                if mid_trial.passed:
                    pass_trial = mid_trial
                else:
                    fail_trial = mid_trial

            if termination_reason not in {"tolerance", "domain_resolution"}:
                termination_reason = (
                    "tolerance"
                    if is_interval_small_enough(
                        self.param_spec,
                        pass_trial.search_value,
                        fail_trial.search_value,
                        self.boundary_cfg.tolerance,
                    )
                    else "max_iters"
                )

        summary = {
            "mode": "robustness-boundary",
            "slam_algorithm": self.slam_algorithm,
            "source_config": str(self.config_path),
            "boundary_config": {
                "name": self.boundary_cfg.name,
                "module": self.boundary_cfg.module,
                "parameter": self.boundary_cfg.parameter,
                "lower_bound": self.boundary_cfg.lower_bound,
                "upper_bound": self.boundary_cfg.upper_bound,
                "tolerance": self.boundary_cfg.tolerance,
                "max_iters": self.boundary_cfg.max_iters,
                "ate_rmse_fail": self.boundary_cfg.ate_rmse_fail,
                "fail_on_tracking_failure": self.boundary_cfg.fail_on_tracking_failure,
            },
            "target_perturbation_name": self.target_perturbation.name,
            "termination_reason": termination_reason,
            "iterations": iteration_count,
            "trial_count": len(trials),
            "trials": [trial.to_dict() for trial in trials],
            "pass_bound": pass_trial.to_dict() if pass_trial else None,
            "fail_bound": fail_trial.to_dict() if fail_trial else None,
        }

        summary_path = self.boundary_dir / "boundary_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Boundary summary written to: %s", summary_path)
        return {"summary_path": summary_path, "summary": summary}

    def _resolve_target_perturbation(self) -> PerturbationConfig:
        target_name = (getattr(self.boundary_cfg, "target_perturbation", "") or "").strip()
        if target_name:
            named_matches = [
                pert
                for pert in self.config.perturbations
                if pert.enabled and pert.name == target_name
            ]
            if not named_matches:
                raise ValueError(
                    f"robustness_boundary.target_perturbation '{target_name}' does not match "
                    "any enabled perturbation name."
                )
            if len(named_matches) > 1:
                raise ValueError(
                    f"robustness_boundary.target_perturbation '{target_name}' is ambiguous "
                    "(multiple enabled perturbations have this name)."
                )

            selected = named_matches[0]
            if selected.type == self.boundary_cfg.module:
                self._target_nested_module_index = None
                return selected

            if selected.type != "composite":
                raise ValueError(
                    f"robustness_boundary.target_perturbation '{target_name}' has type "
                    f"'{selected.type}', which is incompatible with boundary module "
                    f"'{self.boundary_cfg.module}'."
                )

            modules = selected.parameters.get("modules")
            if not isinstance(modules, list):
                raise ValueError(
                    f"Composite perturbation '{selected.name}' is missing a valid modules list."
                )

            matching_indices = [
                idx
                for idx, module_cfg in enumerate(modules)
                if isinstance(module_cfg, dict) and module_cfg.get("type") == self.boundary_cfg.module
            ]
            if not matching_indices:
                raise ValueError(
                    f"Composite perturbation '{selected.name}' does not contain a nested "
                    f"module of type '{self.boundary_cfg.module}'."
                )
            if len(matching_indices) > 1:
                raise ValueError(
                    f"Composite perturbation '{selected.name}' contains multiple nested modules "
                    f"of type '{self.boundary_cfg.module}'. Boundary mode requires exactly one."
                )

            self._target_nested_module_index = matching_indices[0]
            return selected

        direct_matches = [
            pert
            for pert in self.config.perturbations
            if pert.enabled and pert.type == self.boundary_cfg.module
        ]

        if len(direct_matches) == 1:
            self._target_nested_module_index = None
            return direct_matches[0]

        if len(direct_matches) > 1:
            names = ", ".join(p.name for p in direct_matches)
            raise ValueError(
                f"Ambiguous robustness_boundary target for module '{self.boundary_cfg.module}'. "
                f"Multiple enabled perturbations match: {names}. "
                "Keep one enabled perturbation for this module in boundary mode."
            )

        composite_candidates: List[tuple[PerturbationConfig, int]] = []
        for pert in self.config.perturbations:
            if not pert.enabled or pert.type != "composite":
                continue

            modules = pert.parameters.get("modules")
            if not isinstance(modules, list):
                continue

            matching_indices = [
                idx
                for idx, module_cfg in enumerate(modules)
                if isinstance(module_cfg, dict) and module_cfg.get("type") == self.boundary_cfg.module
            ]
            if not matching_indices:
                continue
            if len(matching_indices) > 1:
                raise ValueError(
                    f"Composite perturbation '{pert.name}' contains multiple nested modules "
                    f"of type '{self.boundary_cfg.module}'. Boundary mode requires exactly one."
                )

            composite_candidates.append((pert, matching_indices[0]))

        if not composite_candidates:
            raise ValueError(
                f"No enabled perturbation found with type '{self.boundary_cfg.module}', "
                f"and no enabled composite perturbation contains a nested '{self.boundary_cfg.module}' module."
            )

        if len(composite_candidates) > 1:
            names = ", ".join(candidate.name for candidate, _ in composite_candidates)
            raise ValueError(
                f"Ambiguous robustness_boundary target for module '{self.boundary_cfg.module}'. "
                f"Multiple enabled composite perturbations contain nested matches: {names}. "
                "Set robustness_boundary.target_perturbation to select one."
            )

        selected, module_index = composite_candidates[0]
        self._target_nested_module_index = module_index
        return selected

    def _resolve_param_spec(self) -> BoundaryParamSpec:
        registry = get_module_registry()
        if self.boundary_cfg.module not in registry:
            raise ValueError(f"Unknown module '{self.boundary_cfg.module}'")
        module_class = registry[self.boundary_cfg.module].module_class
        specs = getattr(module_class, "SEARCHABLE_PARAMS", {})
        if self.boundary_cfg.parameter not in specs:
            raise ValueError(
                f"Module '{self.boundary_cfg.module}' does not support parameter "
                f"'{self.boundary_cfg.parameter}' for robustness boundary."
            )
        return specs[self.boundary_cfg.parameter]

    def _get_target_parameter_dict(self, perturbation: PerturbationConfig) -> Dict[str, Any]:
        """Return parameter dictionary for active-if checks and search metadata."""
        if self._target_nested_module_index is None:
            return perturbation.parameters

        modules = perturbation.parameters.get("modules")
        if not isinstance(modules, list):
            raise ValueError(
                f"Composite perturbation '{perturbation.name}' is missing a valid modules list."
            )

        nested_idx = self._target_nested_module_index
        if nested_idx >= len(modules):
            raise ValueError(
                f"Composite perturbation '{perturbation.name}' no longer has nested module index {nested_idx}."
            )

        nested_cfg = modules[nested_idx]
        if not isinstance(nested_cfg, dict):
            raise ValueError(
                f"Composite nested module at index {nested_idx} in '{perturbation.name}' must be a mapping."
            )

        nested_params = nested_cfg.get("parameters", {})
        if not isinstance(nested_params, dict):
            raise ValueError(
                f"Composite nested module at index {nested_idx} in '{perturbation.name}' has invalid parameters."
            )

        return nested_params

    def _build_trial_config(
        self,
        trial_experiment_name: str,
        parameter_value: Any,
    ) -> Config:
        trial_config: Config = copy.deepcopy(self.config)

        trial_config.experiment.name = trial_experiment_name
        trial_config.output.base_dir = str(self.trials_dir)
        trial_config.output.create_timestamp_dir = False
        trial_config.robustness_boundary = None

        for pert in trial_config.perturbations:
            if pert.name != self.target_perturbation.name:
                pert.enabled = False
                continue

            pert.enabled = True
            pert.parameters = dict(pert.parameters)

            if self._target_nested_module_index is None:
                pert.parameters[self.boundary_cfg.parameter] = parameter_value
                continue

            modules = pert.parameters.get("modules")
            if not isinstance(modules, list):
                raise ValueError(
                    f"Composite perturbation '{pert.name}' is missing a valid modules list."
                )

            nested_idx = self._target_nested_module_index
            if nested_idx >= len(modules):
                raise ValueError(
                    f"Composite perturbation '{pert.name}' no longer has nested module index {nested_idx}."
                )

            nested_cfg = modules[nested_idx]
            if not isinstance(nested_cfg, dict):
                raise ValueError(
                    f"Composite nested module at index {nested_idx} in '{pert.name}' must be a mapping."
                )
            if nested_cfg.get("type") != self.boundary_cfg.module:
                raise ValueError(
                    f"Composite nested module at index {nested_idx} in '{pert.name}' "
                    f"is type '{nested_cfg.get('type')}', expected '{self.boundary_cfg.module}'."
                )

            nested_cfg = dict(nested_cfg)
            nested_params = dict(nested_cfg.get("parameters", {}))
            nested_params[self.boundary_cfg.parameter] = parameter_value
            nested_cfg["parameters"] = nested_params

            modules = list(modules)
            modules[nested_idx] = nested_cfg
            pert.parameters["modules"] = modules

        return trial_config

    def _run_trial(self, search_value: float | int, label: str) -> BoundaryTrialResult:
        parameter_value = format_trial_value(self.param_spec, search_value)
        parameter_value = apply_canonicalize(self.param_spec, parameter_value)
        trial_output_dir = self.trials_dir / label
        trial_config_path = self.configs_dir / f"{label}.yaml"

        if trial_output_dir.exists():
            shutil.rmtree(trial_output_dir, ignore_errors=True)

        trial_config = self._build_trial_config(label, parameter_value)
        save_config(trial_config, trial_config_path)

        logger.info(
            "Trial %s: %s.%s=%r",
            label,
            self.boundary_cfg.module,
            self.boundary_cfg.parameter,
            parameter_value,
        )

        metrics_summary_path = (
            trial_output_dir
            / "slam_results"
            / self.slam_algorithm
            / "metrics"
            / "summary.json"
        )

        try:
            run_pipeline = Pipeline(trial_config)
            run_pipeline.setup()
            run_pipeline.run()

            eval_pipeline = EvaluationPipeline(
                config_path=trial_config_path,
                slam_algorithm=self.slam_algorithm,
                slam_config_path=self.slam_config_path,
                compute_metrics=True,
                skip_slam=False,
                num_runs=self.num_runs,
                paper_mode=self.paper_mode,
            )
            trajectories = eval_pipeline.run()

            trial_eval = self._classify_trial(
                trajectories=trajectories,
                metrics_summary_path=metrics_summary_path,
            )
            return BoundaryTrialResult(
                label=label,
                search_value=search_value,
                parameter_value=parameter_value,
                passed=trial_eval["passed"],
                failed=trial_eval["failed"],
                tracking_failure=trial_eval["tracking_failure"],
                ate_rmse=trial_eval["ate_rmse"],
                reason=trial_eval["reason"],
                trial_config_path=trial_config_path,
                trial_output_dir=trial_output_dir,
                metrics_summary_path=metrics_summary_path if metrics_summary_path.exists() else None,
            )
        except Exception as exc:
            logger.warning("Trial %s failed with exception: %s", label, exc)
            return BoundaryTrialResult(
                label=label,
                search_value=search_value,
                parameter_value=parameter_value,
                passed=False,
                failed=True,
                tracking_failure=True,
                ate_rmse=None,
                reason="trial_exception",
                trial_config_path=trial_config_path,
                trial_output_dir=trial_output_dir,
                metrics_summary_path=metrics_summary_path if metrics_summary_path.exists() else None,
                error=str(exc),
            )

    def _classify_trial(
        self,
        trajectories: Dict[str, Path],
        metrics_summary_path: Path,
    ) -> Dict[str, Any]:
        expected = {
            f"{self.target_perturbation.name}_run_{run_id}"
            for run_id in range(self.num_runs)
        }
        produced = {name for name in trajectories if name in expected}
        missing = sorted(expected - produced)
        tracking_failure = bool(missing)

        ate_rmse: Optional[float] = None
        tracking_mean: Optional[float] = None
        if metrics_summary_path.exists():
            with open(metrics_summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            module_stats = (
                summary.get("perturbed_modules", {}).get(self.target_perturbation.name)
            )
            if module_stats:
                # Prefer ate_rmse naming, with backward compatibility for ape_rmse.
                ate_stats = module_stats.get("ate_rmse")
                if not isinstance(ate_stats, dict):
                    ate_stats = module_stats.get("ape_rmse")
                if isinstance(ate_stats, dict):
                    ate_rmse = ate_stats.get("mean")
                tracking_stats = module_stats.get("tracking_completeness")
                if isinstance(tracking_stats, dict):
                    tracking_mean = tracking_stats.get("mean")

        if tracking_mean is not None and tracking_mean < 100.0:
            tracking_failure = True

        threshold = float(self.boundary_cfg.ate_rmse_fail)
        ate_failure = (ate_rmse is None) or (ate_rmse > threshold)
        tracking_failure_is_fatal = bool(self.boundary_cfg.fail_on_tracking_failure)
        failed = (tracking_failure and tracking_failure_is_fatal) or ate_failure
        passed = not failed

        reasons: List[str] = []
        if tracking_failure:
            reasons.append(
                "tracking_failure"
                if tracking_failure_is_fatal
                else "tracking_failure_ignored"
            )
        if missing:
            reasons.append(f"missing_trajectories:{','.join(missing)}")
        if ate_rmse is None:
            reasons.append("ate_unavailable")
        elif ate_rmse > threshold:
            reasons.append(f"ate_above_threshold:{ate_rmse:.6f}>{threshold:.6f}")
        else:
            reasons.append("ate_within_threshold")
        if passed:
            reasons.append("pass")

        return {
            "passed": passed,
            "failed": failed,
            "tracking_failure": tracking_failure,
            "ate_rmse": ate_rmse,
            "reason": ";".join(reasons),
        }
