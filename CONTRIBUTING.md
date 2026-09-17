# Contributing

## Scope

Contributions should stay focused on the framework itself:

- dataset adapters
- perturbation modules
- SLAM and VO integrations
- evaluation and robustness-boundary workflows
- tests and documentation for those areas

Do not commit local datasets, checkpoints, `results/`, or other generated heavyweight artifacts.

## Development Expectations

- Keep changes scoped to one problem or feature.
- Treat the code in `src/` and the config/schema contracts as the source of truth.
- Add or update tests when behavior changes.
- Keep documentation aligned with actual CLI and config behavior.
- Keep the README's "Evaluate Under Runtime Stress" section current when you
  change anything under `src/runtime_stress/`, `src/algorithms/` or `src/config/`.

## Reporting a runtime-stress run

Read the run's `slam_output.log` and report what it says alongside the metrics.
An exit code, a pose count and an ATE are each consistent with a run that
tracked the whole sequence and one that stopped early, so judge the outcome from
the log.

For the ORB-SLAM3 family (`orbslam3`, `orbslam3i`, `nitroslam`) the markers are:

- `Active map reset recieved` -- the system discarded the map. Any occurrence
  means the run failed, whatever its pose count or ATE.
- `Fail to track local map!` counts. One followed by `Relocalized!!` with no
  reset is a recovered stumble.
- `N Frames set to lost`, `New Map created` counts, the final `Map N has K KFs`.

Inertial variants additionally record reset causes (`IMU is not or recently
initialized`, `Timestamp jump`) and `VIBA 1/2`, which marks IMU initialisation
completing. Vision-only `orbslam3` has none of these.

Identify the equivalent markers for any other backend before judging its runs.

## Tests

Install the package in editable mode first (`pip install -e .`), so the tests
resolve `deps/` and `configs/` against your checkout.

Run the narrowest relevant checks first.

A fresh clone reports skips, not failures. Some tests assert against a recorded
campaign under `results/` or a dataset under `datasets/`, and each skips with a
reason naming what it wants. A FAILURE in a fresh clone is a real defect.

```bash
pytest -q tests/modules
pytest -q tests/datasets
pytest -q tests/core/test_cli_list_modules.py
python -m slamadversariallab run configs/slamadversariallab/other/baseline_tum_desk.yaml --dry-run
```

## Dependencies and Submodules

Many integrations under `deps/` are tracked as submodules or maintained forks.

- If you change a forked dependency, make the change in the fork first.
- Update the parent repo submodule pointer only after the dependency commit is pushed.
- Keep `.gitmodules` aligned with the intended canonical remote.
- Do not flatten or replace the dependency layout casually.
