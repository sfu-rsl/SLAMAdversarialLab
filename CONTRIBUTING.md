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

## Tests

Run the narrowest relevant checks first.

Examples:

```bash
pytest -q tests/modules
pytest -q tests/datasets
pytest -q tests/core/test_cli_list_modules.py
python -m slamadverseriallab run configs/slamadverseriallab/other/baseline_tum_desk.yaml --dry-run
```

## Dependencies and Submodules

Many integrations under `deps/` are tracked as submodules or maintained forks.

- If you change a forked dependency, make the change in the fork first.
- Update the parent repo submodule pointer only after the dependency commit is pushed.
- Keep `.gitmodules` aligned with the intended canonical remote.
- Do not flatten or replace the dependency layout casually.
