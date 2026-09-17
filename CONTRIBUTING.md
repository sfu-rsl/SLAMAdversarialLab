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

## Standing rules for runtime-stress work

Two rules bind anyone touching the runtime-stress framework or reporting its
results. They are cited from `src/runtime_stress/health.py` and from the report
tooling.

Some paths below name `tools/`, this project's research apparatus: the
campaign-config generators and the report tools that produce the paper's
figures. It is kept in the working repository and is NOT part of the public
distribution, because its files are bound to specific experiments and operate on
a `results/` tree that only exists where those campaigns were run. Nothing under
`src/` imports it, so the framework itself is unaffected.

### Framework doc sync

After any architectural change -- anything under `src/runtime_stress/`,
`src/algorithms/`, `src/config/` or `tools/gen_campaign_configs.py` -- evaluate
BEFORE committing whether these need updating, and update them in the same
commit when they do:

- `docs/fuzzy-slam/FUZZY_SLAM.md` -- the framework reference (stressor axes,
  controllers, deployment models, headline findings, what is not yet done).
- `docs/fuzzy-slam/EXPERIMENTS.md` -- severity ladders, conditions, and results.

  Both live in the working repository and are NOT part of the public
  distribution, so in a public checkout only the README applies.
- `README.md` -- the "Evaluate Under Runtime Stress" section, new findings
  included.

A pre-commit gate enforces the evaluation: commits staging architecture files
without any framework doc are blocked. Bypass with `SKIP_DOC_SYNC=1 git commit`
only as an explicit attestation that you evaluated and no doc update was needed,
never as a habit. Install once per clone with
`git config core.hooksPath tools/githooks`.

### Run-health verification

Before reporting ANY run's outcome, success or failure alike, read that run's
`slam_output.log` and report what it says alongside the metrics. A run is never
declared healthy from an exit code, a pose count or an ATE: exit 139 fires on
healthy and collapsed runs alike, and **ATE INVERTS past failure**, so a run that
died early scores at or better than baseline because it is scored over a shorter,
easier stretch of the route.

For the ORB-SLAM3 family (`orbslam3`, `orbslam3i`, `nitroslam`) the shared
markers are:

- `Active map reset recieved` -- the system gave up and discarded the map. Any
  occurrence means the run FAILED, whatever its pose count or ATE.
- `Fail to track local map!` counts. One followed by `Relocalized!!` with no
  reset is a recovered stumble, not a failure.
- `N Frames set to lost`, `New Map created` counts, the final `Map N has K KFs`.

Inertial variants additionally record reset causes (`IMU is not or recently
initialized`, `Timestamp jump`) and `VIBA 1/2`, which marks IMU initialisation
completing. Its absence in a stressed run means the bootstrap never finished.
Vision-only `orbslam3` has none of these.

Other systems need their equivalent markers identified before their runs are
judged. `tools/report/verdicts.py` encodes what is known.

## Publishing a release

The public repository receives a curated snapshot, not this repository's history.
Build the tree with `git archive`, which honours the `export-ignore` attributes
in `.gitattributes`:

```bash
git archive --format=tar HEAD | tar -x -C /path/to/public-tree
```

Two paths are marked `export-ignore` and are excluded automatically:

- `docs/fuzzy-slam/` -- the internal research record (experiment trackers,
  close-out notes, pending-decision logs), written for whoever is running the
  campaigns. The results that belong in public are in the paper and in README.
- The top-level ORB-SLAM3 runtime-stress handoff note, a status document.

Add to that list rather than relying on anyone remembering an exclusion. Check
the produced tree before publishing: it should carry no path that does not exist
in the public repository, and no working note addressed to this project's own
contributors.

## Tests

Run the narrowest relevant checks first.

A fresh clone reports skips, not failures. Some tests assert against a recorded
campaign under `results/` or a dataset under `datasets/`, and neither is
distributed. Those skip with a reason naming exactly what is missing. If you see
a FAILURE in a fresh clone, that is a real defect rather than absent data.

Examples:

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
