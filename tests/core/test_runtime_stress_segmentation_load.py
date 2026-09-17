"""The SAM 3 co-tenant load kind: config surface and compile path.

These cover the parts that fail SILENTLY if they regress. A segmentation block
that parses but resolves to nothing produces a cell that runs uncontended and
still reports healthy, which is the one failure this axis cannot survive.
"""
from __future__ import annotations

import pytest

from src.config.schema import LoadSegmentationConfig


class TestSegmentationSchema:
    def test_source_dataset_is_accepted(self):
        LoadSegmentationConfig(source="dataset").validate()

    def test_explicit_frames_dir_is_accepted(self):
        LoadSegmentationConfig(frames_dir="/data/seq").validate()

    def test_neither_source_nor_frames_dir_is_refused(self):
        # Refusing beats guessing: a default would silently segment the wrong
        # images, and the run would look contended either way.
        with pytest.raises(ValueError, match="source: dataset or an explicit"):
            LoadSegmentationConfig().validate()

    def test_both_source_and_frames_dir_is_refused(self):
        # They are alternatives; accepting both hides which one took effect.
        with pytest.raises(ValueError, match="BOTH source and"):
            LoadSegmentationConfig(source="dataset", frames_dir="/data").validate()

    def test_unknown_source_is_refused(self):
        with pytest.raises(ValueError, match="must be 'dataset'"):
            LoadSegmentationConfig(source="slam").validate()

    @pytest.mark.parametrize("prompt", ["", "   "])
    def test_empty_prompt_is_refused(self, prompt):
        with pytest.raises(ValueError, match="non-empty string"):
            LoadSegmentationConfig(frames_dir="/d", prompt=prompt).validate()

    @pytest.mark.parametrize("bad", [0, -5, True, "10"])
    def test_bad_max_frames_is_refused(self, bad):
        with pytest.raises(ValueError, match="positive integer"):
            LoadSegmentationConfig(frames_dir="/d", max_frames=bad).validate()


class TestSegmentationSatisfiesLoadBlock:
    """A load block holding ONLY segmentation is a complete antagonist request."""

    def test_segmentation_alone_is_a_valid_load(self):
        from src.config.schema import LoadControlConfig

        LoadControlConfig(
            segmentation=LoadSegmentationConfig(source="dataset")
        ).validate()

    def test_empty_load_block_still_refused(self):
        from src.config.schema import LoadControlConfig

        with pytest.raises(ValueError, match="at least one antagonist"):
            LoadControlConfig().validate()


class TestSourceResolution:
    """`source: dataset` must become a concrete path at PARSE time.

    Resolving late would mean an unresolvable reference surfaces half-way into a
    campaign instead of when the config is read.
    """

    def _config(self, tmp_path, seg_block):
        import textwrap
        import yaml

        seq = tmp_path / "seq"
        seq.mkdir()
        cfg = {
            "experiment": {"name": "t", "seed": 1},
            "dataset": {"type": "tum", "path": str(seq), "max_frames": 123},
            "perturbations": [{"name": "baseline", "type": "none", "enabled": True}],
            "runtime_stress": {
                "enabled": True,
                "container_runtime": "podman",
                "scenarios": [{
                    "name": "s", "enabled": True,
                    "realtime": {"target_fps": 30, "warmup_frames": 0,
                                 "queue_size": 2, "drop_policy": "drop_oldest"},
                    "phases": [{"name": "stress", "duration_s": 5,
                                "controls": {"load": {"segmentation": seg_block}}}],
                }],
            },
        }
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    def test_source_dataset_resolves_to_the_dataset_path(self, tmp_path):
        from src.config.parser import load_config

        cfg = load_config(str(self._config(tmp_path, {"source": "dataset"})))
        seg = cfg.runtime_stress.scenarios[0].phases[0].controls.load.segmentation
        assert seg.frames_dir == str(tmp_path / "seq")
        # The frame BUDGET is inherited too, so "same dataset" means the same
        # span of the sequence rather than merely the same directory.
        assert seg.max_frames == 123
        # Consumed, so nothing downstream can re-resolve it differently.
        assert seg.source is None

    def test_explicit_max_frames_overrides_the_dataset(self, tmp_path):
        from src.config.parser import load_config

        cfg = load_config(str(self._config(
            tmp_path, {"source": "dataset", "max_frames": 7})))
        seg = cfg.runtime_stress.scenarios[0].phases[0].controls.load.segmentation
        assert seg.max_frames == 7

    def test_explicit_frames_dir_is_left_alone(self, tmp_path):
        from src.config.parser import load_config

        other = tmp_path / "other"
        other.mkdir()
        cfg = load_config(str(self._config(tmp_path, {"frames_dir": str(other)})))
        seg = cfg.runtime_stress.scenarios[0].phases[0].controls.load.segmentation
        assert seg.frames_dir == str(other)


class TestCompile:
    def test_compiles_into_the_phase_controls(self, tmp_path):
        import yaml
        from src.config.parser import load_config
        from src.runtime_stress.models import compile_runtime_stress_request

        seq = tmp_path / "seq"
        seq.mkdir()
        cfg = {
            "experiment": {"name": "t", "seed": 1},
            "dataset": {"type": "tum", "path": str(seq), "max_frames": 50},
            "perturbations": [{"name": "baseline", "type": "none", "enabled": True}],
            "runtime_stress": {
                "enabled": True, "container_runtime": "podman",
                "scenarios": [{
                    "name": "s", "enabled": True,
                    "realtime": {"target_fps": 30, "warmup_frames": 0,
                                 "queue_size": 2, "drop_policy": "drop_oldest"},
                    "phases": [
                        {"name": "warmup", "duration_s": 2},
                        {"name": "stress", "duration_s": 5, "controls": {
                            "load": {"segmentation": {
                                "source": "dataset", "prompt": "chair"}}}},
                    ],
                }],
            },
        }
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump(cfg))
        parsed = load_config(str(path))
        req = compile_runtime_stress_request(
            parsed.runtime_stress, parsed.runtime_stress.scenarios[0])

        # The unstressed warmup phase must carry NO load, or the "grace window"
        # is not actually unstressed.
        assert req.phases[0].controls.load is None
        seg = req.phases[1].controls.load.segmentation
        assert seg.frames_dir == str(seq)
        assert seg.prompt == "chair"
        assert seg.max_frames == 50
        assert seg.conda_env == "sam3"


class TestCoTenantDoesNotOutliveTheRun:
    """The co-tenant MUST die when the SLAM does.

    This is the failure the axis cannot survive quietly. A leaked SAM 3 process
    holds ~95% of the GPU, so it would contend with every later cell in the
    campaign while those cells record NO antagonist -- an invisible confound in
    exactly the direction that manufactures a contention result.

    It leaked once for real: the spawn went through `conda run`, which is a
    wrapper, so Popen returned the wrapper's pid and terminating it orphaned the
    python worker underneath. Measured 4 processes before teardown and 3 after.
    """

    def test_spawn_does_not_use_a_wrapper_process(self):
        # `conda run` hides the real pid. The env's python is invoked directly
        # so the pid we hold is the pid we can kill.
        import inspect
        from src.runtime_stress.load_controller import PodmanLoadController

        src = inspect.getsource(PodmanLoadController._spawn_segmentation)
        assert '"conda", "run"' not in src, (
            "spawning through `conda run` returns the WRAPPER's pid; killing it "
            "orphans the worker. Invoke the env's python directly."
        )
        assert "env_python" in src
        assert "start_new_session=True" in src, (
            "the worker needs its own process group so teardown can signal it "
            "and anything it spawns"
        )

    def test_teardown_signals_the_process_group(self):
        import inspect
        from src.runtime_stress.load_controller import PodmanLoadController

        src = inspect.getsource(PodmanLoadController._teardown_segmentation)
        assert "killpg" in src, (
            "signalling only the group leader leaks any child the worker "
            "spawned; teardown must signal the whole group"
        )
        # And must escalate: a worker mid-CUDA-call can ignore SIGTERM.
        assert "SIGKILL" in src

    def test_release_tears_down_segmentation(self):
        # finalize() -> release() is the path taken when the SLAM exits, so the
        # co-tenant has to be in it.
        import inspect
        from src.runtime_stress.load_controller import PodmanLoadController

        assert "_teardown_segmentation" in inspect.getsource(
            PodmanLoadController.release)


class TestNonContainerAntagonistLiveness:
    """A crowd or co-tenant that dies mid-run must be VISIBLE in the trace.

    `active_container_names()` covers only antagonists that own a container. The
    in-container crowd runs inside the SLAM's cgroup and the segmentation
    co-tenant is a host process, so both were verified once at spawn and never
    again. That left the one failure this axis cannot survive undetectable: a
    crowd dying at minute two leaves the SLAM uncontended for the rest of the
    cell, which then reports as contended and reads as "survived contention".
    """

    def _controller(self):
        from src.runtime_stress.load_controller import PodmanLoadController
        c = PodmanLoadController.__new__(PodmanLoadController)
        c._incontainer_active = False
        c._incontainer_last_spec = None
        c._target_container = None
        c._seg_proc = None
        c._seg_status_path = None
        c._liveness_cache = None
        c._liveness_checked_at = 0.0
        return c

    def test_empty_when_neither_kind_is_active(self):
        # Callers merge unconditionally, so "nothing to report" must be {} and
        # not None or a raise.
        assert self._controller().process_antagonist_liveness() == {}

    def test_reports_worker_count_for_the_in_container_crowd(self):
        from src.runtime_stress.models import LoadInContainer
        c = self._controller()
        c._incontainer_active = True
        c._target_container = "slam-x"
        c._incontainer_last_spec = LoadInContainer(cpu_workers=40)

        class _R:
            returncode = 0
            stdout = "41\n"      # 40 workers + supervisor
        c._run = staticmethod(lambda *a, **k: _R())

        out = c.process_antagonist_liveness()
        assert out["in_container"]["workers_alive"] == 41
        assert out["in_container"]["workers_expected"] == 40
        assert out["in_container"]["died"] is False

    def test_flags_a_crowd_that_died(self):
        from src.runtime_stress.models import LoadInContainer
        c = self._controller()
        c._incontainer_active = True
        c._target_container = "slam-x"
        c._incontainer_last_spec = LoadInContainer(cpu_workers=40)

        class _R:
            returncode = 0
            stdout = "0\n"      # every worker gone
        c._run = staticmethod(lambda *a, **k: _R())

        assert c.process_antagonist_liveness()["in_container"]["died"] is True

    def test_reports_a_dead_segmentation_co_tenant(self):
        c = self._controller()

        class _P:
            def poll(self):
                return 137      # killed
        c._seg_proc = _P()
        c._read_json = staticmethod(lambda p: {"state": "error", "frames_done": 12})

        seg = c.process_antagonist_liveness()["segmentation"]
        assert seg["alive"] is False
        assert seg["returncode"] == 137
        assert seg["frames_done"] == 12

    def test_result_is_cached_so_the_probe_does_not_compete_with_the_workload(self):
        # Counting PIDs costs a `podman exec` inside the SLAM's own container.
        # At the telemetry tick rate that exec would contend with the very thing
        # being measured, so repeated calls must reuse the cached answer.
        from src.runtime_stress.models import LoadInContainer
        c = self._controller()
        c._incontainer_active = True
        c._target_container = "slam-x"
        c._incontainer_last_spec = LoadInContainer(cpu_workers=8)

        calls = {"n": 0}

        class _R:
            returncode = 0
            stdout = "9\n"

        def _run(*a, **k):
            calls["n"] += 1
            return _R()
        c._run = staticmethod(_run)

        for _ in range(10):
            c.process_antagonist_liveness()
        assert calls["n"] == 1, "liveness probe must be throttled, not per-tick"


class TestApplyGuardReachesEveryLoadKind:
    """`apply()`'s early-return guard must list EVERY load kind.

    A kind missing from the guard is torn down and returned past before its own
    dispatch branch is reached. The phase then runs with no antagonist while the
    cell still reports as stressed, and NOTHING errors.

    This is not hypothetical: `segmentation` shipped with a dispatch branch and
    an unupdated guard. A full smoke run through the real pipeline completed
    cleanly, entered the stress phase on time, produced a trajectory, and
    recorded `load_antagonists: []` with `controller_error: None`. Only the
    liveness telemetry showed anything was wrong.
    """

    def _controls(self, **kw):
        from src.runtime_stress.models import LoadControl, RuntimeStressControls
        return RuntimeStressControls(load=LoadControl(**kw))

    def test_segmentation_alone_is_not_returned_past(self, monkeypatch):
        from src.runtime_stress.load_controller import PodmanLoadController
        from src.runtime_stress.models import LoadSegmentation

        c = PodmanLoadController.__new__(PodmanLoadController)
        c._gpu_container = None
        c._seg_proc = None
        c._seg_spec = None
        called = {"spawn": False}
        c._teardown_stress_ng = lambda: None
        c._idle_gpu = lambda: None
        c._teardown_in_container = lambda: None
        c._teardown_segmentation = lambda: None
        c._spec_has_stress_ng = staticmethod(lambda spec: False)

        def _spawn(spec):
            called["spawn"] = True
        c._spawn_segmentation = _spawn

        spec = LoadSegmentation(frames_dir="/d", prompt="person",
                                max_frames=0, conda_env="sam3")
        c.apply(self._controls(segmentation=spec))
        assert called["spawn"], (
            "apply() returned before reaching the segmentation branch; the "
            "phase would run uncontended while reporting as stressed"
        )

    def test_guard_names_every_kind(self):
        # Structural, so a future kind added with a dispatch branch but no guard
        # entry fails here rather than silently in a campaign.
        import inspect
        from src.runtime_stress.load_controller import PodmanLoadController

        src = inspect.getsource(PodmanLoadController.apply)
        # Slice the guard CONDITION itself, from `if load is None or (` to its
        # closing `):`. Splitting on the word "return" would cut at the first
        # occurrence in a comment, which is how this test first failed against
        # correct code.
        start = src.index("if load is None or (")
        guard = src[start:src.index("):", start)]
        for kind in ("gpu is None", "in_container is None", "segmentation is None"):
            assert kind in guard, f"apply()'s early-return guard omits {kind!r}"
