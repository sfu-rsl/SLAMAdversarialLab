"""PodmanLoadController behavior against a stateful fake podman.

House pattern (test_runtime_stress_podman_controller.py): monkeypatch
subprocess.run at the controller's module path with a fake that records
command lists and reflects mutations, so verification readbacks pass or
fail deliberately.
"""

import json

import pytest

from slamadversariallab.runtime_stress.load_controller import (
    CPU_METHOD,
    LOAD_LABEL,
    STRESS_NG_IMAGE,
    PodmanLoadController,
)
from slamadversariallab.runtime_stress.models import (
    LoadControl,
    LoadFence,
    LoadInContainer,
    RuntimeStressControls,
)


class _Result:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakePodman:
    """Stateful podman fake: containers spawn Running with fenced NanoCpus."""

    def __init__(self):
        self.calls = []
        self.containers = {}  # name -> {"running": bool, "nano": int, "workers": int}
        self.fail_image_exists = set()
        self.top_worker_deficit = 0
        self.drop_shares = False  # simulate podman silently ignoring --cpu-shares
        self.burns = {}  # container -> {n, marker}  (in-container workers)
        self.burn_deficit = 0  # top shows this many fewer worker lines than spawned
        self.no_static_binary = False  # simulate missing /sal/stress-ng mount

    def __call__(self, cmd, capture_output=True, text=True, timeout=None):
        self.calls.append(list(cmd))
        prog, verb = cmd[0], cmd[1]
        assert prog == "podman"
        if verb == "image" and cmd[2] == "exists":
            return _Result(1 if cmd[3] in self.fail_image_exists else 0)
        if verb == "container" and cmd[2] == "exists":
            # The SLAM target "exists" as soon as prepare polls for it.
            return _Result(0)
        if verb == "image" and cmd[2] == "inspect":
            return _Result(0, stdout="sha256:feedface\n")
        if verb == "run":
            name = cmd[cmd.index("--name") + 1]
            nano = 0
            if "--cpus" in cmd:
                nano = int(float(cmd[cmd.index("--cpus") + 1]) * 1e9)
            shares = 0
            if "--cpu-shares" in cmd and not self.drop_shares:
                shares = int(cmd[cmd.index("--cpu-shares") + 1])
            workers = 0
            for flag in ("--cpu", "--stream", "--vm"):
                if flag in cmd:
                    workers += int(cmd[cmd.index(flag) + 1])
            self.containers[name] = {
                "running": True, "nano": nano, "shares": shares, "workers": workers
            }
            return _Result(0, stdout="deadbeef\n")
        if verb == "inspect":
            name = cmd[-1]
            state = self.containers.get(name)
            if state is None or not state["running"]:
                return _Result(1, stderr="no such container")
            if "{{.State.Running}}" in cmd:
                return _Result(0, stdout="true\n")
            if "{{.HostConfig.NanoCpus}}" in cmd:
                return _Result(0, stdout=f"{state['nano']}\n")
            if "{{.HostConfig.CpuShares}}" in cmd:
                return _Result(0, stdout=f"{state['shares']}\n")
            return _Result(0, stdout="{}\n")
        if verb == "exec":
            # podman exec [-d] <ctr> <cmd...>: test -x probe, watcher spawn,
            # or sentinel touch.
            rest = cmd[2:]
            if rest and rest[0] == "-d":
                rest = rest[1:]
            ctr = rest[0]
            body = " ".join(str(x) for x in rest[1:])
            if "test -x" in body or (len(rest) >= 2 and rest[1] == "test"):
                return _Result(1 if self.no_static_binary else 0)
            if "touch" in body:
                self.burns.pop(ctr, None)  # sentinel touched -> watcher kills
            elif "/sal/stress-ng" in body:
                import re
                total = sum(int(m) for m in re.findall(
                    r"--(?:cpu|stream|vm) (\d+)", body))
                self.burns[ctr] = {"n": total, "marker": "/sal/stress-ng"}
            return _Result(0)
        if verb == "top":
            name = cmd[2]
            state = self.containers.get(name)
            burn = self.burns.get(name)
            if state is None and burn is None:
                return _Result(1, stderr="no such container")
            lines = ["ARGS"]
            if state is not None:
                lines += [f"stress-ng worker {i}"
                          for i in range(1 + state["workers"] - self.top_worker_deficit)]
            if burn is not None:
                # supervisor + workers, each argv contains the binary path
                for i in range(max(0, 1 + burn["n"] - self.burn_deficit)):
                    lines.append(f"/sal/stress-ng --cpu ... [{i}]")
            return _Result(0, stdout="\n".join(lines) + "\n")
        if verb == "rm":
            name = cmd[-1]
            self.containers.pop(name, None)
            return _Result(0)
        if verb == "ps":
            return _Result(0, stdout="")
        if verb == "logs":
            return _Result(0, stdout="fake logs")
        raise AssertionError(f"unexpected podman verb: {cmd}")


@pytest.fixture()
def fake(monkeypatch):
    fake = _FakePodman()
    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.load_controller.subprocess.run", fake
    )
    return fake


def _prepared(fake, specs):
    ctl = PodmanLoadController(load_specs=specs)
    ctl.prepare(
        object(),
        target_kind="podman_container",
        target_metadata={"container_name": "slam-target-abc"},
    )
    return ctl


CPU_LOAD = LoadControl(cpu_workers=20, fence=LoadFence(cpus=4.0))


def test_rejects_non_podman_target(fake):
    ctl = PodmanLoadController(load_specs=[CPU_LOAD])
    with pytest.raises(RuntimeError, match="podman_container"):
        ctl.prepare(object(), target_kind="host_process_group", target_metadata={})


def test_prepare_fails_loud_on_missing_image(fake):
    fake.fail_image_exists.add(STRESS_NG_IMAGE)
    ctl = PodmanLoadController(load_specs=[CPU_LOAD])
    with pytest.raises(RuntimeError, match="not present"):
        ctl.prepare(
            object(),
            target_kind="podman_container",
            target_metadata={"container_name": "slam-target-abc"},
        )


def test_apply_spawns_fenced_labeled_stress_ng(fake):
    ctl = _prepared(fake, [CPU_LOAD])
    ctl.apply(RuntimeStressControls(load=CPU_LOAD))
    run_cmd = next(c for c in fake.calls if c[1] == "run")
    joined = " ".join(run_cmd)
    assert "--cpus 4" in joined
    assert f"--label {LOAD_LABEL}=1" in joined
    assert "--rm" in run_cmd
    assert STRESS_NG_IMAGE in run_cmd
    assert "--cpu 20" in joined
    assert f"--cpu-method {CPU_METHOD}" in joined


def test_weight_mode_uncapped_high_priority(fake):
    """Weight mode: --cpu-shares set, --cpus (quota) absent (uncapped)."""
    load = LoadControl(cpu_workers=20, fence=LoadFence(cpu_shares=131072))
    ctl = _prepared(fake, [load])
    ctl.apply(RuntimeStressControls(load=load))
    run_cmd = next(c for c in fake.calls if c[1] == "run")
    joined = " ".join(run_cmd)
    assert "--cpu-shares 131072" in joined
    assert "--cpus" not in run_cmd  # uncapped
    assert "--cpu 20" in joined
    # summary records the weight
    (summary,) = ctl.antagonist_summaries()
    assert summary["fence"]["cpu_shares"] == 131072
    assert summary["fence"]["cpus"] is None


def test_weight_mode_verify_fails_loud_on_dropped_shares(fake):
    fake.drop_shares = True  # podman "silently ignores" --cpu-shares
    load = LoadControl(cpu_workers=20, fence=LoadFence(cpu_shares=131072))
    ctl = _prepared(fake, [load])
    with pytest.raises(RuntimeError, match="weight not applied"):
        ctl.apply(RuntimeStressControls(load=load))


def test_apply_verifies_worker_count_fail_loud(fake):
    ctl = _prepared(fake, [CPU_LOAD])
    fake.top_worker_deficit = 3  # workers silently missing
    with pytest.raises(RuntimeError, match="under-contended"):
        ctl.apply(RuntimeStressControls(load=CPU_LOAD))


def test_apply_none_tears_down(fake):
    ctl = _prepared(fake, [CPU_LOAD])
    ctl.apply(RuntimeStressControls(load=CPU_LOAD))
    assert any(n.startswith("sal-load-sng-") for n in fake.containers)
    ctl.apply(RuntimeStressControls())  # phase without load
    assert not any(n.startswith("sal-load-sng-") for n in fake.containers)


def test_release_and_cleanup_remove_tracked_containers(fake):
    ctl = _prepared(fake, [CPU_LOAD])
    ctl.apply(RuntimeStressControls(load=CPU_LOAD))
    ctl.release()
    ctl.cleanup()
    assert not fake.containers
    rm_targets = [c[-1] for c in fake.calls if c[1] == "rm"]
    assert all(t != "slam-target-abc" for t in rm_targets)


def test_slam_container_never_torn_down(fake):
    ctl = _prepared(fake, [CPU_LOAD])
    with pytest.raises(RuntimeError, match="BUG"):
        ctl._rm_container("slam-target-abc")


def test_stream_and_vm_args(fake):
    load = LoadControl(
        stream_workers=4,
        vm_workers=2,
        vm_bytes_mb=2048,
        fence=LoadFence(cpus=4.0, memory_mb=8192),
    )
    ctl = _prepared(fake, [load])
    ctl.apply(RuntimeStressControls(load=load))
    joined = " ".join(next(c for c in fake.calls if c[1] == "run"))
    assert "--stream 4" in joined
    assert "--vm 2 --vm-bytes 2048M" in joined
    assert "--memory 8192m" in joined


def test_prewarm_gpu_spawns_before_target_and_prepare_is_idempotent(fake, monkeypatch):
    from slamadversariallab.runtime_stress.models import LoadGpuAntagonist

    monkeypatch.setattr(
        "slamadversariallab.runtime_stress.load_controller."
        "PodmanLoadController._read_json",
        staticmethod(lambda path: {"state": "ready", "epoch": 0}),
    )
    gpu_load = LoadControl(gpu=LoadGpuAntagonist(vram_mb=1024, matmul_n=2048, duty_cycle=0.5))
    ctl = PodmanLoadController(load_specs=[gpu_load])

    assert ctl.prewarm_gpu() is True
    gpu_runs = [c for c in fake.calls if c[1] == "run" and "sal-load-gpu-" in " ".join(c)]
    assert len(gpu_runs) == 1
    # No target existed yet: prewarm must not have touched `podman container exists`.
    assert not any(c[1] == "container" and c[2] == "exists" for c in fake.calls)

    # prepare() after prewarm must not spawn a second GPU antagonist.
    ctl.prepare(
        object(),
        target_kind="podman_container",
        target_metadata={"container_name": "slam-target-abc"},
    )
    gpu_runs = [c for c in fake.calls if c[1] == "run" and "sal-load-gpu-" in " ".join(c)]
    assert len(gpu_runs) == 1


def test_prewarm_gpu_noop_without_gpu_spec(fake):
    ctl = PodmanLoadController(load_specs=[CPU_LOAD])
    assert ctl.prewarm_gpu() is False
    assert not any(c[1] == "run" for c in fake.calls)


def _apply_capturing_warnings(caplog, ctl, controls_list):
    """Run apply() calls capturing package-logger warnings.

    The package logger sets propagate=False (src/utils/logging.py), so records
    never reach caplog's root handler; flip it for the duration. House pattern
    from test_runtime_stress_hami_controller.py.
    """
    import logging
    caplog.set_level(logging.WARNING)
    package_logger = logging.getLogger("slamadversariallab")
    prior = package_logger.propagate
    package_logger.propagate = True
    try:
        for controls in controls_list:
            ctl.apply(controls)
    finally:
        package_logger.propagate = prior
    # De-duplicate by record IDENTITY. One `logger.warning()` call creates one
    # LogRecord, but caplog can capture that same object more than once: pytest
    # 9.1 changed how it handles a propagating logger, and this test counted 2
    # where 9.0 counted 1, against a controller that had warned exactly once.
    # Counting distinct records measures emissions, which is the claim; counting
    # captures measures the test framework.
    seen, out = set(), []
    for r in caplog.records:
        if "DEPRECATED stressor" in r.message and id(r) not in seen:
            seen.add(id(r))
            out.append(r)
    return out


def test_sibling_path_warns_deprecated_once(fake, caplog):
    """The sibling stress-ng antagonist is deprecated (use in_container), but
    still runs so C8/C9/C11/C12/C14/C15 stay reproducible."""
    ctl = _prepared(fake, [CPU_LOAD])
    warns = _apply_capturing_warnings(
        caplog, ctl,
        [RuntimeStressControls(load=CPU_LOAD), RuntimeStressControls(load=CPU_LOAD)],
    )
    assert len(warns) == 1, "deprecation should warn once per run, not per phase"
    assert "in_container" in warns[0].message
    # still functional: the antagonist really spawned
    assert any(n.startswith("sal-load-sng-") for n in fake.containers)


def test_supported_paths_do_not_warn(fake, caplog):
    """in_container and the GPU antagonist are NOT deprecated."""
    load = LoadControl(in_container=LoadInContainer(cpu_workers=50))
    ctl = _prepared(fake, [load])
    warns = _apply_capturing_warnings(caplog, ctl, [RuntimeStressControls(load=load)])
    assert not warns


def test_in_container_stress_ng_exec_and_teardown(fake):
    load = LoadControl(in_container=LoadInContainer(cpu_workers=200, stream_workers=4))
    ctl = _prepared(fake, [load])
    ctl.apply(RuntimeStressControls(load=load))
    execs = [c for c in fake.calls if c[1] == "exec"]
    assert execs, "expected a podman exec"
    spawn = next(c for c in execs if "--cpu 200" in " ".join(str(x) for x in c))
    joined = " ".join(str(x) for x in spawn)
    assert "slam-target-abc" in spawn          # into the SLAM container
    assert "--cpu 200" in joined and f"--cpu-method {CPU_METHOD}" in joined
    assert "--stream 4" in joined
    assert fake.burns.get("slam-target-abc", {}).get("n") == 204
    summary = next(s for s in ctl.antagonist_summaries() if s["kind"] == "in_container")
    assert summary["cpu_workers"] == 200 and summary["stream_workers"] == 4
    assert summary["container_name"] == "slam-target-abc"
    # phase without load -> sentinel touched, workers gone
    ctl.apply(RuntimeStressControls())
    assert "slam-target-abc" not in fake.burns


def test_in_container_fails_loud_without_static_binary(fake):
    fake.no_static_binary = True  # mount missing / binary not extracted
    load = LoadControl(in_container=LoadInContainer(cpu_workers=200))
    ctl = _prepared(fake, [load])
    with pytest.raises(RuntimeError, match="extract_static.sh"):
        ctl.apply(RuntimeStressControls(load=load))


def test_in_container_verify_fails_loud_on_missing_workers(fake):
    fake.burn_deficit = 50  # 50 of the requested workers never show up
    load = LoadControl(in_container=LoadInContainer(cpu_workers=200))
    ctl = _prepared(fake, [load])
    with pytest.raises(RuntimeError, match="under-contended"):
        ctl.apply(RuntimeStressControls(load=load))


def test_active_container_names_track_lifecycle(fake):
    ctl = _prepared(fake, [CPU_LOAD])
    assert ctl.active_container_names() == []
    ctl.apply(RuntimeStressControls(load=CPU_LOAD))
    names = ctl.active_container_names()
    assert len(names) == 1 and names[0].startswith("sal-load-sng-")
    ctl.apply(RuntimeStressControls())  # phase without load -> teardown
    assert ctl.active_container_names() == []


def test_summaries_report_spec_and_digest(fake):
    ctl = _prepared(fake, [CPU_LOAD])
    ctl.apply(RuntimeStressControls(load=CPU_LOAD))
    (summary,) = ctl.antagonist_summaries()
    assert summary["kind"] == "stress_ng"
    assert summary["cpu_workers"] == 20
    assert summary["fence"]["cpus"] == 4.0
    assert summary["image_digest"] == "sha256:feedface"
