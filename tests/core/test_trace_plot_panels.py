"""Telemetry-chart panel selection and unit handling.

The trace mixes two CPU normalizations that must never share an axis:
``container_cpu_percent`` is percent-of-one-core (1890 means 18.9 cores) while
``system_cpu_percent`` is percent-of-the-whole-machine (never above 100). These
tests pin that separation, plus the rules deciding which panels exist, since a
wrong panel silently misreports what a run consumed.

Panel building is pure stdlib. A single end-to-end test renders a real figure
to guard the matplotlib path.
"""

import json

import pytest

from slamadversariallab.runtime_stress.trace_plot import (
    MissingReadInstant,
    antagonist_series,
    build_panels,
    cores_series,
    cpu_seconds_totals,
    phase_spans,
    plot_run,
    process_group_series,
    series,
    throttling_totals,
    walk_runs,
)


def _sample(elapsed, phase="stress", **fields):
    base = {
        "elapsed_s": elapsed,
        "phase": phase,
        "container_cpu_percent": 200.0,
        "system_cpu_percent": 10.0,
        "rss_bytes": 500_000_000,
        "memory_limit_bytes": 67_000_000_000,
    }
    base.update(fields)
    return base


def _titles(trace, shares_cgroup=False):
    return [p[0] for p in build_panels(trace, shares_cgroup)]


def test_container_cpu_is_converted_to_cores():
    """1890% of one core is 18.9 cores, matching the max_cores knob."""
    trace = [_sample(0.0, container_cpu_percent=1890.0)]
    panels = build_panels(trace)
    title, ylabel, ymax, lines = panels[0]
    assert ylabel == "cores busy"
    assert ymax is None  # cores are unbounded, unlike a percentage
    assert lines[0][1] == pytest.approx([18.9])


def test_cpu_uses_the_counter_not_the_lifetime_average():
    """podman reports CPU == AvgCPU, an average over the container's whole life.

    A container idling then burning 2 cores reads 18%, 47%, 67% ... creeping
    toward 200% and never arriving. Differencing cpu_time_ns must recover the
    true step instead of plotting that ramp.
    """
    samples = [
        # idle for two ticks, then exactly 2 cores/s of CPU per 1 s of wall time
        {"elapsed_s": 0.0, "cpu_time_at_s": 0.0, "cpu_time_ns": 0, "cpu_percent": 0.1},
        {"elapsed_s": 1.0, "cpu_time_at_s": 1.0, "cpu_time_ns": 0, "cpu_percent": 0.09},
        {"elapsed_s": 2.0, "cpu_time_at_s": 2.0, "cpu_time_ns": 2_000_000_000, "cpu_percent": 100.0},
        {"elapsed_s": 3.0, "cpu_time_at_s": 3.0, "cpu_time_ns": 4_000_000_000, "cpu_percent": 133.0},
        {"elapsed_s": 4.0, "cpu_time_at_s": 4.0, "cpu_time_ns": 6_000_000_000, "cpu_percent": 150.0},
    ]
    xs, ys, instantaneous = cores_series(samples)
    assert instantaneous is True
    assert ys == pytest.approx([0.0, 2.0, 2.0, 2.0])  # the step, not 1.0/1.33/1.5
    assert xs == [1.0, 2.0, 3.0, 4.0]


def test_cpu_falls_back_to_the_average_for_legacy_traces():
    """Traces predating the counter keep plotting, flagged as an average."""
    samples = [{"elapsed_s": 0.0, "cpu_percent": 150.0},
               {"elapsed_s": 1.0, "cpu_percent": 200.0}]
    xs, ys, instantaneous = cores_series(samples)
    assert instantaneous is False
    assert ys == pytest.approx([1.5, 2.0])


def test_legacy_cpu_panel_is_labelled_as_an_average():
    legacy = [_sample(t) for t in (0.0, 1.0)]  # no container_cpu_time_ns
    assert "running average" in build_panels(legacy)[0][0]
    current = [dict(_sample(t), container_cpu_time_ns=int(t * 2e9),
                    container_cpu_time_at_s=t) for t in (0.0, 1.0, 2.0)]
    assert "running average" not in build_panels(current)[0][0]


def test_rate_uses_the_counter_read_instant_not_the_sample_timestamp():
    """Collecting the rest of a sample can take seconds under contention.

    Charging that latency to the interval makes a steady 2-core load read as a
    dip. The counter's own read instant is what bounds the interval.
    """
    samples = [
        {"elapsed_s": 0.0, "cpu_time_at_s": 0.0, "cpu_time_ns": 0},
        # counter read at 1.0s, but the sample only finished being built at 8.0s
        {"elapsed_s": 8.0, "cpu_time_at_s": 1.0, "cpu_time_ns": 2_000_000_000},
        {"elapsed_s": 8.6, "cpu_time_at_s": 2.0, "cpu_time_ns": 4_000_000_000},
    ]
    xs, ys, _ = cores_series(samples)
    assert ys == pytest.approx([2.0, 2.0])  # steady, not 0.25 then 3.3
    assert xs == [1.0, 2.0]


def test_counter_reset_or_zero_interval_is_skipped():
    """A restarted container resets the counter; a negative delta is not use."""
    samples = [
        {"elapsed_s": 0.0, "cpu_time_at_s": 0.0, "cpu_time_ns": 5_000_000_000},
        {"elapsed_s": 1.0, "cpu_time_at_s": 1.0, "cpu_time_ns": 0},            # reset -> dropped
        {"elapsed_s": 1.0, "cpu_time_at_s": 1.0, "cpu_time_ns": 1_000_000_000},  # zero interval -> dropped
        {"elapsed_s": 2.0, "cpu_time_at_s": 2.0, "cpu_time_ns": 2_000_000_000},
    ]
    xs, ys, _ = cores_series(samples)
    assert xs == [2.0] and ys == pytest.approx([1.0])


def test_machine_cpu_stays_a_bounded_percentage():
    """The machine-wide series is a different unit and gets its own 0-100 panel."""
    trace = [_sample(0.0, system_cpu_percent=100.0)]
    panels = {p[0]: p for p in build_panels(trace)}
    _, ylabel, ymax, lines = panels["Processor use across the whole machine"]
    assert (ylabel, ymax) == ("percent", 100)
    assert lines[0][1] == [100.0]


def test_cpu_only_run_gets_no_gpu_panels():
    """A CPU-only SLAM reports flat-zero GPU fields; it must not get empty axes."""
    trace = [_sample(t, gpu_util_percent=0.0, gpu_mem_used_mb=0.0) for t in (0.0, 1.0)]
    assert not [t for t in _titles(trace) if "Graphics" in t]


def test_desktop_baseline_vram_is_not_charted():
    """An idle desktop holds a little VRAM; that is not this run's consumption."""
    trace = [_sample(t, gpu_util_percent=0.0, gpu_mem_used_mb=mb)
             for t, mb in ((0.0, 350.0), (1.0, 360.0))]
    assert not [t for t in _titles(trace) if "Graphics" in t]


def test_desktop_compositor_blip_is_not_charted():
    """A live desktop session blips a few percent; that is not the SLAM."""
    trace = [_sample(t, gpu_util_percent=u, gpu_mem_used_mb=340.0)
             for t, u in ((0.0, 0.0), (1.0, 2.0), (2.0, 0.0))]
    assert not [t for t in _titles(trace) if "Graphics" in t]


def test_real_gpu_workload_clears_the_noise_floor():
    trace = [_sample(t, gpu_util_percent=90.0, gpu_mem_used_mb=17000.0)
             for t in (0.0, 1.0)]
    assert len([t for t in _titles(trace) if "Graphics" in t]) == 2


def test_vram_held_without_utilisation_is_charted():
    """OKVIS2-X holds an idle CUDA context: VRAM moves, utilisation stays zero."""
    trace = [_sample(t, gpu_util_percent=0.0, gpu_mem_used_mb=mb)
             for t, mb in ((0.0, 350.0), (1.0, 620.0))]
    assert "Graphics memory, whole card" in _titles(trace)


def test_gpu_active_run_gets_both_gpu_panels():
    trace = [_sample(t, gpu_util_percent=100.0, gpu_mem_used_mb=17000.0,
                     gpu_mem_total_mb=24576.0) for t in (0.0, 1.0)]
    titles = _titles(trace)
    assert "Graphics card use, whole card" in titles
    assert "Graphics memory, whole card" in titles


def test_memory_limit_line_only_when_a_limit_was_imposed():
    """Uncapped runs report host total RAM; drawing it would flatten the panel."""
    uncapped = [_sample(t) for t in (0.0, 1.0)]
    mem = {p[0]: p for p in build_panels(uncapped)}["Memory use by the SLAM container"]
    assert [lb for _, _, lb, _, _ in mem[3]] == ["in use"]

    capped = [_sample(0.0, memory_limit_bytes=67_000_000_000),
              _sample(1.0, memory_limit_bytes=2_000_000_000)]
    mem = {p[0]: p for p in build_panels(capped)}["Memory use by the SLAM container"]
    assert "limit" in [lb for _, _, lb, _, _ in mem[3]]


def test_in_container_load_is_labelled_as_shared():
    """In-container competitors share the SLAM's cgroup, so the line covers both."""
    trace = [_sample(0.0)]
    assert build_panels(trace, shares_cgroup=False)[0][3][0][2] == "SLAM"
    assert build_panels(trace, shares_cgroup=True)[0][3][0][2] == "SLAM and its competitors"


def test_sibling_antagonist_gets_its_own_series():
    def _load(at, ns):
        return {"sal-load-sng-abc": {"cpu_time_ns": ns, "cpu_time_at_s": at,
                                     "cpu_percent": 1650.0}}
    trace = [
        _sample(0.0, container_cpu_time_ns=0, container_cpu_time_at_s=0.0,
                load_antagonists=_load(0.0, 0)),
        _sample(1.0, container_cpu_time_ns=int(2e9), container_cpu_time_at_s=1.0,
                load_antagonists=_load(1.0, int(16.5e9))),
    ]
    assert antagonist_series(trace) == {"sal-load-sng-abc": ([1.0], [16.5])}
    assert len(build_panels(trace)[0][3]) == 2  # SLAM plus the competitor


def test_competitor_line_is_dropped_when_only_a_lifetime_average_exists():
    """A lifetime average is not a rate, and must never share the SLAM's axis.

    Traces written before competitor counter reads were stamped land here: the
    series is omitted and the reason printed, rather than drawing an average
    beside an instantaneous line in the same panel.
    """
    trace = [
        _sample(0.0, load_antagonists={"sal-load-sng-abc": {"cpu_percent": 1650.0}}),
        _sample(1.0, load_antagonists={"sal-load-sng-abc": {"cpu_percent": 1600.0}}),
    ]
    assert antagonist_series(trace) == {}


def test_counter_without_its_read_instant_is_refused():
    """Substituting the sample timestamp yields a believable wrong rate.

    A steady 19.7-core load once read as 1.5 that way, so a trace carrying the
    counter without its read instant is refused rather than guessed at.
    """
    samples = [
        {"elapsed_s": 0.0, "cpu_time_ns": 0},
        {"elapsed_s": 8.0, "cpu_time_ns": 2_000_000_000},
    ]
    with pytest.raises(MissingReadInstant):
        cores_series(samples)


def test_missing_samples_are_skipped_not_zero_filled():
    """A null reading is absent data, not a consumption of zero."""
    trace = [_sample(0.0, rss_bytes=None), _sample(1.0, rss_bytes=1_000_000_000)]
    assert series(trace, "rss_bytes", 1e-9) == ([1.0], [1.0])


def test_phase_spans_prefer_recorded_events(tmp_path):
    """Event timestamps are exact; sample labels only bound the edge to a tick."""
    (tmp_path / "stress_events.json").write_text(json.dumps([
        {"kind": "attach", "elapsed_s": 0.0, "message": "attached"},
        {"kind": "phase_enter", "elapsed_s": 0.0, "message": "warmup"},
        {"kind": "phase_enter", "elapsed_s": 2.5049, "message": "stress"},
    ]))
    trace = [_sample(0.0, phase="warmup"), _sample(10.0)]
    assert phase_spans(str(tmp_path), trace) == [
        [0.0, 2.5049, "warmup"],
        [2.5049, 10.0, "stress"],
    ]


def test_phase_spans_fall_back_to_sample_labels(tmp_path):
    trace = [_sample(0.0, phase="warmup"), _sample(3.0), _sample(9.0)]
    assert phase_spans(str(tmp_path), trace) == [[0.0, 3.0, "warmup"], [3.0, 9.0, "stress"]]


def test_run_without_trace_is_skipped(tmp_path):
    assert plot_run(str(tmp_path), quiet=True) is False


def test_aborted_run_with_empty_trace_is_skipped(tmp_path):
    """Fail-loud drills abort before sampling; a zero-sample trace is correct."""
    (tmp_path / "stress_trace.json").write_text("[]")
    assert plot_run(str(tmp_path), quiet=True) is False


def test_walk_runs_finds_only_dirs_with_traces(tmp_path):
    (tmp_path / "with").mkdir()
    (tmp_path / "with" / "stress_trace.json").write_text("[]")
    (tmp_path / "without").mkdir()
    assert [p.split("/")[-1] for p in walk_runs(str(tmp_path))] == ["with"]


def test_chart_is_written_end_to_end(tmp_path):
    """Guards the matplotlib path the pure-logic tests above cannot reach."""
    trace = [_sample(t, phase="warmup" if t < 2 else "stress",
                     gpu_util_percent=90.0, gpu_mem_used_mb=17000.0,
                     gpu_mem_total_mb=24576.0,
                     load_antagonists={"sal-load-sng-abc": {"cpu_percent": 800.0}})
             for t in (0.0, 1.0, 2.0, 3.0)]
    (tmp_path / "stress_trace.json").write_text(json.dumps(trace))
    (tmp_path / "stress_summary.json").write_text(json.dumps({
        "deadline": {"drop_rate": 0.77},
        "load_antagonists": [{"kind": "stress_ng", "cpu_workers": 8}],
    }))
    assert plot_run(str(tmp_path), quiet=True) is True
    assert (tmp_path / "stress_timeline.png").stat().st_size > 0


def _pg_sample(t, slam_ns, load_ns, slam_delay_ns=0, interval_s=2.0, **extra):
    """A sample carrying the per-process walk output.

    The walk emits per-interval DELTAS, not running totals: processes enter and
    leave the cgroup, so a running total would fall when one exits.
    """
    return dict(_sample(t), container_cpu_time_ns=slam_ns + load_ns,
                container_cpu_time_at_s=t,
                process_groups={"at_s": t, "interval_s": interval_s,
                                "slam": {"cpu_time_ns": slam_ns,
                                         "run_delay_ns": slam_delay_ns,
                                         "procs": 1, "threads": 60},
                                "load": {"cpu_time_ns": load_ns, "run_delay_ns": 0,
                                         "procs": 100, "threads": 100}},
                **extra)


def test_process_walk_splits_a_saturated_cgroup():
    """The cgroup total is identical whether the SLAM gets a third or a tenth.

    Splitting by workload is the only way to show the SLAM's actual share, which
    is what explains a drop-rate difference between two runs that both pin the
    machine at 100%.
    """
    trace = [_pg_sample(0, 0, 0, interval_s=None),
             _pg_sample(2, int(4e9), int(34e9)), _pg_sample(4, int(4e9), int(34e9))]
    s = process_group_series(trace)
    assert s["slam"]["cores"][1] == pytest.approx([2.0, 2.0])
    assert s["load"]["cores"][1] == pytest.approx([17.0, 17.0])


def test_run_delay_is_reported_per_second_of_wall_time():
    """60 threads each waiting the whole time reads as 60 seconds waited per second."""
    trace = [_pg_sample(0, 0, 0, 0, interval_s=None),
             _pg_sample(2, int(2e9), 0, int(120e9))]
    assert process_group_series(trace)["slam"]["delay"][1] == pytest.approx([60.0])


def test_split_panels_appear_only_when_the_walk_ran():
    with_walk = [_pg_sample(0, 0, 0, 0), _pg_sample(2, int(4e9), int(34e9), int(20e9))]
    titles = _titles(with_walk)
    assert "Processor use by the SLAM itself" in titles
    assert "Time the SLAM spent waiting for a processor" in titles
    without = [_sample(t, container_cpu_time_ns=int(t * 2e9)) for t in (0.0, 1.0, 2.0)]
    assert not [t for t in _titles(without) if "by the SLAM itself" in t]


def test_slam_gets_its_own_axis_not_a_shared_one():
    """At 0.16 cores against a load taking 19, a shared axis hides the SLAM."""
    trace = [_pg_sample(0, 0, 0, interval_s=None),
             _pg_sample(2, int(0.32e9), int(38e9), int(40e9))]
    panels = {p[0]: p for p in build_panels(trace)}
    slam = panels["Processor use by the SLAM itself"]
    assert len(slam[3]) == 1, "the load must not share this panel"
    assert slam[3][0][1] == pytest.approx([0.16])


def test_cpu_seconds_totals_are_cumulative_over_the_run():
    trace = [_pg_sample(0, 0, 0, interval_s=None),
             _pg_sample(2, int(4e9), int(34e9)), _pg_sample(4, int(4e9), int(34e9))]
    assert cpu_seconds_totals(trace) == pytest.approx({"slam": 8.0, "load": 68.0})


def test_group_totals_never_go_negative_when_a_process_exits():
    """The bug this contract exists to prevent.

    Summing per-process cumulative counters into a group total and differencing
    the totals reads NEGATIVE when a process exits, because its accumulated time
    leaves the sum. ORB-SLAM3 churns threads constantly, hard during a map reset,
    so this fired on real runs. Deltas are taken per PID and only then summed,
    so a shrinking process set can never produce negative use.
    """
    trace = [_pg_sample(0, 0, 0, interval_s=None),
             _pg_sample(2, int(4e9), int(34e9)),
             _pg_sample(4, 0, int(34e9))]  # every SLAM thread exited this interval
    cores = process_group_series(trace)["slam"]["cores"][1]
    assert all(c >= 0 for c in cores), cores
    assert cores == pytest.approx([2.0, 0.0])


def test_throttling_reported_only_when_a_quota_bit():
    """Load antagonists set no quota, so these counters must stay silent there."""
    none_set = [dict(_sample(t), container_cpu_throttling={
        "nr_periods": 0, "nr_throttled": 0, "throttled_usec": 0}) for t in (0.0, 1.0)]
    assert throttling_totals(none_set) is None

    capped = [dict(_sample(0.0), container_cpu_throttling={
                  "nr_periods": 0, "nr_throttled": 0, "throttled_usec": 0}),
              dict(_sample(2.0), container_cpu_throttling={
                  "nr_periods": 100, "nr_throttled": 80, "throttled_usec": 5_000_000})]
    assert throttling_totals(capped) == (5.0, 100, 80)


def test_legacy_trace_without_walk_still_plots():
    legacy = [_sample(t) for t in (0.0, 1.0)]
    assert process_group_series(legacy) == {}
    assert cpu_seconds_totals(legacy) is None
    assert build_panels(legacy)  # the older panels still render
