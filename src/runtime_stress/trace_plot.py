"""Plot resource consumption over the lifetime of one stressed SLAM run.

Reads a run's stress_trace.json (written on every telemetry tick by the
orchestrator) and draws consumption against elapsed time, one panel per
resource, with the scenario's phases shaded so the moment a limit is applied
and released is visible rather than inferred.

Only resources that actually recorded data get a panel, so a CPU-only SLAM does
not get empty GPU axes.

Units are not uniform in the trace, and the panels keep them apart:
  container_cpu_percent  percent of ONE core, so 1890 means 18.9 cores busy
  system_cpu_percent     percent of the WHOLE machine, so it never exceeds 100
  gpu_*                  whole card, which includes any GPU competitor
The CPU panel is drawn in cores, matching the max_cores knob in the config.

The runtime-stress pipeline calls plot_run() for each run it finishes.

Writes stress_timeline.png next to each trace.
"""
import json
import os
import re

class MissingReadInstant(ValueError):
    """A trace carries a CPU counter but not the instant it was read.

    Raised rather than silently substituting the sample timestamp, which
    produces a believable but wrong rate exactly where load changes.
    """


# Categorical palette chosen to stay separable under the common colour-vision
# deficiencies.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
PLUM = "#8b5fbf"
INK = "#0b0b0b"
MUTED = "#8a8981"
GRID = "#ebebe6"
LIMIT = "#c2402f"
# Phase shading: warmup and recovery are unstressed context, stress is the
# interval under test. Pale, so the data lines stay dominant.
STRESS_FILL = "#fdf0e9"
CALM_FILL = "#f4f4f2"


def phase_fill(name):
    return CALM_FILL if name in ("warmup", "recovery") else STRESS_FILL


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def series(trace, key, scale=1.0, noise_floor=None):
    """(times, values) for samples where `key` was recorded.

    `noise_floor` discards a series that never rises above it. Used for the GPU
    fields: a CPU-only SLAM reports 0 rather than null, and on a machine with a
    desktop session the compositor adds occasional single-digit blips. Every
    GPU-using SLAM here sits far above the floor, so this only removes panels
    that would show nothing.
    """
    xs, ys = [], []
    for s in trace:
        v = s.get(key)
        if v is not None:
            xs.append(s.get("elapsed_s", 0.0))
            ys.append(v * scale)
    if noise_floor is not None and not any(y > noise_floor for y in ys):
        return [], []
    return xs, ys


def cores_series(samples):
    """(times, cores, is_instantaneous) from a list of per-sample dicts.

    Prefers differencing the cumulative CPU-time counter, which gives true
    instantaneous use. The runtime's own CPU percent is an average over the
    container's whole lifetime, so it lags every change and creeps toward the
    real value without reaching it; it is only used when a trace predates the
    counter being recorded, and the caller labels it as an average.
    """
    have_counter = any(s.get("cpu_time_ns") is not None for s in samples)
    if have_counter:
        # The read-instant is REQUIRED, never guessed. Substituting the sample's
        # own timestamp charges the cost of assembling the rest of the sample to
        # this interval, which under contention renders a steady load as a dip
        # that never happened (a real 19.7-core load once read as 1.5). A trace
        # that lacks it cannot yield an instantaneous rate, and saying so beats
        # returning a plausible wrong number.
        if any(s.get("cpu_time_at_s") is None
               for s in samples if s.get("cpu_time_ns") is not None):
            raise MissingReadInstant(
                "trace has cpu_time_ns without cpu_time_at_s; it predates the "
                "read-instant fix and cannot give an instantaneous rate"
            )
        xs, ys, prev = [], [], None
        for s in samples:
            ns = s.get("cpu_time_ns")
            t = s.get("cpu_time_at_s")
            if ns is None or t is None:
                continue
            if prev is not None:
                dt = t - prev[0]
                dns = ns - prev[1]
                # Guard against a zero/backward interval and a counter reset.
                if dt > 0 and dns >= 0:
                    xs.append(t)
                    ys.append(dns / 1e9 / dt)
            prev = (t, ns)
        if xs:
            return xs, ys, True
    xs, ys = [], []
    for s in samples:
        v, t = s.get("cpu_percent"), s.get("elapsed_s")
        if v is not None and t is not None:
            xs.append(t)
            ys.append(v / 100.0)
    return xs, ys, False


def process_group_series(trace):
    """Per-workload cores and cumulative run-delay from the process walk.

    Returns {"slam": {...}, "load": {...}} with `cores` (differenced CPU time)
    and `delay` (run-queue wait, differenced then expressed as thread-seconds
    of waiting per second). Empty when the trace predates the walk.
    """
    samples = [s["process_groups"] for s in trace if s.get("process_groups")]
    if not samples:
        return {}
    out = {}
    for key in ("slam", "load"):
        xs_c, ys_c, xs_d, ys_d = [], [], [], []
        for pg in samples:
            g, t, dt = pg.get(key), pg.get("at_s"), pg.get("interval_s")
            # interval_s is None on the first pass, which has no baseline to
            # difference against.
            if not g or t is None or not dt or dt <= 0:
                continue
            if g.get("cpu_time_ns") is not None:
                xs_c.append(t)
                ys_c.append(g["cpu_time_ns"] / 1e9 / dt)
            if g.get("run_delay_ns") is not None:
                xs_d.append(t)
                ys_d.append(g["run_delay_ns"] / 1e9 / dt)
        if xs_c:
            out[key] = {"cores": (xs_c, ys_c), "delay": (xs_d, ys_d)}
    return out


def throttling_totals(trace):
    """(throttled_seconds, periods, throttled_periods) if a quota ever bit."""
    vals = [s.get("container_cpu_throttling") for s in trace
            if s.get("container_cpu_throttling")]
    if not vals:
        return None
    first, last = vals[0], vals[-1]
    usec = last.get("throttled_usec", 0) - first.get("throttled_usec", 0)
    periods = last.get("nr_periods", 0) - first.get("nr_periods", 0)
    throttled = last.get("nr_throttled", 0) - first.get("nr_throttled", 0)
    if periods <= 0 and usec <= 0:
        return None
    return usec / 1e6, periods, throttled


def antagonist_series(trace):
    """{container_name: (times, cores)} from per-sample antagonist telemetry.

    Only sibling-container antagonists appear here. An in-container antagonist
    shares the SLAM's container, so its cost is already inside the SLAM line.
    """
    by_name = {}
    for s in trace:
        for name, st in (s.get("load_antagonists") or {}).items():
            if not st:
                continue
            by_name.setdefault(name, []).append({
                "elapsed_s": s.get("elapsed_s"),
                "cpu_percent": st.get("cpu_percent"),
                "cpu_time_ns": st.get("cpu_time_ns"),
                "cpu_time_at_s": st.get("cpu_time_at_s"),
            })
    out = {}
    for name, samples in by_name.items():
        # Omit a series we cannot compute honestly, and say which and why.
        # Traces written before load-generator reads were stamped land here.
        try:
            xs, ys, instantaneous = cores_series(samples)
        except MissingReadInstant as exc:
            print(f"  ! load generator '{name}': no CPU line ({exc})")
            continue
        if not instantaneous:
            print(f"  ! load generator '{name}': no CPU line "
                  f"(only a lifetime-average percent is recorded, which is "
                  f"not a rate)")
            continue
        if xs:
            out[name] = (xs, ys)
    return out


def phase_spans(run_dir, trace):
    """[start, end, name] spans. Prefers stress_events.json for exact edges."""
    tmax = trace[-1].get("elapsed_s", 0.0) if trace else 0.0
    events = load_json(os.path.join(run_dir, "stress_events.json"))
    if events:
        enters = [(e.get("elapsed_s", 0.0), e.get("message"))
                  for e in events if e.get("kind") == "phase_enter"]
        if enters:
            spans = []
            for i, (t, name) in enumerate(enters):
                end = enters[i + 1][0] if i + 1 < len(enters) else tmax
                spans.append([t, end, name])
            return spans
    # Fall back to the phase label carried on each sample.
    spans, cur, start = [], None, 0.0
    for s in trace:
        ph, t = s.get("phase"), s.get("elapsed_s", 0.0)
        if ph != cur:
            if cur is not None:
                spans.append([start, t, cur])
            cur, start = ph, t
    if cur is not None:
        spans.append([start, tmax, cur])
    return spans


def dataset_for(run_dir):
    """Human-readable "FAMILY sequence" for the chart title, or None.

    The dataset is not recorded as its own field, but the container name is
    ``{algo}-{sequence}-run_N-{hash}`` and the IO target paths name the family,
    so both are recoverable. Where the paths do not name it, the sequence's own
    shape does: ``freiburg*`` is TUM, ``V*``/``MH*`` EuRoC, all-digits KITTI.
    """
    ss = load_json(os.path.join(run_dir, "stress_summary.json"))
    if not ss:
        return None
    meta = ss.get("target_metadata") or {}
    name = meta.get("container_name") or ""
    m = re.match(r"^[a-z0-9]+-(?P<seq>.+)-run_\d+-[0-9a-f]+$", name)
    if not m:
        return None
    seq = m.group("seq")

    paths = " ".join(meta.get("io_target_paths") or []).lower()
    family = next((f for f in ("euroc", "kitti", "7scenes") if f in paths), None)
    if family is None and "tum" in paths:
        family = "tum"
    if family is None:                       # infer from the sequence's own shape
        if seq.startswith("freiburg"):
            family = "tum"
        elif re.match(r"^(v\d|mh)", seq):
            family = "euroc"
        elif seq.isdigit():
            family = "kitti"
    pretty = {"euroc": "EuRoC", "kitti": "KITTI", "tum": "TUM", "7scenes": "7-Scenes"}
    # EuRoC sequences are written with the rig prefix capitalised and the rest
    # left alone: v1_01_easy -> V1_01_easy, mh_01_easy -> MH_01_easy.
    label = seq
    if family == "euroc":
        pre = re.match(r"^([a-z]+)(.*)$", seq)
        if pre:
            label = pre.group(1).upper() + pre.group(2)
    return f"{pretty.get(family, family or '?')} {label}"


def run_label(run_dir):
    parts = os.path.normpath(os.path.abspath(run_dir)).split(os.sep)

    def after(token, default="?"):
        return parts[parts.index(token) + 1] if token in parts and parts.index(token) + 1 < len(parts) else default

    return after("results"), after("slam_results"), parts[-2], parts[-1]


def cpu_seconds_totals(trace):
    """Total CPU-seconds each workload consumed over the run.

    Sums per-interval deltas. Summing is required rather than differencing a
    running total, because processes enter and leave the cgroup and an exited
    process would otherwise subtract the time it had already used.
    """
    samples = [s["process_groups"] for s in trace if s.get("process_groups")]
    if not samples:
        return None
    out = {}
    for key in ("slam", "load"):
        total = sum(pg[key]["cpu_time_ns"] for pg in samples
                    if pg.get(key) and pg[key].get("cpu_time_ns") is not None
                    and pg.get("interval_s"))
        if total:
            out[key] = total / 1e9
    return out or None


def subtitle_for(run_dir, trace=None):
    """(subtitle, shares_cgroup) from stress_summary.json and the trace."""
    ss = load_json(os.path.join(run_dir, "stress_summary.json"))
    if not ss:
        return "", False
    shares_cgroup = any((a.get("kind") == "in_container")
                        for a in (ss.get("load_antagonists") or []))
    bits = []
    drop = (ss.get("deadline") or {}).get("drop_rate")
    if drop is not None:
        bits.append(f"{drop * 100:.0f}% of frames dropped")
    for a in ss.get("load_antagonists") or []:
        kind = a.get("kind")
        if kind == "in_container":
            bits.append(f"{a.get('cpu_workers')} competing processes inside the SLAM container")
        elif kind == "stress_ng":
            bits.append(f"{a.get('cpu_workers')} competing processes in a separate container")
        elif kind == "gpu":
            bits.append(f"GPU competitor holding {a.get('vram_mb')} MB at "
                        f"{int((a.get('duty_cycle') or 0) * 100)}% duty")
    if trace:
        totals = cpu_seconds_totals(trace)
        if totals and "slam" in totals:
            whole = sum(totals.values())
            share = f" of {whole:.0f} total" if len(totals) > 1 else ""
            bits.append(f"SLAM used {totals['slam']:.0f} CPU-seconds{share}")
        thr = throttling_totals(trace)
        if thr:
            secs, periods, throttled = thr
            if periods:
                bits.append(f"capped in {throttled}/{periods} scheduling periods, "
                            f"{secs:.0f}s of runnable time removed")
    if ss.get("controller_error"):
        bits.append("RUN INVALID: the stressor did not apply")
    return "   ".join(bits), shares_cgroup


def build_panels(trace, shares_cgroup=False):
    """[(title, ylabel, ymax, [(xs, ys, label, colour, style)])] for panels with data."""
    panels = []

    cpu_samples = [{"elapsed_s": s.get("elapsed_s"),
                    "cpu_percent": s.get("container_cpu_percent"),
                    "cpu_time_ns": s.get("container_cpu_time_ns"),
                    "cpu_time_at_s": s.get("container_cpu_time_at_s")} for s in trace]
    # A trace without read-instants cannot give an honest rate. Drop the panel
    # and say so, rather than draw a line that is wrong exactly where the load
    # changes.
    try:
        cpu_x, cpu_y, instantaneous = cores_series(cpu_samples)
    except MissingReadInstant as exc:
        print(f"  ! no processor panel ({exc})")
        cpu_x, cpu_y, instantaneous = [], [], False
    if cpu_x:
        # An in-container antagonist lives in the SLAM's own cgroup, so this
        # figure covers both and cannot be attributed to the SLAM alone.
        cpu_label = "SLAM and its competitors" if shares_cgroup else "SLAM"
        lines = [(cpu_x, cpu_y, cpu_label, BLUE, "-")]
        for i, (name, (ax_, ay)) in enumerate(sorted(antagonist_series(trace).items())):
            lines.append((ax_, ay, f"competitor ({name.split('-')[-1]})",
                          [ORANGE, PLUM][i % 2], "-"))
        title = ("Processor use inside the SLAM container" if instantaneous else
                 "Processor use inside the SLAM container (running average, not instantaneous)")
        panels.append((title, "cores busy", None, lines))

    # When the process walk ran, the cgroup total above can be split by
    # workload. This is the panel that answers "how much did the SLAM itself
    # get", which the cgroup counter cannot: an in-container antagonist shares
    # the cgroup, so the total reads the same whether the SLAM is getting a
    # third of the machine or a tenth.
    pg = process_group_series(trace)
    if pg:
        # The SLAM gets its own panel rather than sharing one with the load.
        # Under heavy contention it runs on a fraction of a core against a load
        # taking ~19, so on a shared axis the line that matters sits flat on the
        # floor. The load's share is the total panel above minus this one.
        if "slam" in pg:
            panels.append(("Processor use by the SLAM itself", "cores busy", None,
                           [(*pg["slam"]["cores"], None, BLUE, "-")]))
        delay_lines = []
        if "slam" in pg and pg["slam"]["delay"][0]:
            delay_lines.append((*pg["slam"]["delay"], "SLAM itself", BLUE, "-"))
        if delay_lines:
            panels.append(("Time the SLAM spent waiting for a processor",
                           "seconds waited\nper second", None, delay_lines))

    sys_x, sys_y = series(trace, "system_cpu_percent")
    if sys_x:
        panels.append(("Processor use across the whole machine", "percent", 100,
                       [(sys_x, sys_y, None, ORANGE, "-")]))

    mem_x, mem_y = series(trace, "rss_bytes", 1e-9)
    if mem_x:
        lines = [(mem_x, mem_y, "in use", BLUE, "-")]
        lim_x, lim_y = series(trace, "memory_limit_bytes", 1e-9)
        # Only draw the ceiling when a limit was actually imposed: uncapped runs
        # report the host's total RAM, which would flatten the panel to nothing.
        if lim_y and min(lim_y) < max(lim_y) * 0.95:
            lines.append((lim_x, lim_y, "limit", LIMIT, "--"))
        panels.append(("Memory use by the SLAM container", "GB", None, lines))

    # 5% of the card: below any real SLAM workload, above desktop compositor blips.
    g_x, g_y = series(trace, "gpu_util_percent", noise_floor=5.0)
    if g_x:
        panels.append(("Graphics card use, whole card", "percent", 100,
                       [(g_x, g_y, None, BLUE, "-")]))

    # 0.5 GB: above an idle desktop's framebuffer, below any SLAM's working set.
    gm_x, gm_y = series(trace, "gpu_mem_used_mb", 1e-3, noise_floor=0.5)
    # A CPU-only SLAM still sees the desktop's flat few hundred MB on the card.
    # Keep this panel only when the card was in use, or when the usage actually
    # moves (which catches a SLAM that allocates VRAM but reports no utilisation).
    if gm_x and (g_x or max(gm_y) - min(gm_y) > 0.2):
        lines = [(gm_x, gm_y, "in use", BLUE, "-")]
        tot_x, tot_y = series(trace, "gpu_mem_total_mb", 1e-3)
        if tot_y:
            lines.append((tot_x, tot_y, "card total", LIMIT, "--"))
        panels.append(("Graphics memory, whole card", "GB", None, lines))

    return panels


def plot_run(run_dir, quiet=False):
    """Draw one run's timeline. Returns True if a chart was written."""
    trace = load_json(os.path.join(run_dir, "stress_trace.json"))
    if not trace:
        if not quiet:
            print(f"  skip, no trace: {run_dir}")
        return False

    # Imported here so `import trace_plot` stays cheap for callers that only
    # want walk_runs().
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    sub, shares_cgroup = subtitle_for(run_dir, trace)
    panels = build_panels(trace, shares_cgroup)
    if not panels:
        if not quiet:
            print(f"  skip, no usable samples: {run_dir}")
        return False

    campaign, algo, scen, run = run_label(run_dir)
    spans = phase_spans(run_dir, trace)
    tmax = trace[-1].get("elapsed_s", 0.0) or 1.0

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 1.9 * len(panels) + 0.9),
                             sharex=True, squeeze=False)
    axes = [a[0] for a in axes]

    for ax, (title, ylabel, ymax, lines) in zip(axes, panels):
        for start, end, name in spans:
            ax.axvspan(start, end, color=phase_fill(name), zorder=0, linewidth=0)
        for xs, ys, label, colour, style in lines:
            ax.plot(xs, ys, color=colour, linewidth=1.9, linestyle=style,
                    label=label, zorder=3)
        ax.set_ylabel(ylabel, fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, loc="left", color=INK, pad=4)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=1)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        # A legend on a single unlabelled series just repeats the panel title.
        if len(lines) > 1:
            ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=len(lines))
        ax.set_xlim(0, tmax)
        # Magnitudes get a zero baseline, so a small wobble reads as small.
        # Unbounded panels get headroom so the legend cannot sit on the data.
        top_limit = ymax
        if top_limit is None:
            peak = max((max(ys) for _, ys, _, _, _ in lines if ys), default=1.0)
            top_limit = (peak * 1.22) or 1.0
        ax.set_ylim(bottom=0, top=top_limit)

    axes[-1].set_xlabel("time since the run started (seconds)", fontsize=9, color=INK)

    # Phase names ride above the top panel, clear of the axis label.
    top = axes[0]
    for start, end, name in spans:
        if end - start > tmax * 0.02:
            top.annotate(name, xy=((start + end) / 2, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(0, 22), textcoords="offset points",
                         ha="center", fontsize=8.5, color=MUTED)

    fig.legend(handles=[Patch(facecolor=STRESS_FILL, label="shaded: the stressed phase")],
               fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(0.99, 0.0))
    fig.tight_layout(rect=[0, 0.025, 1, 0.905])
    # Titles are placed after tight_layout so it cannot reflow them into the plots.
    dataset = dataset_for(run_dir)
    headline = f"{algo}   {dataset}" if dataset else algo
    fig.suptitle(f"{headline}\n{scen}   {run}   {campaign}", fontsize=11, color=INK,
                 ha="left", x=0.007, y=0.995, va="top")
    if sub:
        fig.text(0.007, 0.925, sub, fontsize=8.5, color=MUTED, ha="left", va="top")
    out = os.path.join(run_dir, "stress_timeline.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    if not quiet:
        print(f"  wrote {out}")
    return True


def walk_runs(root):
    """Yield every run directory under `root` that carries a stress trace."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if "stress_trace.json" in filenames:
            yield dirpath
