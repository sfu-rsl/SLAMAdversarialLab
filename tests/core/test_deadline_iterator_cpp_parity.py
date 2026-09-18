"""Bit-exact differential parity test: C++ DeadlineIterator vs the Python one.

Both iterators must make the IDENTICAL drop decisions for the identical timing.
The C++ header is compiled with -DSAL_DEADLINE_TEST_CLOCK, which swaps its real
monotonic clock for a deterministic virtual clock (a test-only seam; production
uses the real clock unchanged). A shared virtual clock is then driven through
both implementations across many randomized adversarial traces, and their
survivor/dropped sets are compared byte-for-byte.

Skipped when g++ is unavailable (e.g. minimal CI). Compiles once, reuses the
binary across the module.
"""
import importlib.util
import random
import shutil
import subprocess
import time as _time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HEADER_DIR = REPO / "src" / "runtime_stress"
HARNESS = Path(__file__).parent / "_deadline_iterator_diff.cpp"

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None or not HARNESS.exists() or
    not (HEADER_DIR / "deadline_iterator.h").exists(),
    reason="g++ or the C++ deadline iterator header/harness is unavailable",
)


def _load_py_iterator():
    spec = importlib.util.spec_from_file_location(
        "dli_parity", HEADER_DIR / "deadline_iterator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DeadlineIterator


@pytest.fixture(scope="module")
def cpp_binary(tmp_path_factory):
    out = tmp_path_factory.mktemp("dlparity") / "dl_diff"
    r = subprocess.run(
        ["g++", "-std=c++14", "-DSAL_DEADLINE_TEST_CLOCK",
         "-I", str(HEADER_DIR), "-O2", str(HARNESS), "-o", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"harness failed to compile:\n{r.stderr}"
    return out


def _make_traces(n_traces, seed=20260713):
    rng = random.Random(seed)
    traces = []
    for _ in range(n_traces):
        n = rng.randint(1, 120)
        fps = rng.choice([5, 8, 10, 15, 20, 30, 40])
        warmup = rng.randint(0, min(10, n))
        queue = rng.randint(1, 5)
        policy = rng.randint(0, 1)
        period = 1.0 / fps
        kind = rng.randint(0, 4)
        pt = []
        for i in range(n):
            if kind == 0:
                pt.append(0.0)
            elif kind == 1:
                pt.append(period * rng.uniform(0.1, 0.9))
            elif kind == 2:
                pt.append(period * rng.uniform(1.1, 4.0))
            elif kind == 3:
                pt.append(period * (5.0 if i % 7 == 0 else 0.0))
            else:
                pt.append(period * rng.uniform(0.0, 6.0))
        traces.append((n, fps, warmup, queue, policy, pt))
    return traces


def _run_python(DeadlineIterator, n, fps, warmup, queue, policy, pt):
    clock = [0.0]
    orig_mono, orig_sleep = _time.monotonic, _time.sleep
    _time.monotonic = lambda: clock[0]

    def _sleep(x):
        assert x >= 0
        clock[0] += x

    _time.sleep = _sleep
    try:
        it = DeadlineIterator(
            list(range(n)), float(fps), warmup_frames=warmup, queue_size=queue,
            drop_policy="drop_oldest" if policy == 0 else "drop_newest")
        for i, _item in enumerate(it):
            clock[0] += pt[i if i < len(pt) else len(pt) - 1]
        return it.survivors, it.dropped
    finally:
        _time.monotonic, _time.sleep = orig_mono, orig_sleep


def test_cpp_python_bit_exact_parity(cpp_binary):
    DeadlineIterator = _load_py_iterator()
    traces = _make_traces(2000)

    # C++ side: feed every trace on stdin, one result line each.
    inp = "".join(
        f"{n} {fps} {w} {q} {pol} {len(pt)} " + " ".join(f"{x:.10g}" for x in pt) + "\n"
        for (n, fps, w, q, pol, pt) in traces
    )
    r = subprocess.run([str(cpp_binary)], input=inp, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    cpp_lines = r.stdout.strip().split("\n")
    assert len(cpp_lines) == len(traces)

    mismatches = []
    for i, (n, fps, w, q, pol, pt) in enumerate(traces):
        ps, pd = _run_python(DeadlineIterator, n, fps, w, q, pol, pt)
        py = f"surv:{','.join(map(str, ps))}|drop:{','.join(map(str, pd))}"
        if py != cpp_lines[i]:
            mismatches.append((i, n, fps, w, q, pol, py, cpp_lines[i]))
            if len(mismatches) >= 3:
                break
    assert not mismatches, f"C++/Python divergence: {mismatches}"
