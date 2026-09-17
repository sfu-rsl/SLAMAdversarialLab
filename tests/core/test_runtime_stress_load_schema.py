"""Schema/parser/compile tests for the load-antagonist control axis."""

import pytest

from slamadversariallab.config.schema import (
    LoadControlConfig,
    LoadFenceConfig,
    LoadGpuAntagonistConfig,
    LoadInContainerConfig,
)
from slamadversariallab.config.parser import _parse_axis_blocks
from slamadversariallab.runtime_stress.models import (
    LoadControl,
    RuntimeStressControls,
    _compile_load,
    _merge_controls_tighter_wins,
)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def test_parse_axis_blocks_returns_load_cfg():
    controls = {
        "load": {
            "cpu_workers": 20,
            "fence": {"cpus": 4.0},
            "gpu": {"vram_mb": 2048, "matmul_n": 4096, "duty_cycle": 0.25},
        }
    }
    cpu, memory, gpu, io, load = _parse_axis_blocks(controls, "ctx")
    assert cpu is None and memory is None and gpu is None and io is None
    assert load.cpu_workers == 20
    assert load.fence.cpus == 4.0
    assert load.gpu.vram_mb == 2048
    assert load.gpu.duty_cycle == 0.25


def test_parse_axis_blocks_rejects_non_dict_load():
    with pytest.raises(ValueError, match=r"ctx\.load must be a dictionary"):
        _parse_axis_blocks({"load": 3}, "ctx")


# ---------------------------------------------------------------------------
# validation (fail loud)
# ---------------------------------------------------------------------------

def test_stress_ng_workers_require_quota_or_weight():
    cfg = LoadControlConfig(cpu_workers=4)
    with pytest.raises(ValueError, match="cpus .quota. OR cpu_shares"):
        cfg.validate()
    # a memory-only fence still leaves it uncapped AND unweighted -> reject
    cfg = LoadControlConfig(cpu_workers=4, fence=LoadFenceConfig(memory_mb=2048))
    with pytest.raises(ValueError, match="cpus .quota. OR cpu_shares"):
        cfg.validate()


def test_weight_only_fence_is_valid():
    # cpu_shares alone (uncapped weight mode) satisfies the calibration rule
    LoadControlConfig(
        cpu_workers=20, fence=LoadFenceConfig(cpu_shares=131072)
    ).validate()


def test_cpu_shares_range_enforced():
    for bad in (1, 300000):
        cfg = LoadControlConfig(
            cpu_workers=4, fence=LoadFenceConfig(cpu_shares=bad)
        )
        with pytest.raises(ValueError, match="cpu_shares must be an integer"):
            cfg.validate()


def test_vm_workers_require_memory_fence():
    cfg = LoadControlConfig(
        vm_workers=2, vm_bytes_mb=1024, fence=LoadFenceConfig(cpus=2.0)
    )
    with pytest.raises(ValueError, match="fence.memory_mb is required"):
        cfg.validate()


def test_vm_workers_and_bytes_come_together():
    cfg = LoadControlConfig(vm_workers=2, fence=LoadFenceConfig(cpus=2.0, memory_mb=2048))
    with pytest.raises(ValueError, match="must be set\\s+together"):
        cfg.validate()


def test_duty_cycle_bounds():
    cfg = LoadGpuAntagonistConfig(vram_mb=1024, duty_cycle=1.5)
    with pytest.raises(ValueError, match=r"duty_cycle must be a number in \[0, 1\]"):
        cfg.validate()


def test_duty_requires_matmul_n():
    cfg = LoadGpuAntagonistConfig(duty_cycle=0.5)
    with pytest.raises(ValueError, match="matmul_n must be a positive integer"):
        cfg.validate()


def test_empty_load_block_rejected():
    with pytest.raises(ValueError, match="at least one antagonist"):
        LoadControlConfig().validate()


def test_in_container_valid_without_fence():
    # in-container load runs in the SLAM's cgroup -> needs no fence
    LoadControlConfig(in_container=LoadInContainerConfig(cpu_workers=200)).validate()


def test_in_container_requires_some_workers():
    with pytest.raises(ValueError, match="at least one of"):
        LoadControlConfig(in_container=LoadInContainerConfig()).validate()


def test_in_container_worker_range():
    cfg = LoadControlConfig(in_container=LoadInContainerConfig(cpu_workers=99999))
    with pytest.raises(ValueError, match="cpu_workers must be an integer"):
        cfg.validate()


def test_in_container_vm_safety_bound():
    # ballast lands in the SLAM's (usually uncapped) cgroup: total capped hard
    cfg = LoadControlConfig(in_container=LoadInContainerConfig(
        vm_workers=8, vm_bytes_mb=8192))  # 64 GB total > 32 GB bound
    with pytest.raises(ValueError, match="safety bound"):
        cfg.validate()


def test_in_container_vm_pairing():
    cfg = LoadControlConfig(in_container=LoadInContainerConfig(
        cpu_workers=4, vm_workers=2))
    with pytest.raises(ValueError, match="must be set together"):
        cfg.validate()


def test_in_container_compiles_and_reaches_runtime():
    from slamadversariallab.runtime_stress.models import LoadInContainer
    compiled = _compile_load(LoadControlConfig(
        in_container=LoadInContainerConfig(cpu_workers=200, stream_workers=4)))
    assert isinstance(compiled.in_container, LoadInContainer)
    assert compiled.in_container.cpu_workers == 200
    assert compiled.in_container.stream_workers == 4


def test_valid_load_block_passes():
    LoadControlConfig(
        cpu_workers=20,
        stream_workers=4,
        vm_workers=2,
        vm_bytes_mb=2048,
        gpu=LoadGpuAntagonistConfig(vram_mb=4096, matmul_n=4096, duty_cycle=0.5),
        fence=LoadFenceConfig(cpus=8.0, memory_mb=8192),
    ).validate()


# ---------------------------------------------------------------------------
# compile + merge
# ---------------------------------------------------------------------------

def test_compile_load_normalizes_and_drops_empty():
    cfg = LoadControlConfig(cpu_workers=4, fence=LoadFenceConfig(cpus=2.0))
    compiled = _compile_load(cfg)
    assert compiled.cpu_workers == 4
    assert compiled.fence.cpus == 2.0
    assert _compile_load(None) is None
    assert _compile_load(LoadControlConfig()) is None  # all-idle -> None


def test_merge_passes_single_load_through():
    load = LoadControl(cpu_workers=4)
    merged = _merge_controls_tighter_wins(
        RuntimeStressControls(load=load), RuntimeStressControls()
    )
    assert merged.load is load


def test_merge_rejects_two_load_sources():
    with pytest.raises(ValueError, match="single-source"):
        _merge_controls_tighter_wins(
            RuntimeStressControls(load=LoadControl(cpu_workers=4)),
            RuntimeStressControls(load=LoadControl(cpu_workers=8)),
        )
