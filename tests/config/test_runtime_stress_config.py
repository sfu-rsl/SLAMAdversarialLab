"""Tests for runtime-stress configuration parsing and validation."""

import pytest

from slamadversariallab.config.parser import parse_runtime_stress


def test_parse_runtime_stress_accepts_cpu_max_cores() -> None:
    runtime_stress = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "telemetry": {"sample_period_ms": 250},
                "scenarios": [
                    {
                        "name": "cpu_medium",
                        "phases": [
                            {"name": "warmup", "duration_s": 5},
                            {
                                "name": "stress",
                                "duration_s": 10,
                                "controls": {"cpu": {"max_cores": 1.5}},
                            },
                        ],
                    }
                ],
            }
        }
    )

    assert runtime_stress is not None
    assert runtime_stress.enabled is True
    assert runtime_stress.telemetry.sample_period_ms == 250
    assert runtime_stress.scenarios[0].phases[1].controls.cpu is not None
    assert runtime_stress.scenarios[0].phases[1].controls.cpu.max_cores == 1.5


def test_parse_runtime_stress_rejects_non_positive_cpu_limit() -> None:
    with pytest.raises(ValueError, match="max_cores"):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "cpu_invalid",
                            "phases": [
                                {
                                    "name": "stress",
                                    "duration_s": 10,
                                    "controls": {"cpu": {"max_cores": 0}},
                                }
                            ],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_accepts_memory_max_mb() -> None:
    runtime_stress = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "telemetry": {"sample_period_ms": 250},
                "scenarios": [
                    {
                        "name": "mem_medium",
                        "phases": [
                            {"name": "warmup", "duration_s": 5},
                            {
                                "name": "stress",
                                "duration_s": 10,
                                "controls": {"memory": {"max_mb": 512}},
                            },
                        ],
                    }
                ],
            }
        }
    )

    assert runtime_stress is not None
    assert runtime_stress.scenarios[0].phases[1].controls.memory is not None
    assert runtime_stress.scenarios[0].phases[1].controls.memory.max_mb == 512


def test_parse_runtime_stress_rejects_non_positive_memory_limit() -> None:
    with pytest.raises(ValueError, match="max_mb"):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "mem_invalid",
                            "phases": [
                                {
                                    "name": "stress",
                                    "duration_s": 10,
                                    "controls": {"memory": {"max_mb": 0}},
                                }
                            ],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_accepts_io_caps() -> None:
    runtime_stress = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "container_runtime": "podman",
                "scenarios": [
                    {
                        "name": "io_caps",
                        "phases": [
                            {"name": "warmup", "duration_s": 1},
                            {
                                "name": "stress",
                                "duration_s": 30,
                                "controls": {
                                    "io": {
                                        "read_bps": 50_000_000,
                                        "write_bps": 10_000_000,
                                        "read_iops": 1000,
                                        "write_iops": 500,
                                    }
                                },
                            },
                        ],
                    }
                ],
            }
        }
    )

    assert runtime_stress is not None
    io = runtime_stress.scenarios[0].phases[1].controls.io
    assert io is not None
    assert io.read_bps == 50_000_000
    assert io.write_bps == 10_000_000
    assert io.read_iops == 1000
    assert io.write_iops == 500


@pytest.mark.parametrize(
    "field_name, bad_value",
    [
        ("read_bps", -1),
        ("read_bps", 0),
        ("read_bps", "fast"),
        ("read_bps", 1.5),
        ("read_bps", True),
        ("write_bps", -1_000),
        ("read_iops", 0),
        ("write_iops", -10),
    ],
)
def test_parse_runtime_stress_rejects_invalid_io_values(field_name, bad_value) -> None:
    with pytest.raises(ValueError, match=field_name):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "container_runtime": "podman",
                    "scenarios": [
                        {
                            "name": "io_invalid",
                            "phases": [
                                {
                                    "name": "stress",
                                    "duration_s": 10,
                                    "controls": {"io": {field_name: bad_value}},
                                }
                            ],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_compiles_io_into_request() -> None:
    """End-to-end: schema -> compile_runtime_stress_request -> IoControl."""
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    runtime_stress = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "container_runtime": "podman",
                "scenarios": [
                    {
                        "name": "io_compile",
                        "phases": [
                            {
                                "name": "stress",
                                "duration_s": 5,
                                "controls": {"io": {"read_bps": 5_000_000}},
                            }
                        ],
                    }
                ],
            }
        }
    )
    assert runtime_stress is not None
    request = compile_runtime_stress_request(runtime_stress, runtime_stress.scenarios[0])
    assert request.phases[0].controls.io is not None
    assert request.phases[0].controls.io.read_bps == 5_000_000
    assert request.phases[0].controls.io.write_bps is None


def test_parse_runtime_stress_defaults_container_runtime_to_docker() -> None:
    runtime_stress = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "default_runtime",
                        "phases": [
                            {
                                "name": "stress",
                                "duration_s": 10,
                                "controls": {"cpu": {"max_cores": 0.5}},
                            }
                        ],
                    }
                ],
            }
        }
    )

    assert runtime_stress is not None
    assert runtime_stress.container_runtime == "docker"


def test_parse_runtime_stress_accepts_container_runtime_podman() -> None:
    runtime_stress = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "container_runtime": "podman",
                "scenarios": [
                    {
                        "name": "podman_scenario",
                        "phases": [
                            {
                                "name": "stress",
                                "duration_s": 10,
                                "controls": {"cpu": {"max_cores": 0.5}},
                            }
                        ],
                    }
                ],
            }
        }
    )

    assert runtime_stress is not None
    assert runtime_stress.container_runtime == "podman"


def test_parse_runtime_stress_rejects_unknown_container_runtime() -> None:
    with pytest.raises(ValueError, match="container_runtime"):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "container_runtime": "lxc",
                    "scenarios": [
                        {
                            "name": "bad_runtime",
                            "phases": [
                                {
                                    "name": "stress",
                                    "duration_s": 10,
                                    "controls": {"cpu": {"max_cores": 0.5}},
                                }
                            ],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_rejects_non_string_container_runtime() -> None:
    with pytest.raises(ValueError, match="container_runtime"):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "container_runtime": 42,
                    "scenarios": [
                        {
                            "name": "bad_runtime",
                            "phases": [
                                {
                                    "name": "stress",
                                    "duration_s": 10,
                                    "controls": {"cpu": {"max_cores": 0.5}},
                                }
                            ],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_accepts_gpu_vram_limit() -> None:
    runtime_stress = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "gpu_vram",
                        "phases": [
                            {
                                "name": "stress",
                                "duration_s": 10,
                                "controls": {"gpu": {"vram_limit_mb": 4096}},
                            }
                        ],
                    }
                ],
            }
        }
    )

    phase = runtime_stress.scenarios[0].phases[0]
    assert phase.controls.gpu is not None
    assert phase.controls.gpu.vram_limit_mb == 4096
    assert phase.controls.gpu.sm_limit_percent is None


def test_parse_runtime_stress_accepts_gpu_sm_limit() -> None:
    runtime_stress = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "gpu_sm",
                        "phases": [
                            {
                                "name": "stress",
                                "duration_s": 10,
                                "controls": {"gpu": {"sm_limit_percent": 50}},
                            }
                        ],
                    }
                ],
            }
        }
    )

    phase = runtime_stress.scenarios[0].phases[0]
    assert phase.controls.gpu is not None
    assert phase.controls.gpu.sm_limit_percent == 50


def test_parse_runtime_stress_accepts_gpu_vram_and_sm_together() -> None:
    runtime_stress = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "gpu_both",
                        "phases": [
                            {
                                "name": "stress",
                                "duration_s": 10,
                                "controls": {
                                    "gpu": {
                                        "vram_limit_mb": 4096,
                                        "sm_limit_percent": 50,
                                    }
                                },
                            }
                        ],
                    }
                ],
            }
        }
    )

    phase = runtime_stress.scenarios[0].phases[0]
    assert phase.controls.gpu.vram_limit_mb == 4096
    assert phase.controls.gpu.sm_limit_percent == 50


def test_parse_runtime_stress_rejects_non_positive_gpu_vram() -> None:
    with pytest.raises(ValueError, match="vram_limit_mb"):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "gpu_bad_vram",
                            "phases": [
                                {
                                    "name": "stress",
                                    "duration_s": 10,
                                    "controls": {"gpu": {"vram_limit_mb": 0}},
                                }
                            ],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_rejects_out_of_range_gpu_sm_limit() -> None:
    with pytest.raises(ValueError, match="sm_limit_percent"):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "gpu_sm_high",
                            "phases": [
                                {
                                    "name": "stress",
                                    "duration_s": 10,
                                    "controls": {"gpu": {"sm_limit_percent": 150}},
                                }
                            ],
                        }
                    ],
                }
            }
        )

    with pytest.raises(ValueError, match="sm_limit_percent"):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "gpu_sm_zero",
                            "phases": [
                                {
                                    "name": "stress",
                                    "duration_s": 10,
                                    "controls": {"gpu": {"sm_limit_percent": 0}},
                                }
                            ],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_accepts_differing_gpu_values_across_phases() -> None:
    """Per-phase GPU caps are now legal: GpuHamiController mutates limit[0]
    in the bind-mounted shared region at phase boundaries, so the previous
    "HAMi launch-time-frozen" restriction no longer applies."""
    runtime_stress = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "gpu_phased",
                        "phases": [
                            {
                                "name": "warmup",
                                "duration_s": 5,
                                "controls": {"gpu": {"vram_limit_mb": 2048}},
                            },
                            {
                                "name": "stress",
                                "duration_s": 10,
                                "controls": {"gpu": {"vram_limit_mb": 4096}},
                            },
                        ],
                    }
                ],
            }
        }
    )
    assert runtime_stress is not None
    phase1, phase2 = runtime_stress.scenarios[0].phases
    assert phase1.controls.gpu.vram_limit_mb == 2048
    assert phase2.controls.gpu.vram_limit_mb == 4096


def test_parse_runtime_stress_accepts_cpu_and_memory_together() -> None:
    runtime_stress = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "combined",
                        "phases": [
                            {
                                "name": "stress",
                                "duration_s": 10,
                                "controls": {
                                    "cpu": {"max_cores": 0.5},
                                    "memory": {"max_mb": 256},
                                },
                            }
                        ],
                    }
                ],
            }
        }
    )

    phase = runtime_stress.scenarios[0].phases[0]
    assert phase.controls.cpu is not None
    assert phase.controls.cpu.max_cores == 0.5
    assert phase.controls.memory is not None
    assert phase.controls.memory.max_mb == 256


# -----------------------------
# Composable stressors (per-phase, multi-axis, tighter-wins)
# -----------------------------


def _composable_yaml(scenario_phases, stressors=None, container_runtime="podman"):
    """Helper to build a runtime_stress YAML dict for composability tests."""
    rt = {
        "enabled": True,
        "container_runtime": container_runtime,
        "scenarios": [{"name": "scn", "phases": scenario_phases}],
    }
    if stressors is not None:
        rt["stressors"] = stressors
    return {"runtime_stress": rt}


def test_parse_runtime_stress_accepts_top_level_stressors_library() -> None:
    rt = parse_runtime_stress(
        _composable_yaml(
            stressors={
                "edge_device": {
                    "cpu": {"max_cores": 0.25},
                    "memory": {"max_mb": 1024},
                    "io": {"read_bps": 10_000_000},
                },
                "cpu_storm": {"cpu": {"max_cores": 0.05}},
                "io_storm": {"io": {"read_bps": 1_000_000}},
            },
            scenario_phases=[
                {"name": "stress", "duration_s": 5, "stressors": ["edge_device"]}
            ],
        )
    )
    assert rt is not None
    assert set(rt.stressors.keys()) == {"edge_device", "cpu_storm", "io_storm"}
    assert rt.stressors["edge_device"].cpu is not None
    assert rt.stressors["edge_device"].cpu.max_cores == 0.25
    assert rt.stressors["edge_device"].memory.max_mb == 1024
    assert rt.stressors["edge_device"].io.read_bps == 10_000_000
    assert rt.stressors["cpu_storm"].memory is None  # absent axes stay None


def test_compile_resolves_phase_stressor_ref_into_runtime_controls() -> None:
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        _composable_yaml(
            stressors={
                "edge_device": {
                    "cpu": {"max_cores": 0.25},
                    "memory": {"max_mb": 1024},
                    "io": {"read_bps": 10_000_000},
                }
            },
            scenario_phases=[
                {"name": "stress", "duration_s": 5, "stressors": ["edge_device"]}
            ],
        )
    )
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    controls = request.phases[0].controls
    assert controls.cpu is not None and controls.cpu.max_cores == 0.25
    assert controls.memory is not None and controls.memory.max_mb == 1024
    assert controls.io is not None and controls.io.read_bps == 10_000_000
    assert controls.io.write_bps is None  # not set anywhere


def test_compile_tighter_wins_when_two_stressors_overlap_on_same_axis() -> None:
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        _composable_yaml(
            stressors={
                "edge_device": {
                    "cpu": {"max_cores": 0.25},
                    "memory": {"max_mb": 1024},
                    "io": {"read_bps": 10_000_000},
                },
                "cpu_storm": {"cpu": {"max_cores": 0.05}},
            },
            scenario_phases=[
                {
                    "name": "stress",
                    "duration_s": 5,
                    "stressors": ["edge_device", "cpu_storm"],
                }
            ],
        )
    )
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    controls = request.phases[0].controls
    # cpu took the smaller of {0.25, 0.05}
    assert controls.cpu.max_cores == 0.05
    # memory only set by edge_device
    assert controls.memory.max_mb == 1024
    # io only set by edge_device
    assert controls.io.read_bps == 10_000_000


def test_compile_tighter_wins_is_order_independent() -> None:
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    def _request_for(order):
        return compile_runtime_stress_request(
            parse_runtime_stress(
                _composable_yaml(
                    stressors={
                        "edge_device": {"cpu": {"max_cores": 0.25}},
                        "cpu_storm": {"cpu": {"max_cores": 0.05}},
                    },
                    scenario_phases=[
                        {"name": "stress", "duration_s": 5, "stressors": list(order)}
                    ],
                )
            ),
            None,  # scenario passed below
        )

    # compile takes (rt, scenario); use a single rt and pull scenarios[0] both ways
    rt_a = parse_runtime_stress(
        _composable_yaml(
            stressors={
                "edge_device": {"cpu": {"max_cores": 0.25}},
                "cpu_storm": {"cpu": {"max_cores": 0.05}},
            },
            scenario_phases=[
                {"name": "stress", "duration_s": 5, "stressors": ["edge_device", "cpu_storm"]}
            ],
        )
    )
    rt_b = parse_runtime_stress(
        _composable_yaml(
            stressors={
                "edge_device": {"cpu": {"max_cores": 0.25}},
                "cpu_storm": {"cpu": {"max_cores": 0.05}},
            },
            scenario_phases=[
                {"name": "stress", "duration_s": 5, "stressors": ["cpu_storm", "edge_device"]}
            ],
        )
    )
    a = compile_runtime_stress_request(rt_a, rt_a.scenarios[0])
    b = compile_runtime_stress_request(rt_b, rt_b.scenarios[0])
    assert a.phases[0].controls.cpu.max_cores == 0.05
    assert b.phases[0].controls.cpu.max_cores == 0.05


def test_compile_inline_controls_merged_with_stressors_tighter_wins() -> None:
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        _composable_yaml(
            stressors={
                "edge_device": {
                    "cpu": {"max_cores": 0.25},
                    "io": {"read_bps": 10_000_000},
                }
            },
            scenario_phases=[
                {
                    "name": "stress",
                    "duration_s": 5,
                    "stressors": ["edge_device"],
                    "controls": {
                        # tighter on io.read_bps -> wins; looser on cpu -> ignored
                        "io": {"read_bps": 1_000_000},
                        "cpu": {"max_cores": 0.99},
                    },
                }
            ],
        )
    )
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    controls = request.phases[0].controls
    assert controls.cpu.max_cores == 0.25  # edge_device wins (tighter)
    assert controls.io.read_bps == 1_000_000  # inline wins (tighter)


def test_parse_rejects_phase_referencing_undeclared_stressor() -> None:
    with pytest.raises(ValueError, match="not declared"):
        parse_runtime_stress(
            _composable_yaml(
                stressors={"edge_device": {"cpu": {"max_cores": 0.25}}},
                scenario_phases=[
                    {"name": "stress", "duration_s": 5, "stressors": ["typo_name"]}
                ],
            )
        )


def test_parse_rejects_stressor_with_invalid_axis_value() -> None:
    with pytest.raises(ValueError, match="max_cores"):
        parse_runtime_stress(
            _composable_yaml(
                stressors={"bad": {"cpu": {"max_cores": -1}}},
                scenario_phases=[
                    {"name": "stress", "duration_s": 5, "stressors": ["bad"]}
                ],
            )
        )


def test_parse_rejects_stressor_with_no_axes() -> None:
    with pytest.raises(ValueError, match="at least one of"):
        parse_runtime_stress(
            _composable_yaml(
                stressors={"empty": {}},
                scenario_phases=[
                    {"name": "stress", "duration_s": 5, "stressors": ["empty"]}
                ],
            )
        )


def test_parse_rejects_phase_stressors_not_a_list() -> None:
    with pytest.raises(ValueError, match="stressors must be a list"):
        parse_runtime_stress(
            _composable_yaml(
                stressors={"edge_device": {"cpu": {"max_cores": 0.25}}},
                scenario_phases=[
                    {"name": "stress", "duration_s": 5, "stressors": "edge_device"}
                ],
            )
        )


def test_compile_phase_with_no_stressors_or_controls_yields_empty_controls() -> None:
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        _composable_yaml(scenario_phases=[{"name": "warmup", "duration_s": 5}])
    )
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    controls = request.phases[0].controls
    assert controls.cpu is None
    assert controls.memory is None
    assert controls.gpu is None
    assert controls.io is None


def test_existing_inline_only_phase_unchanged_after_composability() -> None:
    """Backward compat: a phase that uses only `controls:` (no stressors)
    must produce the same compiled output as before this feature shipped."""
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        _composable_yaml(
            scenario_phases=[
                {
                    "name": "stress",
                    "duration_s": 10,
                    "controls": {
                        "cpu": {"max_cores": 0.5},
                        "memory": {"max_mb": 512},
                    },
                }
            ]
        )
    )
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    controls = request.phases[0].controls
    assert controls.cpu.max_cores == 0.5
    assert controls.memory.max_mb == 512
    assert controls.gpu is None
    assert controls.io is None


def test_scenario_with_per_phase_gpu_caps_via_stressors_compiles() -> None:
    """Phases referencing GPU stressors with different caps are legal now —
    GpuHamiController mutates limit[0] at phase boundaries."""
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        _composable_yaml(
            stressors={
                "low_vram": {"gpu": {"vram_limit_mb": 4096}},
                "high_vram": {"gpu": {"vram_limit_mb": 8192}},
            },
            scenario_phases=[
                {"name": "phase_a", "duration_s": 5, "stressors": ["low_vram"]},
                {"name": "phase_b", "duration_s": 5, "stressors": ["high_vram"]},
            ],
        )
    )
    assert rt is not None
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    assert request.phases[0].controls.gpu.vram_limit_mb == 4096
    assert request.phases[1].controls.gpu.vram_limit_mb == 8192


def test_parse_runtime_stress_accepts_realtime_target_fps() -> None:
    """Schema-level: realtime block on a scenario parses and validates."""
    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "rt_scenario",
                        "realtime": {"target_fps": 10},
                        "phases": [
                            {"name": "stress", "duration_s": 5},
                        ],
                    }
                ],
            }
        }
    )
    assert rt is not None
    assert rt.scenarios[0].realtime is not None
    assert rt.scenarios[0].realtime.target_fps == 10


def test_parse_runtime_stress_omits_realtime_by_default() -> None:
    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "no_rt",
                        "phases": [{"name": "stress", "duration_s": 5}],
                    }
                ],
            }
        }
    )
    assert rt is not None
    assert rt.scenarios[0].realtime is None


def test_parse_runtime_stress_rejects_non_positive_realtime_fps() -> None:
    with pytest.raises(ValueError):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "bad_rt",
                            "realtime": {"target_fps": 0},
                            "phases": [{"name": "stress", "duration_s": 5}],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_compiles_realtime_into_request() -> None:
    """End-to-end: schema -> compile_runtime_stress_request -> RealtimeDeadline."""
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "rt_compile",
                        "realtime": {"target_fps": 30},
                        "phases": [{"name": "stress", "duration_s": 10}],
                    }
                ],
            }
        }
    )
    assert rt is not None
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    assert request.realtime is not None
    assert request.realtime.target_fps == 30.0


def test_parse_runtime_stress_no_realtime_compiles_to_none() -> None:
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "no_rt_compile",
                        "phases": [{"name": "stress", "duration_s": 5}],
                    }
                ],
            }
        }
    )
    assert rt is not None
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    assert request.realtime is None


def test_parse_runtime_stress_realtime_warmup_frames_default_zero() -> None:
    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "rt_default_warmup",
                        "realtime": {"target_fps": 10},
                        "phases": [{"name": "stress", "duration_s": 5}],
                    }
                ],
            }
        }
    )
    assert rt is not None
    assert rt.scenarios[0].realtime.warmup_frames == 0


def test_parse_runtime_stress_accepts_realtime_warmup_frames() -> None:
    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "rt_with_warmup",
                        "realtime": {"target_fps": 10, "warmup_frames": 5},
                        "phases": [{"name": "stress", "duration_s": 5}],
                    }
                ],
            }
        }
    )
    assert rt is not None
    assert rt.scenarios[0].realtime.warmup_frames == 5


def test_parse_runtime_stress_rejects_negative_warmup_frames() -> None:
    with pytest.raises(ValueError):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "bad_warmup",
                            "realtime": {"target_fps": 10, "warmup_frames": -1},
                            "phases": [{"name": "stress", "duration_s": 5}],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_compiles_warmup_frames() -> None:
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "rt_warmup_compile",
                        "realtime": {"target_fps": 10, "warmup_frames": 5},
                        "phases": [{"name": "stress", "duration_s": 5}],
                    }
                ],
            }
        }
    )
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    assert request.realtime.warmup_frames == 5


def test_parse_runtime_stress_realtime_queue_size_default_one() -> None:
    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "rt_default_queue",
                        "realtime": {"target_fps": 10},
                        "phases": [{"name": "stress", "duration_s": 5}],
                    }
                ],
            }
        }
    )
    assert rt is not None
    assert rt.scenarios[0].realtime.queue_size == 1


def test_parse_runtime_stress_accepts_realtime_queue_size() -> None:
    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "rt_with_queue",
                        "realtime": {"target_fps": 10, "queue_size": 8},
                        "phases": [{"name": "stress", "duration_s": 5}],
                    }
                ],
            }
        }
    )
    assert rt is not None
    assert rt.scenarios[0].realtime.queue_size == 8


def test_parse_runtime_stress_rejects_zero_queue_size() -> None:
    with pytest.raises(ValueError):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "bad_queue",
                            "realtime": {"target_fps": 10, "queue_size": 0},
                            "phases": [{"name": "stress", "duration_s": 5}],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_compiles_queue_size() -> None:
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "rt_queue_compile",
                        "realtime": {"target_fps": 10, "queue_size": 4},
                        "phases": [{"name": "stress", "duration_s": 5}],
                    }
                ],
            }
        }
    )
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    assert request.realtime.queue_size == 4


def test_parse_runtime_stress_realtime_drop_policy_default() -> None:
    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "rt_default_policy",
                        "realtime": {"target_fps": 10},
                        "phases": [{"name": "stress", "duration_s": 5}],
                    }
                ],
            }
        }
    )
    assert rt is not None
    assert rt.scenarios[0].realtime.drop_policy == "drop_oldest"


def test_parse_runtime_stress_accepts_drop_newest() -> None:
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "rt_drop_newest",
                        "realtime": {"target_fps": 10, "drop_policy": "drop_newest"},
                        "phases": [{"name": "stress", "duration_s": 5}],
                    }
                ],
            }
        }
    )
    assert rt is not None
    assert rt.scenarios[0].realtime.drop_policy == "drop_newest"
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    assert request.realtime.drop_policy == "drop_newest"


def test_parse_runtime_stress_rejects_invalid_drop_policy() -> None:
    with pytest.raises(ValueError):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "bad_policy",
                            "realtime": {"target_fps": 10, "drop_policy": "sideways"},
                            "phases": [{"name": "stress", "duration_s": 5}],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_frame_anchored_phases() -> None:
    from slamadversariallab.runtime_stress.models import compile_runtime_stress_request

    rt = parse_runtime_stress(
        {
            "runtime_stress": {
                "enabled": True,
                "scenarios": [
                    {
                        "name": "frame_phases",
                        "realtime": {"target_fps": 15},
                        "phases": [
                            {"name": "a", "until_frame": 100,
                             "controls": {"cpu": {"max_cores": 0.5}}},
                            {"name": "b", "until_frame": 300},
                        ],
                    }
                ],
            }
        }
    )
    assert rt is not None
    request = compile_runtime_stress_request(rt, rt.scenarios[0])
    assert request.frame_anchored is True
    assert [p.until_frame for p in request.phases] == [100, 300]
    assert all(p.duration_s is None for p in request.phases)


def test_parse_runtime_stress_rejects_frame_phases_without_realtime() -> None:
    with pytest.raises(ValueError):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "no_realtime",
                            "phases": [{"name": "a", "until_frame": 100}],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_rejects_mixed_time_and_frame_phases() -> None:
    with pytest.raises(ValueError):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "mixed",
                            "realtime": {"target_fps": 15},
                            "phases": [
                                {"name": "a", "duration_s": 5},
                                {"name": "b", "until_frame": 100},
                            ],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_rejects_non_increasing_until_frame() -> None:
    with pytest.raises(ValueError):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "backwards",
                            "realtime": {"target_fps": 15},
                            "phases": [
                                {"name": "a", "until_frame": 200},
                                {"name": "b", "until_frame": 100},
                            ],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_rejects_phase_with_both_anchors() -> None:
    with pytest.raises(ValueError):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "both",
                            "realtime": {"target_fps": 15},
                            "phases": [
                                {"name": "a", "duration_s": 5, "until_frame": 100},
                            ],
                        }
                    ],
                }
            }
        )


def test_parse_runtime_stress_rejects_phase_with_no_anchor() -> None:
    with pytest.raises(ValueError):
        parse_runtime_stress(
            {
                "runtime_stress": {
                    "enabled": True,
                    "scenarios": [
                        {
                            "name": "none",
                            "phases": [{"name": "a"}],
                        }
                    ],
                }
            }
        )
