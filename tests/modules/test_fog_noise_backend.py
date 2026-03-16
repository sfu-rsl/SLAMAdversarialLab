"""Fog noise backend selection tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slamadverseriallab.config.schema import PerturbationConfig
from slamadverseriallab.modules.base import ModuleSetupContext


def _setup_context(tmp_path: Path) -> ModuleSetupContext:
    return ModuleSetupContext(
        dataset=object(),
        dataset_path=tmp_path,
        total_frames=3,
        input_path=None,
    )


def _stub_depth_setup(self, context=None):
    self.depth_dirs = {"left": Path("/tmp/fake_depth")}
    self.cameras = ["left"]
    self._depth_setup_complete = True


def test_fog_noise_backend_falls_back_to_perlin_when_simplex_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from slamadverseriallab.modules.scene.fog import FogModule
    import slamadverseriallab.utils.noise as noise_utils

    monkeypatch.setattr(FogModule, "_setup_depth_estimation", _stub_depth_setup)

    def _raise_import_error(self):
        raise ImportError("mock missing SimplexNoise")

    monkeypatch.setattr(FogModule, "_import_simplex_noise_module", _raise_import_error)

    perlin_calls = {"count": 0}

    def _fake_perlin(*args, **kwargs):
        perlin_calls["count"] += 1
        shape = kwargs.get("shape", (4, 4))
        return np.full(shape, 0.5, dtype=np.float32)

    monkeypatch.setattr(noise_utils, "generate_perlin_noise_2d", _fake_perlin)

    module = FogModule(
        PerturbationConfig(
            name="fog_fallback",
            type="none",
            parameters={"add_noise": True, "noise_backend": "auto"},
        )
    )
    module.setup(_setup_context(tmp_path))

    assert module.use_simplex is False
    assert module.noise_backend == "perlin_fallback"

    _ = module._apply_3d_noise(np.ones((4, 4), dtype=np.float32), beta=0.1, frame_idx=0)
    assert perlin_calls["count"] == 1


def test_fog_noise_backend_uses_simplex_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from slamadverseriallab.modules.scene.fog import FogModule

    monkeypatch.setattr(FogModule, "_setup_depth_estimation", _stub_depth_setup)

    class _FakeSimplexBackend:
        class SimplexNoise:
            def setup(self, depth):
                return None

            def noise3d(self, x, y, z):
                return np.zeros_like(x, dtype=np.float32)

    monkeypatch.setattr(
        FogModule,
        "_import_simplex_noise_module",
        lambda self: (_FakeSimplexBackend, "mock_simplex"),
    )

    module = FogModule(
        PerturbationConfig(
            name="fog_simplex",
            type="none",
            parameters={"add_noise": True, "noise_steps": 3, "noise_backend": "auto"},
        )
    )
    module.setup(_setup_context(tmp_path))

    assert module.use_simplex is True
    assert module.noise_backend == "simplex"

    beta_map = module._apply_3d_noise(np.ones((6, 6), dtype=np.float32), beta=0.1, frame_idx=0)
    assert beta_map.shape == (6, 6)


def test_fog_noise_backend_simplex_raises_when_simplex_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from slamadverseriallab.modules.scene.fog import FogModule

    monkeypatch.setattr(FogModule, "_setup_depth_estimation", _stub_depth_setup)
    monkeypatch.setattr(
        FogModule,
        "_import_simplex_noise_module",
        lambda self: (_ for _ in ()).throw(ImportError("mock missing SimplexNoise")),
    )

    module = FogModule(
        PerturbationConfig(
            name="fog_simplex_required",
            type="none",
            parameters={"add_noise": True, "noise_backend": "simplex"},
        )
    )

    with pytest.raises(RuntimeError, match="noise_backend='simplex'"):
        module.setup(_setup_context(tmp_path))


def test_fog_noise_backend_perlin_forces_perlin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from slamadverseriallab.modules.scene.fog import FogModule
    import slamadverseriallab.utils.noise as noise_utils

    monkeypatch.setattr(FogModule, "_setup_depth_estimation", _stub_depth_setup)

    calls = {"simplex_import": 0, "perlin": 0}

    def _should_not_be_called(self):
        calls["simplex_import"] += 1
        raise AssertionError("Simplex import should not be called for forced perlin")

    def _fake_perlin(*args, **kwargs):
        calls["perlin"] += 1
        shape = kwargs.get("shape", (4, 4))
        return np.full(shape, 0.5, dtype=np.float32)

    monkeypatch.setattr(FogModule, "_import_simplex_noise_module", _should_not_be_called)
    monkeypatch.setattr(noise_utils, "generate_perlin_noise_2d", _fake_perlin)

    module = FogModule(
        PerturbationConfig(
            name="fog_forced_perlin",
            type="none",
            parameters={"add_noise": True, "noise_backend": "perlin"},
        )
    )
    module.setup(_setup_context(tmp_path))

    assert module.use_simplex is False
    assert module.noise_backend == "perlin_forced"

    _ = module._apply_3d_noise(np.ones((4, 4), dtype=np.float32), beta=0.1, frame_idx=0)
    assert calls["simplex_import"] == 0
    assert calls["perlin"] == 1


def test_fog_noise_backend_ignored_when_noise_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from slamadverseriallab.modules.scene.fog import FogModule

    monkeypatch.setattr(FogModule, "_setup_depth_estimation", _stub_depth_setup)

    module = FogModule(
        PerturbationConfig(
            name="fog_noise_disabled",
            type="none",
            parameters={"add_noise": False, "noise_backend": "perlin"},
        )
    )
    module.setup(_setup_context(tmp_path))
    assert module.noise_backend == "disabled"


def test_fog_get_config_reports_selected_noise_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from slamadverseriallab.modules.scene.fog import FogModule

    monkeypatch.setattr(FogModule, "_setup_depth_estimation", _stub_depth_setup)
    monkeypatch.setattr(
        FogModule,
        "_import_simplex_noise_module",
        lambda self: (_ for _ in ()).throw(ImportError("mock missing SimplexNoise")),
    )

    module = FogModule(
        PerturbationConfig(
            name="fog_config_backend",
            type="none",
            parameters={"add_noise": True, "noise_backend": "auto"},
        )
    )
    module.setup(_setup_context(tmp_path))

    cfg = module.get_config()
    assert cfg["noise_backend_requested"] == "auto"
    assert cfg["noise_backend"] == "perlin_fallback"
