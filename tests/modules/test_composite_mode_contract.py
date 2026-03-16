"""Tests for sequential-only composite mode contract."""

from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from slamadverseriallab.config.schema import PerturbationConfig
from slamadverseriallab.modules.base import (
    CompositeModule,
    CompositionMode,
    ModuleSetupContext,
    PerturbationModule,
)


class _AddValueModule(PerturbationModule):
    """Simple deterministic module that adds a constant to the image."""

    def __init__(self, config: PerturbationConfig, increment: int) -> None:
        super().__init__(config)
        self.increment = increment
        self.calls = 0

    def _setup(self, context: ModuleSetupContext) -> None:
        return

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str,
        **kwargs,
    ) -> np.ndarray:
        self.calls += 1
        out = image.astype(np.int16) + self.increment
        return np.clip(out, 0, 255).astype(np.uint8)


class _DropModule(PerturbationModule):
    """Module that drops every frame by returning None."""

    def __init__(self, config: PerturbationConfig) -> None:
        super().__init__(config)
        self.calls = 0

    def _setup(self, context: ModuleSetupContext) -> None:
        return

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str,
        **kwargs,
    ) -> Optional[np.ndarray]:
        self.calls += 1
        return None


def _setup_context(tmp_path: Path) -> ModuleSetupContext:
    return ModuleSetupContext(
        dataset=object(),
        dataset_path=tmp_path,
        total_frames=1,
        input_path=None,
    )


def test_composite_mode_enum_is_sequential_only() -> None:
    assert [mode.value for mode in CompositionMode] == ["sequential"]


def test_composite_config_validation_rejects_parallel_mode() -> None:
    cfg = PerturbationConfig(
        name="composite_parallel_invalid",
        type="composite",
        parameters={
            "mode": "parallel",
            "modules": [{"name": "noop", "type": "none"}],
        },
    )

    with pytest.raises(ValueError, match="Invalid composition mode 'parallel'"):
        cfg.validate()


def test_composite_constructor_rejects_parallel_mode(tmp_path: Path) -> None:
    child = _AddValueModule(PerturbationConfig(name="child", type="none"), increment=1)

    with pytest.raises(ValueError, match="Only 'sequential' is supported"):
        CompositeModule(
            PerturbationConfig(name="composite_invalid", type="composite"),
            modules=[child],
            mode="parallel",
        )


def test_composite_sequential_applies_modules_in_order(tmp_path: Path) -> None:
    m1 = _AddValueModule(PerturbationConfig(name="m1", type="none"), increment=10)
    m2 = _AddValueModule(PerturbationConfig(name="m2", type="none"), increment=20)
    composite = CompositeModule(
        PerturbationConfig(name="composite_seq", type="composite"),
        modules=[m1, m2],
        mode="sequential",
    )
    composite.setup(_setup_context(tmp_path))

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    out = composite.apply(image, depth=None, frame_idx=0, camera="left")

    assert out is not None
    assert int(out[0, 0, 0]) == 30
    assert m1.calls == 1
    assert m2.calls == 1


def test_composite_sequential_short_circuits_when_child_drops_frame(tmp_path: Path) -> None:
    dropper = _DropModule(PerturbationConfig(name="dropper", type="none"))
    after = _AddValueModule(PerturbationConfig(name="after", type="none"), increment=10)
    composite = CompositeModule(
        PerturbationConfig(name="composite_drop", type="composite"),
        modules=[dropper, after],
        mode="sequential",
    )
    composite.setup(_setup_context(tmp_path))

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    out = composite.apply(image, depth=None, frame_idx=0, camera="left")

    assert out is None
    assert dropper.calls == 1
    assert after.calls == 0
