"""Tests for depth backend resolver fallback ordering."""

from __future__ import annotations

from pathlib import Path

import pytest

from slamadverseriallab.depth.providers import (
    DepthProviderNotApplicable,
    ProviderResult,
)
from slamadverseriallab.depth.resolver import DepthBackendResolver


def test_default_resolver_provider_order_is_da3_chain() -> None:
    resolver = DepthBackendResolver()
    assert [provider.backend_name for provider in resolver.providers] == [
        "existing",
        "foundation_stereo",
        "da3",
    ]


def test_resolver_records_candidates_across_skip_fail_and_select(tmp_path: Path) -> None:
    class _SkipProvider:
        backend_name = "existing"

        def setup(self, module, source_path: Path, dataset, cameras):
            raise DepthProviderNotApplicable("no reusable depth")

    class _FailProvider:
        backend_name = "foundation_stereo"

        def setup(self, module, source_path: Path, dataset, cameras):
            raise RuntimeError("stereo backend failed")

    class _SelectedProvider:
        backend_name = "da3"

        def setup(self, module, source_path: Path, dataset, cameras):
            return ProviderResult(
                backend="da3",
                depth_source="da3",
                depth_dirs={"left": tmp_path / "left_da3_depth"},
            )

    resolver = DepthBackendResolver()
    resolver.providers = [_SkipProvider(), _FailProvider(), _SelectedProvider()]

    result = resolver.resolve(
        module=object(),
        source_path=tmp_path,
        dataset=object(),
        cameras=["left"],
    )

    assert result.backend == "da3"
    assert result.depth_source == "da3"
    assert result.depth_dirs == {"left": tmp_path / "left_da3_depth"}
    assert result.candidates == [
        {
            "provider": "existing",
            "status": "skipped",
            "message": "no reusable depth",
        },
        {
            "provider": "foundation_stereo",
            "status": "failed",
            "message": "stereo backend failed",
        },
        {
            "provider": "da3",
            "status": "selected",
            "message": "ok",
        },
    ]


def test_resolver_forced_da2_uses_only_da2_provider(tmp_path: Path) -> None:
    class _ShouldNotRun:
        backend_name = "da3"

        def setup(self, module, source_path: Path, dataset, cameras):
            raise AssertionError("auto-chain provider should not run in forced mode")

    class _ForcedDA2Provider:
        backend_name = "da2"

        def setup(self, module, source_path: Path, dataset, cameras):
            return ProviderResult(
                backend="da2",
                depth_source="dav2",
                depth_dirs={"left": tmp_path / "left_depth"},
            )

    resolver = DepthBackendResolver()
    resolver.providers = [_ShouldNotRun()]
    resolver.providers_by_name = {"da3": _ShouldNotRun(), "da2": _ForcedDA2Provider()}

    result = resolver.resolve(
        module=object(),
        source_path=tmp_path,
        dataset=object(),
        cameras=["left"],
        preferred_backend="da2",
    )

    assert result.backend == "da2"
    assert result.depth_source == "dav2"
    assert result.depth_dirs == {"left": tmp_path / "left_depth"}
    assert result.candidates == [
        {
            "provider": "da2",
            "status": "selected",
            "message": "ok",
        }
    ]


def test_resolver_forced_backend_does_not_fallback(tmp_path: Path) -> None:
    called = {"da3": False}

    class _FailDA2Provider:
        backend_name = "da2"

        def setup(self, module, source_path: Path, dataset, cameras):
            raise RuntimeError("da2 unavailable")

    class _DA3Provider:
        backend_name = "da3"

        def setup(self, module, source_path: Path, dataset, cameras):
            called["da3"] = True
            return ProviderResult(
                backend="da3",
                depth_source="da3",
                depth_dirs={"left": tmp_path / "left_da3_depth"},
            )

    resolver = DepthBackendResolver()
    resolver.providers_by_name = {"da2": _FailDA2Provider(), "da3": _DA3Provider()}

    with pytest.raises(RuntimeError, match="Requested depth backend 'da2' failed"):
        resolver.resolve(
            module=object(),
            source_path=tmp_path,
            dataset=object(),
            cameras=["left"],
            preferred_backend="da2",
        )

    assert called["da3"] is False
