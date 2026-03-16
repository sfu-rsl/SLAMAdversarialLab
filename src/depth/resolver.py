"""Depth backend resolver with deterministic fallback order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..utils import get_logger
from .providers import (
    DA2Provider,
    DA3Provider,
    DepthProvider,
    DepthProviderNotApplicable,
    ExistingDepthProvider,
    FoundationStereoProvider,
)

logger = get_logger(__name__)


@dataclass
class DepthResolution:
    """Depth backend resolution output."""

    backend: str
    depth_source: str
    depth_dirs: Dict[str, Path]
    candidates: List[Dict[str, str]]


class DepthBackendResolver:
    """Resolve depth backend with auto fallback chain or explicit selection.

    Auto order:
      1) Existing depth declared by dataset
      2) FoundationStereo (stereo + calibration)
      3) DA3
    """

    def __init__(self) -> None:
        self.providers: List[DepthProvider] = [
            ExistingDepthProvider(),
            FoundationStereoProvider(),
            DA3Provider(),
        ]
        self.providers_by_name: Dict[str, DepthProvider] = {
            provider.backend_name: provider for provider in self.providers
        }
        # DA2 is explicit-only: available when requested, not part of auto chain.
        self.providers_by_name["da2"] = DA2Provider()

    def resolve(
        self,
        module: Any,
        source_path: Path,
        dataset: Any,
        cameras: List[str],
        preferred_backend: str = "auto",
    ) -> DepthResolution:
        preferred = str(preferred_backend or "auto").strip().lower()
        if not preferred:
            preferred = "auto"

        forced_mode = preferred != "auto"
        if forced_mode:
            provider = self.providers_by_name.get(preferred)
            if provider is None:
                available = ", ".join(sorted(["auto", *self.providers_by_name.keys()]))
                raise ValueError(
                    f"Unknown depth backend preference '{preferred_backend}'. "
                    f"Supported values: {available}"
                )
            providers = [provider]
        else:
            providers = self.providers

        candidates: List[Dict[str, str]] = []

        for provider in providers:
            try:
                result = provider.setup(module, source_path, dataset, cameras)
                candidates.append({
                    "provider": provider.backend_name,
                    "status": "selected",
                    "message": "ok",
                })
                logger.info("Depth backend selected: %s", result.backend)
                return DepthResolution(
                    backend=result.backend,
                    depth_source=result.depth_source,
                    depth_dirs=result.depth_dirs,
                    candidates=candidates,
                )
            except DepthProviderNotApplicable as e:
                candidates.append({
                    "provider": provider.backend_name,
                    "status": "skipped",
                    "message": str(e),
                })
                logger.info("Depth backend skipped (%s): %s", provider.backend_name, e)
                if forced_mode:
                    raise RuntimeError(
                        f"Requested depth backend '{preferred}' is not applicable: {e}"
                    ) from e
            except Exception as e:  # noqa: BLE001 - must continue fallback chain
                candidates.append({
                    "provider": provider.backend_name,
                    "status": "failed",
                    "message": str(e),
                })
                logger.warning("Depth backend failed (%s): %s", provider.backend_name, e)
                if forced_mode:
                    raise RuntimeError(
                        f"Requested depth backend '{preferred}' failed: {e}"
                    ) from e

        messages = "; ".join(f"{c['provider']}={c['status']} ({c['message']})" for c in candidates)
        raise RuntimeError(f"Could not resolve any depth backend: {messages}")


def resolve_depth_backend(
    module: Any,
    source_path: Path,
    dataset: Any,
    cameras: List[str],
    preferred_backend: str = "auto",
) -> DepthResolution:
    """Convenience helper to resolve depth backend."""
    return DepthBackendResolver().resolve(
        module,
        source_path,
        dataset,
        cameras,
        preferred_backend=preferred_backend,
    )
