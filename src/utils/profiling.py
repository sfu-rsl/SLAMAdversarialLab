"""Simple timing utilities for SLAMAdverserialLab.

Provides basic execution time measurement and logging.
"""

import time
from contextlib import contextmanager
from typing import Dict, Optional, Any
from pathlib import Path
import json

from ..utils import get_logger

logger = get_logger(__name__)


class SimpleProfiler:
    """Simple profiler for basic timing measurements."""

    def __init__(self, enabled: bool = True, verbose: bool = False):
        """Initialize simple profiler.

        Args:
            enabled: Whether profiling is enabled
            verbose: Print timing information as it's collected
        """
        self.enabled = enabled
        self.verbose = verbose
        self.timings: Dict[str, float] = {}

        if self.enabled:
            logger.info("Simple profiler initialized")

    @contextmanager
    def timer(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for timing a code block.

        Args:
            name: Name of the operation being timed
            metadata: Optional metadata (ignored in simple profiler)
        """
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.timings[name] = self.timings.get(name, 0) + duration

            if self.verbose:
                logger.info(f"{name}: {duration:.3f}s")

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        if not self.enabled:
            return
        self._start_times = getattr(self, '_start_times', {})
        self._start_times[name] = time.perf_counter()

    def stop_timer(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """Stop a named timer and return duration."""
        if not self.enabled:
            return None

        start_times = getattr(self, '_start_times', {})
        if name not in start_times:
            logger.warning(f"Timer '{name}' was not started")
            return None

        duration = time.perf_counter() - start_times[name]
        del start_times[name]

        self.timings[name] = self.timings.get(name, 0) + duration

        if self.verbose:
            logger.info(f"{name}: {duration:.3f}s")

        return duration

    def get_summary(self) -> Dict[str, Any]:
        """Get timing summary."""
        return {
            'timings': self.timings.copy(),
            'total_operations': len(self.timings),
            'total_time': sum(self.timings.values())
        }

    def print_summary(self) -> None:
        """Print timing summary."""
        if not self.timings:
            logger.info("No timing data collected")
            return

        logger.info("=== Timing Summary ===")
        total_time = sum(self.timings.values())

        for name, duration in sorted(self.timings.items()):
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            logger.info(f"{name}: {duration:.3f}s ({percentage:.1f}%)")

        logger.info(f"Total: {total_time:.3f}s")

    def save_report(self, filepath: Path, format: str = "json") -> None:
        """Save timing report to file."""
        if not self.enabled:
            return

        summary = self.get_summary()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
        else:
            # Text format
            with open(filepath, 'w') as f:
                f.write("=== Timing Summary ===\n")
                total_time = summary['total_time']

                for name, duration in sorted(summary['timings'].items()):
                    percentage = (duration / total_time * 100) if total_time > 0 else 0
                    f.write(f"{name}: {duration:.3f}s ({percentage:.1f}%)\n")

                f.write(f"Total: {total_time:.3f}s\n")

        logger.info(f"Timing report saved to {filepath}")

    def reset(self) -> None:
        """Reset all timing data."""
        self.timings.clear()
        if hasattr(self, '_start_times'):
            self._start_times.clear()


def create_profiler(config: Optional[Dict[str, Any]] = None) -> SimpleProfiler:
    """Create a profiler instance.

    Args:
        config: Configuration dictionary with keys:
            - enabled: bool (default True)
            - verbose: bool (default False)

    Returns:
        SimpleProfiler instance
    """
    if config is None:
        config = {}

    return SimpleProfiler(
        enabled=config.get('enabled', True),
        verbose=config.get('verbose', False)
    )


# Global profiler instance (disabled by default)
_global_profiler: Optional[SimpleProfiler] = None


def get_profiler() -> Optional[SimpleProfiler]:
    """Get the global profiler instance."""
    return _global_profiler


def set_profiler(profiler: Optional[SimpleProfiler]) -> None:
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler


@contextmanager
def profile(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Convenience function for timing with global profiler."""
    profiler = get_profiler()
    if profiler:
        with profiler.timer(name, metadata):
            yield
    else:
        yield
