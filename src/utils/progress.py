"""Progress tracking utilities for SLAMAdverserialLab."""

import sys
import time
from contextlib import contextmanager
from typing import Optional, Iterator, Any, Dict
from tqdm import tqdm
from tqdm.auto import tqdm as auto_tqdm

from . import get_logger

logger = get_logger(__name__)


class ProgressTracker:
    """Manage progress bars used across the pipeline."""

    def __init__(self, disable: bool = False, use_auto: bool = False):
        """
        Initialize progress tracker.

        Args:
            disable: If True, disable all progress bars
            use_auto: If True, use auto-detect for notebook/console
        """
        self.disable = disable
        self.use_auto = use_auto
        self._active_bars = []
        self._start_times = {}

        # Choose tqdm implementation
        self.tqdm_class = auto_tqdm if use_auto else tqdm

    def track_frames(
        self,
        frames: Iterator,
        total: int,
        desc: str = "Processing frames",
        unit: str = "frames",
        leave: bool = True
    ) -> Iterator:
        """
        Track progress through frame iteration.

        Args:
            frames: Iterator over frames
            total: Total number of frames
            desc: Description for progress bar
            unit: Unit name for display
            leave: Whether to leave progress bar after completion

        Yields:
            Frame from the iterator
        """
        with self.tqdm_class(
            total=total,
            desc=desc,
            unit=unit,
            disable=self.disable,
            leave=leave,
            file=sys.stdout,
            dynamic_ncols=True
        ) as pbar:
            self._active_bars.append(pbar)
            try:
                for frame in frames:
                    yield frame
                    pbar.update(1)
            finally:
                self._active_bars.remove(pbar)

    def track_modules(
        self,
        modules: list,
        desc: str = "Initializing modules"
    ) -> Iterator:
        """
        Track progress through module operations.

        Args:
            modules: List of modules to process
            desc: Description for progress bar

        Yields:
            Module from the list
        """
        with self.tqdm_class(
            total=len(modules),
            desc=desc,
            unit="modules",
            disable=self.disable,
            leave=False,
            file=sys.stdout
        ) as pbar:
            self._active_bars.append(pbar)
            try:
                for module in modules:
                    yield module
                    pbar.update(1)
            finally:
                self._active_bars.remove(pbar)

    def track_dataset_loading(
        self,
        dataset_type: str,
        path: str
    ) -> 'ProgressContext':
        """
        Create a progress context for dataset loading.

        Args:
            dataset_type: Type of dataset being loaded
            path: Dataset path

        Returns:
            Progress context manager
        """
        desc = f"Loading {dataset_type} dataset"
        return ProgressContext(self, desc, unit="files")

    def track_pipeline_stage(
        self,
        stage_name: str,
        total_steps: Optional[int] = None
    ) -> 'ProgressContext':
        """
        Track a pipeline stage.

        Args:
            stage_name: Name of the stage
            total_steps: Total number of steps (if known)

        Returns:
            Progress context manager
        """
        return ProgressContext(self, stage_name, total=total_steps)

    def update_status(self, message: str, level: str = "info") -> None:
        """
        Update status message.

        Args:
            message: Status message
            level: Log level (info, warning, error)
        """
        if not self.disable:
            # Clear line and print status
            print(f"\r{' ' * 80}\r{message}", end="", flush=True)

        # Also log the message
        getattr(logger, level)(message)

    def close_all(self) -> None:
        """Close all active progress bars."""
        for pbar in self._active_bars:
            pbar.close()
        self._active_bars.clear()


class ProgressContext:
    """Context manager for progress tracking."""

    def __init__(
        self,
        tracker: ProgressTracker,
        desc: str,
        total: Optional[int] = None,
        unit: str = "steps"
    ):
        """
        Initialize progress context.

        Args:
            tracker: Parent progress tracker
            desc: Description for progress bar
            total: Total number of steps
            unit: Unit name
        """
        self.tracker = tracker
        self.desc = desc
        self.total = total
        self.unit = unit
        self.pbar = None
        self.start_time = None

    def __enter__(self) -> 'ProgressContext':
        """Enter context."""
        self.start_time = time.time()

        if self.total is not None:
            self.pbar = self.tracker.tqdm_class(
                total=self.total,
                desc=self.desc,
                unit=self.unit,
                disable=self.tracker.disable,
                leave=False,
                file=sys.stdout
            )
            self.tracker._active_bars.append(self.pbar)
        else:
            # Just print status for unknown total
            self.tracker.update_status(f"{self.desc}...")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        elapsed = time.time() - self.start_time

        if self.pbar is not None:
            self.pbar.close()
            self.tracker._active_bars.remove(self.pbar)

        # Log completion
        if exc_type is None:
            logger.debug(f"{self.desc} completed in {elapsed:.2f}s")
        else:
            logger.error(f"{self.desc} failed after {elapsed:.2f}s")

    def update(self, n: int = 1, status: Optional[str] = None) -> None:
        """
        Update progress.

        Args:
            n: Number of steps to advance
            status: Optional status message
        """
        if self.pbar is not None:
            self.pbar.update(n)
            if status:
                self.pbar.set_description(f"{self.desc} - {status}")
        elif status:
            self.tracker.update_status(f"{self.desc} - {status}")

    def set_total(self, total: int) -> None:
        """
        Set total steps (for initially unknown totals).

        Args:
            total: Total number of steps
        """
        if self.pbar is None and total > 0:
            self.pbar = self.tracker.tqdm_class(
                total=total,
                desc=self.desc,
                unit=self.unit,
                disable=self.tracker.disable,
                leave=False,
                file=sys.stdout
            )
            self.tracker._active_bars.append(self.pbar)
        elif self.pbar is not None:
            self.pbar.total = total
            self.pbar.refresh()


def create_progress_tracker(config: Optional[Dict[str, Any]] = None) -> ProgressTracker:
    """
    Create a progress tracker with configuration.

    Args:
        config: Configuration dictionary with keys:
            - disable: Disable progress bars
            - use_auto: Use auto-detect for notebook/console

    Returns:
        Configured ProgressTracker instance
    """
    if config is None:
        config = {}

    return ProgressTracker(
        disable=config.get('disable', False),
        use_auto=config.get('use_auto', False)
    )


@contextmanager
def track_time(operation: str):
    """
    Simple context manager to track operation time.

    Args:
        operation: Description of operation
    """
    logger.info(f"Starting: {operation}")
    start_time = time.time()

    try:
        yield
        elapsed = time.time() - start_time
        logger.info(f"Completed: {operation} ({elapsed:.2f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed: {operation} ({elapsed:.2f}s) - {e}")
        raise


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_throughput(
    items: int,
    seconds: float,
    unit: str = "items"
) -> str:
    """
    Format throughput rate.

    Args:
        items: Number of items processed
        seconds: Time taken
        unit: Unit name

    Returns:
        Formatted throughput string
    """
    if seconds > 0:
        rate = items / seconds
        if rate >= 1:
            return f"{rate:.1f} {unit}/s"
        else:
            # Show inverse for slow rates
            time_per = seconds / items
            return f"{time_per:.1f}s/{unit}"
    else:
        return f"-- {unit}/s"


def estimate_remaining_time(
    processed: int,
    total: int,
    elapsed: float
) -> Optional[float]:
    """
    Estimate remaining time for operation.

    Args:
        processed: Number of items processed
        total: Total number of items
        elapsed: Elapsed time in seconds

    Returns:
        Estimated remaining time in seconds, or None if cannot estimate
    """
    if processed == 0 or processed >= total:
        return None

    rate = processed / elapsed
    remaining = total - processed

    return remaining / rate
