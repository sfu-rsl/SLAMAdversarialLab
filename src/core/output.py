"""Output saving utilities for perturbed datasets."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from datetime import datetime

from ..utils import get_logger, write_image, write_depth, ensure_dir
from .frame import Frame

logger = get_logger(__name__)


class OutputFormat(Enum):
    """Supported output formats."""

    IMAGES = "images"
    """Individual image files"""


@dataclass
class OutputConfig:
    """Configuration for output saving."""

    format: OutputFormat = OutputFormat.IMAGES
    """Output format to use"""

    base_dir: Path = Path("./results")
    """Base output directory"""

    save_images: bool = True
    """Whether to save RGB images"""

    save_depth: bool = True
    """Whether to save depth maps"""

    save_metadata: bool = True
    """Whether to save metadata"""

    image_format: str = "png"
    """Image file format (png, jpg)"""


    compression: bool = False
    """Whether to compress output"""

    organize_by_module: bool = True
    """Whether to organize output by perturbation module"""

    organize_by_sequence: bool = True
    """Whether to organize output by sequence"""


    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.format, str):
            self.format = OutputFormat(self.format.lower())
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)


class OutputWriter:
    """Base class for output writers."""

    def __init__(self, config: OutputConfig):
        """
        Initialize output writer.

        Args:
            config: Output configuration
        """
        self.config = config
        self.output_dir = config.base_dir

    def setup(self, experiment_name: str, module_name: Optional[str] = None):
        """
        Set up output directory structure.

        Args:
            experiment_name: Name of the experiment
            module_name: Name of the perturbation module
        """
        if experiment_name:
            self.output_dir = self.config.base_dir / experiment_name
        else:
            self.output_dir = self.config.base_dir

        if module_name and self.config.organize_by_module:
            self.output_dir = self.output_dir / "images" / module_name
        elif self.config.organize_by_module and self.config.save_images:
            self.output_dir = self.output_dir / "images"

        ensure_dir(self.output_dir)

        if self.config.format == OutputFormat.IMAGES:
            if self.config.save_images:
                ensure_dir(self.output_dir / "image_2")
                ensure_dir(self.output_dir / "image_3")
            if self.config.save_depth:
                ensure_dir(self.output_dir / "depth")

        logger.info(f"Output directory set up: {self.output_dir}")

    def write_frame(
        self,
        frame: Union[Frame, Dict[str, Any]],
        frame_idx: int,
        sequence_id: Optional[str] = None
    ):
        """
        Write a single frame to output.

        Args:
            frame: Frame data
            frame_idx: Frame index
            sequence_id: Optional sequence identifier
        """
        raise NotImplementedError


    def finalize(self):
        """Finalize output (close files, create archives, etc.)."""
        pass

    def write_metadata(self, metadata: Dict[str, Any]):
        """
        Write metadata to output.

        Args:
            metadata: Metadata dictionary
        """
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.debug(f"Metadata written to {metadata_file}")


class ImageWriter(OutputWriter):
    """Writer for individual image files."""

    def write_frame(
        self,
        frame: Union[Frame, Dict[str, Any]],
        frame_idx: int,
        sequence_id: Optional[str] = None
    ):
        """Write frame as individual image files."""
        # Extract data from frame
        if isinstance(frame, Frame):
            image = frame.image
            depth = frame.depth
            timestamp = frame.timestamp
            rgb_filename = None
            depth_filename = None
            image_right = None
            rgb_filename_right = None
        else:
            image = frame.get('image')
            depth = frame.get('depth')
            timestamp = frame.get('timestamp', frame_idx)
            rgb_filename = frame.get('rgb_filename')
            depth_filename = frame.get('depth_filename')
            image_right = frame.get('image_right')
            rgb_filename_right = frame.get('rgb_filename_right')

        if rgb_filename:
            from pathlib import Path
            base_name = Path(rgb_filename).stem
        elif sequence_id and self.config.organize_by_sequence:
            base_name = f"{sequence_id}_{frame_idx:06d}"
        else:
            base_name = f"{frame_idx:06d}"

        if self.config.save_images and image is not None:
            image_path = self.output_dir / "image_2" / f"{base_name}.{self.config.image_format}"
            write_image(image, image_path)

        if self.config.save_images and image_right is not None:
            if rgb_filename_right:
                from pathlib import Path
                base_name_right = Path(rgb_filename_right).stem
            else:
                base_name_right = base_name

            image_right_path = self.output_dir / "image_3" / f"{base_name_right}.{self.config.image_format}"
            logger.debug(f"Saving right image to: {image_right_path}")
            write_image(image_right, image_right_path)

        if self.config.save_depth and depth is not None:
            if depth_filename:
                depth_base_name = Path(depth_filename).stem
            else:
                depth_base_name = base_name
            depth_path = self.output_dir / "depth" / f"{depth_base_name}.png"
            write_depth(depth, depth_path, scale=5000.0)  # TUM scale



class OutputManager:
    """Manager for handling multiple output formats."""

    def __init__(self, config: OutputConfig):
        """
        Initialize output manager.

        Args:
            config: Output configuration
        """
        self.config = config
        self.writers: Dict[str, OutputWriter] = {}

    def setup(
        self,
        experiment_name: str,
        modules: Optional[List[str]] = None,
        formats: Optional[List[OutputFormat]] = None
    ):
        """
        Set up output writers.

        Args:
            experiment_name: Name of the experiment
            modules: List of module names
            formats: Output formats to use (None = use config)
        """
        formats = formats or [self.config.format]

        for format_type in formats:
            if format_type != OutputFormat.IMAGES:
                logger.warning(f"Unsupported output format: {format_type}")
                continue

            if modules and self.config.organize_by_module:
                for module in modules:
                    writer = ImageWriter(self.config)
                    writer.setup(experiment_name, module)
                    self.writers[f"images_{module}"] = writer
            else:
                writer = ImageWriter(self.config)
                writer.setup(experiment_name)
                self.writers["images"] = writer

    def write_frame(
        self,
        frame: Union[Frame, Dict[str, Any]],
        frame_idx: int,
        module_name: Optional[str] = None,
        sequence_id: Optional[str] = None
    ):
        """
        Write frame to appropriate writers.

        Args:
            frame: Frame data
            frame_idx: Frame index
            module_name: Perturbation module name
            sequence_id: Sequence identifier
        """
        for writer_key, writer in self.writers.items():
            if module_name and self.config.organize_by_module:
                if module_name not in writer_key:
                    continue

            writer.write_frame(frame, frame_idx, sequence_id)


    def finalize(self):
        """Finalize all writers."""
        for writer in self.writers.values():
            writer.finalize()

        if self.config.compression:
            self._create_archive()

    def _create_archive(self):
        """Create compressed archive of output."""
        import tarfile

        archive_path = self.config.base_dir / f"results_{datetime.now():%Y%m%d_%H%M%S}.tar.gz"

        with tarfile.open(archive_path, 'w:gz') as tar:
            for writer in self.writers.values():
                if writer.output_dir.exists():
                    tar.add(writer.output_dir, arcname=writer.output_dir.name)

        logger.info(f"Archive created: {archive_path}")

    def write_metadata(self, metadata: Dict[str, Any]):
        """
        Write metadata to all writers.

        Args:
            metadata: Metadata dictionary
        """
        for writer in self.writers.values():
            writer.write_metadata(metadata)


def create_output_manager(
    format: Union[str, OutputFormat],
    base_dir: Union[str, Path],
    **kwargs
) -> OutputManager:
    """
    Convenience function to create an output manager.

    Args:
        format: Output format
        base_dir: Base output directory
        **kwargs: Additional configuration options

    Returns:
        Configured OutputManager instance
    """
    config = OutputConfig(
        format=format,
        base_dir=base_dir,
        **kwargs
    )
    return OutputManager(config)
