"""Main pipeline for running SLAMAdverserialLab experiments."""

import os
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from ..utils import get_logger, write_image, ensure_dir, ProgressTracker, track_time, copy_or_truncate_text_file
from ..config import Config
from ..modules import PerturbationModule, create_module
from ..modules.base import ModuleSetupContext
from .output import OutputManager, OutputFormat, create_output_manager

logger = get_logger(__name__)


class Pipeline:
    """Run dataset perturbation experiments and write outputs."""

    def __init__(
        self,
        config: Config,
        show_progress: bool = True,
        enable_profiling: Optional[bool] = None  # None = use config value
    ):
        """
        Initialize the pipeline.

        Args:
            config: Complete experiment configuration
            show_progress: Whether to show progress bars
            enable_profiling: Whether to enable profiling (None = use config)
        """
        self.config = config
        self.dataset: Optional[Any] = None  # Will be a Dataset instance
        self.modules: List[PerturbationModule] = []
        self.output_dir: Optional[Path] = None
        self.output_manager: Optional[OutputManager] = None
        self.start_time: Optional[float] = None
        self.progress = ProgressTracker(disable=not show_progress)
        self.profiler = None  # Will be initialized if profiling enabled

        if enable_profiling is not None:
            self.enable_profiling = enable_profiling
        else:
            self.enable_profiling = getattr(config, 'profiling', None) and config.profiling.enabled

        logger.info(f"Initializing pipeline for experiment: {config.experiment.name}")

    def setup(self) -> None:
        """
        Set up the pipeline components.

        This includes:
        - Creating output directory
        - Loading dataset
        - Initializing perturbation modules
        """
        logger.info("Setting up pipeline...")

        self._create_output_dir()

        self.dataset = self.load_dataset()

        self._copy_dataset_files()

        self.modules = self.load_modules()

        if self.config.output.save_images:
            self._setup_output_manager()

        if self.enable_profiling:
            from ..utils.profiling import create_profiler, set_profiler

            profiling_config = {}
            if hasattr(self.config, 'profiling'):
                profiling_config = {
                    'enabled': True,
                    'verbose': self.config.profiling.verbose
                }
            else:
                profiling_config = {'enabled': True}

            self.profiler = create_profiler(profiling_config)
            set_profiler(self.profiler)  # Set as global profiler
            logger.info("Performance profiling enabled")

        logger.info("Pipeline setup complete")

    def _create_output_dir(self) -> None:
        """Create output directory structure."""
        base_dir = Path(self.config.output.base_dir)

        if self.config.output.create_timestamp_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = base_dir / f"{self.config.experiment.name}_{timestamp}"
        else:
            self.output_dir = base_dir / self.config.experiment.name

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory created: {self.output_dir}")

    def _copy_dataset_files(self) -> None:
        """Copy dataset-specific files to output directory based on dataset type."""
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before copying dataset files")

        dataset_type = self.config.dataset.type
        if not dataset_type:
            logger.debug("No dataset type specified, skipping dataset file copying")
            return

        dataset_type = dataset_type.lower()

        if dataset_type == "mock":
            return

        logger.info(f"Dataset type '{dataset_type}' detected, preparing to copy dataset files...")

        dataset = self.dataset
        max_frames = self.config.dataset.max_frames

        metadata_files = dataset.get_metadata_files_with_dest()

        destination_to_sources: Dict[str, List[Path]] = {}
        for file_path, dest_name, _ in metadata_files:
            destination_to_sources.setdefault(dest_name, []).append(file_path)

        duplicate_destinations = {
            dest_name: sources
            for dest_name, sources in destination_to_sources.items()
            if len(sources) > 1
        }
        if duplicate_destinations:
            collision_lines = []
            for dest_name, sources in sorted(duplicate_destinations.items()):
                source_list = ", ".join(str(path) for path in sources)
                collision_lines.append(f"{dest_name}: {source_list}")
            collisions = "; ".join(collision_lines)
            raise RuntimeError(
                f"Duplicate metadata destination filenames detected for dataset "
                f"'{self.config.dataset.type}': {collisions}"
            )

        for file_path, dest_name, should_truncate in metadata_files:
            if file_path.exists():
                dst_path = self.output_dir / dest_name
                if should_truncate and max_frames:
                    copy_or_truncate_text_file(
                        file_path,
                        dst_path,
                        max_lines=max_frames,
                        preserve_comments=True
                    )
                    logger.info(
                        f"Copied {file_path} to {dst_path} "
                        f"(truncated to {max_frames} frames)"
                    )
                else:
                    shutil.copy2(file_path, dst_path)
                    logger.info(f"Copied {file_path} to {dst_path}")

    def _setup_output_manager(self) -> None:
        """Set up output manager for saving perturbed frames."""
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before output manager setup")

        output_format = OutputFormat.IMAGES
        dataset = self.dataset

        # Don't save depth for datasets that skip depth or don't have native depth
        save_depth = not self.config.dataset.skip_depth and dataset.has_depth()

        is_kitti = self.config.dataset.type.lower() == 'kitti'

        from .output import OutputConfig as OutputSaveConfig
        save_config = OutputSaveConfig(
            format=output_format,
            base_dir=self.output_dir,
            save_images=self.config.output.save_images,
            save_depth=save_depth,
            save_metadata=False,  # Metadata not used in current implementation
            organize_by_module=True,
            organize_by_sequence=not is_kitti  # Don't organize by sequence for KITTI (already in filename)
        )

        self.output_manager = OutputManager(save_config)

        module_names = [m.name for m in self.modules if m.enabled]

        self.output_manager.setup(
            experiment_name="",
            modules=module_names
        )

        logger.info(f"Output manager configured for {len(module_names)} modules")

        if self.config.output.save_images and self.config.dataset.type.lower() != "mock":
            max_frames = self.config.dataset.max_frames

            metadata_files = dataset.get_metadata_files_with_dest()

            for module_name in module_names:
                module_dir = self.output_dir / "images" / module_name
                module_dir.mkdir(parents=True, exist_ok=True)

                for file_path, dest_name, should_truncate in metadata_files:
                    if file_path.exists():
                        if should_truncate and max_frames:
                            copy_or_truncate_text_file(
                                file_path,
                                module_dir / dest_name,
                                max_lines=max_frames,
                                preserve_comments=True
                            )
                        else:
                            shutil.copy2(file_path, module_dir / dest_name)

                self._create_output_camera_aliases(dataset, module_dir)

                logger.info(f"Copied metadata files to {module_dir}")

    def _create_output_camera_aliases(self, dataset: Any, module_dir: Path) -> None:
        """Create dataset-camera aliases to canonical output camera directories."""
        camera_roles = dataset.get_active_camera_roles()
        if "left" not in camera_roles:
            raise RuntimeError(
                "Dataset must expose an active 'left' camera role during output camera alias creation."
            )

        for camera_role in camera_roles:
            dataset_camera = dataset.get_image_directory_name(camera_role)
            canonical_camera = dataset.get_canonical_camera_name(camera_role)

            if not dataset_camera or dataset_camera == canonical_camera:
                continue

            target_dir = module_dir / canonical_camera
            alias_path = module_dir / dataset_camera

            if not target_dir.exists():
                logger.warning(
                    f"Cannot create camera alias '{dataset_camera}' -> '{canonical_camera}' "
                    f"because target directory does not exist: {target_dir}"
                )
                continue

            if alias_path.exists() or alias_path.is_symlink():
                continue

            alias_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                relative_target = os.path.relpath(target_dir, start=alias_path.parent)
                os.symlink(relative_target, alias_path)
                logger.debug(
                    f"Created camera alias {alias_path} -> {relative_target}"
                )
            except OSError as e:
                logger.warning(
                    f"Could not create camera alias {alias_path} -> {target_dir}: {e}"
                )


    def load_dataset(self):
        """
        Load the dataset based on configuration.

        Returns:
            Loaded dataset instance
        """
        logger.info(f"Loading dataset: {self.config.dataset.type}")

        path_desc = self.config.dataset.path or f"sequence:{self.config.dataset.sequence}"

        with self.progress.track_dataset_loading(
            self.config.dataset.type,
            str(path_desc)
        ):
            from ..datasets import create_dataset
            dataset = create_dataset(self.config.dataset)

        # Log dataset summary
        info = dataset.get_sequence_info()
        sequences_str = ', '.join(info['sequences'])
        logger.info(f"Dataset loaded: {len(dataset)} frames from sequences [{sequences_str}]")

        return dataset

    def load_modules(self) -> List[PerturbationModule]:
        """
        Initialize perturbation modules based on configuration.

        Returns:
            List of initialized perturbation modules
        """
        logger.info(f"Loading {len(self.config.perturbations)} perturbation modules")
        if self.dataset is None:
            raise RuntimeError(
                "Dataset must be loaded before module setup so ModuleSetupContext "
                "contains a concrete dataset object."
            )

        modules = []
        enabled_configs = [c for c in self.config.perturbations if c.enabled]
        total_frames = len(self.dataset)
        dataset_path_value = getattr(self.dataset, "path", None)
        dataset_path = Path(dataset_path_value) if dataset_path_value is not None else None

        setup_context = ModuleSetupContext(
            dataset=self.dataset,
            dataset_path=dataset_path,
            total_frames=total_frames,
            input_path=None,
        )

        for pert_config in self.progress.track_modules(enabled_configs, desc="Loading modules"):
            try:
                module = create_module(pert_config)
                module.setup(setup_context)
                modules.append(module)
                logger.info(f"Loaded module: {module}")
            except Exception as e:
                logger.error(f"Failed to load module {pert_config.name}: {e}")
                raise

        # Log skipped modules
        for pert_config in self.config.perturbations:
            if not pert_config.enabled:
                logger.info(f"Skipped disabled module: {pert_config.name}")

        logger.info(f"Loaded {len(modules)} active modules")
        return modules

    def run(self) -> Dict[str, Any]:
        """
        Execute the main pipeline.

        Returns:
            Dictionary containing execution results and statistics
        """
        logger.info("=" * 60)
        logger.info("Starting pipeline execution")
        logger.info("=" * 60)

        self.start_time = time.time()

        # Start profiling if enabled
        if self.profiler:
            self.profiler.start_timer("pipeline_total")

        results = {
            'experiment': self.config.experiment.name,
            'start_time': datetime.now().isoformat(),
            'dataset': {
                'type': self.config.dataset.type,
                'frames': len(self.dataset) if self.dataset else 0
            },
            'modules': [m.get_info() for m in self.modules],
            'frames_processed': 0,
            'errors': []
        }

        try:
            # Ensure setup is complete
            if not self.dataset or not self.output_dir:
                if self.profiler:
                    with self.profiler.timer("setup"):
                        self.setup()
                else:
                    self.setup()

            # Process frames
            if self.profiler:
                with self.profiler.timer("process_frames"):
                    results.update(self._process_frames())
            else:
                results.update(self._process_frames())

            self._save_results(results)

            # Finalize output
            if self.output_manager:
                self.output_manager.finalize()
                logger.info("Output manager finalized")


            if self.profiler:
                with self.profiler.timer("cleanup"):
                    self._cleanup()
            else:
                self._cleanup()

            if self.profiler:
                self.profiler.stop_timer("pipeline_total")
                results['profiling'] = self.profiler.get_summary()

                if hasattr(self.config, 'profiling') and self.config.profiling.save_report:
                    report_path = self.config.profiling.report_path
                    if not report_path:
                        report_path = self.output_dir / "profiling" / "profile_report.json"
                    else:
                        report_path = Path(report_path)

                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    self.profiler.save_report(
                        report_path,
                        format=self.config.profiling.report_format
                    )
                    results['profiling_report'] = str(report_path)

                # Print summary if verbose
                if hasattr(self.config, 'profiling') and self.config.profiling.verbose:
                    self.profiler.print_summary()

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            results['errors'].append(str(e))
            results['status'] = 'failed'
            # Persist failure summary when possible for easier debugging.
            if self.output_dir is not None:
                try:
                    self._save_results(results)
                except Exception as save_error:
                    logger.warning(f"Failed to save pipeline failure results: {save_error}")
            # Best-effort cleanup on failure.
            try:
                self._cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Error during failure cleanup: {cleanup_error}")
            raise
        finally:
            elapsed = time.time() - self.start_time
            results['elapsed_time'] = elapsed
            logger.info(f"Pipeline execution completed in {elapsed:.2f} seconds")

        results['status'] = 'success'
        return results

    def _process_frames(self) -> Dict[str, Any]:
        """
        Process all frames through the perturbation modules.

        Returns:
            Processing statistics
        """
        return self._process_frames_single()

    def _process_frames_single(self) -> Dict[str, Any]:
        """
        Process all frames through the perturbation modules.

        Returns:
            Processing statistics
        """
        logger.info(f"Processing {len(self.dataset)} frames through {len(self.modules)} modules")

        stats = {
            'frames_processed': 0,
            'processing_times': {},
            'module_stats': {}
        }

        for module in self.modules:
            stats['module_stats'][module.name] = {
                'frames': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'dropped_frames': []  # Track dropped frame indices
            }

        frame_iterator = self.progress.track_frames(
            self.dataset,
            total=len(self.dataset),
            desc="Processing frames",
            unit="frames"
        )

        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before frame processing.")

        camera_roles = self.dataset.get_active_camera_roles()
        if "left" not in camera_roles:
            raise RuntimeError(
                "Dataset must expose an active 'left' camera role during frame processing."
            )

        left_camera = "left"
        right_camera = "right" if "right" in camera_roles else None

        for idx, frame_data in enumerate(frame_iterator):

            image = frame_data['image']
            depth = frame_data.get('depth')
            image_right = frame_data.get('image_right')

            for module in self.modules:
                module_start = time.time()

                if idx == 0:
                    logger.info(f"Processing module: {module.name}")

                try:
                    rgb_filename = frame_data.get('rgb_filename')
                    rgb_filename_right = frame_data.get('rgb_filename_right')

                    if image_right is not None:
                        if right_camera is None:
                            raise RuntimeError(
                                "Stereo frame data provided but dataset did not provide a right camera name."
                            )
                        perturbed_image = module.apply(
                            image,
                            depth,
                            frame_idx=idx,
                            camera=left_camera,
                            rgb_filename=rgb_filename,
                        )
                        perturbed_image_right = module.apply(
                            image_right,
                            depth,
                            frame_idx=idx,
                            camera=right_camera,
                            rgb_filename=rgb_filename_right,
                        )
                        logger.debug(f"Stereo processing frame {idx}: left={perturbed_image is not None}, right={perturbed_image_right is not None}")
                    else:
                        if self.profiler:
                            with self.profiler.timer(f"module_{module.name}",
                                                    metadata={"frame": idx}):
                                perturbed_image = module.apply(
                                    image,
                                    depth,
                                    frame_idx=idx,
                                    camera=left_camera,
                                    rgb_filename=rgb_filename,
                                )
                        else:
                            perturbed_image = module.apply(
                                image,
                                depth,
                                frame_idx=idx,
                                camera=left_camera,
                                rgb_filename=rgb_filename,
                            )

                        perturbed_image_right = None

                    if perturbed_image is None:
                        stats['module_stats'][module.name]['dropped_frames'].append(idx)
                        logger.debug(f"Frame {idx} dropped by module {module.name}")
                        continue  # Skip to next module/frame

                    # For stereo, if right image is None but left is not, also skip (synchronized drop)
                    if image_right is not None and perturbed_image_right is None:
                        stats['module_stats'][module.name]['dropped_frames'].append(idx)
                        logger.debug(f"Frame {idx} dropped (stereo sync) by module {module.name}")
                        continue

                    if self.config.output.save_images:
                        if self.output_manager:
                            perturbed_frame = {
                                'image': perturbed_image,
                                'depth': depth,
                                'timestamp': frame_data.get('timestamp', idx),
                                'sequence_id': frame_data.get('sequence_id', 'default'),
                                'frame_id': idx,
                                'rgb_filename': frame_data.get('rgb_filename'),  # Preserve original filename
                                'depth_filename': frame_data.get('depth_filename'),
                            }
                            if perturbed_image_right is not None:
                                perturbed_frame['image_right'] = perturbed_image_right
                                perturbed_frame['rgb_filename_right'] = frame_data.get('rgb_filename_right')

                            self.output_manager.write_frame(
                                perturbed_frame,
                                idx,
                                module.name,
                                frame_data.get('sequence_id')
                            )
                        else:
                            # Fall back to old method
                            self._save_image(
                                perturbed_image,
                                module.name,
                                idx,
                                frame_data.get('sequence_id', 'default'),
                                frame_data.get('rgb_filename')
                            )

                    # Update stats
                    module_time = time.time() - module_start
                    module_stats = stats['module_stats'][module.name]
                    module_stats['frames'] += 1
                    module_stats['total_time'] += module_time

                except Exception as e:
                    error_msg = (
                        f"Error processing frame {idx} with module {module.name}: {e}"
                    )
                    logger.error(error_msg)
                    # Fail fast: any module/frame exception aborts the run.
                    raise RuntimeError(error_msg) from e

            stats['frames_processed'] += 1


        # Calculate average times
        for module_name, module_stats in stats['module_stats'].items():
            if module_stats['frames'] > 0:
                module_stats['avg_time'] = module_stats['total_time'] / module_stats['frames']

        # Write dropped_frames.txt for each module that dropped frames
        # and filter metadata files to remove dropped frame entries
        for module_name, module_stats in stats['module_stats'].items():
            dropped = module_stats.get('dropped_frames', [])
            if dropped:
                module_dir = self.output_dir / "images" / module_name
                module_dir.mkdir(parents=True, exist_ok=True)

                # Write dropped frames list
                dropped_file = module_dir / "dropped_frames.txt"
                with open(dropped_file, 'w') as f:
                    f.write(f"# Dropped frames for module: {module_name}\n")
                    f.write(f"# Total dropped: {len(dropped)}\n")
                    for frame_idx in dropped:
                        f.write(f"{frame_idx}\n")
                logger.info(f"Module {module_name}: {len(dropped)} frames dropped, saved to {dropped_file}")

                if self.dataset:
                    total_frames = len(self.dataset)
                    dropped_set = set(dropped)
                    kept_indices = [i for i in range(total_frames) if i not in dropped_set]
                    self.dataset.filter_metadata_for_kept_frames(module_dir, kept_indices, total_frames)

        logger.info(f"Processed {stats['frames_processed']} frames successfully")
        return stats


    def _save_image(self, image: Any, module_name: str, frame_idx: int, sequence_id: str, rgb_filename: Optional[str] = None) -> None:
        """
        Save a processed image to disk.

        This is a fallback method when OutputManager is not available.

        Args:
            image: Image data
            module_name: Name of the perturbation module
            frame_idx: Frame index
            sequence_id: Sequence identifier
            rgb_filename: Original filename to preserve (optional)
        """
        module_dir = self.output_dir / "images" / module_name
        module_dir.mkdir(parents=True, exist_ok=True)

        if rgb_filename:
            base_name = Path(rgb_filename).stem
            output_path = module_dir / f"{base_name}.png"
        else:
            output_path = module_dir / f"{sequence_id}_{frame_idx:06d}.png"

        try:
            write_image(image, output_path)
            logger.debug(f"Saved image to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save image to {output_path}: {e}")

    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save execution results to disk.

        Args:
            results: Results dictionary
        """
        import json

        results_file = self.output_dir / "pipeline_results.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.debug(f"Pipeline results saved to: {results_file}")

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up pipeline resources")

        # Cleanup modules
        for module in self.modules:
            try:
                module.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up module {module.name}: {e}")

        self.modules.clear()

    def get_info(self) -> Dict[str, Any]:
        """
        Get pipeline information.

        Returns:
            Dictionary containing pipeline state information
        """
        return {
            'experiment': self.config.experiment.name,
            'dataset': {
                'type': self.config.dataset.type,
                'loaded': self.dataset is not None,
                'frames': len(self.dataset) if self.dataset else 0
            },
            'modules': [m.get_info() for m in self.modules],
            'output_dir': str(self.output_dir) if self.output_dir else None
        }

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        dataset_info = f"{len(self.dataset)} frames" if self.dataset else "not loaded"
        modules_info = f"{len(self.modules)} modules" if self.modules else "no modules"
        return (
            f"Pipeline("
            f"experiment='{self.config.experiment.name}', "
            f"dataset={dataset_info}, "
            f"{modules_info})"
        )
