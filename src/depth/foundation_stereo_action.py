"""FoundationStereo integration for sequence depth generation."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class _FoundationStereoRuntime:
    """Resolved runtime paths and CLI mappings for FoundationStereo."""

    repo_dir: Path
    script_path: Path
    checkpoint_path: Path
    python_executable: str
    cli_flags: Dict[str, str]
    save_numpy_flag: Optional[str]
    optional_zero_flags: Dict[str, str]


class FoundationStereoAction:
    """Run FoundationStereo and convert its disparity output to metric depth."""

    def __init__(
        self,
        repo_dir: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        python_executable: Optional[str] = None,
        conda_env: str = "foundation_stereo",
        min_disparity: float = 0.05,
        max_depth_m: float = 200.0,
    ) -> None:
        default_repo = Path(__file__).resolve().parents[2] / "deps" / "depth-estimation" / "FoundationStereo"
        self.repo_dir = Path(repo_dir) if repo_dir is not None else default_repo
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.python_executable = python_executable
        self.conda_env = conda_env
        self.min_disparity = min_disparity
        self.max_depth_m = max_depth_m
        self._runtime: Optional[_FoundationStereoRuntime] = None

    def run_sequence(
        self,
        left_images_dir: Path,
        right_images_dir: Path,
        output_dir: Path,
        fx: float,
        baseline: float,
        max_frames: Optional[int] = None,
    ) -> None:
        """Run FoundationStereo over a stereo sequence."""
        runtime = self._get_runtime()

        left_images = self._collect_image_files(left_images_dir)
        if not left_images:
            raise RuntimeError(f"No images found in {left_images_dir}")
        if max_frames is not None and max_frames > 0:
            left_images = left_images[:max_frames]

        right_lookup = {p.name: p for p in self._collect_image_files(right_images_dir)}
        if not right_lookup:
            raise RuntimeError(f"No images found in {right_images_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="foundation_stereo_frames_") as temp_root:
            temp_root_path = Path(temp_root)

            for idx, left_path in enumerate(left_images, start=1):
                right_path = right_lookup.get(left_path.name)
                if right_path is None:
                    raise RuntimeError(
                        f"Missing right image for {left_path.name} in {right_images_dir}"
                    )

                depth_out_path = output_dir / left_path.name
                if depth_out_path.exists():
                    continue

                frame_output_dir = temp_root_path / f"{idx:06d}"
                frame_output_dir.mkdir(parents=True, exist_ok=True)

                cmd = self._build_inference_command(
                    runtime=runtime,
                    left_path=left_path,
                    right_path=right_path,
                    output_dir=frame_output_dir,
                )
                self._run_command(cmd, cwd=runtime.repo_dir)

                disparity = self._load_disparity_output(frame_output_dir)
                depth = self._disparity_to_depth(disparity, fx=fx, baseline=baseline)

                left_image = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
                if left_image is None:
                    raise RuntimeError(f"Failed to read source image: {left_path}")
                expected_h, expected_w = left_image.shape[:2]
                if depth.shape != (expected_h, expected_w):
                    depth = cv2.resize(depth, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)

                depth_encoded = np.clip(depth * 256.0, 0.0, 65535.0).astype(np.uint16)
                if not cv2.imwrite(str(depth_out_path), depth_encoded):
                    raise RuntimeError(f"Failed to write depth map: {depth_out_path}")

                if idx == 1 or idx % 25 == 0 or idx == len(left_images):
                    logger.info(
                        "FoundationStereo progress: %d/%d (%s)",
                        idx,
                        len(left_images),
                        left_path.name,
                    )

    def _get_runtime(self) -> _FoundationStereoRuntime:
        if self._runtime is not None:
            return self._runtime

        repo_dir = self.repo_dir
        if not repo_dir.exists():
            raise RuntimeError(
                f"FoundationStereo repository not found at {repo_dir}. "
                "Please clone https://github.com/NVlabs/FoundationStereo into "
                "deps/depth-estimation/FoundationStereo."
            )

        script_path = repo_dir / "scripts" / "run_demo.py"
        if not script_path.exists():
            raise RuntimeError(
                f"FoundationStereo demo script not found: {script_path}. "
                "Expected scripts/run_demo.py from NVlabs/FoundationStereo."
            )

        checkpoint_path = self._resolve_checkpoint(repo_dir)
        python_executable = self._resolve_python_executable()
        cli_flags, save_numpy_flag, optional_zero_flags = self._resolve_cli_flags(
            python_executable=python_executable,
            script_path=script_path,
            repo_dir=repo_dir,
        )

        self._runtime = _FoundationStereoRuntime(
            repo_dir=repo_dir,
            script_path=script_path,
            checkpoint_path=checkpoint_path,
            python_executable=python_executable,
            cli_flags=cli_flags,
            save_numpy_flag=save_numpy_flag,
            optional_zero_flags=optional_zero_flags,
        )
        return self._runtime

    def _resolve_checkpoint(self, repo_dir: Path) -> Path:
        if self.checkpoint_path is not None:
            checkpoint = self.checkpoint_path
            if not checkpoint.is_absolute():
                checkpoint = (repo_dir / checkpoint).resolve()
            if checkpoint.exists():
                return checkpoint
            raise RuntimeError(f"FoundationStereo checkpoint not found: {checkpoint}")

        candidate_paths = [
            repo_dir / "pretrained_models" / "23-51-11" / "model_best_bp2.pth",
            repo_dir / "pretrained_models" / "model_best_bp2.pth",
        ]

        for candidate in candidate_paths:
            if candidate.exists():
                return candidate

        pretrained_root = repo_dir / "pretrained_models"
        matches = sorted(pretrained_root.glob("**/model_best_bp2.pth")) if pretrained_root.exists() else []
        if matches:
            return matches[0]

        raise RuntimeError(
            "FoundationStereo checkpoint not found. Expected model_best_bp2.pth under "
            f"{pretrained_root}. Download the official checkpoint from the FoundationStereo repo "
            "and place it under pretrained_models/."
        )

    def _resolve_python_executable(self) -> str:
        if self.python_executable:
            python_path = Path(self.python_executable)
            if python_path.exists():
                return str(python_path)
            raise RuntimeError(f"Configured FoundationStereo python not found: {python_path}")

        for base in [
            os.path.expanduser("~/miniconda3"),
            os.path.expanduser("~/anaconda3"),
            "/opt/conda",
        ]:
            candidate = Path(base) / "envs" / self.conda_env / "bin" / "python"
            if candidate.exists():
                return str(candidate)

        logger.warning(
            "FoundationStereo conda env '%s' not found, using current Python: %s",
            self.conda_env,
            sys.executable,
        )
        return sys.executable

    def _resolve_cli_flags(
        self,
        python_executable: str,
        script_path: Path,
        repo_dir: Path,
    ) -> Tuple[Dict[str, str], Optional[str], Dict[str, str]]:
        help_cmd = [python_executable, str(script_path), "--help"]
        result = subprocess.run(help_cmd, cwd=repo_dir, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to query FoundationStereo CLI arguments via --help: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        help_text = f"{result.stdout}\n{result.stderr}"
        available_flags = set(re.findall(r"--[a-zA-Z0-9_-]+", help_text))

        left_flag = self._pick_flag(
            available_flags=available_flags,
            candidates=["--left_path", "--left_file", "--left_img", "--left"],
            label="left image path",
        )
        right_flag = self._pick_flag(
            available_flags=available_flags,
            candidates=["--right_path", "--right_file", "--right_img", "--right"],
            label="right image path",
        )
        output_flag = self._pick_flag(
            available_flags=available_flags,
            candidates=["--out_dir", "--output_dir", "--save_dir", "--save_path", "--out_path"],
            label="output directory",
        )
        checkpoint_flag = self._pick_flag(
            available_flags=available_flags,
            candidates=["--ckpt_dir", "--ckpt", "--checkpoint", "--model_path", "--pretrained"],
            label="checkpoint path",
        )

        save_numpy_flag = None
        for candidate in ["--save_numpy", "--save_npy", "--save_npz"]:
            if candidate in available_flags:
                save_numpy_flag = candidate
                break

        optional_zero_flags: Dict[str, str] = {}
        for key, candidates in {
            "get_pc": ["--get_pc"],
            "remove_invisible": ["--remove_invisible"],
            "denoise_cloud": ["--denoise_cloud"],
        }.items():
            for candidate in candidates:
                if candidate in available_flags:
                    optional_zero_flags[key] = candidate
                    break

        return {
            "left": left_flag,
            "right": right_flag,
            "output": output_flag,
            "checkpoint": checkpoint_flag,
        }, save_numpy_flag, optional_zero_flags

    @staticmethod
    def _pick_flag(available_flags: set[str], candidates: Sequence[str], label: str) -> str:
        for candidate in candidates:
            if candidate in available_flags:
                return candidate
        raise RuntimeError(
            f"FoundationStereo CLI does not expose a '{label}' option. "
            f"Available flags: {sorted(available_flags)}"
        )

    def _build_inference_command(
        self,
        runtime: _FoundationStereoRuntime,
        left_path: Path,
        right_path: Path,
        output_dir: Path,
    ) -> List[str]:
        cmd = [
            runtime.python_executable,
            str(runtime.script_path),
            runtime.cli_flags["left"],
            str(left_path),
            runtime.cli_flags["right"],
            str(right_path),
            runtime.cli_flags["output"],
            str(output_dir),
            runtime.cli_flags["checkpoint"],
            str(runtime.checkpoint_path),
        ]

        if runtime.save_numpy_flag is not None:
            cmd.append(runtime.save_numpy_flag)

        # Force headless mode to avoid Open3D visualization failures in non-interactive runs.
        for key in ("get_pc", "remove_invisible", "denoise_cloud"):
            flag = runtime.optional_zero_flags.get(key)
            if flag is not None:
                cmd.extend([flag, "0"])

        return cmd

    @staticmethod
    def _run_command(cmd: List[str], cwd: Path) -> None:
        process = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
        if process.returncode != 0:
            output = process.stderr.strip() or process.stdout.strip()
            raise RuntimeError(f"FoundationStereo inference failed: {output}")

    @staticmethod
    def _collect_image_files(image_dir: Path) -> List[Path]:
        files = sorted(image_dir.glob("*.png"))
        if files:
            return files
        return sorted(image_dir.glob("*.jpg"))

    def _load_disparity_output(self, output_dir: Path) -> np.ndarray:
        npy_candidates = self._find_files(output_dir, patterns=["*disp*.npy", "*dispar*.npy", "*.npy"])
        for path in npy_candidates:
            disparity = np.load(path)
            if disparity.size > 0:
                return self._ensure_2d(disparity)

        npz_candidates = self._find_files(output_dir, patterns=["*disp*.npz", "*dispar*.npz", "*.npz"])
        for path in npz_candidates:
            with np.load(path) as data:
                for key in ["disp", "disparity", "pred_disp", "pred", "output"]:
                    if key in data:
                        return self._ensure_2d(data[key])
                if data.files:
                    return self._ensure_2d(data[data.files[0]])

        pfm_candidates = self._find_files(output_dir, patterns=["*disp*.pfm", "*dispar*.pfm", "*.pfm"])
        for path in pfm_candidates:
            return self._ensure_2d(self._read_pfm(path))

        png_candidates = self._find_files(output_dir, patterns=["*disp*.png", "*dispar*.png", "*.png"])
        for path in png_candidates:
            disparity = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if disparity is None:
                continue
            if disparity.dtype == np.uint8:
                # 8-bit outputs are usually color visualizations, not metric disparity.
                continue
            return self._ensure_2d(disparity.astype(np.float32))

        raise RuntimeError(
            f"FoundationStereo did not produce a supported disparity file in {output_dir}. "
            "Expected one of: .npy, .npz, .pfm, or 16-bit .png disparity output."
        )

    @staticmethod
    def _find_files(root: Path, patterns: Sequence[str]) -> List[Path]:
        files: List[Path] = []
        for pattern in patterns:
            files.extend(sorted(root.rglob(pattern)))
        return files

    @staticmethod
    def _ensure_2d(array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array).astype(np.float32)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return arr[..., 0]
        raise RuntimeError(f"Unsupported disparity shape from FoundationStereo: {arr.shape}")

    @staticmethod
    def _read_pfm(path: Path) -> np.ndarray:
        with open(path, "rb") as f:
            header = f.readline().decode("ascii", errors="ignore").rstrip()
            if header not in {"PF", "Pf"}:
                raise RuntimeError(f"Invalid PFM header in {path}: {header}")
            color = header == "PF"

            dims_line = f.readline().decode("ascii", errors="ignore").strip()
            while dims_line.startswith("#"):
                dims_line = f.readline().decode("ascii", errors="ignore").strip()
            match = re.match(r"^(\d+)\s+(\d+)$", dims_line)
            if match is None:
                raise RuntimeError(f"Invalid PFM dimensions in {path}: {dims_line}")
            width, height = int(match.group(1)), int(match.group(2))

            scale_line = f.readline().decode("ascii", errors="ignore").strip()
            scale = float(scale_line)
            endian = "<" if scale < 0 else ">"

            data = np.fromfile(f, f"{endian}f")
            expected = width * height * (3 if color else 1)
            if data.size != expected:
                raise RuntimeError(
                    f"Invalid PFM payload size in {path}: expected {expected}, got {data.size}"
                )

            if color:
                data = data.reshape((height, width, 3))
                data = data[..., 0]
            else:
                data = data.reshape((height, width))

            data = np.flipud(data)
            return data.astype(np.float32)

    def _disparity_to_depth(self, disparity: np.ndarray, fx: float, baseline: float) -> np.ndarray:
        disp = disparity.astype(np.float32)
        valid = np.isfinite(disp) & (disp > self.min_disparity)

        depth = np.zeros_like(disp, dtype=np.float32)
        depth[valid] = (fx * baseline) / disp[valid]
        depth = np.clip(depth, 0.0, self.max_depth_m)
        depth[~valid] = 0.0
        return depth
