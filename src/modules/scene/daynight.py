"""Day/night image translation using img2img-turbo."""

import sys
import numpy as np
import torch
from enum import Enum
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from PIL import Image

from ..base import PerturbationModule
from ...utils import get_logger

logger = get_logger(__name__)


class TransformDirection(str, Enum):
    """CycleGAN-Turbo transformation directions."""
    DAY_TO_NIGHT = "day_to_night"
    """Convert daytime scenes to nighttime"""
    NIGHT_TO_DAY = "night_to_day"
    """Convert nighttime scenes to daytime"""
    CLEAR_TO_RAINY = "clear_to_rainy"
    """Add rain effects to clear weather scenes"""
    RAINY_TO_CLEAR = "rainy_to_clear"
    """Remove rain effects from rainy scenes"""


@dataclass
class DayNightParameters:
    """Parameters for day-to-night transformation."""

    direction: TransformDirection = TransformDirection.DAY_TO_NIGHT
    """Transform direction for image translation"""

    caption: Optional[str] = None
    """Custom text prompt to guide the transformation (optional)"""

    enable_fp16: bool = True
    """Use FP16 for faster inference"""

IMG2IMG_TURBO_PATH = Path(__file__).parent.parent.parent.parent / "deps" / "perturbations" / "img2img-turbo"
IMG2IMG_TURBO_SRC_PATH = IMG2IMG_TURBO_PATH / "src"
if str(IMG2IMG_TURBO_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(IMG2IMG_TURBO_SRC_PATH))

try:
    from cyclegan_turbo import CycleGAN_Turbo
    from torchvision import transforms as T
    DAYNIGHT_AVAILABLE = True
except ImportError as e:
    DAYNIGHT_AVAILABLE = False
    _DAYNIGHT_IMPORT_ERROR = str(e)
    logger.debug(f"img2img-turbo not available: {e}")


class DayNightModule(PerturbationModule):
    """Day/night image translation using CycleGAN-Turbo."""

    module_name = "daynight"
    module_description = "Day-to-night transformation using img2img-turbo"

    PARAMETERS_CLASS = DayNightParameters

    def _setup(self, context) -> None:
        """Setup img2img-turbo model."""
        if not DAYNIGHT_AVAILABLE:
            error_detail = globals().get('_DAYNIGHT_IMPORT_ERROR', 'unknown error')
            raise ImportError(
                f"img2img-turbo not available ({error_detail}). Please install:\n"
                "  git submodule update --init deps/perturbations/img2img-turbo\n"
                "  pip install -r deps/perturbations/img2img-turbo/requirements.txt\n"
                "Model weights download automatically from Hugging Face on first run.\n"
            )

        # Extract parameters
        params = self.parameters or {}
        self.direction = params.get('direction', 'day_to_night')
        self.caption = params.get('caption', None)  # Custom caption override
        self.enable_fp16 = params.get('enable_fp16', True)

        valid_directions = ['day_to_night', 'night_to_day', 'clear_to_rainy', 'rainy_to_clear']
        if self.direction not in valid_directions:
            raise ValueError(f"Invalid direction '{self.direction}'. Must be one of: {valid_directions}")

        logger.info(f"Setting up DayNightModule '{self.name}'")
        logger.info(f"  Direction: {self.direction}")
        if self.caption:
            logger.info(f"  Custom caption: {self.caption}")
        logger.info(f"  FP16: {self.enable_fp16}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            logger.warning("CUDA not available, using CPU (will be very slow)")

        logger.info(f"Loading CycleGAN-Turbo model: {self.direction}")
        self.model = CycleGAN_Turbo(
            pretrained_name=self.direction,
            pretrained_path=None  # Auto-download from HuggingFace
        )
        self.model.eval()

        self.model.unet.enable_xformers_memory_efficient_attention()

        if self.enable_fp16 and self.device.type == 'cuda':
            self.model.half()
            logger.info("  FP16 optimization enabled")

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

        logger.info(f"DayNightModule '{self.name}' ready")

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        """Apply day/night transformation to a single frame.

        Args:
            image: Input RGB image (H, W, 3), uint8
            depth: Optional depth map (not used)
            frame_idx: Frame index
            **kwargs: Additional context (unused)

        Returns:
            Transformed RGB image (H, W, 3), uint8
        """
        h, w = image.shape[:2]

        pil_image = Image.fromarray(image).convert("RGB")

        # Ensure dimensions are divisible by 8 (VAE requirement)
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        if new_h != h or new_w != w:
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

        # Preprocess
        x_t = self.normalize(pil_image).unsqueeze(0).to(self.device)

        if self.enable_fp16 and self.device.type == 'cuda':
            x_t = x_t.half()

        # Inference (one-step)
        with torch.no_grad():
            # Model forward: direction=None uses default pretrained direction
            # caption=None uses default, or pass custom caption
            output = self.model(x_t, direction=None, caption=self.caption)

        # Post-process: denormalize and convert to uint8
        output = (output[0] + 0.5).clamp(0, 1)  # [-0.5, 0.5] -> [0, 1]
        output = output.cpu().float().numpy().transpose(1, 2, 0)
        output = (output * 255).astype(np.uint8)

        # Resize back to original dimensions
        output_pil = Image.fromarray(output)
        output_pil = output_pil.resize((w, h), Image.LANCZOS)
        output = np.array(output_pil)

        return output

    def _cleanup(self) -> None:
        """Release GPU memory."""
        logger.info(f"Cleaning up DayNightModule '{self.name}'")

        # Delete model
        if hasattr(self, 'model'):
            del self.model

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"DayNightModule '{self.name}' cleaned up")
