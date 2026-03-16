"""CRF-based compression degradation using real video encoding."""

from dataclasses import dataclass
from typing import List

from .video_encoding_base import VideoEncodingModuleBase, VideoCodec, EncodingPreset
from ...utils import get_logger

logger = get_logger(__name__)


@dataclass
class BitrateDegradationParameters:
    """Parameters for CRF-based bitrate degradation."""

    crf: int = 35
    """Constant Rate Factor (0=lossless, 51=worst). Typical: 23=good, 35=degraded, 45=severe"""

    encoding_preset: EncodingPreset = EncodingPreset.MEDIUM
    """Encoding preset (speed vs compression tradeoff)"""

    codec: VideoCodec = VideoCodec.LIBX264
    """Video codec for encoding"""

    framerate: int = 30
    """Framerate for encoding"""


class BitrateDegradationPresets:
    """Common CRF-based bitrate degradation presets."""

    @staticmethod
    def get_preset(name: str) -> BitrateDegradationParameters:
        """Get preset configuration by name.

        Args:
            name: Preset name

        Returns:
            BitrateDegradationParameters for the preset

        Raises:
            ValueError: If preset name is unknown
        """
        presets = {
            # Lossless - no degradation (baseline)
            "lossless": BitrateDegradationParameters(
                crf=0,
                encoding_preset=EncodingPreset.VERYSLOW,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),

            # High quality - minimal artifacts
            "high_quality": BitrateDegradationParameters(
                crf=18,
                encoding_preset=EncodingPreset.MEDIUM,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),

            # Medium quality - slight artifacts visible
            "medium_quality": BitrateDegradationParameters(
                crf=28,
                encoding_preset=EncodingPreset.MEDIUM,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),

            # Low quality - noticeable blocking and artifacts
            "low_quality": BitrateDegradationParameters(
                crf=35,
                encoding_preset=EncodingPreset.FAST,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),

            # Very low quality - significant degradation
            "very_low_quality": BitrateDegradationParameters(
                crf=42,
                encoding_preset=EncodingPreset.FAST,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),

            # Extreme degradation - maximum compression artifacts
            "extreme_degradation": BitrateDegradationParameters(
                crf=51,
                encoding_preset=EncodingPreset.ULTRAFAST,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),
        }

        if name not in presets:
            raise ValueError(
                f"Unknown bitrate degradation preset: {name}. "
                f"Available presets: {', '.join(presets.keys())}"
            )

        return presets[name]


class BitrateDegradationModule(VideoEncodingModuleBase):
    """CRF-based compression degradation using real video encoding."""

    module_name = "bitrate_degradation"
    module_description = "CRF-based compression artifacts (storage/archival)"
    deprecated = True
    deprecation_message = "CRF-based encoding does not model network bandwidth constraints"
    replacement = "network_degradation"

    PARAMETERS_CLASS = BitrateDegradationParameters

    def _setup_encoding_params(self, params: dict) -> None:
        """Parse CRF-specific parameters."""
        if 'preset' in params:
            preset_name = params['preset']
            preset_params = BitrateDegradationPresets.get_preset(preset_name)

            # Start with preset values
            self.crf = preset_params.crf
            self.encoding_preset = preset_params.encoding_preset
            self.codec = preset_params.codec
            self.framerate = preset_params.framerate

            # Allow preset override with explicit parameters
            self.crf = params.get('crf', self.crf)
            self.encoding_preset = params.get('encoding_preset', self.encoding_preset)
            self.codec = params.get('codec', self.codec)
            self.framerate = params.get('framerate', self.framerate)
        else:
            # No preset - use explicit parameters or defaults
            self.crf = params.get('crf', 35)

    def _get_encoding_flags(self) -> List[str]:
        """Return CRF encoding flags."""
        return ['-crf', str(self.crf)]

    def _get_log_info(self) -> str:
        """Return CRF info for logging."""
        return f"CRF: {self.crf}"
