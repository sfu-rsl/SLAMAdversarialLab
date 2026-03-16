"""CBR-based network degradation using real video encoding."""

from dataclasses import dataclass
from typing import List

from .video_encoding_base import VideoEncodingModuleBase, VideoCodec, EncodingPreset
from ...utils import get_logger

logger = get_logger(__name__)


@dataclass
class NetworkDegradationParameters:
    """Parameters for CBR-based network degradation."""

    target_bitrate: str = "2M"
    """Target bitrate (e.g., '10M', '2M', '500k'). Encoder aims for this rate."""

    maxrate: str = "2M"
    """Maximum bitrate cap. Hard limit the encoder cannot exceed."""

    bufsize: str = "4M"
    """VBV buffer size. Typically 1-2x maxrate. Affects rate variation."""

    encoding_preset: EncodingPreset = EncodingPreset.MEDIUM
    """Encoding preset (speed vs compression tradeoff)"""

    codec: VideoCodec = VideoCodec.LIBX264
    """Video codec for encoding"""

    framerate: int = 30
    """Framerate for encoding"""


class NetworkDegradationPresets:
    """Common CBR presets for network degradation."""

    @staticmethod
    def get_preset(name: str) -> NetworkDegradationParameters:
        """Get preset configuration by name.

        Args:
            name: Preset name

        Returns:
            NetworkDegradationParameters for the preset

        Raises:
            ValueError: If preset name is unknown
        """
        presets = {
            # Level 1: No degradation (baseline reference)
            # 6 Mbps - Upper bound for 1080p30 streaming [1]
            "level_0": NetworkDegradationParameters(
                target_bitrate="6M",
                maxrate="6M",
                bufsize="12M",
                encoding_preset=EncodingPreset.MEDIUM,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),

            # Level 1: Minimal degradation
            # 3 Mbps - Lower bound for 1080p30 streaming [1]
            "level_1": NetworkDegradationParameters(
                target_bitrate="3M",
                maxrate="3M",
                bufsize="6M",
                encoding_preset=EncodingPreset.MEDIUM,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),

            # Level 2: Light degradation
            # 1 Mbps - Optimal for 720p mobile video conferencing [2]
            "level_2": NetworkDegradationParameters(
                target_bitrate="1M",
                maxrate="1M",
                bufsize="2M",
                encoding_preset=EncodingPreset.FAST,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),

            # Level 3: Moderate degradation
            # 750 kbps - Minimum acceptable for 720p at reduced framerate [2]
            "level_3": NetworkDegradationParameters(
                target_bitrate="750k",
                maxrate="750k",
                bufsize="1500k",
                encoding_preset=EncodingPreset.FAST,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),

            # Level 4: Heavy degradation
            # 500 kbps - Below recommended minimum, visible artifacts [3]
            "level_4": NetworkDegradationParameters(
                target_bitrate="500k",
                maxrate="500k",
                bufsize="1M",
                encoding_preset=EncodingPreset.FAST,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),

            # Level 5: Severe degradation
            # 250 kbps - Near practical minimum for video streaming [3]
            "level_5": NetworkDegradationParameters(
                target_bitrate="250k",
                maxrate="250k",
                bufsize="500k",
                encoding_preset=EncodingPreset.ULTRAFAST,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),

            # Level 6: Extreme degradation
            # 100 kbps - Below practical thresholds, severe artifacts
            "level_6": NetworkDegradationParameters(
                target_bitrate="100k",
                maxrate="100k",
                bufsize="200k",
                encoding_preset=EncodingPreset.ULTRAFAST,
                codec=VideoCodec.LIBX264,
                framerate=30
            ),
        }

        if name not in presets:
            raise ValueError(
                f"Unknown network degradation preset: {name}. "
                f"Available presets: {', '.join(presets.keys())}"
            )

        return presets[name]


class NetworkDegradationModule(VideoEncodingModuleBase):
    """CBR-based network degradation using real video encoding."""

    module_name = "network_degradation"
    module_description = "CBR-based network bandwidth degradation (streaming)"

    PARAMETERS_CLASS = NetworkDegradationParameters

    def _setup_encoding_params(self, params: dict) -> None:
        """Parse CBR-specific parameters."""
        if 'preset' in params:
            preset_name = params['preset']
            preset_params = NetworkDegradationPresets.get_preset(preset_name)

            # Start with preset values
            self.target_bitrate = preset_params.target_bitrate
            self.maxrate = preset_params.maxrate
            self.bufsize = preset_params.bufsize
            self.encoding_preset = preset_params.encoding_preset
            self.codec = preset_params.codec
            self.framerate = preset_params.framerate

            # Allow preset override with explicit parameters
            self.target_bitrate = params.get('target_bitrate', self.target_bitrate)
            self.maxrate = params.get('maxrate', self.maxrate)
            self.bufsize = params.get('bufsize', self.bufsize)
            self.encoding_preset = params.get('encoding_preset', self.encoding_preset)
            self.codec = params.get('codec', self.codec)
            self.framerate = params.get('framerate', self.framerate)
        else:
            # No preset - use explicit parameters or defaults
            self.target_bitrate = params.get('target_bitrate', '2M')
            self.maxrate = params.get('maxrate', '2M')
            self.bufsize = params.get('bufsize', '4M')

    def _get_encoding_flags(self) -> List[str]:
        """Return CBR encoding flags."""
        return [
            '-b:v', self.target_bitrate,
            '-maxrate', self.maxrate,
            '-bufsize', self.bufsize
        ]

    def _get_log_info(self) -> str:
        """Return CBR info for logging."""
        return f"Bitrate: {self.target_bitrate} (max: {self.maxrate}, buf: {self.bufsize})"
