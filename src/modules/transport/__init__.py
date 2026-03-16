"""Video Transport perturbation modules.

This category includes modules that reproduce compression and transmission effects:
- Network degradation (CBR bandwidth-limited streaming)
- Bitrate degradation (CRF storage compression) [deprecated]
- Frame drop (temporal discontinuities)
"""

from .video_encoding_base import VideoEncodingModuleBase

__all__ = [
    'VideoEncodingModuleBase',
]

# Lazy imports for modules
def __getattr__(name):
    if name == 'NetworkDegradationModule':
        from .network_degradation import NetworkDegradationModule
        return NetworkDegradationModule
    elif name == 'BitrateDegradationModule':
        from .bitrate_degradation import BitrateDegradationModule
        return BitrateDegradationModule
    elif name == 'FrameDropModule':
        from .frame_drop import FrameDropModule
        return FrameDropModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
