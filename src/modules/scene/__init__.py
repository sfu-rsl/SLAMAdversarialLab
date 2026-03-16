"""Scene and Illumination perturbation modules.

This category includes modules that simulate weather and lighting changes:
- Fog (Koschmieder model with depth estimation)
- Rain (physics-based particle rendering)
- Day-to-night transformation
"""

from .fog_base import FogParametersBase, FogPresetsBase, FogModuleBase

__all__ = [
    'FogParametersBase',
    'FogPresetsBase',
    'FogModuleBase',
    'FogModule',
    'FogDepthAnythingParameters',
    'FogDepthAnythingPresets',
]

# Lazy imports for heavy modules
def __getattr__(name):
    if name == 'FogModule':
        from .fog import FogModule
        return FogModule
    elif name == 'FogDepthAnythingParameters':
        from .fog import FogDepthAnythingParameters
        return FogDepthAnythingParameters
    elif name == 'FogDepthAnythingPresets':
        from .fog import FogDepthAnythingPresets
        return FogDepthAnythingPresets
    elif name == 'FogParameters':
        from .fog import FogDepthAnythingParameters
        return FogDepthAnythingParameters
    elif name == 'FogPresets':
        from .fog import FogDepthAnythingPresets
        return FogDepthAnythingPresets
    elif name == 'RainModule':
        from .rain import RainModule
        return RainModule
    elif name == 'DayNightModule':
        from .daynight import DayNightModule
        return DayNightModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
