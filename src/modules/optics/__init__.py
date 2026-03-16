"""Optics and Sensor perturbation modules.

This category includes modules that model lens contamination and optical artifacts:
- Lens soiling (bokeh particles)
- Cracked lens (stress propagation fractures)
- Vignetting (edge darkening)
- Motion blur (camera movement)
- Lens flare (bright sources)
- Lens patch (occlusions)
- Flickering (brightness/contrast changes)
"""

__all__ = []

# Lazy imports for modules
def __getattr__(name):
    if name == 'LensSoilingModule':
        from .lens_soiling import LensSoilingModule
        return LensSoilingModule
    elif name == 'CrackedLensPhysicsModule':
        from .cracked_lens_physics import CrackedLensPhysicsModule
        return CrackedLensPhysicsModule
    elif name == 'VignetteModule':
        from .vignetting import VignetteModule
        return VignetteModule
    elif name == 'MotionBlurModule':
        from .motion_blur import MotionBlurModule
        return MotionBlurModule
    elif name == 'MotionBlurParameters':
        from .motion_blur import MotionBlurParameters
        return MotionBlurParameters
    elif name == 'MotionBlurPresets':
        from .motion_blur import MotionBlurPresets
        return MotionBlurPresets
    elif name == 'LensFlareModule':
        from .lens_flare import LensFlareModule
        return LensFlareModule
    elif name == 'LensFlareParameters':
        from .lens_flare import LensFlareParameters
        return LensFlareParameters
    elif name == 'LensFlarePresets':
        from .lens_flare import LensFlarePresets
        return LensFlarePresets
    elif name == 'LensPatchModule':
        from .lens_patch import LensPatchModule
        return LensPatchModule
    elif name == 'LensPatchParameters':
        from .lens_patch import LensPatchParameters
        return LensPatchParameters
    elif name == 'LensPatchPresets':
        from .lens_patch import LensPatchPresets
        return LensPatchPresets
    elif name == 'FlickerModule':
        from .flickering import FlickerModule
        return FlickerModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
