"""Perturbation modules for SLAMAdverserialLab.

Modules are organized into three categories:
- scene: Scene and illumination conditions (fog, rain, day-to-night)
- optics: Optics and sensor conditions (lens soiling, cracked lens, vignetting, motion blur)
- transport: Video transport conditions (bitrate degradation, frame drop)

Modules are automatically registered when they define `module_name` class attribute.
"""

from .base import (
    PerturbationModule,
    ModuleSetupContext,
    NullModule,
    CompositeModule,
    CompositionMode,
    get_module_registry,
)
from .registry import (
    get_module_class,
    create_module,
    create_composite_from_list,
    list_modules,
    clear_registry,
    get_module_documentation,
    discover_modules,
    DeprecatedModuleError
)

__all__ = [
    # Base classes
    'PerturbationModule',
    'ModuleSetupContext',
    'NullModule',
    'CompositeModule',
    'CompositionMode',
    # Registry functions
    'get_module_registry',
    'get_module_class',
    'create_module',
    'create_composite_from_list',
    'list_modules',
    'clear_registry',
    'get_module_documentation',
    'discover_modules',
    # Exceptions
    'DeprecatedModuleError',
]
