"""Module registry with auto-discovery via inheritance.

Modules are automatically registered when they inherit from PerturbationModule
and define `module_name`. This file provides discovery and lookup utilities.
"""

import ast
import importlib
import inspect
import pkgutil
import textwrap
from typing import Dict, Type, Optional, Any
from dataclasses import dataclass, MISSING
from enum import Enum

from ..utils import get_logger
from ..config.schema import PerturbationConfig
from .base import (
    PerturbationModule,
    NullModule,
    CompositeModule,
    CompositionMode,
    get_module_registry,
    ModuleRegistration,
)

logger = get_logger(__name__)


@dataclass
class ModuleInfo:
    """Extended information about a registered module."""
    name: str
    module_class: Type[PerturbationModule]
    description: str = ""
    parameters: Optional[Dict[str, Any]] = None
    deprecated: bool = False
    deprecation_message: str = ""
    replacement: str = ""

    @classmethod
    def from_registration(cls, reg: ModuleRegistration) -> 'ModuleInfo':
        """Create ModuleInfo from a ModuleRegistration."""
        info = cls(
            name=reg.name,
            module_class=reg.module_class,
            description=reg.description,
            deprecated=reg.deprecated,
            deprecation_message=reg.deprecation_message,
            replacement=reg.replacement,
        )
        info._extract_parameters()
        return info

    def _extract_parameters(self):
        """Extract parameter documentation from module's PARAMETERS_CLASS."""
        if not self.module_class:
            return

        params_class = getattr(self.module_class, 'PARAMETERS_CLASS', None)
        if params_class is not None and hasattr(params_class, '__dataclass_fields__'):
            params = {}

            field_docs = _extract_field_docstrings(params_class)

            for field_name, field_info in params_class.__dataclass_fields__.items():
                default = field_info.default
                if default is MISSING:
                    if field_info.default_factory is not MISSING:
                        try:
                            default = field_info.default_factory()
                        except Exception:
                            default = None
                    else:
                        default = None

                field_type = field_info.type
                type_str = _format_type(field_type)
                choices = field_info.metadata.get('choices')

                if choices is None and isinstance(field_type, type) and issubclass(field_type, Enum):
                    choices = [e.value for e in field_type]
                elif choices is None and hasattr(field_type, '__origin__'):
                    args = getattr(field_type, '__args__', ())
                    for arg in args:
                        if isinstance(arg, type) and issubclass(arg, Enum):
                            choices = [e.value for e in arg]
                            break

                description = field_docs.get(field_name, '')
                if not description:
                    description = field_info.metadata.get('description', '')

                param_info = {
                    'type': type_str,
                    'default': default,
                    'description': description
                }
                if choices:
                    param_info['choices'] = choices

                params[field_name] = param_info

            self.parameters = params if params else None
            return

        # Fallback: Look for legacy PARAMETERS dict attribute
        if hasattr(self.module_class, 'PARAMETERS'):
            self.parameters = self.module_class.PARAMETERS
            return

        self.parameters = None


def _format_type(field_type) -> str:
    """Format a type annotation as a readable string."""
    import typing

    if field_type is type(None):
        return 'None'

    if hasattr(field_type, '__name__'):
        return field_type.__name__

    origin = getattr(field_type, '__origin__', None)
    args = getattr(field_type, '__args__', ())

    if origin is typing.Union:
        if len(args) == 2 and type(None) in args:
            non_none = [a for a in args if a is not type(None)][0]
            return f'Optional[{_format_type(non_none)}]'
        return f'Union[{", ".join(_format_type(a) for a in args)}]'

    if origin is list:
        if args:
            return f'List[{_format_type(args[0])}]'
        return 'List'

    if origin is tuple:
        if args:
            return f'Tuple[{", ".join(_format_type(a) for a in args)}]'
        return 'Tuple'

    if origin is dict:
        if args:
            return f'Dict[{_format_type(args[0])}, {_format_type(args[1])}]'
        return 'Dict'

    return str(field_type).replace('typing.', '')


def _extract_field_docstrings(cls) -> dict:
    """Extract dataclass field docstrings from a class and its bases."""
    docs = {}
    for base in reversed(cls.__mro__):
        if base is object or not hasattr(base, '__dataclass_fields__'):
            continue
        docs.update(_extract_field_docstrings_from_source(base))
    return docs


def _extract_field_docstrings_from_source(cls) -> dict:
    """Extract field docstrings from a single dataclass definition."""
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return {}

    class_node = next(
        (
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == cls.__name__
        ),
        None,
    )
    if class_node is None:
        return {}

    docs = {}
    body = class_node.body
    for idx, node in enumerate(body):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if idx + 1 >= len(body):
            continue
        next_node = body[idx + 1]
        if (
            isinstance(next_node, ast.Expr)
            and isinstance(getattr(next_node, "value", None), ast.Constant)
            and isinstance(next_node.value.value, str)
        ):
            docs[node.target.id] = next_node.value.value.strip()

    return docs


def _normalize_value_for_text(value: Any) -> Any:
    """Normalize values for documentation text output."""
    if isinstance(value, Enum):
        return value.value
    return value


def _normalize_value_for_yaml(value: Any) -> Any:
    """Normalize Python values for YAML-oriented output."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_normalize_value_for_yaml(item) for item in value]
    if isinstance(value, list):
        return [_normalize_value_for_yaml(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_value_for_yaml(item) for key, item in value.items()}
    return value


def _format_default_value(value: Any) -> str:
    """Format a parameter default value for human-readable text."""
    value = _normalize_value_for_text(value)
    if value is None:
        return 'None'
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _format_yaml_scalar(value: Any) -> str:
    """Format a value for inline YAML output."""
    value = _normalize_value_for_yaml(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        if value == "" or any(char in value for char in [":", "#", "{", "}", "[", "]", "\n"]):
            return f'"{value}"'
        return value
    if isinstance(value, list):
        return "[" + ", ".join(_format_yaml_scalar(item) for item in value) + "]"
    return str(value)


def _build_module_yaml_example(info: ModuleInfo) -> str:
    """Build a YAML starter snippet for a module."""
    lines = [
        "perturbations:",
        f"  - name: {info.name}_example",
        f"    type: {info.name}",
        "    enabled: true",
    ]

    parameters = info.parameters or {}
    if not parameters:
        lines.append("    parameters: {}")
        return "\n".join(lines)

    required_or_default = []
    optional = []
    for param_name, param_info in parameters.items():
        if not isinstance(param_info, dict):
            continue
        default = _normalize_value_for_yaml(param_info.get('default'))
        if default is None:
            optional.append((param_name, param_info))
            continue
        required_or_default.append((param_name, default))

    lines.append("    parameters:")
    if required_or_default:
        for param_name, default in required_or_default:
            lines.append(f"      {param_name}: {_format_yaml_scalar(default)}")
    else:
        lines.append("      {}")

    if optional:
        lines.append("      # Optional parameters:")
        for param_name, param_info in optional:
            description = param_info.get('description', '')
            comment = f"  # {description}" if description else ""
            lines.append(f"      # {param_name}: null{comment}")

    return "\n".join(lines)


def discover_modules() -> int:
    """
    Discover and import all module files to trigger auto-registration.

    This recursively scans the modules directory and imports all Python files,
    which triggers __init_subclass__ and registers modules that define module_name.

    Returns:
        Number of modules discovered
    """
    discovered = 0

    # Recursively walk the current modules package and import leaf modules.
    # This keeps discovery working whether code is loaded as `src` or an
    # installed package namespace.
    package_name = __package__ or "src.modules"
    modules_pkg = importlib.import_module(package_name)
    package_prefix = f"{modules_pkg.__name__}."

    for module_info in pkgutil.walk_packages(modules_pkg.__path__, prefix=package_prefix):
        if module_info.ispkg:
            continue

        module_path = module_info.name
        module_name = module_path.rsplit('.', 1)[-1]

        if module_name.startswith('_'):
            continue
        if module_name in {'__init__', 'registry', 'base'}:
            continue
        if module_name.endswith('_base'):
            continue

        try:
            importlib.import_module(module_path)
            discovered += 1
            logger.debug(f"Imported module file: {module_path}")
        except Exception as e:
            logger.debug(f"Could not import {module_path}: {e}")

    return discovered


def get_module_class(name: str) -> Optional[Type[PerturbationModule]]:
    """
    Get a module class by name.

    Args:
        name: Registered module name

    Returns:
        Module class or None if not found
    """
    registry = get_module_registry()
    if name not in registry:
        return None
    return registry[name].module_class


class DeprecatedModuleError(Exception):
    """Raised when attempting to use a deprecated module."""
    pass


def create_module(config: PerturbationConfig) -> PerturbationModule:
    """
    Create a module instance from configuration.

    Args:
        config: Perturbation configuration

    Returns:
        Instantiated module

    Raises:
        ValueError: If module type not found in registry
        DeprecatedModuleError: If module is deprecated
    """
    module_type = config.type

    if module_type == "none":
        logger.info(f"Creating NullModule for baseline: {config.name}")
        return NullModule(config)

    if module_type == "composite":
        return _create_composite_module(config)

    registry = get_module_registry()
    if module_type not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown module type '{module_type}'. "
            f"Available types: none, composite, {available}"
        )

    reg = registry[module_type]
    if reg.deprecated:
        error_msg = f"Module '{module_type}' is deprecated"
        if reg.deprecation_message:
            error_msg += f": {reg.deprecation_message}"
        if reg.replacement:
            error_msg += f". Use '{reg.replacement}' instead."
        raise DeprecatedModuleError(error_msg)

    module_class = reg.module_class
    logger.info(f"Creating {module_class.__name__} module: {config.name}")
    return module_class(config)


def _create_composite_module(config: PerturbationConfig) -> CompositeModule:
    """
    Create a composite module from configuration.

    Expected config.parameters:
    - modules: List of module configurations
    - mode: Composition mode (sequential only)

    Args:
        config: Composite module configuration

    Returns:
        Instantiated CompositeModule

    Raises:
        ValueError: If configuration is invalid
    """
    params = config.parameters

    if "modules" not in params:
        raise ValueError("Composite module requires 'modules' parameter")

    module_configs = params["modules"]
    if not isinstance(module_configs, list) or not module_configs:
        raise ValueError("'modules' must be a non-empty list")

    child_modules = []
    for i, module_config in enumerate(module_configs):
        if isinstance(module_config, dict):
            if "name" not in module_config:
                module_config["name"] = f"{config.name}_module_{i}"
            if "type" not in module_config:
                raise ValueError(f"Module {i} missing 'type' field")

            child_config = PerturbationConfig(**module_config)
        else:
            child_config = module_config

        child_module = create_module(child_config)
        child_modules.append(child_module)

    mode = params.get("mode", "sequential")

    logger.info(
        f"Creating CompositeModule '{config.name}' with "
        f"{len(child_modules)} modules in {mode} mode"
    )

    return CompositeModule(
        config=config,
        modules=child_modules,
        mode=mode
    )


def create_composite_from_list(
    name: str,
    modules: list,
    mode: str = "sequential"
) -> CompositeModule:
    """
    Helper function to create a composite module from a list of modules.

    Args:
        name: Name for the composite module
        modules: List of module instances
        mode: Composition mode (sequential only)

    Returns:
        CompositeModule instance
    """
    config = PerturbationConfig(
        name=name,
        type="composite",
        enabled=True,
        parameters={
            "mode": mode
        }
    )

    return CompositeModule(
        config=config,
        modules=modules,
        mode=mode
    )


def list_modules(detailed: bool = False, include_deprecated: bool = True) -> Dict[str, Any]:
    """
    Get all registered modules.

    Args:
        detailed: If True, include descriptions and parameters
        include_deprecated: If True, include deprecated modules (default True)

    Returns:
        Dictionary mapping module names to info
    """
    registry = get_module_registry()

    if not detailed:
        result = {}
        for name, reg in registry.items():
            if not include_deprecated and reg.deprecated:
                continue
            result[name] = name
        return result

    result = {}
    for name, reg in registry.items():
        if not include_deprecated and reg.deprecated:
            continue

        info = ModuleInfo.from_registration(reg)

        result[name] = {
            'name': info.name,
            'description': info.description or 'No description available',
            'parameters': info.parameters or {},
            'loaded': True,
            'deprecated': info.deprecated,
            'deprecation_message': info.deprecation_message,
            'replacement': info.replacement
        }

    return result


def clear_registry() -> None:
    """Clear the module registry (mainly for testing)."""
    registry = get_module_registry()
    registry.clear()
    logger.debug("Module registry cleared")


def get_module_documentation(module_name: str, output_format: str = "text") -> str:
    """
    Get detailed documentation for a module.

    Args:
        module_name: Name of the module
        output_format: Output format: "text" or "yaml"

    Returns:
        Formatted documentation string
    """
    registry = get_module_registry()

    if module_name not in registry:
        return f"Module '{module_name}' not found"

    reg = registry[module_name]
    info = ModuleInfo.from_registration(reg)

    if output_format == "yaml":
        return _build_module_yaml_example(info)
    if output_format != "text":
        raise ValueError(f"Unsupported module documentation format: {output_format}")

    doc_lines = [
        f"Module: {info.name}",
        "=" * 40,
    ]

    # Show deprecation warning prominently
    if info.deprecated:
        doc_lines.append("")
        doc_lines.append("*** DEPRECATED ***")
        if info.deprecation_message:
            doc_lines.append(f"Reason: {info.deprecation_message}")
        if info.replacement:
            doc_lines.append(f"Use '{info.replacement}' instead.")
        doc_lines.append("")

    doc_lines.extend([
        f"Description: {info.description or 'No description available'}",
        ""
    ])

    if info.parameters:
        doc_lines.append("Parameters:")
        doc_lines.append("-" * 20)
        for param_name, param_info in info.parameters.items():
            if isinstance(param_info, dict):
                param_type = param_info.get('type', 'unknown')
                param_default = param_info.get('default', None)
                param_desc = param_info.get('description', '')
                param_choices = param_info.get('choices', None)

                doc_lines.append(f"  {param_name}:")
                doc_lines.append(f"    Type: {param_type}")
                doc_lines.append(f"    Default: {_format_default_value(param_default)}")
                if param_desc:
                    doc_lines.append(f"    Description: {param_desc}")
                if param_choices:
                    doc_lines.append(f"    Choices: {param_choices}")
            else:
                doc_lines.append(f"  {param_name}: {param_info}")
        doc_lines.append("")

    doc_lines.append("Example YAML:")
    doc_lines.append("-" * 20)
    doc_lines.append(_build_module_yaml_example(info))
    doc_lines.append("")

    if info.module_class and info.module_class.__doc__:
        doc_lines.append("Full Documentation:")
        doc_lines.append("-" * 20)
        doc_lines.append(info.module_class.__doc__)

    return "\n".join(doc_lines)


# Discover all modules on import
discover_modules()
