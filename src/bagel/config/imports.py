"""Import utilities for loading classes and callables from config strings."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import pathlib as pl
import sys
from types import ModuleType
from typing import Any


class ImportResolutionError(ValueError):
    """Raised when class/callable import resolution fails."""


def _split_target(target: str) -> tuple[str, str]:
    if ':' not in target:
        raise ImportResolutionError(
            f"Invalid target '{target}'. Expected '<module_or_file>:<attribute>' format."
        )
    source, attr = target.rsplit(':', 1)
    source = source.strip()
    attr = attr.strip()
    if not source or not attr:
        raise ImportResolutionError(
            f"Invalid target '{target}'. Both source and attribute must be non-empty."
        )
    return source, attr


def _looks_like_file_source(source: str) -> bool:
    return source.endswith('.py') or source.startswith('./') or source.startswith('../') or source.startswith('/')


def _module_name_for_file(path: pl.Path) -> str:
    digest = hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:12]
    stem = path.stem.replace('-', '_')
    return f'bagel_user_plugin_{stem}_{digest}'


def _load_module_from_file(source: str, base_dir: pl.Path) -> ModuleType:
    source_path = pl.Path(source)
    if not source_path.is_absolute():
        source_path = (base_dir / source_path).resolve()

    if not source_path.exists() or source_path.suffix != '.py':
        raise ImportResolutionError(f"Plugin file '{source_path}' does not exist or is not a .py file")

    module_name = _module_name_for_file(source_path)
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportResolutionError(f"Could not create module spec for plugin file '{source_path}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise ImportResolutionError(f"Failed to import plugin file '{source_path}': {exc}") from exc
    return module


def _load_module_from_source(source: str, base_dir: pl.Path) -> ModuleType:
    if _looks_like_file_source(source):
        return _load_module_from_file(source, base_dir)

    try:
        return importlib.import_module(source)
    except Exception as exc:  # noqa: BLE001
        raise ImportResolutionError(f"Failed to import module '{source}': {exc}") from exc


def resolve_attribute(target: str, base_dir: pl.Path) -> Any:
    """Resolve any attribute from target '<module_or_file>:<attribute>' string."""
    source, attr = _split_target(target)
    module = _load_module_from_source(source, base_dir)

    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportResolutionError(f"Attribute '{attr}' not found in '{source}'") from exc


def resolve_class(target: str, base_dir: pl.Path) -> type[Any]:
    """Resolve a class from target '<module_or_file>:<ClassName>' string."""
    resolved = resolve_attribute(target, base_dir)
    if not inspect.isclass(resolved):
        raise ImportResolutionError(f"Target '{target}' did not resolve to a class")
    return resolved


def resolve_callable(target: str, base_dir: pl.Path) -> Any:
    """Resolve a callable from target '<module_or_file>:<callable_name>' string."""
    resolved = resolve_attribute(target, base_dir)
    if not callable(resolved):
        raise ImportResolutionError(f"Target '{target}' did not resolve to a callable")
    return resolved
