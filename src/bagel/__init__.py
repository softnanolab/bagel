"""
Bagel
-------

Protein design library used to find the amino acid sequences that give rise to structures that meet the given design
criteria.
"""

import importlib
import os
from types import ModuleType
from typing import Any

from .utils import get_version_from_pyproject, resolve_and_set_model_dir

if os.environ.get('BAGEL_SKIP_MODEL_SETUP') != '1':
    resolve_and_set_model_dir()

__version__ = get_version_from_pyproject()

__all__ = ['Chain', 'Residue', 'State', 'System', 'constants', 'energies', 'minimizer', 'mutation', 'oracles']

_LAZY_ATTRS = {
    'Chain': '.chain',
    'Residue': '.chain',
    'State': '.state',
    'System': '.system',
}

_LAZY_MODULES = {
    'constants': '.constants',
    'energies': '.energies',
    'minimizer': '.minimizer',
    'mutation': '.mutation',
    'oracles': '.oracles',
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module = importlib.import_module(_LAZY_ATTRS[name], __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    if name in _LAZY_MODULES:
        module = importlib.import_module(_LAZY_MODULES[name], __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    default_dir = set(globals().keys())
    default_dir.update(__all__)
    return sorted(default_dir)
