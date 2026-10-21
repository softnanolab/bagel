"""
Bagel
-------

Protein design library used to find the amino acid sequences that give rise to structures that meet the given design
criteria.
"""

import importlib
import os
from typing import Any

from .utils import get_version_from_pyproject

__version__ = get_version_from_pyproject()

# boileroom derives its runtime image tag from its own version. The commit pinned in pyproject.toml reports
# 0.4.5, which is the target of boileroom's next release and was never published as an image -- main builds
# publish prereleases such as 0.4.5-alpha.N instead. Left alone, the lookup asks Docker for a tag that does
# not exist and every modal/apptainer oracle fails to pull its image. So name the prerelease this pin was
# tested against.
#
# The image only carries the runtime environment (CUDA, torch, the esm SDK); boileroom's own code is shipped
# from the installed package at call time, so this tag does not have to match the pinned commit.
#
# This is NOT inert once boileroom publishes a release: boileroom honours BOILEROOM_IMAGE_TAG over the tag it
# derives from its own version, so leaving this in place would keep forcing the old prerelease image. Bump it
# alongside the boileroom pin, and delete it once the pinned version has a published image of its own. A
# tripwire test fails if the pinned version moves without this being revisited.
DEFAULT_BOILEROOM_IMAGE_TAG = '0.4.5-alpha.16'
# setdefault, so an explicit BOILEROOM_IMAGE_TAG still wins. This runs before any boileroom model module is
# imported (the oracles import boileroom lazily, inside their _load), which is when the tag is read.
os.environ.setdefault('BOILEROOM_IMAGE_TAG', DEFAULT_BOILEROOM_IMAGE_TAG)

__all__ = [
    'Chain',
    'Residue',
    'State',
    'System',
    'constants',
    'energies',
    'minimizer',
    'mutation',
    'oracles',
    'callbacks',
]

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
    'callbacks': '.callbacks',
}


def __getattr__(name: str) -> Any:
    """
    Lazily import attributes and modules on first access.

    This reduces initial import time by deferring heavy imports until needed.
    This is especially useful for the analyzer module, which is not used in the main package.
    """
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
