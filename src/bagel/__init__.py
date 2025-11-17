"""
Bagel
-------

Protein design library used to find the amino acid sequences that give rise to structures that meet the given design
criteria.
"""

from .utils import get_version_from_pyproject, resolve_and_set_model_dir

resolve_and_set_model_dir()

__version__ = get_version_from_pyproject()

from .chain import Chain, Residue
from .state import State
from .system import System
from . import constants, energies, minimizer, mutation, oracles


__all__ = ['Chain', 'Residue', 'State', 'System', 'constants', 'energies', 'minimizer', 'mutation', 'oracles']
