import re
import pathlib
import copy
import numpy as np
from typing import Optional

from biotite.structure import AtomArray

from .constants import aa_dict

aa_dict_3to1 = {v: k for k, v in aa_dict.items()}


def get_version_from_pyproject(pyproject_path: Optional[str] = None) -> Optional[str]:
    """Parse the version string from pyproject.toml."""
    if pyproject_path is None:
        path = pathlib.Path(__file__).parent.parent.parent / 'pyproject.toml'
    else:
        path = pathlib.Path(pyproject_path)
    version = None
    if path.exists():
        with open(path) as f:
            for line in f:
                m = re.match(r'version\s*=\s*[\'"]([^\'"]+)[\'"]', line)
                if m:
                    version = m.group(1)
                    break
    return version


def get_atomarray_in_residue_range(
    atoms: AtomArray,
    start: Optional[int] = None,
    end: Optional[int] = None,
    chain: Optional[str] = None,
) -> AtomArray:
    """
    Extract a range of residues, optionally restricted to a chain, from an AtomArray.

    Parameters
    ----------
    atoms: AtomArray
        AtomArray containing all atoms.
    start: int or None, default=None
        Starting residue index (inclusive). If None, uses the minimum `res_id` in `atoms`.
    end: int or None, default=None
        Ending residue index (inclusive). If None, uses the maximum `res_id` in `atoms`.
    chain: str or None, default=None
        Chain identifier (e.g., 'A', 'B'). If None, all chains are considered.

    Returns
    -------
    AtomArray
        The subset of atoms within the requested residue index range and chain.
    """
    slice_atoms = atoms[atoms.chain_id == chain] if chain is not None else atoms

    if start is None:
        start = int(slice_atoms.res_id.min())
    if end is None:
        end = int(slice_atoms.res_id.max())

    sel = (slice_atoms.res_id >= start) & (slice_atoms.res_id <= end)
    return slice_atoms[sel]


def sequence_from_atomarray(atoms: AtomArray) -> str:
    """
    Extract the amino acid sequence in 1-letter code from an AtomArray.

    Parameters
    ----------
    atoms: AtomArray
        AtomArray containing all atoms.

    Returns
    -------
    str
        Amino acid sequence in 1-letter convention, derived from residues with 'CA' atoms.
    Notes
    -----
    Any non-protein atoms (e.g., water, ions, ligands) are removed before extracting the sequence.
    If such atoms are present, they are ignored in the sequence extraction.
    """
    protein_mask = np.isin(atoms.res_name, list(aa_dict_3to1.keys()))
    protein_atoms = atoms[protein_mask]
    ca_mask = protein_atoms.atom_name == 'CA'
    return ''.join([aa_dict_3to1[res_name] for res_name in protein_atoms[ca_mask].res_name])
