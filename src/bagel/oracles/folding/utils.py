import os
from typing import Union, List
from io import StringIO
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile
from bagel.constants import atom_order

import pandas as pd  # This is necessary because its "unique" method does not sort elements and leaves them as they are
import numpy as np


def pdb_file_to_atomarray(pdb_path: Union[str, StringIO]) -> AtomArray:
    assert isinstance(pdb_path, (str, StringIO)), 'pdb_path must be a string or StringIO'
    if isinstance(pdb_path, str):
        assert os.path.exists(pdb_path), 'pdb_path must be a valid path'
    return PDBFile.read(pdb_path).get_structure(model=1)


def pdb_string_to_atomarray(pdb_string: str) -> AtomArray:
    assert isinstance(pdb_string, str), 'pdb_string must be a string'
    return pdb_file_to_atomarray(StringIO(pdb_string))


def reindex_chains(atomarray: AtomArray, custom_chain_idx: List[str]) -> AtomArray:
    """
    Reindex the chains of an AtomArray based on a chain index map.
    This is necessary because ESMFold uses a different chain indexing than the
    flexible one use in desprot.
    """
    assert isinstance(atomarray, list), 'atomarray must be a list'
    assert len(atomarray) == 1, (
        'Atomarray must contain a single structure, this program is incompatible batch predictions from ESMFold'
    )
    assert isinstance(atomarray[0], AtomArray), 'atomarray must contain AtomArray elements'
    atomarray_chain_ids = atomarray[0].get_annotation('chain_id')
    assert len(pd.unique(atomarray_chain_ids)) == len(custom_chain_idx), (
        'number of independent chains in atomarray and custom_chain_idx must be the same'
    )

    id_conversion = {old_id: new_id for old_id, new_id in zip(pd.unique(atomarray_chain_ids), custom_chain_idx)}

    atoms = atomarray[0].copy()
    # Now associate to every atom with a chain_id in atomarray the corresponding chain_id in custom_chain_idx
    for i in range(len(atoms)):
        current_id = atoms.chain_id[i]
        new_id = id_conversion[current_id]
        atoms.chain_id[i] = new_id
    return atoms


### Reordering atoms to match ESMFold output ###
def get_unique_residues(atom_array: AtomArray):
    residues = []
    for i in range(len(atom_array)):
        res_key = (atom_array.res_id[i], atom_array.chain_id[i])
        if res_key not in residues:
            residues.append(res_key)
    return residues


def reorder_atoms_in_template(atom_array: AtomArray) -> AtomArray:
    residues = get_unique_residues(atom_array)

    reordered_indices = []
    for res_id, chain_id in residues:
        # get atom indices for this residue
        indices = np.where((atom_array.res_id == res_id) & (atom_array.chain_id == chain_id))[0]
        atoms = atom_array[indices]

        # sort using atom_order
        sort_keys = [atom_order.get(name, float('inf')) for name in atoms.atom_name]
        sorted_indices = indices[np.argsort(sort_keys)]

        reordered_indices.extend(sorted_indices)

    # Return a reordered AtomArray
    return atom_array[reordered_indices]
