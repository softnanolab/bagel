import os
import logging
from typing import Union, List
from io import StringIO

import pandas as pd  # This is necessary because its "unique" method does not sort elements and leaves them as they are
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

from bagel.constants import atom_order, aa_dict

logger = logging.getLogger(__name__)

aa_dict_3to1 = {v: k for k, v in aa_dict.items()}


def sequence_from_atomarray(atoms: AtomArray) -> str:
    return ''.join([aa_dict_3to1[aa] for aa in atoms[atoms.atom_name == 'CA'].res_name])


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


def get_unique_residues(atom_array: AtomArray):
    residues, seen = [], set()
    for i in range(len(atom_array)):
        res_key = (atom_array.res_id[i], atom_array.chain_id[i])
        if res_key not in seen:
            seen.add(res_key)
            residues.append(res_key)
    return residues


### Reordering atoms to match ESMFold output ###
def reorder_atoms_in_template(atom_array: AtomArray) -> AtomArray:
    reordered_indices = []
    for res_id, chain_id in get_unique_residues(atom_array):
        indices = np.where((atom_array.res_id == res_id) & (atom_array.chain_id == chain_id))[0]
        atoms = atom_array[indices]

        # Skip non–amino-acid residues (e.g., HOH, ligands)
        res_name = atoms.res_name[0]
        if res_name not in aa_dict_3to1:
            logger.warning(f'Skipping non-amino-acid residue {res_name} (res_id={res_id}, chain_id={chain_id}).')
            continue

        # Filter and report atoms not in atom_order
        protein_mask = np.array([name in atom_order for name in atoms.atom_name])
        for name in atoms.atom_name[~protein_mask]:
            logger.warning(
                f"Removed non-protein atom '{name}' from residue {atoms.res_name[0]} "
                f'(res_id={res_id}, chain_id={chain_id}).'
            )

        # Reorder protein atoms
        protein_indices = indices[protein_mask]
        sort_keys = [atom_order[name] for name in atoms.atom_name[protein_mask]]
        sorted_indices = protein_indices[np.argsort(sort_keys)]
        reordered_indices.extend(sorted_indices)

    return atom_array[reordered_indices]
