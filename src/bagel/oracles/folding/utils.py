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
    Reindex chain identifiers of a single-structure AtomArray to a provided list of chain IDs.
    
    Takes a one-element list containing an AtomArray and remaps its atom-level chain_id values so the set of unique chain IDs
    (in their order of first appearance) is replaced by the values from custom_chain_idx. The function operates on and
    returns a shallow copy of the contained AtomArray; the original AtomArray in the input list is not modified.
    
    Parameters:
        atomarray (List[AtomArray]): A list containing exactly one AtomArray.
        custom_chain_idx (List[str]): New chain identifiers; its length must equal the number of unique chain IDs in the AtomArray.
    
    Returns:
        AtomArray: A copy of the input AtomArray with chain_id values replaced according to custom_chain_idx.
    
    Raises:
        AssertionError: If input types/lengths are invalid or the number of unique chain IDs does not match custom_chain_idx.
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
    """
    Return the list of unique residues found in the atom array as (res_id, chain_id) pairs, preserving the order of first encounter.
    
    Parameters:
        atom_array (AtomArray): AtomArray-like object with indexable attributes `res_id` and `chain_id` per atom.
    
    Returns:
        list[tuple]: Ordered list of unique (res_id, chain_id) tuples representing residues.
    """
    residues, seen = [], set()
    for i in range(len(atom_array)):
        res_key = (atom_array.res_id[i], atom_array.chain_id[i])
        if res_key not in seen:
            seen.add(res_key)
            residues.append(res_key)
    return residues


### Reordering atoms to match ESMFold output ###
def reorder_atoms_in_template(atom_array: AtomArray) -> AtomArray:
    """
    Reorder atoms in an AtomArray to match the canonical atom ordering used by the template.
    
    This function walks residues in encounter order and, for each residue that is a recognized amino acid, selects only the protein atom names present in the global `atom_order`, sorts those atoms according to `atom_order`, and builds a new AtomArray containing atoms in that template order. Non-amino-acid residues (e.g., water or ligands) are skipped; atom names not present in `atom_order` are removed from their residue. Warnings are emitted for skipped residues and removed atoms via the module logger.
    
    Parameters:
        atom_array (AtomArray): The input structure; must support boolean indexing and attributes `res_id`, `chain_id`, `res_name`, and `atom_name`.
    
    Returns:
        AtomArray: A new AtomArray containing the subset of atoms reordered to the template/expected atom order.
    """
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
