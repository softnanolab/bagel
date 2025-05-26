import os
from typing import Union, List
from io import StringIO
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

import pandas as pd  # This is necessary because its "unique" method does not sort elements and leaves them as they are


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
