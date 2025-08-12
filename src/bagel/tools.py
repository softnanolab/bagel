import random
import bagel as bg
import os
import numpy as np
from biotite.structure import AtomArray
from biotite.database.rcsb import fetch
import copy

aa_dict_3to1 = {v: k for k, v in bg.constants.aa_dict.items()}
# Functions to combine different chains / extract them from a .pdb file to make composite chains

def get_atomarray_in_residue_range(atoms: AtomArray, start: int, end: int , chain: str) -> AtomArray:
    """Extract a range of residues with a specific chain_id from an AtomArray object.
    Input:
    - atoms: AtomArray object containing all atoms
    - start: starting residue index (inclusive)
    - end: ending residue index (exclusive)
    - chain: chain identifier (e.g. 'A', 'B')
    Output: 
    - AtomArray object containing the extracted atoms
    """
    chain_atoms = copy.deepcopy( atoms[atoms.chain_id == chain] )
    return chain_atoms[np.logical_and(chain_atoms.res_id >= start, chain_atoms.res_id < end)]

def sequence_from_atomarray(atoms: AtomArray) -> str:
    """Extract the sequence of aminoacids written in 1-Letter convention 
    from an AtomArray object.
    Input:
    - atoms: AtomArray object containing all atoms
    Output:
    - str: amino acid sequence in 1-letter convention
    """
    return "".join(
        [aa_dict_3to1[aa] for aa in atoms[atoms.atom_name == "CA"].res_name]
    )