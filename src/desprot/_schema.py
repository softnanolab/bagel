# type: ignore
"""Sudocode reference for key data structures."""

import numpy as np
from biotite.structure import AtomArray
from dataclasses import dataclass

n_atoms = 0  # Number of atoms in any given state


@dataclass
class structure(AtomArray):
    """Encodes atom information of a state. Each attribute below is in an n_atoms length array"""

    # Core Biotite labels
    coord = np.ndarray[3, float]  # atom positions
    element = str  # upper case element code ('C', 'O', 'N', ...) #? Hydrogen atoms not listed sometimes?
    atom_name = str  # upper case code for atom identify ('N', 'C', 'CA', ...)
    res_name = str[3]  # upper case 3 letter code for each amino acid ('ARG', 'VAL', 'PRO', ...)
    res_id = int  # position of residue in given chain. #! Starts from 1 (or even 2 sometimes). Maybe shouldn't change
    chain_id = str[1]  # chain the atom belongs to ('A', 'B', 'C', ...)
    hetero = bool  # if True, atom is part of isolated, free single monomeric residue.
    ins_code = None  # ignore. Allows very similar proteins to have the same name but include different inserted atoms


folding_metrics = {
    'PTM': float,  # Predicted Template Modelling score {0-1}
    'PLDDT': float,  # Predicted Local Distance Difference Test score averaged over all residues #? {0-1}
    'local_PLDDT': np.ndarray[n_atoms, float],  # Predicted Local Distance Difference Test score for each residues
    'PAE': np.ndarray[(n_atoms, n_atoms), float],  # Pairwise predicted alignment error for each atom to each other.
}
