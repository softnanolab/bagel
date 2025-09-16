import os
import pytest
from biotite.structure import AtomArray
from biotite.structure.io import load_structure
from bagel.constants import atom_order
from bagel.oracles.folding.utils import reorder_atoms_in_template, get_unique_residues


def test_reorder_atoms_in_template(formolase_structure: AtomArray) -> None:
    structure = formolase_structure

    # reorder and check all residues are in correct order
    reordered = reorder_atoms_in_template(structure)

    # Assert that the structure and reordered are not the same
    assert list(structure.atom_name) != list(reordered.atom_name), (
        'Reordered structure should differ from original structure'
    )

    for res_id, chain_id in get_unique_residues(reordered):
        res_mask = (reordered.res_id == res_id) & (reordered.chain_id == chain_id)
        atom_names = reordered.atom_name[res_mask]
        indices = [atom_order[name] for name in atom_names]
        assert indices == sorted(indices), (
            f'Residue ({chain_id}, {res_id}) atoms {list(atom_names)} '
            f'not correctly reordered by reorder_atoms_in_template'
        )
