import pytest
from biotite.structure import AtomArray
from biotite.structure.io import load_structure
from bagel.constants import atom_order
from bagel.oracles.folding.utils import reorder_atoms_in_template, get_unique_residues


def test_reorder_atoms_in_template(pdb_path):
    # Load the structure from the PDB file
    structure = load_structure(pdb_path)

    # Check original order is not correct for at least one residue
    out_of_order = False
    for res_id, chain_id in get_unique_residues(structure):
        res_mask = (structure.res_id == res_id) & (structure.chain_id == chain_id)
        atom_names = structure.atom_name[res_mask]
        indices = [atom_order.get(atom, float('inf')) for atom in atom_names]
        if indices != sorted(indices):
            out_of_order = True
            break

    assert out_of_order, 'Original structure is already ordered by atom_order; test cannot fail as expected.'

    # Now reorder and check all residues are in correct order
    reordered = reorder_atoms_in_template(structure)
    for res_id, chain_id in get_unique_residues(reordered):
        res_mask = (reordered.res_id == res_id) & (reordered.chain_id == chain_id)
        atom_names = reordered.atom_name[res_mask]
        indices = [atom_order.get(atom, float('inf')) for atom in atom_names]
        assert indices == sorted(indices), (
            f'Residue ({chain_id}, {res_id}) atoms {list(atom_names)} '
            f'not correctly reordered by reorder_atoms_in_template'
        )
