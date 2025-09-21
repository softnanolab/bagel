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


def test_reorder_skips_non_protein_residues() -> None:
    pdb_path = os.path.join(os.path.dirname(__file__), '..', 'structures', 'fake_with_ligands.pdb')
    pdb_path = os.path.abspath(pdb_path)

    structure: AtomArray = load_structure(pdb_path)

    # Sanity check: input contains HOH and LIG residues alongside amino acids
    unique_residues_before = {
        (int(res_id), chain_id, res_name)
        for res_id, chain_id, res_name in zip(structure.res_id, structure.chain_id, structure.res_name)
    }
    assert any(res_name == 'HOH' for (_, _, res_name) in unique_residues_before)
    assert any(res_name == 'LIG' for (_, _, res_name) in unique_residues_before)
    assert any(res_name in ('ALA', 'GLY') for (_, _, res_name) in unique_residues_before)

    reordered = reorder_atoms_in_template(structure)

    # Collect residues after reordering
    unique_residues_after = {
        (int(res_id), chain_id, res_name)
        for res_id, chain_id, res_name in zip(reordered.res_id, reordered.chain_id, reordered.res_name)
    }

    # Ensure non-protein residues are removed
    assert all(res_name not in ('HOH', 'LIG') for (_, _, res_name) in unique_residues_after), (
        f'Found non-protein residues after reorder: {unique_residues_after}'
    )

    # Ensure protein residues remain
    assert any(res_name in ('ALA', 'GLY') for (_, _, res_name) in unique_residues_after)

    # Within remaining residues, atoms should be ordered by atom_order
    for res_id, chain_id in {(r, c) for (r, c, _) in unique_residues_after}:
        mask = (reordered.res_id == res_id) & (reordered.chain_id == chain_id)
        atom_names = reordered.atom_name[mask]
        # Only consider atoms that appear in atom_order
        indices = [atom_order[name] for name in atom_names if name in atom_order]
        assert indices == sorted(indices), (
            f'Residue ({chain_id}, {res_id}) atoms {list(atom_names)} not correctly reordered'
        )
