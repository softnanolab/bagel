import os
import bagel as bg
from bagel.oracles import OraclesResultDict
from biotite.structure import AtomArray, sasa, annotate_sse, get_residue_count, concatenate, Atom, array
from biotite.structure.io import load_structure
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import copy
import inspect
import pytest


def test_residue_list_to_group_function(residues: list[bg.Residue]) -> None:
    residue_group = bg.energies.residue_list_to_group(residues)
    chain_ids, res_ids = residue_group
    assert np.all(chain_ids == np.array(['A'] * 5 + ['B'])), 'function returned wrong chain ids'
    assert np.all(res_ids == np.array(list(range(5)) + [0])), 'function returned wrong res ids'


def test_energies_properly_update_residue_group_after_residue_index_shifted_after_removal(
    fake_esmfold: bg.oracles.folding.ESMFold,
    residues: list[bg.Residue],
) -> None:
    energy = bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=residues)
    energy.remove_residue(chain_id='A', res_index=2)
    energy.shift_residues_indices_after_removal(chain_id='A', res_index=2)
    assert all(energy.residue_groups[0][0] == np.array(['A', 'A', 'A', 'A', 'B'])), 'incorrect chain_IDs'
    assert all(energy.residue_groups[0][1] == np.array([0, 1, 2, 3, 0])), 'incorrect res_indices'


def test_energies_properly_update_residue_group_before_residue_index_shifted_for_addition(
    fake_esmfold: bg.oracles.folding.ESMFold,
    residues: list[bg.Residue],
) -> None:
    energy = bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=residues)
    energy.shift_residues_indices_before_addition(chain_id='A', res_index=1)
    assert all(energy.residue_groups[0][0] == np.array(['A', 'A', 'A', 'A', 'A', 'B'])), 'incorrect chain_IDs'
    assert all(energy.residue_groups[0][1] == np.array([0, 2, 3, 4, 5, 0])), 'incorrect res_indices'


def test_energies_properly_update_residue_group_after_remove_residue(
    fake_esmfold: bg.oracles.folding.ESMFold,
    residues: list[bg.Residue],
) -> None:
    energy = bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=residues)
    energy.remove_residue(chain_id='A', res_index=2)
    assert all(energy.residue_groups[0][0] == np.array(['A', 'A', 'A', 'A', 'B'])), 'incorrect chain_IDs'
    assert all(energy.residue_groups[0][1] == np.array([0, 1, 3, 4, 0])), 'incorrect res_indices'


def test_energies_properly_update_residue_group_before_add_residue(
    fake_esmfold: bg.oracles.folding.ESMFold,
    residues: list[bg.Residue],
) -> None:
    energy = bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=residues)
    energy.add_residue(chain_id='A', res_index=2, parent_res_index=1)
    assert all(energy.residue_groups[0][0] == np.array(['A', 'A', 'A', 'A', 'A', 'B', 'A'])), 'incorrect chain_IDs'
    assert all(energy.residue_groups[0][1] == np.array([0, 1, 2, 3, 4, 0, 2])), 'incorrect  res_indices'


def test_energies_get_correct_residue_mask(
    fake_esmfold: bg.oracles.folding.ESMFold, small_structure: AtomArray
) -> None:
    energy = bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=[bg.Residue(name='V', chain_ID='A', index=1)])
    mask = energy.get_residue_mask(structure=small_structure, residue_group_index=0)
    assert all(mask == np.array([False, True, False]))


def test_energies_get_correct_residue_mask_for_multimer(
    fake_esmfold: bg.oracles.folding.ESMFold,
    square_structure: AtomArray,
    line_structure: AtomArray,
) -> None:
    residues = [
        bg.Residue(name='V', chain_ID='E', index=2),
        bg.Residue(name='V', chain_ID='D', index=0),
        bg.Residue(name='V', chain_ID='E', index=3),
        bg.Residue(name='V', chain_ID='D', index=1),
        bg.Residue(name='V', chain_ID='E', index=1),
    ]
    energy = bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=residues)
    structure = concatenate([square_structure, line_structure])
    mask = energy.get_residue_mask(structure, residue_group_index=0)
    assert all(mask == [False, True, True, True, False, True, True]), AssertionError(
        f'Incorrect residue mask for multimer {mask}'
    )


def test_energies_get_correct_atom_mask(
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure: AtomArray,
) -> None:
    energy = bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=[bg.Residue(name='V', chain_ID='A', index=0)])
    mask = energy.get_atom_mask(structure=small_structure, residue_group_index=0)
    assert all(mask == np.array([True, True, False, False, False]))

    # Create individual atoms
    atoms = []
    # Chain C (GLY, ALA)
    atoms.extend(
        [
            Atom([0, 0, 0], chain_id='C', res_id=0, atom_name='N', res_name='GLY'),  # C0 N
            Atom([0.5, 0, 0], chain_id='C', res_id=0, atom_name='CA', res_name='GLY'),  # C0 CA
            Atom([1, 0, 0], chain_id='C', res_id=1, atom_name='N', res_name='ALA'),  # C1 N
            Atom([1.5, 0, 0], chain_id='C', res_id=1, atom_name='CA', res_name='ALA'),  # C1 CA
        ]
    )
    # Chain B (SER, THR)
    atoms.extend(
        [
            Atom([0, 1, 0], chain_id='B', res_id=0, atom_name='N', res_name='SER'),  # B0 N
            Atom([0.5, 1, 0], chain_id='B', res_id=0, atom_name='CA', res_name='SER'),  # B0 CA
            Atom([1, 1, 0], chain_id='B', res_id=1, atom_name='N', res_name='THR'),  # B1 N
            Atom([1.5, 1, 0], chain_id='B', res_id=1, atom_name='CA', res_name='THR'),  # B1 CA
        ]
    )
    # Chain A (VAL, LEU)
    atoms.extend(
        [
            Atom([0, 0, 1], chain_id='A', res_id=0, atom_name='N', res_name='VAL'),  # A0 N
            Atom([0.5, 0, 1], chain_id='A', res_id=0, atom_name='CA', res_name='VAL'),  # A0 CA
            Atom([1, 0, 1], chain_id='A', res_id=1, atom_name='N', res_name='LEU'),  # A1 N
            Atom([1.5, 0, 1], chain_id='A', res_id=1, atom_name='CA', res_name='LEU'),  # A1 CA
        ]
    )

    # Create the structure
    structure = array(atoms)

    # Create energy term with residues in different order than structure
    # This tests if order preservation matters
    group1_residues = [
        bg.Residue(name='V', chain_ID='A', index=0),  # A0
        bg.Residue(name='G', chain_ID='C', index=0),  # C0
        bg.Residue(name='A', chain_ID='C', index=1),  # C1
        bg.Residue(name='S', chain_ID='B', index=0),  # B0
    ]

    group2_residues = [
        bg.Residue(name='L', chain_ID='A', index=1),  # A1
        bg.Residue(name='T', chain_ID='B', index=1),  # B1
    ]

    energy = bg.energies.SeparationEnergy(oracle=fake_esmfold, residues=(group1_residues, group2_residues))

    # Test first residue group mask
    mask1 = energy.get_atom_mask(structure, residue_group_index=0)
    expected_mask1 = np.array(
        [
            True,
            True,  # C0 atoms
            True,
            True,  # C1 atoms
            True,
            True,  # B0 atoms
            False,
            False,  # B1 atoms
            True,
            True,  # A0 atoms
            False,
            False,  # A1 atoms
        ]
    )

    assert np.array_equal(mask1, expected_mask1), (
        f'First group mask incorrect. Expected:\n{expected_mask1}\nGot:\n{mask1}'
    )

    # Test second residue group mask
    mask2 = energy.get_atom_mask(structure, residue_group_index=1)
    expected_mask2 = np.array(
        [
            False,
            False,  # C0 atoms
            False,
            False,  # C1 atoms
            False,
            False,  # B0 atoms
            True,
            True,  # B1 atoms
            False,
            False,  # A0 atoms
            True,
            True,  # A1 atoms
        ]
    )

    assert np.array_equal(mask2, expected_mask2), (
        f'Second group mask incorrect. Expected:\n{expected_mask2}\nGot:\n{mask2}'
    )

    # Additional test: verify that the masked atoms for first group have the expected coordinates
    masked_atoms1 = structure[mask1]
    expected_coords1 = np.array(
        [
            [0, 0, 0],  # C0 N
            [0.5, 0, 0],  # C0 CA
            [1, 0, 0],  # C1 N
            [1.5, 0, 0],  # C1 CA
            [0, 1, 0],  # B0 N
            [0.5, 1, 0],  # B0 CA
            [0, 0, 1],  # A0 N
            [0.5, 0, 1],  # A0 CA
        ]
    )

    assert np.allclose(masked_atoms1.coord, expected_coords1), (
        f'First group masked atom coordinates incorrect. Expected:\n{expected_coords1}\nGot:\n{masked_atoms1.coord}'
    )

    # Additional test: verify that the masked atoms for second group have the expected coordinates
    masked_atoms2 = structure[mask2]
    expected_coords2 = np.array(
        [
            [1, 1, 0],  # B1 N
            [1.5, 1, 0],  # B1 CA
            [1, 0, 1],  # A1 N
            [1.5, 0, 1],  # A1 CA
        ]
    )

    assert np.allclose(masked_atoms2.coord, expected_coords2), (
        f'Second group masked atom coordinates incorrect. Expected:\n{expected_coords2}\nGot:\n{masked_atoms2.coord}'
    )


# Note that here we do ESMFold specific tests, but these should similar extend to other FoldingOracles
# Once we employ more FoldingOracles, we can make these tests more general, by tackling the FoldingOracle class directly
def test_PTMEnergy(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.ptm = 0.7
    energy = bg.energies.PTMEnergy(oracle=fake_esmfold, weight=2.0)
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    assert np.isclose(unweighted_energy, -0.7)
    assert np.isclose(weighted_energy, -0.7 * 2.0)


def test_PLDDTEnergy(
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure_residues: list[bg.Residue],
    small_structure: AtomArray,
) -> None:
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.local_plddt = np.array([0.2, 0.4, 0.6]).reshape(1, 3)
    mock_folding_result.structure = small_structure
    energy = bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=small_structure_residues[:2], weight=2.0)
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    assert np.isclose(unweighted_energy, -0.3)  # avoids float rounding errors
    assert np.isclose(weighted_energy, -0.6)


def test_OverallPLDDTEnergy(
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure: AtomArray,
    small_structure_chains: list[bg.Chain],
) -> None:
    folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    folding_result.input_chains = small_structure_chains
    folding_result.structure = small_structure
    folding_result.local_plddt = np.array([0.2, 0.4, 0.6]).reshape(1, 3)
    energy = bg.energies.OverallPLDDTEnergy(oracle=fake_esmfold, weight=2.0)
    oracles_result = OraclesResultDict({fake_esmfold: folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    assert np.isclose(unweighted_energy, -0.4), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, -0.8), 'weighted energy is incorrect'


def test_solvent_accessible_surface_area_function_gives_expected_return_array(small_structure: AtomArray) -> None:
    sasa_result = sasa(small_structure)
    assert len(sasa_result) == len(small_structure), 'sasa does not return one number for each atom'
    assert np.issubdtype(sasa_result.dtype, np.floating), 'sasa does not return floats for each atom'


@patch('bagel.energies.sasa')
def test_SurfaceAreaEnergy(
    mock_sasa: Mock,
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure_residues: list[bg.Residue],
    small_structure: AtomArray,
) -> None:
    mock_sasa.return_value = np.arange(5, dtype=float)
    energy = bg.energies.SurfaceAreaEnergy(oracle=fake_esmfold, residues=small_structure_residues[:1], weight=2.0)
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    # returns mean of normalized sasa over given residues
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    value = 1 / (22 * 2)  # max sasa is 22, and there are 2 atoms in the first residue
    assert np.isclose(unweighted_energy, value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, value * 2), 'weighted energy is incorrect'


@patch('bagel.energies.sasa')
def test_HydrophobicEnergy(
    mock_sasa: Mock,
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure_residues: list[bg.Residue],
    small_structure: AtomArray,
) -> None:
    mock_sasa.return_value = np.array([22, 22, 22, 22, 0])  # atoms of first 2 residues are given max sasa
    energy = bg.energies.HydrophobicEnergy(
        oracle=fake_esmfold, residues=small_structure_residues[:2], mode='surface', weight=2.0
    )
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    # returns sum of normalized sasa for hydrophobic atoms, divided by the number of atoms in given residues
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    value = 2 / 4  # 4 atoms in given residues, only 2 are part of hydrophobic residue
    assert np.isclose(unweighted_energy, value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, value * 2), 'weighted energy is incorrect'


def test_PAEEnergy_with_cross_term_only(
    fake_esmfold: bg.oracles.folding.ESMFold,
    mixed_structure_state: bg.State,
) -> None:
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.pae = np.arange(7**2).reshape((1, 7, 7))
    mock_folding_result.structure = mixed_structure_state._oracles_result[fake_esmfold].structure
    residues = sum([chain.residues for chain in mixed_structure_state.chains], start=[])
    energy = bg.energies.PAEEnergy(oracle=fake_esmfold, residues=[residues[1:6:2], residues[2:7:2]], weight=2.0)
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    relevant_PAEs = [9, 11, 13, 15, 17, 19, 23, 25, 27, 29, 31, 33, 37, 39, 41, 43, 45, 47]
    # sum of relevant PAEs / (num PAEs * max PAE)
    assert np.isclose(unweighted_energy, np.mean(relevant_PAEs) / 30), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, np.mean(relevant_PAEs) / 30 * 2), 'weighted energy is incorrect'


def test_PAEEnergy_without_cross_term_only(
    fake_esmfold: bg.oracles.folding.ESMFold,
    mixed_structure_state: bg.State,
) -> None:
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.pae = np.arange(7**2).reshape((1, 7, 7))
    mock_folding_result.structure = mixed_structure_state._oracles_result[fake_esmfold].structure
    residues = sum([chain.residues for chain in mixed_structure_state.chains], start=[])
    energy = bg.energies.PAEEnergy(
        oracle=fake_esmfold,
        residues=[residues[1:6:4], residues[2:7:4]],
        cross_term_only=False,
        weight=2.0,
    )
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    relevant_PAEs = [9, 12, 13, 15, 19, 20, 36, 37, 41, 43, 44, 47]
    # sum of relevant PAEs / (num PAEs * max PAE)
    assert np.isclose(unweighted_energy, np.mean(relevant_PAEs) / 30), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, np.mean(relevant_PAEs) / 30 * 2), 'weighted energy is incorrect'


def test_PAEEnergy_of_residues_with_itself(
    fake_esmfold: bg.oracles.folding.ESMFold,
    mixed_structure_state: bg.State,
) -> None:
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.pae = np.arange(7**2).reshape((1, 7, 7))
    mock_folding_result.structure = mixed_structure_state._oracles_result[fake_esmfold].structure
    residues = sum([chain.residues for chain in mixed_structure_state.chains], start=[])
    energy = bg.energies.PAEEnergy(
        oracle=fake_esmfold,
        residues=[residues[1:6:2]],
        cross_term_only=False,
        weight=2.0,
    )
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    relevant_PAEs = [8, 10, 12, 22, 24, 26, 36, 38, 40]
    # sum of relevant PAEs / (num PAEs * max PAE)
    assert np.isclose(unweighted_energy, np.mean(relevant_PAEs) / 30), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, np.mean(relevant_PAEs) / 30 * 2), 'weighted energy is incorrect'


def test_FlexEvoBindEnergy_Unsymmetrized(
    fake_esmfold: bg.oracles.folding.ESMFold,
    simplest_dimer_state: bg.State,
) -> None:
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = simplest_dimer_state._oracles_result[fake_esmfold].structure
    mock_folding_result.local_plddt = simplest_dimer_state._oracles_result[fake_esmfold].local_plddt
    residues = sum([chain.residues for chain in simplest_dimer_state.chains], start=[])
    energy = bg.energies.FlexEvoBindEnergy(
        oracle=fake_esmfold,
        residues=[[residues[2]], residues[0:2]],
        plddt_weighted=True,
        symmetrized=False,
        weight=2.0,
    )
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    # print all attributes of oracles_result

    print(f'PLDDT = {oracles_result[fake_esmfold].local_plddt}')
    print(f'structure = {oracles_result[fake_esmfold].structure}')
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)

    # PLDDT weight with PLDDT = 0.5 (in denominator) and min_dist = 1.0 (in numerator)
    assert np.isclose(unweighted_energy, 2.0 * np.sqrt(5) / 2), f'unweighted energy is incorrect {unweighted_energy}'
    assert np.isclose(weighted_energy, 4.0 * np.sqrt(5) / 2), f'weighted energy is incorrect {weighted_energy}'

    energy = bg.energies.FlexEvoBindEnergy(
        oracle=fake_esmfold,
        residues=[[residues[2]], residues[0:2]],
        plddt_weighted=False,
        symmetrized=False,
        weight=2.0,
    )
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)

    assert np.isclose(unweighted_energy, np.sqrt(5) / 2), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, np.sqrt(5)), 'weighted energy is incorrect'


def test_FlexEvobindEnergy_Symmetrized(
    fake_esmfold: bg.oracles.folding.ESMFold,
    simplest_dimer_state: bg.State,
) -> None:
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = simplest_dimer_state._oracles_result[fake_esmfold].structure
    mock_folding_result.local_plddt = simplest_dimer_state._oracles_result[fake_esmfold].local_plddt
    residues = sum([chain.residues for chain in simplest_dimer_state.chains], start=[])
    energy = bg.energies.FlexEvoBindEnergy(
        oracle=fake_esmfold,
        residues=[[residues[2]], residues[0:2]],
        plddt_weighted=True,
        symmetrized=True,
        weight=2.0,
    )
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    # print all attributes of oracles_result

    print(f'PLDDT = {oracles_result[fake_esmfold].local_plddt}')
    print(f'structure = {oracles_result[fake_esmfold].structure}')
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)

    # PLDDT weight with PLDDT = 0.5 (in denominator) and min_dist = 1.0 (in numerator)
    assert np.isclose(unweighted_energy, np.sqrt(5.0)), f'unweighted energy is incorrect {unweighted_energy}'
    assert np.isclose(weighted_energy, 2.0 * np.sqrt(5.0)), f'weighted energy is incorrect {weighted_energy}'

    energy = bg.energies.FlexEvoBindEnergy(
        oracle=fake_esmfold,
        residues=[[residues[2]], residues[0:2]],
        plddt_weighted=False,
        symmetrized=True,
        weight=2.0,
    )
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)

    assert np.isclose(unweighted_energy, np.sqrt(5.0) / 2.0), 'no-plddt and unweighted energy is incorrect'
    assert np.isclose(weighted_energy, np.sqrt(5.0)), 'no-plddt and weighted energy is incorrect'


def test_RingSymmetryEnergy(
    fake_esmfold: bg.oracles.folding.ESMFold,
    square_structure_residues: list[bg.Residue],
    square_structure: AtomArray,
) -> None:
    energy = bg.energies.RingSymmetryEnergy(
        oracle=fake_esmfold,
        symmetry_groups=[[residue] for residue in square_structure_residues],
        weight=2.0,
    )
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = square_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    # centroids of each residue backbone make a 2d square of length 1
    value = np.std([1, 1, 2**0.5] * 4)
    assert np.isclose(unweighted_energy, value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, value * 2), 'weighted energy is incorrect'


# --- ChemicalPotentialEnergy ---


def test_ChemicalPotentialEnergy(
    fake_esmfold: bg.oracles.folding.ESMFold,
    square_structure_residues: list[bg.Residue],
    square_structure: AtomArray,
) -> None:
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = square_structure
    mock_folding_result.input_chains = [bg.Chain(residues=square_structure_residues)]
    energy = bg.energies.ChemicalPotentialEnergy(
        oracle=fake_esmfold, chemical_potential=-1.0, target_size=8.0, power=0.5, weight=2.0
    )
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    # Energy should be: chemical_potential * ( abs( number_of_residues - target_size ) )**power
    # -1.0 * ( abs( 4.0 - 8.0 )**0.5 ) = -1.0 * ( 4**0.5 ) = -1.0 * 2.0 = -2.0
    value = -2.0
    assert np.isclose(unweighted_energy, value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, value * 2), 'weighted energy is incorrect'


def test_ChemicalPotentialEnergy_with_embedding_oracle(
    fake_esm2: bg.oracles.embedding.ESM2,
):
    # Create a mock embedding oracle and result
    fake_embedding_result = Mock(bg.oracles.embedding.ESM2Result)
    # Create a chain with 3 residues
    residues = [bg.Residue(name='A', chain_ID='X', index=i) for i in range(3)]
    chain = bg.Chain(residues=residues)
    fake_embedding_result.input_chains = [chain]

    # Insert into OraclesResultDict
    oracles_result = OraclesResultDict({fake_esm2: fake_embedding_result})

    # Create ChemicalPotentialEnergy with target_size = 5, power = 2, chemical_potential = 1.5, weight = 2.0
    energy = bg.energies.ChemicalPotentialEnergy(
        oracle=fake_esm2, power=2.0, target_size=5, chemical_potential=1.5, weight=2.0
    )
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    # Should be: 1.5 * (abs(3-5))**2 = 1.5 * 4 = 6.0
    assert np.isclose(unweighted_energy, 6.0), f'unweighted energy is incorrect: {unweighted_energy}'
    assert np.isclose(weighted_energy, 12.0), f'weighted energy is incorrect: {weighted_energy}'


def test_RingSymmetryEnergy_with_direct_neighbours_only(
    fake_esmfold: bg.oracles.folding.ESMFold,
    square_structure_residues: list[bg.Residue],
    square_structure: AtomArray,
) -> None:
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = square_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    energy = bg.energies.RingSymmetryEnergy(
        oracle=fake_esmfold,
        symmetry_groups=[[residue] for residue in square_structure_residues],
        direct_neighbours_only=True,
        weight=2.0,
    )
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    # centroids of each residue make a 2d square of length 1. The direct neighbour distance for each atom is 1
    assert np.isclose(unweighted_energy, 0), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, 0 * 2), 'weighted energy is incorrect'


def test_SeparationEnergy(
    fake_esmfold: bg.oracles.folding.ESMFold,
    square_structure_residues: list[bg.Residue],
    square_structure: AtomArray,
) -> None:
    energy = bg.energies.SeparationEnergy(
        oracle=fake_esmfold,
        residues=[square_structure_residues[:2], square_structure_residues[2:]],
        weight=2.0,
    )
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = square_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    # distance between the centroids of the bottom corners and top corners for a square of length 1 is 1
    value = 1.0
    assert np.isclose(unweighted_energy, value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, value * 2), 'weighted energy is incorrect'


def make_harmonic_function(cutoff: float, stiffness: float):
    def harmonic_distance_to_energy(distance: float) -> float:
        if distance < cutoff:
            return 0.0
        return 0.5 * stiffness * (distance - cutoff) ** 2

    return harmonic_distance_to_energy


def test_SeparationEnergyNonLinear(
    fake_esmfold: bg.oracles.folding.ESMFold,
    square_structure_residues: list[bg.Residue],
    square_structure: AtomArray,
) -> None:
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = square_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    # residues chosen: [square_structure_residues[:2], square_structure_residues[2:]],
    # distance between the centroids of the bottom corners and top corners for a square of length 1 is 1

    energy = bg.energies.SeparationEnergy(
        oracle=fake_esmfold,
        residues=[square_structure_residues[:2], square_structure_residues[2:]],
        function=lambda x, x0=1.01, k=1.0: 0.0 if x < x0 else 0.5 * k * (x - x0) ** 2,
        weight=2.0,
    )
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    # harmonic potential but below cutoff distance, this should be 0 always
    assert np.isclose(unweighted_energy, 0.0), (
        f'unweighted energy is incorrect, 0 expected below harmonic cutoff but found {unweighted_energy}'
    )
    assert np.isclose(weighted_energy, 0.0), (
        f'weighted energy is incorrect, 0 expected below harmonic cutoff but found {weighted_energy}'
    )

    energy = bg.energies.SeparationEnergy(
        oracle=fake_esmfold,
        residues=[square_structure_residues[:2], square_structure_residues[2:]],
        function=make_harmonic_function(0.5, 1.0),
        weight=2.0,
    )
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    # harmonic potential 0.5 above cutoff distance, this should be 1/2 * (x-x0)**2 = 1/8
    assert np.isclose(unweighted_energy, 1.0 / 8.0), (
        f'unweighted energy is incorrect, {1.0 / 8.0} expected but found {unweighted_energy}'
    )
    assert np.isclose(weighted_energy, 1.0 / 4.0), (
        f'weighted energy is incorrect, {1.0 / 4.0} expected but found {weighted_energy}'
    )

    energy = bg.energies.SeparationEnergy(
        oracle=fake_esmfold,
        residues=[square_structure_residues[:2], square_structure_residues[2:]],
        function=lambda x, x0=1.0, k=10.0: 1.0 / (1.0 + np.exp(-k * (x - x0))),
        weight=2.0,
    )
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    # sigmoidal potential at cutoff distance, this should be 1.0/2.0 regardless of value of k
    assert np.isclose(unweighted_energy, 0.5), (
        f'unweighted energy is incorrect, 0.5 expected at x0 but found {unweighted_energy}'
    )
    assert np.isclose(weighted_energy, 1.0), (
        f'weighted energy is incorrect, 1.0 expected at x0 but found {weighted_energy}'
    )

    kk = 10.0
    energy = bg.energies.SeparationEnergy(
        oracle=fake_esmfold,
        residues=[square_structure_residues[:2], square_structure_residues[2:]],
        function=lambda x, x0=0.0, k=kk: 1.0 / (1.0 + np.exp(-k * (x - x0))),
        weight=2.0,
    )
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)

    # sigmoidal potential 1.0 above cutoff distance, this should be 1.0 / (1.0 + exp(-k))
    value = 1.0 / (1.0 + np.exp(-kk))
    assert np.isclose(unweighted_energy, value), (
        f'unweighted energy is incorrect, {value} expected but found {unweighted_energy}'
    )
    assert np.isclose(weighted_energy, 2.0 * value), (
        f'weighted energy is incorrect, {2.0 * value} expected but found {weighted_energy}'
    )


def test_GlobularEnergy(
    fake_esmfold: bg.oracles.folding.ESMFold,
    square_structure_residues: list[bg.Residue],
    square_structure: AtomArray,
) -> None:
    energy = bg.energies.GlobularEnergy(
        oracle=fake_esmfold,
        residues=square_structure_residues[:2],
        weight=2.0,
    )
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = square_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    # the centroid of the first 2 residue backbones are at [0, 0.5, 0] coords. The 4 atoms form a square of length
    # 1 around the centroid, equidistance from it.
    assert np.isclose(unweighted_energy, 0), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, 0 * 2), 'weighted energy is incorrect'


def test_TemplateMatchEnergy_gives_zero_distance_for_rotated_and_shifted_structure(
    fake_esmfold: bg.oracles.folding.ESMFold,
    square_structure_residues: list[bg.Residue],
    square_structure: AtomArray,
) -> None:
    # Comparing one of the diagonals of the square to the other
    template_atoms = copy.deepcopy(square_structure[np.isin(square_structure.res_id, [0, 2])])
    template_atoms.coord[:, 1:] += 3.0  # shifting strucutre in y and z
    energy = bg.energies.TemplateMatchEnergy(
        oracle=fake_esmfold,
        template_atoms=template_atoms,
        residues=square_structure_residues[1::2],
        backbone_only=True,
        weight=2.0,
    )
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = square_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    assert np.isclose(unweighted_energy, 0, atol=1e-7), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, 0 * 2, atol=1e-7), 'weighted energy is incorrect'


def test_TemplateMatchEnergy_gives_zero_distance_for_rotated_and_shifted_structure_using_distogram_metric(
    fake_esmfold: bg.oracles.folding.ESMFold,
    square_structure_residues: list[bg.Residue],
    square_structure: AtomArray,
) -> None:
    # Comparing one of the diagonals of the square to the other
    template_atoms = copy.deepcopy(square_structure[np.isin(square_structure.res_id, [0, 2])])
    template_atoms.coord[:, 1:] += 3.0  # shifting strucutre in y and z
    energy = bg.energies.TemplateMatchEnergy(
        oracle=fake_esmfold,
        template_atoms=template_atoms,
        residues=square_structure_residues[1::2],
        backbone_only=True,
        distogram_separation=True,
        weight=2.0,
    )
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = square_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    assert np.isclose(unweighted_energy, 0, atol=1e-7), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, 0 * 2), 'weighted energy is incorrect'


def test_TemplateMatchEnergy_is_correct_with_simple_structure(
    fake_esmfold: bg.oracles.folding.ESMFold,
    line_structure_residues: list[bg.Residue],
    line_structure: AtomArray,
) -> None:
    template_atoms = copy.deepcopy(line_structure[line_structure.res_id < 1])
    template_atoms.coord[0, :] -= [0.1, 0.1, 0]  # shifting back atom sqrt(0.02) backwards in direction of line
    template_atoms.coord[4, :] += [0.1, 0.1, 0]  # shifting front atom sqrt(0.02) forwards in direction of line
    energy = bg.energies.TemplateMatchEnergy(
        oracle=fake_esmfold,
        template_atoms=template_atoms,
        residues=line_structure_residues[:2],
        backbone_only=True,
        weight=2.0,
    )
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = line_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    # it is only 3 atoms because you only count for backbone atoms, defined as of type C, N and CA
    value = np.mean([0.02, 0.0, 0.02]) ** 0.5
    assert np.isclose(unweighted_energy, value), (
        'unweighted energy is incorrect'
    )  # first and last template atoms sqrt(0.02) away
    assert np.isclose(weighted_energy, value * 2), 'weighted energy is incorrect'


def test_TemplateMatchEnergy_is_correct_with_simple_structure_using_distogram_metric(
    fake_esmfold: bg.oracles.folding.ESMFold,
    line_structure_residues: list[bg.Residue],
    line_structure: AtomArray,
) -> None:
    template_atoms = copy.deepcopy(line_structure[line_structure.res_id < 1])
    template_atoms.coord[0, :] -= [0.1, 0.1, 0]  # shifting back atom sqrt(0.02) backwards in direction of line
    template_atoms.coord[4, :] += [0.1, 0.1, 0]  # shifting front atom sqrt(0.02) forwards in direction of line
    energy = bg.energies.TemplateMatchEnergy(
        oracle=fake_esmfold,
        template_atoms=template_atoms,
        residues=line_structure_residues[:2],
        backbone_only=True,
        distogram_separation=True,
        weight=2.0,
    )
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = line_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    unique_distogram_distances_squared = [0.02, 0.08, 0.02] * 2  # requires a small sketch to make sense of
    value = np.mean(unique_distogram_distances_squared) ** 0.5
    assert np.isclose(unweighted_energy, value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, value * 2), 'weighted energy is incorrect'


def test_TemplateMatchEnergy_is_correct_with_different_atom_order(
    fake_esmfold: bg.oracles.folding.ESMFold,
    formolase_ordered_residues: list[bg.Residue],
    formolase_ordered_structure: AtomArray,
    formolase_structure: AtomArray,
) -> None:
    template_atoms = formolase_structure

    energy = bg.energies.TemplateMatchEnergy(
        oracle=fake_esmfold,
        template_atoms=template_atoms,
        residues=formolase_ordered_residues,
        backbone_only=False,
        weight=2.0,
    )
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = formolase_ordered_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    value = 0.0
    assert np.isclose(unweighted_energy, value, atol=1e-5), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, value * 2, atol=1e-5), 'weighted energy is incorrect'


def test_secondary_structure_elements_function_gives_expected_return_array(small_structure: AtomArray) -> None:
    sse_labels = annotate_sse(small_structure)
    assert len(sse_labels) == get_residue_count(small_structure), 'sse does not return one number for each residue'
    assert np.issubdtype(sse_labels.dtype, np.str_), 'sse does not return strings for each atom'


@patch('bagel.energies.annotate_sse')
def test_SecondaryStructureEnergy(
    mock_annotate_sse: Mock,
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure_residues: list[bg.Residue],
    small_structure: AtomArray,
) -> None:
    mock_annotate_sse.return_value = np.array(['a', '', 'c'])
    energy = bg.energies.SecondaryStructureEnergy(
        oracle=fake_esmfold, residues=small_structure_residues, target_secondary_structure='coil', weight=2.0
    )
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    value = 2 / 3
    assert np.isclose(unweighted_energy, value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, value * 2), 'weighted energy is incorrect'


def test_embeddings_similarity_energy(
    square_structure_residues: list[bg.Residue],
    esm2: bg.oracles.embedding.ESM2,
):
    esmfold = bg.oracles.folding.ESMFold()

    # Enforce that the oracle is an instance of EmbeddingOracle
    with pytest.raises(AssertionError) as e:
        energy = bg.energies.EmbeddingsSimilarityEnergy(
            oracle=esmfold,
            residues=square_structure_residues,
            reference_embeddings=np.zeros((len(square_structure_residues), 1280)),  # Using typical ESM2 embedding size
        )
    # Assert outside the `with` block: inside it the construction above raises first, so the
    # message check would never execute.
    assert 'Oracle must be an instance of EmbeddingOracle' in str(e.value)

    # Enforce correct number of reference embeddings
    with pytest.raises(AssertionError) as e:
        energy = bg.energies.EmbeddingsSimilarityEnergy(
            oracle=esm2,
            residues=square_structure_residues,
            reference_embeddings=np.zeros(
                (len(square_structure_residues) - 1, 1280)
            ),  # Using typical ESM2 embedding size
        )
    assert 'Number of reference embeddings (1) does not match number of residues to include in energy term (2)' in str(
        e.value
    )

    # Test dynamic reference embeddings
    # Create initial two-chain multimer state
    chain_A = bg.Chain(
        [
            bg.Residue(name='A', chain_ID='A', index=0),
            bg.Residue(name='R', chain_ID='A', index=1),
            bg.Residue(name='N', chain_ID='A', index=2),
        ]
    )
    chain_B = bg.Chain(
        [
            bg.Residue(name='D', chain_ID='B', index=0),
            bg.Residue(name='C', chain_ID='B', index=1),
        ]
    )

    # Create energy term tracking specific residues across both chains
    tracked_residues = [
        chain_A.residues[1],  # A1
        chain_B.residues[0],  # B0
        chain_A.residues[2],  # A2
    ]

    energy = bg.energies.EmbeddingsSimilarityEnergy(
        oracle=esm2,
        residues=tracked_residues,
        reference_embeddings=np.zeros((len(tracked_residues), 1280)),  # Using typical ESM2 embedding size
    )

    # Initial state - verify correct indices
    # Expected: [1, 3, 2] because:
    # Chain A: indices 0,1,2 (first 3 positions)
    # Chain B: indices 0,1 (next 2 positions)
    # So B0 is at global position 3
    initial_indices = energy.conserved_index_list([chain_A, chain_B])
    assert initial_indices == [1, 3, 2], f'Initial indices incorrect: {initial_indices}'

    # Test dynamic changes:
    # 1. Add residue before tracked residue in chain A
    new_residue = bg.Residue(name='W', chain_ID='A', index=1)
    energy.shift_residues_indices_before_addition(chain_id=new_residue.chain_ID, res_index=new_residue.index)
    chain_A.add_residue(amino_acid=new_residue.name, index=new_residue.index)

    # Now indices should be [2, 4, 3] because:
    # - A1 moved to position 2
    # - B0 moved to position 4 (due to new residue in chain A)
    # - A2 moved to position 3
    indices_after_addition = energy.conserved_index_list([chain_A, chain_B])
    assert indices_after_addition == [2, 4, 3], f'Indices after addition incorrect: {indices_after_addition}'

    # 2. Remove a residue from chain A that affects positions
    chain_A.remove_residue(index=0)
    # Update energy term indices
    energy.remove_residue(chain_id='A', res_index=0)
    energy.shift_residues_indices_after_removal(chain_id='A', res_index=0)

    # Now indices should be [1, 3, 2] because:
    # - Removing first residue shifts everything back
    indices_after_removal = energy.conserved_index_list([chain_A, chain_B])
    assert indices_after_removal == [1, 3, 2], f'Indices after removal incorrect: {indices_after_removal}'


def test_LISEnergy(
    fake_esmfold: bg.oracles.folding.ESMFold,
    mixed_structure_state: bg.State,
) -> None:
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.pae = np.arange(7**2).reshape((1, 7, 7))
    mock_folding_result.structure = mixed_structure_state._oracles_result[fake_esmfold].structure
    residues = sum([chain.residues for chain in mixed_structure_state.chains], start=[])
    energy = bg.energies.LISEnergy(oracle=fake_esmfold, residues=[residues[0:2], residues[2:4]], weight=2.0)
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)
    # relevant_PAEs = np.array( [2, 3, 9, 10, 14, 15, 21, 22] )
    # mask = relevant_PAEs <= 12.0
    # relevant_PAEs = relevant_PAEs[mask]
    # expected = -np.mean( (12.0 - relevant_PAEs ) / 12.0 )
    expected = -0.5  # Calculated by hand
    assert np.isclose(unweighted_energy, expected), (
        f'unweighted energy is incorrect, expected 0.5, found {unweighted_energy}'
    )
    assert np.isclose(weighted_energy, 2.0 * expected), (
        f'weighted energy is incorrect, expected {2.0 * expected}, found {weighted_energy}'
    )


def test_HydropathyEnergy_all_mode(
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure: AtomArray,
) -> None:
    """Test HydropathyEnergy computation in 'all' mode with correct values."""
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    energy = bg.energies.HydropathyEnergy(
        oracle=fake_esmfold,
        mode='all',
        weight=2.0,
    )

    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)

    # Compute expected manually using hydropathy index
    from bagel.constants import hydropathy_index

    # Get unique residues (chain_id, res_id pairs) while preserving order
    unique_indices = []
    seen = set()
    for i, (chain, res) in enumerate(zip(small_structure.chain_id, small_structure.res_id)):
        pair = (chain, res)
        if pair not in seen:
            seen.add(pair)
            unique_indices.append(i)

    expected_values = []
    for idx in unique_indices:
        res_name = small_structure.res_name[idx]
        expected_values.append(hydropathy_index.get(res_name, 0.0))

    expected = np.mean(expected_values)

    assert np.isclose(unweighted_energy, expected), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, expected * 2.0), 'weighted energy is incorrect'


@patch('bagel.energies.sasa')
def test_HydropathyEnergy_surface_mode(
    mock_sasa: Mock,
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure: AtomArray,
) -> None:
    """Test HydropathyEnergy computation in 'surface' mode with SASA weighting."""
    mock_sasa.return_value = np.array([22, 22, 22, 22, 0])  # atoms of first 2 residues are given max sasa

    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    energy = bg.energies.HydropathyEnergy(
        oracle=fake_esmfold,
        mode='surface',
        weight=2.0,
    )

    from bagel.constants import hydropathy_index, max_theoretical_sasa_for_residues

    residues = ['GLY', 'VAL', 'VAL']
    residue_sasa = np.array([44, 44, 0])  # SASA for each residue by summing atomic SASA values
    max_sasa = np.array([max_theoretical_sasa_for_residues[res] for res in residues])
    normalized_sasa = np.clip(residue_sasa / max_sasa, 0.0, 1.0)
    hydropathy = np.array([hydropathy_index[res] for res in residues])

    # Weighted mean: sum(h_i * rel_sasa_i) / sum(rel_sasa_i)
    expected_value = np.sum(hydropathy * normalized_sasa) / np.sum(normalized_sasa)

    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)

    # Verify weight is applied correctly
    assert np.isfinite(unweighted_energy), 'energy should be finite'
    assert np.isclose(unweighted_energy, expected_value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, expected_value * 2.0), 'weight should be applied correctly'


@patch('bagel.energies.sasa')
def test_HydropathyEnergy_core_mode(
    mock_sasa: Mock,
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure: AtomArray,
) -> None:
    """Test HydropathyEnergy computation in 'core' mode with inverted SASA weighting."""
    mock_sasa.return_value = np.zeros(len(small_structure))

    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    energy = bg.energies.HydropathyEnergy(
        oracle=fake_esmfold,
        mode='core',
        weight=2.0,
    )

    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)

    # With SASA=0, normalized weight for core mode is 1.0 (1.0 - 0.0)
    from bagel.constants import hydropathy_index

    # Small_structure has 3 residues: GLY (-0.4), VAL (4.2), VAL (4.2)
    # For core mode with all SASA=0: weight = 1.0 - 0.0 = 1.0 for all
    # expected_value = sum(h_i * 1.0) / sum(1.0) = (-0.4 + 4.2 + 4.2) / 3
    hydropathy = np.array(
        [
            hydropathy_index['GLY'],
            hydropathy_index['VAL'],
            hydropathy_index['VAL'],
        ]
    )
    core_weights = np.ones(3)  # 1.0 - 0.0 for all
    expected_value = np.sum(hydropathy * core_weights) / np.sum(core_weights)

    assert np.isfinite(unweighted_energy), 'energy should be finite'
    assert np.isclose(unweighted_energy, expected_value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, expected_value * 2.0), 'weighted energy is incorrect'


def test_HydropathyEnergy_with_selected_residues(
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure_residues: list[bg.Residue],
    small_structure: AtomArray,
) -> None:
    """Test HydropathyEnergy computation with selected residues subset."""
    from bagel.constants import hydropathy_index

    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    energy = bg.energies.HydropathyEnergy(
        oracle=fake_esmfold,
        residues=small_structure_residues[:1],
        weight=2.0,
    )

    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)

    # Expected energy is the GRAVY value of the single selected residue (GLY) in mode='all' (no SASA weighting)
    expected_value = hydropathy_index['GLY']

    assert np.isfinite(unweighted_energy), 'energy should be finite'
    assert np.isclose(unweighted_energy, expected_value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, expected_value * 2.0), 'weight applied correctly'


def test_HydropathyEnergy_unknown_residue_handling(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    """Test that unknown residues are filtered out by mocking structure."""
    from biotite.structure import Atom, array
    from bagel.constants import hydropathy_index

    # Create a small structure with an unknown residue
    atoms = [
        Atom(coord=[0, 0, 0], chain_id='A', res_id=0, res_name='XXX', element='C', atom_name='CA'),  # Unknown
        Atom(coord=[1, 0, 0], chain_id='A', res_id=1, res_name='VAL', element='C', atom_name='CA'),  # Known
        Atom(coord=[2, 0, 0], chain_id='A', res_id=2, res_name='VAL', element='C', atom_name='CA'),  # Known
    ]
    structure = array(atoms)

    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = structure

    energy = bg.energies.HydropathyEnergy(
        oracle=fake_esmfold,
        weight=1.0,
    )
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})

    # Should issue a deterministic warning about unknown residue
    with pytest.warns(UserWarning, match=r"Unknown residues encountered: \('XXX',\) \(count=1\)") as warning_record:
        unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)

    assert len(warning_record) == 1

    # Unknown residue gets removed from calculation, so only the two VAL residues contribute
    # Mean of two identical GRAVY values is just that value
    expected_value = hydropathy_index['VAL']

    assert np.isclose(unweighted_energy, expected_value), (
        'energy with unknown residue should exclude it from calculation'
    )
    assert np.isclose(weighted_energy, expected_value * 1.0), 'weighted energy should match'


def test_HydropathyEnergy_missing_residue_warning(
    fake_esmfold,
    small_structure,
) -> None:
    """Test that selecting residues not present in structure should warn and skip them."""
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    # Residue that does NOT exist
    fake_residues = [
        bg.Residue(name='A', chain_ID='Z', index=999),
    ]

    energy = bg.energies.HydropathyEnergy(
        oracle=fake_esmfold,
        residues=fake_residues,
        weight=1.0,
    )

    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})

    with pytest.warns(UserWarning, match='not found in structure'):
        value, weighted = energy.compute(oracles_result)

    assert value == 0.0
    assert weighted == 0.0


def test_HydropathyEnergy_empty_residue_selection(
    fake_esmfold,
    small_structure,
) -> None:
    """If selected residues exist but none match structure → return 0."""
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    # Residue group that won't match anything
    energy = bg.energies.HydropathyEnergy(
        oracle=fake_esmfold,
        residues=[bg.Residue(name='A', chain_ID='Z', index=999)],
        weight=1.0,
    )

    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})

    # Warning for missing residues
    with pytest.warns(UserWarning):
        value, weighted = energy.compute(oracles_result)

    assert value == 0.0
    assert weighted == 0.0


def test_HydropathyEnergy_empty_structure_returns_zero(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    """Test that empty structure returns zero energy."""
    from biotite.structure import AtomArray

    # Create truly empty structure
    empty_structure = AtomArray(0)

    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = empty_structure

    energy = bg.energies.HydropathyEnergy(
        oracle=fake_esmfold,
        weight=2.0,
    )

    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result=oracles_result)

    assert unweighted_energy == 0.0, 'unweighted energy should be 0 for empty structure'
    assert weighted_energy == 0.0, 'weighted energy should be 0 for empty structure'


# ---------------------------------------------------------------------------------------------------------------
# ShapeComplementarityEnergy
# ---------------------------------------------------------------------------------------------------------------

CARBON_CONTACT = 2 * 1.87  # separation of two carbon nuclei whose united-atom vdW surfaces just touch


def build_plate(
    n_side: int,
    height: float,
    chain_id: str,
    res_name: str = 'LEU',
    spacing: float = 2.5,
    corrugation: float = 0.0,
) -> tuple[list[Atom], list[bg.Residue]]:
    """Builds a square slab of atoms in the xy plane, one atom per residue, optionally corrugated in z."""
    atoms, residues = [], []
    letter = {v: k for k, v in bg.constants.aa_dict.items()}[res_name]
    for index in range(n_side * n_side):
        i, j = divmod(index, n_side)
        z = height + corrugation * np.sin(0.9 * i) * np.cos(0.9 * j)
        atoms.append(
            Atom(
                coord=[i * spacing, j * spacing, z],
                chain_id=chain_id,
                res_id=index,
                res_name=res_name,
                element='C',
                atom_name='CA',
            )
        )
        residues.append(bg.Residue(name=letter, chain_ID=chain_id, index=index))
    return atoms, residues


def facing_plates(
    separation: float, n_side: int = 8, corrugation: float = 0.0, mirror: bool = False, res_name: str = 'LEU'
) -> tuple[AtomArray, tuple[list[bg.Residue], list[bg.Residue]]]:
    """Builds two slabs facing each other across `separation`, optionally with matching or opposed corrugation."""
    bottom_atoms, bottom_residues = build_plate(n_side, 0.0, 'A', res_name, corrugation=corrugation)
    top_atoms, top_residues = build_plate(
        n_side, separation, 'B', res_name, corrugation=-corrugation if mirror else corrugation
    )
    return array(bottom_atoms + top_atoms), (bottom_residues, top_residues)


def shape_complementarity(
    oracle: bg.oracles.folding.ESMFold,
    structure: AtomArray,
    groups: tuple[list[bg.Residue], list[bg.Residue]],
    **kwargs,
) -> float:
    """Returns minus the energy for the given structure and residue groups, so that a better fit reads higher."""
    energy = bg.energies.ShapeComplementarityEnergy(oracle=oracle, residues=groups, **kwargs)
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = structure
    unweighted_energy, _ = energy.compute(oracles_result=OraclesResultDict({oracle: mock_folding_result}))
    return -unweighted_energy


def test_fibonacci_sphere_returns_evenly_spread_unit_vectors() -> None:
    points = bg.energies._fibonacci_sphere(500)
    assert points.shape == (500, 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0), 'points must lie on the unit sphere'
    assert np.allclose(points.mean(axis=0), 0.0, atol=1e-2), 'points must be spread evenly over the whole sphere'
    assert np.array_equal(points, bg.energies._fibonacci_sphere(500)), 'point generation must be deterministic'


def test_statistic_handles_both_modes_and_empty_input() -> None:
    values = np.array([0.0, 1.0, 2.0, 30.0])
    assert np.isclose(bg.energies._statistic(values, 'mean'), 8.25)
    # the median must ignore the outlier that drags the mean
    assert np.isclose(bg.energies._statistic(values, 'median'), 1.5)
    assert bg.energies._statistic(np.zeros(0), 'mean') == 0.0, 'empty input must not produce a nan'
    assert bg.energies._statistic(np.zeros(0), 'median') == 0.0, 'empty input must not produce a nan'


def test_molecular_surface_dots_of_an_isolated_atom_lie_on_its_vdw_sphere() -> None:
    coords = np.zeros((1, 3))
    radii = np.array([1.87])
    unit_sphere = bg.energies._fibonacci_sphere(100)
    dots, normals, parents = bg.energies._molecular_surface_dots(coords, radii, 1.4, unit_sphere, np.array([True]))
    assert len(dots) == 100, 'nothing can occlude a lone atom, so every dot must survive'
    assert np.allclose(np.linalg.norm(dots, axis=1), 1.87), 'dots must sit on the vdW sphere, not the probe sphere'
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0), 'normals must be unit vectors'
    assert np.all(parents == 0)


def test_molecular_surface_dots_removes_crevices_a_probe_cannot_enter() -> None:
    # two atoms far enough apart not to overlap, but too close for a 1.4 A probe to reach between them
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    radii = np.array([1.87, 1.87])
    unit_sphere = bg.energies._fibonacci_sphere(400)
    seed = np.array([True, True])
    with_probe, _, _ = bg.energies._molecular_surface_dots(coords, radii, 1.4, unit_sphere, seed)
    without_probe, _, _ = bg.energies._molecular_surface_dots(coords, radii, 0.0, unit_sphere, seed)
    assert len(with_probe) < len(without_probe), 'the probe must exclude the crevice between the two atoms'
    # without a probe the surfaces reach right up to where the two atoms almost meet; with one they must not
    midplane_gap_with_probe = np.abs(with_probe[:, 0] - 2.0).min()
    midplane_gap_without_probe = np.abs(without_probe[:, 0] - 2.0).min()
    assert midplane_gap_with_probe > midplane_gap_without_probe + 0.5, (
        'the probe should hold the surface well back from the crevice between the two atoms'
    )


def test_buried_by_partner_only_flags_dots_the_partner_shields() -> None:
    dots = np.array([[1.87, 0.0, 0.0], [-1.87, 0.0, 0.0]])  # one dot facing the partner, one facing away
    normals = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    partner_coords = np.array([[CARBON_CONTACT, 0.0, 0.0]])
    buried = bg.energies._buried_by_partner(dots, normals, partner_coords, np.array([1.87]), 1.4)
    assert buried[0] and not buried[1], 'only the dot pressed against the partner is buried'
    assert not bg.energies._buried_by_partner(dots, normals, np.zeros((0, 3)), np.zeros(0), 1.4).any()


def test_ShapeComplementarityEnergy_rewards_well_fitting_surfaces(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    structure, groups = facing_plates(CARBON_CONTACT)
    touching = shape_complementarity(fake_esmfold, structure, groups, scaling='intensive')

    structure, groups = facing_plates(CARBON_CONTACT + 1.5)
    separated = shape_complementarity(fake_esmfold, structure, groups, scaling='intensive')

    assert touching > 0.6, f'two flat surfaces in contact should be highly complementary, found {touching}'
    assert touching > separated, 'pulling the surfaces apart must reduce complementarity'
    assert separated > 0.0, 'surfaces 1.5 A apart still partly face each other'


def test_ShapeComplementarityEnergy_prefers_interlocking_to_clashing_corrugation(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    # same corrugation on both slabs means every bump sits in the opposing groove: a lock and key fit
    structure, groups = facing_plates(CARBON_CONTACT, corrugation=1.2, mirror=False)
    interlocking = shape_complementarity(fake_esmfold, structure, groups)
    # mirrored corrugation puts bump against bump instead
    structure, groups = facing_plates(CARBON_CONTACT, corrugation=1.2, mirror=True)
    clashing = shape_complementarity(fake_esmfold, structure, groups)
    assert interlocking > clashing, f'interlocking ({interlocking}) should beat clashing ({clashing}) surfaces'


def test_ShapeComplementarityEnergy_is_zero_when_the_groups_do_not_touch(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    structure, groups = facing_plates(40.0)
    assert shape_complementarity(fake_esmfold, structure, groups) == 0.0, 'no interface means no complementarity'


def test_ShapeComplementarityEnergy_is_bounded_symmetric_and_deterministic(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    structure, (group_1, group_2) = facing_plates(CARBON_CONTACT, corrugation=0.8)
    value = shape_complementarity(fake_esmfold, structure, (group_1, group_2), scaling='intensive')
    assert -1.0 <= value <= 1.0, 'the intensive statistic is a weighted mean of quantities bounded by 1'
    assert np.isclose(value, shape_complementarity(fake_esmfold, structure, (group_2, group_1), scaling='intensive')), (
        'the energy must not depend on which group is passed first'
    )
    assert value == shape_complementarity(fake_esmfold, structure, (group_1, group_2), scaling='intensive'), (
        're-evaluating the same structure must give a bit-identical energy'
    )


def test_ShapeComplementarityEnergy_applies_its_weight(fake_esmfold: bg.oracles.folding.ESMFold) -> None:
    structure, groups = facing_plates(CARBON_CONTACT)
    energy = bg.energies.ShapeComplementarityEnergy(oracle=fake_esmfold, residues=groups, weight=3.0)
    assert energy.scaling == 'extensive', 'the default scaling should be the one suitable for an energy'
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = structure
    unweighted_energy, weighted_energy = energy.compute(OraclesResultDict({fake_esmfold: mock_folding_result}))
    assert unweighted_energy < 0, 'a good fit must lower the energy'
    assert np.isclose(weighted_energy, unweighted_energy * 3.0), 'weighted energy is incorrect'


@pytest.mark.parametrize('scaling', ['extensive', 'intensive'])
def test_ShapeComplementarityEnergy_is_blind_to_residue_chemistry(
    fake_esmfold: bg.oracles.folding.ESMFold, scaling: str
) -> None:
    # The term is purely geometric: identical geometry must give an identical energy whatever the residues are.
    # This is the property that the removed hydrophobicity weighting used to break, and it is worth pinning.
    leucine_structure, leucine_groups = facing_plates(CARBON_CONTACT, corrugation=0.8, res_name='LEU')
    arginine_structure, arginine_groups = facing_plates(CARBON_CONTACT, corrugation=0.8, res_name='ARG')
    leucine = shape_complementarity(fake_esmfold, leucine_structure, leucine_groups, scaling=scaling)
    arginine = shape_complementarity(fake_esmfold, arginine_structure, arginine_groups, scaling=scaling)
    assert np.isclose(leucine, arginine), (
        'an all-leucine and an all-arginine interface of the same shape must score the same'
    )


def test_ShapeComplementarityEnergy_takes_no_chemistry_arguments(fake_esmfold: bg.oracles.folding.ESMFold) -> None:
    # Guards against the hydrophobicity weighting being reintroduced without a deliberate decision.
    _, groups = facing_plates(CARBON_CONTACT, n_side=2)
    parameters = inspect.signature(bg.energies.ShapeComplementarityEnergy.__init__).parameters
    assert 'hydrophobic_weight' not in parameters
    assert 'hydrophobicity' not in parameters
    with pytest.raises(TypeError):
        bg.energies.ShapeComplementarityEnergy(oracle=fake_esmfold, residues=groups, hydrophobic_weight=4.0)  # type: ignore[call-arg]


def test_ShapeComplementarityEnergy_rejects_invalid_arguments(fake_esmfold: bg.oracles.folding.ESMFold) -> None:
    _, groups = facing_plates(CARBON_CONTACT, n_side=2)
    valid = dict(oracle=fake_esmfold, residues=groups)
    with pytest.raises(AssertionError):
        bg.energies.ShapeComplementarityEnergy(oracle=fake_esmfold, residues=([], groups[1]))
    with pytest.raises(AssertionError):
        bg.energies.ShapeComplementarityEnergy(**valid, interface_cutoff=0.0)
    with pytest.raises(AssertionError):
        bg.energies.ShapeComplementarityEnergy(**valid, distance_decay=-1.0)
    with pytest.raises(AssertionError):
        bg.energies.ShapeComplementarityEnergy(**valid, n_surface_points=0)
    with pytest.raises(AssertionError):
        bg.energies.ShapeComplementarityEnergy(**valid, probe_radius=-0.1)
    with pytest.raises(AssertionError):
        bg.energies.ShapeComplementarityEnergy(**valid, statistic='wrong')  # type: ignore[arg-type]


def test_ShapeComplementarityEnergy_warns_when_the_two_groups_overlap(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    _, (group_1, group_2) = facing_plates(CARBON_CONTACT, n_side=2)
    with pytest.warns(UserWarning, match='appear in both groups'):
        bg.energies.ShapeComplementarityEnergy(oracle=fake_esmfold, residues=(group_1, group_1 + group_2))


@pytest.mark.parametrize('scaling', ['extensive', 'intensive'])
def test_ShapeComplementarityEnergy_ranks_a_real_interface_above_a_decoy(
    fake_esmfold: bg.oracles.folding.ESMFold, formolase_structure: AtomArray, scaling: str
) -> None:
    chain_ids = list(pd.unique(formolase_structure.chain_id))[:2]
    groups = []
    for chain_id in chain_ids:
        chain = formolase_structure[formolase_structure.chain_id == chain_id]
        sequence = bg.oracles.folding.utils.sequence_from_atomarray(chain)
        # index must be the structure's res_id, not a 0-based position: get_atom_mask matches
        # bg.Residue.index against AtomArray.res_id, so enumerate() would drop the residues whose
        # res_id falls outside 0..len-1 (here res_id runs 2..564).
        res_ids = list(dict.fromkeys(int(r) for r in chain.res_id))
        groups.append([bg.Residue(name=aa, chain_ID=chain_id, index=res_id) for aa, res_id in zip(sequence, res_ids)])
    native_groups = (groups[0], groups[1])

    native = shape_complementarity(fake_esmfold, formolase_structure, native_groups, scaling=scaling)
    # slide one subunit sideways: the same residues are still in contact, but they no longer interlock
    decoy_structure = copy.deepcopy(formolase_structure)
    decoy_structure.coord[decoy_structure.chain_id == chain_ids[1]] += np.array([3.0, 0.0, 0.0])
    decoy = shape_complementarity(fake_esmfold, decoy_structure, native_groups, scaling=scaling)

    assert 0.3 < native < 1.0, f'a real protein-protein interface should score well, found {native}'
    assert native > decoy + 0.1, f'the native packing ({native}) should clearly beat the decoy ({decoy})'


def test_ShapeComplementarityEnergy_is_short_ranged(fake_esmfold: bg.oracles.folding.ESMFold) -> None:
    # a dot only counts while the partner shields it from the solvent, so the term must die once a probe fits
    # between the two surfaces, i.e. beyond about twice the probe radius of clearance
    probe_radius = bg.constants.probe_radius_water
    in_range = shape_complementarity(fake_esmfold, *facing_plates(CARBON_CONTACT + 1.5 * probe_radius))
    out_of_range = shape_complementarity(fake_esmfold, *facing_plates(CARBON_CONTACT + 2.5 * probe_radius))
    assert in_range > 0.0, 'surfaces closer together than two probe radii must still see each other'
    assert out_of_range == 0.0, 'once a solvent molecule fits between the surfaces the energy must vanish exactly'


def test_ShapeComplementarityEnergy_intensive_scaling_grows_with_size_but_saturates(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    # The statistic is a per-dot average, so it is not extensive. It still rises with contact area through a
    # perimeter effect - a small patch is dominated by its poorly facing rim - but the rise must flatten off,
    # unlike a buried-area term which would keep growing in proportion to the number of residues.
    scores = [
        shape_complementarity(fake_esmfold, *facing_plates(CARBON_CONTACT, n_side=n), scaling='intensive')
        for n in (3, 6, 12)
    ]
    small, medium, large = scores
    assert small < medium < large, 'a larger contact patch is less dominated by its rim and should score higher'
    assert (large - medium) < (medium - small), 'the gain from extra interface area must saturate, not stay linear'
    assert large < 4 * small, 'the statistic is an average, so it must not scale with the number of residues'


def build_ragged_interface() -> tuple[
    AtomArray, tuple[list[bg.Residue], list[bg.Residue]], tuple[list[bg.Residue], list[bg.Residue]]
]:
    """Builds an interface whose first half packs tightly and whose second half is held 1.5 A too far apart."""
    n_side = 8
    atoms, group_1, group_2, tight_1, tight_2 = [], [], [], [], []
    for chain_id, base_height in (('A', 0.0), ('B', CARBON_CONTACT)):
        for index in range(n_side * n_side):
            i, j = divmod(index, n_side)
            tightly_packed = i < n_side // 2
            height = base_height if tightly_packed or chain_id == 'A' else base_height + 1.5
            atoms.append(
                Atom(
                    coord=[i * 2.5, j * 2.5, height],
                    chain_id=chain_id,
                    res_id=index,
                    res_name='LEU',
                    element='C',
                    atom_name='CA',
                )
            )
            residue = bg.Residue(name='L', chain_ID=chain_id, index=index)
            group = group_1 if chain_id == 'A' else group_2
            group.append(residue)
            if tightly_packed:
                (tight_1 if chain_id == 'A' else tight_2).append(residue)

    return array(atoms), (group_1, group_2), (tight_1, tight_2)


def test_ShapeComplementarityEnergy_extensive_scaling_penalises_shedding_a_badly_packed_region(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    # The whole point of integrating rather than averaging: dropping part of the interface must cost energy, even
    # if the part dropped was the badly packed part. Otherwise a design can "improve" by shedding bad contacts.
    structure, whole_groups, tight_groups = build_ragged_interface()
    whole = shape_complementarity(fake_esmfold, structure, whole_groups, scaling='extensive')
    trimmed = shape_complementarity(fake_esmfold, structure, tight_groups, scaling='extensive')
    assert whole > trimmed > 0, (
        f'losing the loose half must cost energy under extensive scaling, but the score went {whole} -> {trimmed}'
    )


def test_ShapeComplementarityEnergy_intensive_scaling_rewards_shedding_a_badly_packed_region(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    # Documents the pathology that motivates the extensive default, and guards the docstring warning about it.
    structure, whole_groups, tight_groups = build_ragged_interface()
    whole = shape_complementarity(fake_esmfold, structure, whole_groups, scaling='intensive')
    trimmed = shape_complementarity(fake_esmfold, structure, tight_groups, scaling='intensive')
    assert trimmed > whole, (
        f'averaging should reward dropping the loose half ({whole} -> {trimmed}); if this ever fails the docstring '
        f'warning about intensive scaling needs revisiting'
    )


def test_ShapeComplementarityEnergy_is_extensive_in_the_contact_area(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    # An energy should be linear in the amount of interface, up to a perimeter correction. Fit E = a n^2 + b n
    # against square patches of side n and check the bulk term dominates and the model is accurate.
    sides = np.array([4, 6, 8, 10, 12, 14], dtype=float)
    values = np.array(
        [
            shape_complementarity(fake_esmfold, *facing_plates(CARBON_CONTACT, n_side=int(n)), scaling='extensive')
            for n in sides
        ]
    )
    design = np.stack([sides**2, sides], axis=1)
    (bulk, perimeter), *_ = np.linalg.lstsq(design, values, rcond=None)
    predicted = design @ np.array([bulk, perimeter])

    assert np.all(values > 0), 'a well packed interface must give a favourable (negative) energy at every size'
    assert np.max(np.abs(predicted - values) / values) < 0.05, 'area plus perimeter should describe the energy well'
    assert bulk > 0, 'the bulk term must reward interface area'
    # the perimeter correction must be a correction, not the leading behaviour, for a large patch
    assert abs(perimeter * sides[-1]) < 0.25 * abs(bulk * sides[-1] ** 2)
    # doubling the area roughly doubles the energy, which an intensive statistic could never do
    small = shape_complementarity(fake_esmfold, *facing_plates(CARBON_CONTACT, n_side=8), scaling='extensive')
    large = shape_complementarity(fake_esmfold, *facing_plates(CARBON_CONTACT, n_side=16), scaling='extensive')
    assert 3.4 < large / small < 4.6, (
        f'four times the area should give roughly four times the energy, got {large / small}'
    )


def test_ShapeComplementarityEnergy_extensive_scaling_is_still_zero_out_of_contact(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    apart = shape_complementarity(fake_esmfold, *facing_plates(40.0), scaling='extensive')
    assert apart == 0.0, 'integrating over an empty interface must give exactly zero, not a nan'


def test_ShapeComplementarityEnergy_rejects_invalid_scaling(fake_esmfold: bg.oracles.folding.ESMFold) -> None:
    _, groups = facing_plates(CARBON_CONTACT, n_side=2)
    with pytest.raises(AssertionError):
        bg.energies.ShapeComplementarityEnergy(oracle=fake_esmfold, residues=groups, scaling='wrong')  # type: ignore[arg-type]
    with pytest.raises(AssertionError):
        bg.energies.ShapeComplementarityEnergy(oracle=fake_esmfold, residues=groups, area_scale=0.0)
