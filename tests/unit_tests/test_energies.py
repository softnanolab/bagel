import os
import bagel as bg
from bagel.oracles import OraclesResultDict
from biotite.structure import AtomArray, sasa, annotate_sse, get_residue_count, concatenate, Atom, array
from biotite.structure.io import load_structure
import numpy as np
from unittest.mock import Mock, patch
import copy
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
        assert (
            'Number of reference embeddings (1) does not match number of residues to include in energy term (2)'
            in str(e.value)
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


def test_HydrogenBondEnergy_instantiation(fake_esmfold: bg.oracles.folding.ESMFold) -> None:
    """Test that HydrogenBondEnergy can be instantiated with default parameters."""
    energy = bg.energies.HydrogenBondEnergy(oracle=fake_esmfold)
    assert energy.name == 'hbond', 'Expected default name for HydrogenBondEnergy'
    assert energy.weight == 1.0, 'Expected default weight for HydrogenBondEnergy'
    assert energy.inheritable is True, 'Expected inheritable to be True for HydrogenBondEnergy'
    assert energy.cutoff_dist == 2.5, 'Expected default cutoff distance for HydrogenBondEnergy'
    assert energy.cutoff_angle == 120, 'Expected default cutoff angle for HydrogenBondEnergy'


def test_HydrogenBondEnergy_with_residues(
    fake_esmfold: bg.oracles.folding.ESMFold,
    residues: list[bg.Residue],
) -> None:
    """Test that HydrogenBondEnergy can be instantiated with specific residues."""
    energy = bg.energies.HydrogenBondEnergy(oracle=fake_esmfold, residues=residues)
    assert len(energy.residue_groups) == 1, 'Expected one group of residues in HydrogenBondEnergy'
    chain_ids, res_ids = energy.residue_groups[0]
    assert np.array_equal(chain_ids, np.array(['A'] * 5 + ['B'])), 'Expected chain IDs to match input residues'
    assert np.array_equal(res_ids, np.array(list(range(5)) + [0])), 'Expected residue IDs to match input residues'


def test_HydrogenBondEnergy_with_custom_parameters(fake_esmfold: bg.oracles.folding.ESMFold) -> None:
    """Test that HydrogenBondEnergy accepts custom cutoff parameters."""
    energy = bg.energies.HydrogenBondEnergy(
        oracle=fake_esmfold,
        cutoff_dist=3.0,
        cutoff_angle=130,
        weight=2.0,
        name='custom',
    )
    assert energy.cutoff_dist == 3.0, 'Expected custom cutoff distance for HydrogenBondEnergy'
    assert energy.cutoff_angle == 130, 'Expected custom cutoff angle for HydrogenBondEnergy'
    assert energy.weight == 2.0, 'Expected custom weight for HydrogenBondEnergy'
    assert energy.name == 'hbond_custom', 'Expected custom name for HydrogenBondEnergy'


def test_HydrogenBondEnergy_no_hbonds_in_structure(
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure: AtomArray,
) -> None:
    """Test HydrogenBondEnergy with a structure that has no hydrogen atoms (thus no hydrogen bonds)."""
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    energy = bg.energies.HydrogenBondEnergy(oracle=fake_esmfold, weight=1.0)
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result)

    # No hydrogen atoms = no hydrogen bonds, so energy should be 0
    assert unweighted_energy == 0.0, 'unweighted energy should be 0 when no hydrogen bonds are present'
    assert weighted_energy == 0.0, 'weighted energy should be 0 when no hydrogen bonds are present'


def test_HydrogenBondEnergy_empty_structure(
    fake_esmfold: bg.oracles.folding.ESMFold,
) -> None:
    """Test HydrogenBondEnergy with an empty structure."""
    from biotite.structure import AtomArray

    empty_structure = AtomArray(0)

    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = empty_structure

    energy = bg.energies.HydrogenBondEnergy(oracle=fake_esmfold, weight=1.0)
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result)

    # Empty structure = 0 residues, 0 hydrogen bonds, avoid division by zero
    assert unweighted_energy == 0.0, 'unweighted energy should be 0 for empty structure'
    assert weighted_energy == 0.0, 'weighted energy should be 0 for empty structure'


def test_HydrogenBondEnergy_weight_applied(
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure: AtomArray,
) -> None:
    """Test that the weight is correctly applied to the energy."""
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    weight = 3.0
    energy = bg.energies.HydrogenBondEnergy(oracle=fake_esmfold, weight=weight)
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result)

    # Weighted energy should be unweighted energy * weight
    assert weighted_energy == unweighted_energy * weight, 'Weighted energy should be unweighted energy * weight'


def test_HydrogenBondEnergy_residue_selection(
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure: AtomArray,
    small_structure_residues: list[bg.Residue],
) -> None:
    """Test that HydrogenBondEnergy respects residue selection."""
    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    energy = bg.energies.HydrogenBondEnergy(
        oracle=fake_esmfold,
        residues=small_structure_residues[:2],
    )
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result)

    # Should compute without error
    assert isinstance(unweighted_energy, (float, np.floating)), 'unweighted energy should be a float'
    assert isinstance(weighted_energy, (float, np.floating)), 'weighted energy should be a float'


@patch('bagel.energies.hbond')
def test_HydrogenBondEnergy_energy_values(
    mock_hbond: Mock,
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure: AtomArray,
) -> None:
    """Test that HydrogenBondEnergy computes correct energy values."""
    # Mock hbond to return 4 hydrogen bonds in the small_structure (3 residues)
    # triplets are (donor_idx, hydrogen_idx, acceptor_idx)
    mock_hbond.return_value = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 0],
        ]
    )

    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    weight = 2.0
    energy = bg.energies.HydrogenBondEnergy(oracle=fake_esmfold, weight=weight)
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result)

    # Expected: small_structure has 3 unique residues
    # hbond_count = 4 (from mock)
    # n_residues = 3
    # value = -4 / 3 ≈ -1.333...
    expected_value = -4.0 / 3.0

    assert np.isfinite(unweighted_energy), 'energy should be finite'
    assert np.isclose(unweighted_energy, expected_value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, expected_value * weight), 'weighted energy is incorrect'


@patch('bagel.energies.hbond')
def test_HydrogenBondEnergy_energy_values_with_residue_selection(
    mock_hbond: Mock,
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure: AtomArray,
    small_structure_residues: list[bg.Residue],
) -> None:
    """Test that HydrogenBondEnergy normalizes by selected residue count."""
    # Mock hbond to return 2 hydrogen bonds
    # Both involve atoms from the selected residues (first 2)
    mock_hbond.return_value = np.array(
        [
            [0, 1, 2],
            [2, 3, 4],
        ]
    )

    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    weight = 1.5
    energy = bg.energies.HydrogenBondEnergy(
        oracle=fake_esmfold,
        residues=small_structure_residues[:2],  # Select first 2 residues
        weight=weight,
    )
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result)

    # Expected: only 2 residues selected
    # Both hbonds involve atoms in those 2 residues: hbond_count = 2
    # n_residues = 2 (selected count, not total)
    # value = -2 / 2 = -1.0
    expected_value = -2.0 / 2.0

    assert np.isfinite(unweighted_energy), 'energy should be finite'
    assert np.isclose(unweighted_energy, expected_value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, expected_value * weight), 'weighted energy is incorrect'


@patch('bagel.energies.hbond')
def test_HydrogenBondEnergy_zero_hbonds(
    mock_hbond: Mock,
    fake_esmfold: bg.oracles.folding.ESMFold,
    small_structure: AtomArray,
) -> None:
    """Test HydrogenBondEnergy returns zero when no hydrogen bonds are found."""
    # Mock hbond to return empty array (no hydrogen bonds)
    mock_hbond.return_value = np.empty((0, 3), dtype=int)

    mock_folding_result = Mock(bg.oracles.folding.ESMFoldResult)
    mock_folding_result.structure = small_structure

    weight = 2.0
    energy = bg.energies.HydrogenBondEnergy(oracle=fake_esmfold, weight=weight)
    oracles_result = OraclesResultDict({fake_esmfold: mock_folding_result})
    unweighted_energy, weighted_energy = energy.compute(oracles_result)

    # Expected: 0 hydrogen bonds
    # hbond_count = 0
    # n_residues = 3
    # value = -0 / 3 = 0.0
    expected_value = 0.0

    assert np.isfinite(unweighted_energy), 'energy should be finite'
    assert np.isclose(unweighted_energy, expected_value), 'unweighted energy is incorrect'
    assert np.isclose(weighted_energy, expected_value * weight), 'weighted energy is incorrect'
