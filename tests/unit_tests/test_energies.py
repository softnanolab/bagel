import bricklane as bl
from biotite.structure import AtomArray, sasa, annotate_sse, get_residue_count, concatenate
import numpy as np
from unittest.mock import Mock, patch
import copy


def test_residue_list_to_group_function(residues: list[bl.Residue]) -> None:
    residue_group = bl.energies.residue_list_to_group(residues)
    chain_ids, res_ids = residue_group
    assert np.all(chain_ids == np.array(['A'] * 5 + ['B'])), 'function returned wrong chain ids'
    assert np.all(res_ids == np.array(list(range(5)) + [0])), 'function returned wrong res ids'


def test_energies_properly_update_residue_group_after_residue_index_shifted_after_removal(residues: list[bl.Residue]) -> None:
    energy = bl.energies.PLDDTEnergy(residues)
    energy.remove_residue(chain_id='A', res_index=2)
    energy.shift_residues_indices_after_removal(chain_id='A', res_index=2)
    assert all(energy.residue_groups[0][0] == np.array(['A', 'A', 'A', 'A', 'B'])), 'incorrect chain_IDs'
    assert all(energy.residue_groups[0][1] == np.array([0, 1, 2, 3, 0])), 'incorrect res_indices'


def test_energies_properly_update_residue_group_before_residue_index_shifted_for_addition(residues: list[bl.Residue]) -> None:
    energy = bl.energies.PLDDTEnergy(residues)
    energy.shift_residues_indices_before_addition(chain_id='A', res_index=1)
    assert all(energy.residue_groups[0][0] == np.array(['A', 'A', 'A', 'A', 'A', 'B'])), 'incorrect chain_IDs'
    assert all(energy.residue_groups[0][1] == np.array([0, 2, 3, 4, 5, 0])), 'incorrect res_indices'


def test_energies_properly_update_residue_group_after_remove_residue(residues: list[bl.Residue]) -> None:
    energy = bl.energies.PLDDTEnergy(residues)
    energy.remove_residue(chain_id='A', res_index=2)
    assert all(energy.residue_groups[0][0] == np.array(['A', 'A', 'A', 'A', 'B'])), 'incorrect chain_IDs'
    assert all(energy.residue_groups[0][1] == np.array([0, 1, 3, 4, 0])), 'incorrect res_indices'


def test_energies_properly_update_residue_group_before_add_residue(residues: list[bl.Residue]) -> None:
    energy = bl.energies.PLDDTEnergy(residues)
    energy.add_residue(chain_id='A', res_index=2, parent_res_index=1)
    assert all(energy.residue_groups[0][0] == np.array(['A', 'A', 'A', 'A', 'A', 'B', 'A'])), 'incorrect chain_IDs'
    assert all(energy.residue_groups[0][1] == np.array([0, 1, 2, 3, 4, 0, 2])), 'incorrect  res_indices'


def test_energies_get_correct_residue_mask(small_structure: AtomArray) -> None:
    energy = bl.energies.PLDDTEnergy(residues=[bl.Residue(name='V', chain_ID='A', index=1)])
    mask = energy.get_residue_mask(structure=small_structure, residue_group_index=0)
    assert all(mask == np.array([False, True, False]))


def test_energies_get_correct_residue_mask_for_multimer(square_structure: AtomArray, line_structure: AtomArray) -> None:
    residues = [
        bl.Residue(name='V', chain_ID='E', index=2),
        bl.Residue(name='V', chain_ID='D', index=0),
        bl.Residue(name='V', chain_ID='E', index=3),
        bl.Residue(name='V', chain_ID='D', index=1),
        bl.Residue(name='V', chain_ID='E', index=1),
    ]
    energy = bl.energies.PLDDTEnergy(residues)
    structure = concatenate([square_structure, line_structure])
    mask = energy.get_residue_mask(structure, residue_group_index=0)
    assert all(mask == [False, True, True, True, False, True, True]), AssertionError( f'Incorrect residue mask for multimer {mask}')


def test_energies_get_correct_atom_mask(small_structure: AtomArray) -> None:
    energy = bl.energies.PLDDTEnergy(residues=[bl.Residue(name='V', chain_ID='A', index=0)])
    mask = energy.get_atom_mask(structure=small_structure, residue_group_index=0)
    assert all(mask == np.array([True, True, False, False, False]))


def test_PTMEnergy() -> None:
    mock_folding_metrics = Mock(bl.folding.FoldingMetrics)
    mock_folding_metrics.ptm = 0.7
    energy = bl.energies.PTMEnergy().compute(structure=None, folding_metrics=mock_folding_metrics)
    assert np.isclose(energy, -0.7)


def test_PLDDTEnergy(small_structure_residues: list[bl.Residue], small_structure: AtomArray) -> None:
    mock_folding_metrics = Mock(bl.folding.FoldingMetrics)
    mock_folding_metrics.local_plddt = np.array([0.2, 0.4, 0.6]).reshape(1, 3)
    energy = bl.energies.PLDDTEnergy(residues=small_structure_residues[:2])
    energy.compute(structure=small_structure, folding_metrics=mock_folding_metrics)
    assert np.isclose(energy.value, -0.3)  # avoids float rounding errors


def test_OverallPLDDTEnergy() -> None:
    mock_folding_metrics = Mock(bl.folding.FoldingMetrics)
    mock_folding_metrics.local_plddt = np.array([0.2, 0.4, 0.6]).reshape(1, 3)
    energy = bl.energies.OverallPLDDTEnergy()
    energy.compute(structure=None, folding_metrics=mock_folding_metrics)
    assert np.isclose(energy.value, -0.4)  # avoids float rounding errors


def test_solvent_accessible_surface_area_function_gives_expected_return_array(small_structure: AtomArray) -> None:
    sasa_result = sasa(small_structure)
    assert len(sasa_result) == len(small_structure), 'sasa does not return one number for each atom'
    assert np.issubdtype(sasa_result.dtype, np.floating), 'sasa does not return floats for each atom'


@patch('bricklane.energies.sasa')
def test_SurfaceAreaEnergy(
    mock_sasa: Mock, small_structure_residues: list[bl.Residue], small_structure: AtomArray
) -> None:
    mock_sasa.return_value = np.arange(5, dtype=float)
    energy = bl.energies.SurfaceAreaEnergy(residues=small_structure_residues[:1])
    # returns mean of normalized sasa over given residues
    energy.compute(structure=small_structure, folding_metrics=None)
    assert np.isclose(energy.value, 1 / (22 * 2))  # max sasa is 22, and there are 2 atoms in the first residue


@patch('bricklane.energies.sasa')
def test_HydrophobicEnergy(
    mock_sasa: Mock, small_structure_residues: list[bl.Residue], small_structure: AtomArray
) -> None:
    mock_sasa.return_value = np.array([22, 22, 22, 22, 0])  # atoms of first 2 residues are given max sasa
    energy = bl.energies.HydrophobicEnergy(residues=small_structure_residues[:2], surface_only=True)
    # returns sum of normalized sasa for hydrophobic atoms, divided by the number of atoms in given residues
    energy.compute(structure=small_structure, folding_metrics=None)
    assert np.isclose(energy.value, 2 / 4)  # 4 atoms in given residues, only 2 are part of hydrophobic residue


def test_PAEEnergy_with_cross_term_only(mixed_structure_state: bl.State) -> None:
    mock_folding_metrics = Mock(bl.folding.FoldingMetrics)
    mock_folding_metrics.pae = np.arange(7**2).reshape((1, 7, 7))
    residues = sum([chain.residues for chain in mixed_structure_state.chains], start=[])
    energy = bl.energies.PAEEnergy(group_1_residues=residues[1:6:2], group_2_residues=residues[2:7:2])
    energy.compute(structure=mixed_structure_state._structure, folding_metrics=mock_folding_metrics)
    relevant_PAEs = [9, 11, 13, 15, 17, 19, 23, 25, 27, 29, 31, 33, 37, 39, 41, 43, 45, 47]
    assert np.allclose(energy.value, np.mean(relevant_PAEs) / 30)  # sum of relevant PAEs / (num PAEs * max PAE)


def test_PAEEnergy_without_cross_term_only(mixed_structure_state: bl.State) -> None:
    mock_folding_metrics = Mock(bl.folding.FoldingMetrics)
    mock_folding_metrics.pae = np.arange(7**2).reshape((1, 7, 7))
    residues = sum([chain.residues for chain in mixed_structure_state.chains], start=[])
    energy = bl.energies.PAEEnergy(
        group_1_residues=residues[1:6:4],
        group_2_residues=residues[2:7:4],
        cross_term_only=False,
    )
    energy.compute(structure=mixed_structure_state._structure, folding_metrics=mock_folding_metrics)
    relevant_PAEs = [9, 12, 13, 15, 19, 20, 36, 37, 41, 43, 44, 47]
    assert np.allclose(energy.value, np.mean(relevant_PAEs) / 30)  # sum of relevant PAEs / (num PAEs * max PAE)


def test_PAEEnergy_of_residues_with_itself(mixed_structure_state: bl.State) -> None:
    mock_folding_metrics = Mock(bl.folding.FoldingMetrics)
    mock_folding_metrics.pae = np.arange(7**2).reshape((1, 7, 7))
    residues = sum([chain.residues for chain in mixed_structure_state.chains], start=[])
    energy = bl.energies.PAEEnergy(group_1_residues=residues[1:6:2])
    energy.compute(structure=mixed_structure_state._structure, folding_metrics=mock_folding_metrics)
    relevant_PAEs = [8, 10, 12, 22, 24, 26, 36, 38, 40]
    assert np.allclose(energy.value, np.mean(relevant_PAEs) / 30)  # sum of relevant PAEs / (num PAEs * max PAE)


def test_RingSymmetryEnergy(square_structure_residues: list[bl.Residue], square_structure: AtomArray) -> None:
    energy = bl.energies.RingSymmetryEnergy(symmetry_groups=[[residue] for residue in square_structure_residues])
    energy.compute(structure=square_structure, folding_metrics=None)
    # centroids of each residue backbone make a 2d square of length 1
    assert np.isclose(energy.value, np.std([1, 1, 2**0.5] * 4))  # Neighbour distances for each atom are 1, 1, and √2


def test_RingSymmetryEnergy_with_direct_neighbours_only(
    square_structure_residues: list[bl.Residue], square_structure: AtomArray
) -> None:
    energy = bl.energies.RingSymmetryEnergy(
        symmetry_groups=[[residue] for residue in square_structure_residues],
        direct_neighbours_only=True,
    )
    energy.compute(structure=square_structure, folding_metrics=None)
    # centroids of each residue make a 2d square of length 1. The direct neighbour distance for each atom is 1
    assert np.isclose(energy.value, 0)


def test_SeparationEnergy(square_structure_residues: list[bl.Residue], square_structure: AtomArray) -> None:
    energy = bl.energies.SeparationEnergy(
        group_1_residues=square_structure_residues[:2],
        group_2_residues=square_structure_residues[2:],
    )
    energy.compute(structure=square_structure, folding_metrics=None)
    # distance between the centroids of the bottom corners and top corners for a square of length 1 is 1
    assert np.isclose(energy.value, 1 / 8)  # distance / total number of atoms (2 at each corner)


def test_GlobularEnergy(square_structure_residues: list[bl.Residue], square_structure: AtomArray) -> None:
    energy = bl.energies.GlobularEnergy(residues=square_structure_residues[:2])
    # the centroid of the first 2 residue backbones are at [0, 0.5, 0] coords. The 4 atoms form a square of length
    # 1 around the centroid, equidistance from it.
    energy.compute(structure=square_structure, folding_metrics=None)
    assert np.isclose(energy.value, 0)


def test_TemplateMatchEnergy_gives_zero_distance_for_rotated_and_shifted_structure(
    square_structure_residues: list[bl.Residue], square_structure: AtomArray
) -> None:
    # Comparing one of the diagonals of the square to the other
    template_atoms = copy.deepcopy(square_structure[np.isin(square_structure.res_id, [0, 2])])
    template_atoms.coord[:, 1:] += 3.0  # shifting strucutre in y and z
    energy = bl.energies.TemplateMatchEnergy(
        template_atoms, residues=square_structure_residues[1::2], backbone_only=True
    )
    energy.compute(structure=square_structure, folding_metrics=None)
    assert np.isclose(energy.value, 0, atol=1e-7)  # superimposing template onto structure adds some error


def test_TemplateMatchEnergy_gives_zero_distance_for_rotated_and_shifted_structure_using_distogram_metric(
    square_structure_residues: list[bl.Residue], square_structure: AtomArray
) -> None:
    # Comparing one of the diagonals of the square to the other
    template_atoms = copy.deepcopy(square_structure[np.isin(square_structure.res_id, [0, 2])])
    template_atoms.coord[:, 1:] += 3.0  # shifting strucutre in y and z
    energy = bl.energies.TemplateMatchEnergy(
        template_atoms, residues=square_structure_residues[1::2], backbone_only=True, distogram_separation=True
    )
    energy.compute(structure=square_structure, folding_metrics=None)
    assert np.isclose(energy.value, 0, atol=1e-7)  # superimposing template onto structure adds some error


def test_TemplateMatchEnergy_is_correct_with_simple_structure(
    line_structure_residues: list[bl.Residue],
    line_structure: AtomArray,
) -> None:
    template_atoms = copy.deepcopy(line_structure[line_structure.res_id < 1])
    template_atoms.coord[0, :] -= [0.1, 0.1, 0]  # shifting back atom sqrt(0.02) backwards in direction of line
    template_atoms.coord[4, :] += [0.1, 0.1, 0]  # shifting front atom sqrt(0.02) forwards in direction of line
    energy = bl.energies.TemplateMatchEnergy(template_atoms, residues=line_structure_residues[:2], backbone_only=True)
    energy.compute(structure=line_structure, folding_metrics=None)
    # it is only 3 atoms because you only count for backbone atoms, defined as of type C, N and CA
    assert np.isclose(energy.value, np.mean( [0.02, 0.0, 0.02] )**0.5 )  # first and last template atoms sqrt(0.02) away


def test_TemplateMatchEnergy_is_correct_with_simple_structure_using_distogram_metric(
    line_structure_residues: list[bl.Residue],
    line_structure: AtomArray,
) -> None:
    template_atoms = copy.deepcopy(line_structure[line_structure.res_id < 1])
    template_atoms.coord[0, :] -= [0.1, 0.1, 0]  # shifting back atom sqrt(0.02) backwards in direction of line
    template_atoms.coord[4, :] += [0.1, 0.1, 0]  # shifting front atom sqrt(0.02) forwards in direction of line
    energy = bl.energies.TemplateMatchEnergy(
        template_atoms, residues=line_structure_residues[:2], backbone_only=True, distogram_separation=True
    )
    energy.compute(structure=line_structure, folding_metrics=None)
    unique_distogram_distances_squared = [0.02, 0.08, 0.02] * 2  # requires a small sketch to make sense of
    assert np.isclose(energy.value, np.mean(unique_distogram_distances_squared) ** 0.5)


def test_secondary_structure_elements_function_gives_expected_return_array(small_structure: AtomArray) -> None:
    sse_labels = annotate_sse(small_structure)
    assert len(sse_labels) == get_residue_count(small_structure), 'sse does not return one number for each residue'
    assert np.issubdtype(sse_labels.dtype, np.str_), 'sse does not return strings for each atom'


@patch('bricklane.energies.annotate_sse')
def test_SecondaryStructureEnergy(
    mock_annotate_sse: Mock, small_structure_residues: list[bl.Residue], small_structure: AtomArray
) -> None:
    mock_annotate_sse.return_value = np.array(['a', '', 'c'])
    energy = bl.energies.SecondaryStructureEnergy(residues=small_structure_residues, target_secondary_structure='coil')
    energy.compute(small_structure, folding_metrics=None)
    assert np.isclose(energy.value, 2 / 3)
