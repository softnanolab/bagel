import bagel as bg
from unittest.mock import Mock
import numpy as np


def test_state_calculate_internal_structure_and_energies_method_outputs_correct_value(simple_state: bg.State) -> None:
    mock_folding_metrics = Mock(bg.folding.FoldingMetrics)
    mock_folding_metrics.ptm = 3.0
    mock_folding_metrics.local_plddt = [[2.0, 2.0]]
    mock_folder = Mock(bg.folding.ESMFolder)
    mock_folder.fold = Mock(return_value=[1.0, mock_folding_metrics])

    simple_state.get_energy(mock_folder)
    # energy term energies = -3.0 and -2.0. energy term weights = 1.0 and 2.0
    assert np.isclose(simple_state._energy, (-1 * 1 + -0.5 * 1.0)), AssertionError(
        f'simple_state.energy = {simple_state.energy} != -1.5'
    )


def test_state_chemical_potential_energy_outputs_correct_value(mixed_structure_state: bg.State) -> None:
    # this state has chain lengths of [4, 1, 2] and a  a chemical potential of 2.0
    mixed_structure_state.chemical_potential = 2.0
    assert np.isclose(mixed_structure_state.get_chemical_potential_contribution(), 14.0)


def test_state_remove_residue_from_all_energy_terms_removes_correct_residue(mixed_structure_state: bg.State) -> None:
    mixed_structure_state.chains[2].remove_residue(index=2)
    mixed_structure_state.remove_residue_from_all_energy_terms(chain_ID='E', residue_index=2)

    chain_ids, res_ids = mixed_structure_state.energy_terms[0].residue_groups[0]
    # started at ['C', 'D', 'D', 'E', 'E', 'E', 'E'], [0, 0, 1, 0, 1, 2, 3], remove index 2 at chain E
    # removing index 2 E residue means the rest shift down and the max index tracked should be 2
    assert np.all(chain_ids == ['C', 'D', 'D', 'E', 'E', 'E']) and np.all(res_ids == [0, 0, 1, 0, 1, 2])

    chain_ids, res_ids = mixed_structure_state.energy_terms[1].residue_groups[0]
    # started at ['C', 'D', 'D'], [0, 0, 1], should remain the same
    assert np.all(chain_ids == ['C', 'D', 'D']) and np.all(res_ids == [0, 0, 1])

    chain_ids, res_ids = mixed_structure_state.energy_terms[1].residue_groups[1]
    # started at ['E', 'E', 'E', 'E'], [0, 1, 2, 3], remove index 2 at chain E
    # removing index 2 E residue means the rest shift down and the max index tracked should be 2
    assert np.all(chain_ids == ['E', 'E', 'E']) and np.all(res_ids == [0, 1, 2])


def test_state_add_residue_to_all_energy_terms_adds_residue_to_residue_group(mixed_structure_state: bg.State) -> None:
    mixed_structure_state.chains[1].add_residue(amino_acid='A', index=1)
    mixed_structure_state.add_residue_to_all_energy_terms(chain_ID='D', residue_index=1)

    # started at ['C', 'D', 'D', 'E', 'E', 'E', 'E'], [0, 0, 1, 0, 1, 2, 3]
    chain_ids, res_ids = mixed_structure_state.energy_terms[0].residue_groups[0]  # inheritable energy
    assert np.all(chain_ids == ['C', 'D', 'D', 'E', 'E', 'E', 'E', 'D'])  # extra D added at end
    assert np.all(
        res_ids == [0, 0, 2, 0, 1, 2, 3, 1]
    )  # index of original residue shifted up and added new residue at end

    # started at ['C', 'D', 'D'], [0, 0, 1] and shift of last index in chain D, but no residue added since non inheritable
    chain_ids, res_ids = mixed_structure_state.energy_terms[1].residue_groups[0]  # non inheritable energy
    assert np.all(chain_ids == ['C', 'D', 'D'])  # extra D not added
    assert np.all(res_ids == [0, 0, 2])  # index of original residue shifted up

    # started at ['E', 'E', 'E', 'E'], [0, 1, 2, 3] and remains the same because chain D is not part of this energy term
    chain_ids, res_ids = mixed_structure_state.energy_terms[1].residue_groups[1]  # non inheritable energy
    assert np.all(chain_ids == ['E', 'E', 'E', 'E'])  # extra D not added
    assert np.all(res_ids == [0, 1, 2, 3])  # index of original residue shifted up
