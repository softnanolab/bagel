from unittest.mock import patch, Mock
import bricklane as br


#! Incompatible with current code, unique_chains does not exist
#def test_MutationProtocol_random_residue_method_returns_residue_from_system(mixed_system: br.System) -> None:
#    residue, chain = br.mutation.GrandCanonical().random_mutable_residue(mixed_system)
#    all_chains = mixed_system.unique_chains()
#    assert chain in all_chains, 'Did not return chain from system.'
#    all_residues = sum([chain.residues for chain in all_chains], start=[])
#    assert residue in all_residues, 'Did not return residue from system.'
#    assert residue.mutable, 'Did not return mutable residue.'

#! Incompatible with current code, random_mutable_residue does not exist
#def test_MutationProtocol_random_residue_method_returns_mutable_residue(mixed_system: br.System) -> None:
#    all_chains = mixed_system.unique_chains()
#    residues = sum([chain.residues for chain in all_chains], start=[])
#    for i in range(len(residues) - 1):
#        residues[i].mutable = False
#    only_mutable_residue = residues[-1]
#    residue, _ = br.mutation.GrandCanonical().random_mutable_residue(mixed_system)
#    assert residue == only_mutable_residue


#! Incompatible with current code, sample_different_amino_acid does not exist
#def test_MutationProtocol_sample_differen_amino_acid_method_returns_correct_output() -> None:
#    mutator = br.mutation.GrandCanonical(mutation_bias={'A': 0.999_999, 'B': 0.000_001})
#    assert mutator.sample_different_amino_acid(origional_amino_acid='A') == 'B'


#! Could be compatible if one corrects .one_step to return a tuple System, float, float
#! Also, patch.object should be get_total_energy
#@patch.object(br.System, 'calculate_system_energies')  # prevents unnecessary folding
#def test_Canonical_MutationProtocol_one_step_method_output_is_correct_for_1_mutation(
#    mocked_method: Mock, mixed_system: br.System
#) -> None:
#    original_sequence = ''.join([chain.sequence for chain in mixed_system.unique_chains()])
#    mutated_system = br.mutation.Canonical().one_step(folder=None, system=mixed_system, n_mutations=1)
#    assert mocked_method.called, 'mutator did not recalculate structure and energies.'
#    mutated_sequence = ''.join([chain.sequence for chain in mutated_system.unique_chains()])
#    assert len(original_sequence) == len(mutated_sequence), 'Canonical should not add or remove residues!'
#    changed_residues = [j for i, j in zip(original_sequence, mutated_sequence) if i != j]
#    assert len(changed_residues) == 1, 'More or less than 1 residue was mutated'
#    assert changed_residues[0] in br.constants.aminoacids_letters, 'Did not mutate residue into correct amino acid'


#! Could be compatible if one corrects .one_step to return a tuple System, float, float
#! Also, patch.object should be get_total_energy
#@patch.object(br.System, 'calculate_system_energies')  # prevents unnecessary folding
#def test_Canonical_MutationProtocol_one_step_method_output_is_correct_for_2_mutations(
#    mocked_method: Mock, huge_system: br.System
#) -> None:
#    origional_sequence = ''.join([chain.sequence for chain in huge_system.unique_chains()])
#    mutated_system = br.mutation.Canonical().one_step(folder=None, system=huge_system, n_mutations=2)
#    assert mocked_method.called, 'mutator did not recalculate structure and energies.'
#    mutated_sequence = ''.join([chain.sequence for chain in mutated_system.unique_chains()])
#    assert len(origional_sequence) == len(mutated_sequence), 'Canonical should not add or remove residues!'
#    changed_residues = [j for i, j in zip(origional_sequence, mutated_sequence) if i != j]
#    # currently mutator can mutate same residue multiple times in one_step
#    assert len(changed_residues) == 2, 'incorrect number of residues were mutated'
#    assert all([res in br.constants.aminoacids_letters for res in changed_residues]), (
#        'Did not mutate residues into correct amino acids'
#    )


#! Could be compatible if one corrects .one_step to return a tuple System, float, float
#! Also, patch.object should be get_total_energy
#@patch.object(br.System, 'calculate_system_energies')  # prevents unnecessary folding
#def test_GrandCanonical_MutationProtocol_one_step_method_gives_correct_output_for_mutation_move(
#    mocked_method: Mock,
#    mixed_system: br.System,
#) -> None:
#    mutator = br.mutation.GrandCanonical(move_probabilities={'mutation': 1.0, 'addition': 0.0, 'subtraction': 0.0})
#    original_sequence = ''.join([chain.sequence for chain in mixed_system.unique_chains()])
#    mutated_system = mutator.one_step(folder=None, system=mixed_system)
#    assert mocked_method.called, 'mutator did not recalculate structure and energies.'
#    mutated_sequence = ''.join([chain.sequence for chain in mutated_system.unique_chains()])
#    changed_residues = [j for i, j in zip(original_sequence, mutated_sequence) if i != j]
#    assert len(changed_residues) == 1, 'More or less than 1 residue was mutated'
#    assert changed_residues[0] in br.constants.aminoacids_letters, 'Did not mutate residue into correct amino acid'


#! Could be compatible if one corrects .one_step to return a tuple System, float, float
#! Also, patch.object should be get_total_energy
#@patch.object(br.System, 'calculate_system_energies')  # prevents unnecessary folding
#def test_GrandCanonical_MutationProtocol_one_step_method_gives_correct_output_for_addition_move(
#    mocked_calculate_method: Mock,
#    energies_system: br.System,
#) -> None:
#    mutator = br.mutation.GrandCanonical(move_probabilities={'mutation': 0.0, 'addition': 1.0, 'subtraction': 0.0})
#    mutated_system = mutator.one_step(folder=None, system=energies_system)
#    assert mocked_calculate_method.called, 'mutator did not recalculate structure and energies.'
#    assert any([len(state.chains[0].residues) == 6 for state in mutated_system.states]), 'residue not added'
#        num_res = len(state.chains[0].residues)
#    for state in mutated_system.states:
#        state_type = 'mutated' if num_res == 6 else 'origional'
#        for term in state.energy_terms:
#            num_tracked_res = len(term.residue_groups[0][0])
#            # note energies must not track residues of chains from a different state!
#            assert num_tracked_res == num_res, f'incorrect number of energy terms in {state_type} state'


#! Could be compatible if one corrects .one_step to return a tuple System, float, float
#! Also, patch.object should be get_total_energy
#@patch.object(br.System, 'calculate_system_energies')  # prevents unnecessary folding
#def test_GrandCanonical_MutationProtocol_one_step_method_gives_correct_output_for_removal_move(
#    mocked_calculate_method: Mock,
#    energies_system: br.System,
#) -> None:
#    mutator = br.mutation.GrandCanonical(move_probabilities={'mutation': 0.0, 'addition': 0.0, 'subtraction': 1.0})
#    mutated_system = mutator.one_step(folder=None, system=energies_system)
#    assert mocked_calculate_method.called, 'mutator did not recalculate structure and energies.'
#    assert any([len(state.chains[0].residues) == 4 for state in mutated_system.states]), 'residue not removed'
#    for state in mutated_system.states:
#        num_res = len(state.chains[0].residues)
#        state_type = 'mutated' if num_res == 4 else 'origional'
#        for term in state.energy_terms:
#            num_tracked_res = len(term.residue_groups[0][0])
#            assert num_tracked_res == num_res, f'incorrect number of energy terms in {state_type} state'


@patch.object(br.System, 'get_total_energy')  # prevents unnecessary folding
#@patch.object(br.System, 'calculate_system_energies')  # prevents unnecessary folding
def test_GrandCanonical_MutationProtocol_does_not_remove_all_residues_in_chain(
    mocked_calculate_method: Mock, residues: list[br.Residue]
) -> None:
    chain = br.Chain(residues[:1])
    state = br.State([chain], energy_terms=[br.energies.PTMEnergy()], energy_terms_weights=[1.0], name='A')
    #state = br.State([chain], energy_terms=[br.energies.PTMEnergy()], energy_term_weights=[1.0], name='A')
    single_residue_system = br.System([state])
    mutator = br.mutation.GrandCanonical(move_probabilities={'mutation': 0.0, 'addition': 0.0, 'subtraction': 1.0})
    mutated_system, _, _ = mutator.one_step(folding_algorithm=None, system=single_residue_system, old_system=single_residue_system.__copy__())
    assert len(mutated_system.states[0].chains[0].residues) > 0