from unittest.mock import patch, Mock
import bagel as bg


@patch.object(bg.System, 'get_total_energy')  # prevents unnecessary folding
def test_GrandCanonical_MutationProtocol_one_step_method_gives_correct_output_for_addition_move(
    mocked_calculate_method: Mock,
    energies_system: bg.System,
) -> None:
    mutator = bg.mutation.GrandCanonical(move_probabilities={'substitution': 0.0, 'addition': 1.0, 'removal': 0.0})
    mutated_system, _ = mutator.one_step(system=energies_system, old_system=energies_system.__copy__())
    assert mocked_calculate_method.called, 'mutator did not recalculate structure and energies.'
    #! The following check is ACTUALLY correct, because the residue MUST BE ADDED AT LEAST IN ONE STATE (as checked by "any")
    assert any([len(state.chains[0].residues) == 6 for state in mutated_system.states]), AssertionError(
        f'residue not added {[len(state.chains[0].residues) == 6 for state in mutated_system.states]}'
    )
    for state in mutated_system.states:
        num_res = len(state.chains[0].residues)
        state_type = 'mutated' if num_res == 6 else 'original'
        for term in state.energy_terms:
            num_tracked_res = len(term.residue_groups[0][0])
            # note energies must not track residues of chains from a different state!
            assert num_tracked_res == num_res, f'incorrect number of energy terms in {state_type} state'


@patch.object(bg.System, 'get_total_energy')  # prevents unnecessary folding
def test_GrandCanonical_MutationProtocol_one_step_method_gives_correct_output_for_removal_move(
    mocked_calculate_method: Mock,
    energies_system: bg.System,
) -> None:
    mutator = bg.mutation.GrandCanonical(move_probabilities={'substitution': 0.0, 'addition': 0.0, 'removal': 1.0})
    mutated_system, _ = mutator.one_step(system=energies_system, old_system=energies_system.__copy__())
    assert mocked_calculate_method.called, 'mutator did not recalculate structure and energies.'
    assert any([len(state.chains[0].residues) == 4 for state in mutated_system.states]), AssertionError(
        f'residue not removed {mutated_system.states[0].chains[0].residues}'
    )
    for state in mutated_system.states:
        num_res = len(state.chains[0].residues)
        state_type = 'mutated' if num_res == 4 else 'original'
        for term in state.energy_terms:
            num_tracked_res = len(term.residue_groups[0][0])
            assert num_tracked_res == num_res, f'incorrect number of energy terms in {state_type} state'


@patch.object(bg.System, 'get_total_energy')  # prevents unnecessary folding
def test_GrandCanonical_MutationProtocol_does_not_remove_all_residues_in_chain(
    mocked_calculate_method: Mock, fake_esmfold: bg.oracles.folding.ESMFold, residues: list[bg.Residue]
) -> None:
    chain = bg.Chain(residues[:1])
    state = bg.State(
        name='A',
        chains=[chain],
        energy_terms=[bg.energies.PTMEnergy(oracle=fake_esmfold, weight=1.0)],
    )
    single_residue_system = bg.System(states=[state])
    mutator = bg.mutation.GrandCanonical(move_probabilities={'substitution': 0.0, 'addition': 0.0, 'removal': 1.0})
    mutated_system, _ = mutator.one_step(system=single_residue_system, old_system=single_residue_system.__copy__())
    assert len(mutated_system.states[0].chains[0].residues) > 0


def test_mutation_protocol_resets_system_total_energy(
    real_simple_state: bg.State,
) -> None:
    system = bg.System(states=[real_simple_state, real_simple_state])
    system.get_total_energy()
    assert system.total_energy is not None, 'system total energy is None'
    assert system.total_energy + 1.4 < 0.5, 'system total energy is not correct'

    for state in system.states:
        assert len(state._energy_terms_value) == 2, 'system energy terms value is not correct'
        assert state._oracles_result is not None, 'system oracles result is None'
        assert len(state._oracles_result) == 1, 'system oracles result is not correct'

    mutator = bg.mutation.Canonical()
    mutator.reset_system(system)
    assert system.total_energy is None, 'system total energy not reset'

    for state in system.states:
        assert len(state._energy_terms_value) == 0, 'system energy terms value is not correct'
        assert state._oracles_result is not None, 'system oracles result is None'
        assert len(state._oracles_result) == 0, 'system oracles result is not correct'
