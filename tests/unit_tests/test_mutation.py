from unittest.mock import patch, Mock
import numpy as np
import bagel as bg


def _energy_term_topology(
    system: bg.System,
) -> dict[str, dict[str, tuple[tuple[tuple[str, ...], tuple[int, ...]], ...]]]:
    """
    Helper used to compare how energy terms track residues across systems.
    Returns a nested mapping: state -> energy term -> ordered residue groups.
    """
    topology: dict[str, dict[str, tuple[tuple[tuple[str, ...], tuple[int, ...]], ...]]] = {}
    for state in system.states:
        term_mapping: dict[str, tuple[tuple[tuple[str, ...], tuple[int, ...]], ...]] = {}
        for term in state.energy_terms:
            residue_groups = []
            for chain_ids, residue_indices in term.residue_groups:
                residue_groups.append((tuple(chain_ids.tolist()), tuple(residue_indices.tolist())))
            term_mapping[term.name] = tuple(residue_groups)
        topology[state.name] = term_mapping
    return topology


@patch.object(bg.System, 'get_total_energy')  # prevents unnecessary folding
def test_GrandCanonical_MutationProtocol_one_step_method_gives_correct_output_for_addition_move(
    mocked_calculate_method: Mock,
    energies_system: bg.System,
) -> None:
    mutator = bg.mutation.GrandCanonical(move_probabilities={'substitution': 0.0, 'addition': 1.0, 'removal': 0.0})
    mutated_system, mutation_record = mutator.one_step(system=energies_system)
    assert isinstance(mutation_record, bg.mutation.MutationRecord), 'mutator did not return MutationRecord'
    assert len(mutation_record.mutations) > 0, 'mutation record should contain mutations'
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
    mutated_system, mutation_record = mutator.one_step(system=energies_system)
    assert isinstance(mutation_record, bg.mutation.MutationRecord), 'mutator did not return MutationRecord'
    assert len(mutation_record.mutations) > 0, 'mutation record should contain mutations'
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
    mutated_system, mutation_record = mutator.one_step(system=single_residue_system)
    assert len(mutated_system.states[0].chains[0].residues) > 0
    # Check that removal was skipped
    assert any(m.move_type is None for m in mutation_record.mutations), (
        'removal should be skipped when chain.length == 1'
    )


def test_mutate_random_residue_excludes_self_when_flag_true() -> None:
    np.random.seed(42)
    # Create a simple one-residue chain
    residue = bg.Residue(name='A', chain_ID='X', index=0, mutable=True)
    chain = bg.Chain(residues=[residue])

    # Exclude self should never produce identity substitutions
    mutator = bg.mutation.Canonical(exclude_self=True)

    previous = chain.residues[0].name
    for _ in range(200):
        mutator.mutate_random_residue(chain)
        current = chain.residues[0].name
        assert current != previous, 'exclude_self=True produced a self-substitution'
        previous = current


def test_mutate_random_residue_includes_self_at_expected_rate_when_flag_false() -> None:
    np.random.seed(42)
    # Create a simple one-residue chain
    residue = bg.Residue(name='A', chain_ID='X', index=0, mutable=True)
    chain = bg.Chain(residues=[residue])

    # Using default mutation_bias_no_cystein (uniform over 19, zero for C)
    # With exclude_self=False, probability of self-substitution is 1/19 per step
    mutator = bg.mutation.Canonical(exclude_self=False)

    num_steps = 300
    num_self = 0
    previous = chain.residues[0].name
    for _ in range(num_steps):
        mutator.mutate_random_residue(chain)
        current = chain.residues[0].name
        if current == previous:
            num_self += 1
        previous = current

    observed = num_self / num_steps
    expected = 1.0 / 19.0
    # Loose tolerance to avoid flakiness but still meaningful
    assert abs(observed - expected) < 0.05, f'self-substitution rate {observed} deviates from expected {expected}'


@patch.object(bg.System, 'get_total_energy')  # prevents unnecessary folding
def test_grand_canonical_replay_matches_sequences_and_topology(
    mocked_calculate_method: Mock,
    energies_system: bg.System,
) -> None:
    """Run a longer GC trajectory and ensure replay reproduces sequences and energy topology."""
    np.random.seed(1234)
    steps = 25
    mutator = bg.mutation.GrandCanonical(
        n_mutations=1, move_probabilities={'substitution': 0.4, 'addition': 0.3, 'removal': 0.3}
    )

    def capture_state(system: bg.System) -> tuple[list[list[str]], dict[str, dict[str, tuple]]]:
        sequences = [[chain.sequence for chain in state.chains] for state in system.states]
        topology = _energy_term_topology(system)
        return sequences, topology

    current_system = energies_system.__copy__()
    mutation_records: list[bg.mutation.MutationRecord] = []
    sequence_trajectory: list[list[list[str]]] = []
    topology_trajectory: list[dict[str, dict[str, tuple]]] = []

    sequences, topology = capture_state(current_system)
    sequence_trajectory.append(sequences)
    topology_trajectory.append(topology)

    for _ in range(steps):
        mutated_system, mutation_record = mutator.one_step(system=current_system)
        mutation_records.append(mutation_record)
        current_system = mutated_system
        sequences, topology = capture_state(current_system)
        sequence_trajectory.append(sequences)
        topology_trajectory.append(topology)

    assert len(mutation_records) == steps, 'Did not record the expected number of mutation steps'

    replayed_system = energies_system.__copy__()
    replay_sequence_traj: list[list[list[str]]] = []
    replay_topology_traj: list[dict[str, dict[str, tuple]]] = []

    sequences, topology = capture_state(replayed_system)
    replay_sequence_traj.append(sequences)
    replay_topology_traj.append(topology)

    for mutation_record in mutation_records:
        replayed_system = mutator.replay(replayed_system, mutation_record)
        sequences, topology = capture_state(replayed_system)
        replay_sequence_traj.append(sequences)
        replay_topology_traj.append(topology)

    assert sequence_trajectory == replay_sequence_traj, 'Replayed sequences diverged from recorded trajectory'
    assert topology_trajectory == replay_topology_traj, 'Energy term residue groups diverged during replayed trajectory'


@patch.object(bg.System, 'get_total_energy')
def test_GrandCanonical_allows_insertion_and_deletion_at_chain_start_and_end(
    mocked_calculate_method: Mock,
    energies_system: bg.System,
) -> None:
    """Verify GrandCanonical allows insertion and deletion at the start and end of chains."""
    # Test addition at start (position 0)
    system_copy = energies_system.__copy__()
    mutator = bg.mutation.GrandCanonical(move_probabilities={'substitution': 0.0, 'addition': 1.0, 'removal': 0.0})

    with patch('numpy.random.choice') as mock_choice:
        # Call order: choose_chain → move_type → position → amino_acid
        target_chain = system_copy.states[0].chains[0]
        mock_choice.side_effect = [
            target_chain,  # choose_chain
            'addition',  # move_type (deterministic with prob=1.0, but still called)
            0,  # position in add_random_residue
            list(mutator.mutation_bias.keys())[0],  # amino_acid
        ]
        _, mutation_record = mutator.one_step(system=system_copy)
        assert len(mutation_record.mutations) == 1
        assert mutation_record.mutations[0].residue_index == 0
        assert mutation_record.mutations[0].move_type == 'addition'

    # Test addition at end (position chain.length)
    system_copy = energies_system.__copy__()
    original_length = len(system_copy.states[0].chains[0].residues)

    with patch('numpy.random.choice') as mock_choice:
        target_chain = system_copy.states[0].chains[0]
        mock_choice.side_effect = [
            target_chain,
            'addition',
            original_length,  # end position
            list(mutator.mutation_bias.keys())[0],
        ]
        _, mutation_record = mutator.one_step(system=system_copy)
        assert mutation_record.mutations[0].residue_index == original_length
        assert mutation_record.mutations[0].move_type == 'addition'

    # Test removal at start (position 0)
    system_copy = energies_system.__copy__()
    mutator_removal = bg.mutation.GrandCanonical(
        move_probabilities={'substitution': 0.0, 'addition': 0.0, 'removal': 1.0}
    )

    with patch('numpy.random.choice') as mock_choice:
        target_chain = system_copy.states[0].chains[0]
        mock_choice.side_effect = [
            target_chain,  # choose_chain
            'removal',  # move_type
            0,  # mutable_residue_index in remove_random_residue
        ]
        _, mutation_record = mutator_removal.one_step(system=system_copy)
        assert mutation_record.mutations[0].residue_index == 0
        assert mutation_record.mutations[0].move_type == 'removal'

    # Test removal at end (last position)
    system_copy = energies_system.__copy__()
    last_mutable_index = system_copy.states[0].chains[0].mutable_residue_indexes[-1]

    with patch('numpy.random.choice') as mock_choice:
        target_chain = system_copy.states[0].chains[0]
        mock_choice.side_effect = [
            target_chain,
            'removal',
            last_mutable_index,
        ]
        mutated_system, mutation_record = mutator_removal.one_step(system=system_copy)
        assert mutation_record.mutations[0].residue_index == last_mutable_index
        assert mutation_record.mutations[0].move_type == 'removal'
