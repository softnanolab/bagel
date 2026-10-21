from unittest.mock import patch, Mock
import pytest
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


class _StubInverseFoldingOracle:
    """Records calls and returns a fixed distribution over the 20 amino acids."""

    def __init__(self, probabilities: dict[str, float], generated: dict[int, str] | None = None) -> None:
        self.probabilities = probabilities
        self.generated = generated or {}  # residue index -> residue inverse_generate should return
        self.calls: list[dict] = []

    def inverse_fold(self, chains, structure, chain_id, residue_index):
        # deliberately the pre-temperature signature: tempering is the mutator's job, not the oracle's
        self.calls.append(
            {'sequences': [c.sequence for c in chains], 'chain_id': chain_id, 'residue_index': residue_index}
        )
        return dict(self.probabilities)

    def inverse_generate(
        self,
        chains,
        structure,
        chain_id,
        residue_indices,
        temperature=0.1,
        excluded_amino_acids='',
    ):
        # Joint decoding: the model samples, so the mutator does no tempering or exclusion of its own.
        self.calls.append(
            {
                'sequences': [c.sequence for c in chains],
                'chain_id': chain_id,
                'residue_indices': list(residue_indices),
                'temperature': temperature,
                'excluded_amino_acids': excluded_amino_acids,
            }
        )
        return {index: self.generated.get(index, 'W') for index in residue_indices}


_AAS = 'ACDEFGHIKLMNPQRSTVWY'


def test_InverseFoldingSampling_rejects_unsupported_method() -> None:
    import pytest

    with pytest.raises(ValueError, match='Unsupported inverse folding method'):
        bg.mutation.InverseFoldingSampling(inverse_folding_method='ProteinMPNN')
    assert bg.mutation.InverseFoldingSampling().inverse_folding_method == 'ESM3'


def test_InverseFoldingSampling_propose_amino_acid_never_returns_current_and_follows_distribution() -> None:
    mutator = bg.mutation.InverseFoldingSampling(oracle=object(), mutation_bias={aa: 1.0 for aa in _AAS})
    # 'A' (current) has most of the mass; of the rest, 'C':'D' = 1:3 once renormalised
    probabilities = {aa: 0.0 for aa in _AAS} | {'A': 0.6, 'C': 0.1, 'D': 0.3}
    np.random.seed(0)
    draws = [mutator.propose_amino_acid(probabilities, 'A') for _ in range(4000)]
    assert set(draws) == {'C', 'D'}
    assert abs(draws.count('D') / len(draws) - 0.75) < 0.03


def test_InverseFoldingSampling_propose_amino_acid_raises_when_no_alternative() -> None:
    import pytest

    mutator = bg.mutation.InverseFoldingSampling(oracle=object(), mutation_bias={aa: 1.0 for aa in _AAS})
    probabilities = {aa: 0.0 for aa in _AAS} | {'A': 1.0}
    with pytest.raises(ValueError, match='No valid mutation targets'):
        mutator.propose_amino_acid(probabilities, 'A')


@patch.object(bg.System, 'get_total_energy')
def test_InverseFoldingSampling_one_step_mutates_single_residue_and_leaves_original_untouched(
    mocked_energy: Mock,
    energies_system: bg.System,
) -> None:
    oracle = _StubInverseFoldingOracle({aa: 1 / 20 for aa in _AAS})
    mutator = bg.mutation.InverseFoldingSampling(oracle=oracle)
    original_sequences = [state.chains[0].sequence for state in energies_system.states]

    mutated_system, record = mutator.one_step(energies_system)

    assert [state.chains[0].sequence for state in energies_system.states] == original_sequences
    assert len(record.mutations) == 1
    mutation = record.mutations[0]
    assert mutation.move_type == 'substitution'
    assert mutation.new_amino_acid != mutation.old_amino_acid
    # exactly one residue differs in the mutated chain, at the recorded index
    chain = next(c for s in mutated_system.states for c in s.chains if c.chain_ID == mutation.chain_id)
    original = next(
        seq
        for state, seq in zip(energies_system.states, original_sequences)
        if state.chains[0].chain_ID == mutation.chain_id
    )
    diffs = [i for i, (a, b) in enumerate(zip(original, chain.sequence)) if a != b]
    assert diffs == [mutation.residue_index]
    assert chain.sequence[mutation.residue_index] == mutation.new_amino_acid
    # the oracle was asked about the (single) masked residue, with the unmutated sequence
    assert len(oracle.calls) == 1
    assert oracle.calls[0]['residue_index'] == mutation.residue_index
    assert oracle.calls[0]['sequences'] == [original]


@patch.object(bg.System, 'get_total_energy')
def test_InverseFoldingSampling_one_step_samples_from_oracle_distribution(
    mocked_energy: Mock,
    energies_system: bg.System,
) -> None:
    # all the mass on W, so any (non-W) residue must become W; a residue that is already W gets nothing to move to
    oracle = _StubInverseFoldingOracle({aa: float(aa == 'W') for aa in _AAS} | {'Y': 1e-12})
    mutator = bg.mutation.InverseFoldingSampling(oracle=oracle)
    for state in energies_system.states:
        for residue in state.chains[0].residues:
            residue.name = 'A'
    _, record = mutator.one_step(energies_system)
    assert record.mutations[0].new_amino_acid == 'W'


@patch.object(bg.System, 'get_total_energy')
def test_InverseFoldingSampling_computes_one_structure_per_step(
    mocked_energy: Mock,
    energies_system: bg.System,
    monkeypatch,
) -> None:
    """A step folds once, whatever n_mutations is: the whole proposal is conditioned on one structure."""
    # the stub proposes 'W' everywhere, so start from 'A' or a residue that is already 'W' is not a change
    for state in energies_system.states:
        for residue in state.chains[0].residues:
            residue.name = 'A'
    oracle = _StubInverseFoldingOracle({aa: 1 / 20 for aa in _AAS})
    mutator = bg.mutation.InverseFoldingSampling(oracle=oracle, n_mutations=3)
    structure_calls: list[object] = []
    original = mutator._get_structure

    monkeypatch.setattr(mutator, '_get_structure', lambda state: (structure_calls.append(state), original(state))[1])

    _, record = mutator.one_step(energies_system)

    assert len(structure_calls) == 1
    assert len(oracle.calls) == 1
    assert len(record.mutations) == 3


def test_InverseFoldingSampling_state_without_folding_oracle_raises() -> None:
    import pytest

    chain = bg.Chain([bg.Residue(name='A', chain_ID='A', index=i) for i in range(3)])
    system = bg.System([bg.State(name='s', chains=[chain], energy_terms=[])])
    mutator = bg.mutation.InverseFoldingSampling(oracle=_StubInverseFoldingOracle({aa: 0.05 for aa in _AAS}))
    with pytest.raises(ValueError, match='no folding oracle'):
        mutator.one_step(system)


def test_InverseFoldingSampling_zero_mutation_bias_residues_are_excluded_and_rest_renormalised() -> None:
    bias = {aa: 1.0 for aa in _AAS} | {'C': 0.0, 'D': 0.0}
    mutator = bg.mutation.InverseFoldingSampling(oracle=object(), mutation_bias=bias)
    # C and D carry most of the ESM3 mass but are excluded; A is current; remaining E:F = 1:3
    probabilities = {aa: 0.0 for aa in _AAS} | {'A': 0.2, 'C': 0.3, 'D': 0.3, 'E': 0.05, 'F': 0.15}
    np.random.seed(0)
    draws = [mutator.propose_amino_acid(probabilities, 'A') for _ in range(4000)]
    assert set(draws) == {'E', 'F'}
    assert abs(draws.count('F') / len(draws) - 0.75) < 0.03


def test_InverseFoldingSampling_default_bias_never_proposes_cysteine() -> None:
    mutator = bg.mutation.InverseFoldingSampling(oracle=object())
    probabilities = {aa: 0.0 for aa in _AAS} | {'C': 0.99, 'W': 0.01}
    assert {mutator.propose_amino_acid(probabilities, 'A') for _ in range(50)} == {'W'}


def test_InverseFoldingSampling_structures_come_from_unmutated_states_when_chain_is_shared(
    fake_esmfold: bg.oracles.folding.ESMFold, monkeypatch
) -> None:
    folded_sequences: list[list[str]] = []
    original_fold = fake_esmfold.fold

    def recording_fold(chains):
        folded_sequences.append([c.sequence for c in chains])
        return original_fold(chains)

    monkeypatch.setattr(fake_esmfold, 'fold', recording_fold)

    residues = [bg.Residue(name='A', chain_ID='A', index=i, mutable=True) for i in range(5)]
    shared_chain = bg.Chain(residues)
    states = [
        bg.State(
            name=name,
            chains=[shared_chain],
            energy_terms=[bg.energies.PLDDTEnergy(oracle=fake_esmfold, residues=residues, weight=1.0)],
        )
        for name in ('S1', 'S2')
    ]
    system = bg.System(states)
    oracle = _StubInverseFoldingOracle({aa: 1 / 20 for aa in _AAS})
    mutator = bg.mutation.InverseFoldingSampling(oracle=oracle, n_mutations=2)

    monkeypatch.setattr(np.random, 'randint', lambda n: 0)  # pick state S1
    _, record = mutator.one_step(system)

    assert len(record.mutations) == 2
    # the structure is taken from the unmutated input system, so the fold saw the original sequence
    assert folded_sequences == [['AAAAA']]
    assert oracle.calls[0]['sequences'] == ['AAAAA']


def test_InverseFoldingSampling_t_sampling_proposals_defaults_to_0_1() -> None:
    """ESM3 advises against sampling above 0.1, so that is the default; it is not a cap."""
    assert bg.mutation.InverseFoldingSampling.DEFAULT_T_SAMPLING_PROPOSALS == 0.1
    mutator = bg.mutation.InverseFoldingSampling(oracle=object())
    assert mutator.proposal_temperature(0) == pytest.approx(0.1)
    # A higher value is the caller's choice and is passed through untouched.
    hot = bg.mutation.InverseFoldingSampling(oracle=object(), t_sampling_proposals=0.7)
    assert hot.proposal_temperature(0) == pytest.approx(0.7)


def test_InverseFoldingSampling_t_sampling_proposals_accepts_a_schedule() -> None:
    """A sequence is read per step and a callable is handed the step, so a scheduler can drive it."""
    scheduled = bg.mutation.InverseFoldingSampling(oracle=object(), t_sampling_proposals=[0.5, 0.3, 0.1])
    assert [scheduled.proposal_temperature(step) for step in range(3)] == pytest.approx([0.5, 0.3, 0.1])
    with pytest.raises(ValueError, match='too few for step 3'):
        scheduled.proposal_temperature(3)

    callable_schedule = bg.mutation.InverseFoldingSampling(
        oracle=object(), t_sampling_proposals=lambda step: 1.0 / (step + 1)
    )
    assert [callable_schedule.proposal_temperature(step) for step in range(3)] == pytest.approx([1.0, 0.5, 1 / 3])


@pytest.mark.parametrize('temperature', [0.0, -1.0, float('nan'), float('inf'), -float('inf')])
def test_InverseFoldingSampling_t_sampling_proposals_rejects_non_positive(temperature) -> None:
    """NaN and infinity survive a bare `<= 0` check, so they are rejected explicitly."""
    mutator = bg.mutation.InverseFoldingSampling(oracle=object(), t_sampling_proposals=temperature)
    with pytest.raises(ValueError, match='finite positive temperature'):
        mutator.proposal_temperature(0)


@patch.object(bg.System, 'get_total_energy')
def test_InverseFoldingSampling_one_step_ignores_the_monte_carlo_temperature(
    mocked_energy: Mock, energies_system: bg.System, monkeypatch
) -> None:
    """The proposal temperature is the protocol's own, not the one the minimizer accepts or rejects with."""
    oracle = _StubInverseFoldingOracle({aa: 1 / 20 for aa in _AAS})
    mutator = bg.mutation.InverseFoldingSampling(oracle=oracle, t_sampling_proposals=0.4)
    used: list[float] = []
    original = mutator.propose_amino_acid

    def recording(probabilities, current_aa, temperature=1.0):
        used.append(temperature)
        return original(probabilities, current_aa, temperature=temperature)

    monkeypatch.setattr(mutator, 'propose_amino_acid', recording)

    mutator.one_step(energies_system, temperature=2.0)  # Monte Carlo temperature, deliberately far from 0.4

    assert used == [pytest.approx(0.4)]
    assert len(oracle.calls) == 1  # the oracle keeps its original call signature (no temperature argument)


@patch.object(bg.System, 'get_total_energy')
def test_InverseFoldingSampling_one_step_advances_the_schedule(
    mocked_energy: Mock, energies_system: bg.System, monkeypatch
) -> None:
    """Each call to one_step reads the next entry of the schedule."""
    oracle = _StubInverseFoldingOracle({aa: 1 / 20 for aa in _AAS})
    mutator = bg.mutation.InverseFoldingSampling(oracle=oracle, t_sampling_proposals=[0.5, 0.2])
    used: list[float] = []
    original = mutator.propose_amino_acid

    def recording(probabilities, current_aa, temperature=1.0):
        used.append(temperature)
        return original(probabilities, current_aa, temperature=temperature)

    monkeypatch.setattr(mutator, 'propose_amino_acid', recording)

    mutator.one_step(energies_system)
    mutator.one_step(energies_system)

    assert used == pytest.approx([0.5, 0.2])


def test_InverseFoldingSampling_propose_amino_acid_tempers_allowed_alternatives() -> None:
    mutator = bg.mutation.InverseFoldingSampling(oracle=object(), mutation_bias={aa: 1.0 for aa in _AAS})
    probabilities = {aa: 0.0 for aa in _AAS} | {'A': 0.5, 'W': 0.3, 'Y': 0.2}
    # at T=0.1 the W:Y odds go from 3:2 to 3**10 : 2**10, i.e. P(W) = 0.6**10 / (0.6**10 + 0.4**10) ~ 0.983
    expected = 0.6**10 / (0.6**10 + 0.4**10)
    np.random.seed(0)
    draws = [mutator.propose_amino_acid(probabilities, 'A', temperature=0.1) for _ in range(4000)]
    assert set(draws) == {'W', 'Y'}
    assert abs(draws.count('W') / len(draws) - expected) < 0.01


def test_InverseFoldingSampling_propose_amino_acid_does_not_underflow_at_low_temperature() -> None:
    """A dominant current amino acid must not zero out every alternative when the temperature is very low."""
    mutator = bg.mutation.InverseFoldingSampling(oracle=object(), mutation_bias={aa: 1.0 for aa in _AAS})
    probabilities = {aa: 1e-11 for aa in _AAS} | {'A': 1 - 19e-11, 'W': 1e-10}
    # naively, (1e-10) ** (1 / 0.001) underflows to 0 and sampling would fail; in log space W is simply the argmax
    draws = {mutator.propose_amino_acid(probabilities, 'A', temperature=0.001) for _ in range(50)}
    assert draws == {'W'}


def test_InverseFoldingSampling_propose_amino_acid_handles_tiny_positive_temperature() -> None:
    """Even a denormal temperature must give a valid (greedy) proposal rather than NaN weights."""
    mutator = bg.mutation.InverseFoldingSampling(oracle=object(), mutation_bias={aa: 1.0 for aa in _AAS})
    probabilities = {aa: 0.0 for aa in _AAS} | {'A': 0.5, 'W': 0.3, 'Y': 0.2}
    draws = {mutator.propose_amino_acid(probabilities, 'A', temperature=1e-310) for _ in range(50)}
    assert draws == {'W'}


@pytest.mark.parametrize('temperature', [0.0, -1.0, float('nan'), float('inf')])
def test_InverseFoldingSampling_propose_amino_acid_rejects_non_positive_temperature(temperature) -> None:
    """A bare `<= 0` check lets NaN through to numpy as NaN weights, and infinity through as a silent
    uniform distribution, which would turn inverse folding into random substitution without raising."""
    mutator = bg.mutation.InverseFoldingSampling(oracle=object(), mutation_bias={aa: 1.0 for aa in _AAS})
    with pytest.raises(ValueError, match='finite and positive'):
        mutator.propose_amino_acid({aa: 0.05 for aa in _AAS}, 'A', temperature=temperature)


@pytest.mark.parametrize('protocol', [bg.mutation.Canonical, bg.mutation.GrandCanonical])
@patch.object(bg.System, 'get_total_energy')
def test_random_protocols_accept_and_ignore_temperature(
    mocked_energy: Mock, energies_system: bg.System, protocol
) -> None:
    np.random.seed(0)
    _, record_without = protocol().one_step(energies_system)
    np.random.seed(0)
    _, record_with = protocol().one_step(energies_system, temperature=5.0)
    assert record_with == record_without


def test_minimizer_passes_step_temperature_to_mutator(energies_system: bg.System, tmp_path) -> None:
    """The mutator receives the same temperature the minimizer uses to accept or reject that step."""
    received: list[float] = []

    def one_step(system, temperature=None):
        received.append(temperature)
        return system.__copy__(), Mock()

    mutator = Mock()
    mutator.one_step = one_step
    schedule = [0.3, 0.2, 0.1]
    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=mutator, temperature=schedule, n_steps=len(schedule), experiment_name='test', log_path=tmp_path
    )
    minimizer.minimize_system(energies_system)
    assert received == pytest.approx(schedule)


def test_minimizer_supports_legacy_mutator_without_temperature_argument(energies_system: bg.System, tmp_path) -> None:
    """Custom protocols written before ``temperature`` was added to ``one_step`` keep working."""

    class LegacyMutator(bg.mutation.MutationProtocol):
        def one_step(self, system):  # type: ignore[override]
            return system.__copy__(), bg.mutation.MutationRecord(mutations=[])

    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=LegacyMutator(), temperature=1.0, n_steps=2, experiment_name='test', log_path=tmp_path
    )
    minimizer.minimize_system(energies_system)  # must not raise TypeError


def test_InverseFoldingSampling_one_mutation_keeps_the_single_residue_path() -> None:
    """The default stays on inverse_fold, and still refuses to re-propose the current residue."""
    mutator = bg.mutation.InverseFoldingSampling(oracle=object())
    assert mutator.n_mutations == 1
    assert mutator.exclude_self is True
    # Joint decoding cannot exclude the current residue, so the flag follows n_mutations.
    assert bg.mutation.InverseFoldingSampling(oracle=object(), n_mutations=3).exclude_self is False


def test_InverseFoldingSampling_excluded_amino_acids_follows_mutation_bias() -> None:
    """Zero-bias and missing amino acids are the ones the model is told not to sample."""
    assert bg.mutation.InverseFoldingSampling(oracle=object()).excluded_amino_acids() == 'C'
    bias = {aa: 1.0 for aa in _AAS} | {'C': 0.0, 'W': 0.0}
    assert bg.mutation.InverseFoldingSampling(oracle=object(), mutation_bias=bias).excluded_amino_acids() == 'CW'
    partial = {'A': 1.0, 'D': 1.0}  # every other amino acid is missing, so excluded
    excluded = bg.mutation.InverseFoldingSampling(oracle=object(), mutation_bias=partial).excluded_amino_acids()
    assert set(excluded) == set(_AAS) - {'A', 'D'}


@patch.object(bg.System, 'get_total_energy')
def test_InverseFoldingSampling_decodes_several_residues_in_one_call(
    mocked_energy: Mock,
    energies_system: bg.System,
) -> None:
    """One joint proposal masks n_mutations residues, calls the oracle once, and records every change."""
    for state in energies_system.states:
        for residue in state.chains[0].residues:
            residue.name = 'A'
    oracle = _StubInverseFoldingOracle({aa: 1 / 20 for aa in _AAS})
    mutator = bg.mutation.InverseFoldingSampling(oracle=oracle, n_mutations=3)
    original_sequences = [state.chains[0].sequence for state in energies_system.states]

    mutated_system, record = mutator.one_step(energies_system)

    assert [state.chains[0].sequence for state in energies_system.states] == original_sequences
    assert len(oracle.calls) == 1  # one joint call, not one per residue
    call = oracle.calls[0]
    assert len(call['residue_indices']) == 3 and len(set(call['residue_indices'])) == 3
    assert call['temperature'] == bg.mutation.InverseFoldingSampling.DEFAULT_T_SAMPLING_PROPOSALS
    assert call['excluded_amino_acids'] == 'C'  # the default bias, enforced inside the model

    assert len(record.mutations) == 3
    assert {m.residue_index for m in record.mutations} == set(call['residue_indices'])
    assert all(m.move_type == 'substitution' and m.new_amino_acid == 'W' for m in record.mutations)
    chain = next(c for s in mutated_system.states for c in s.chains if c.chain_ID == record.mutations[0].chain_id)
    assert all(chain.sequence[m.residue_index] == 'W' for m in record.mutations)


@patch.object(bg.System, 'get_total_energy')
def test_InverseFoldingSampling_does_not_record_unchanged_residues(
    mocked_energy: Mock,
    energies_system: bg.System,
) -> None:
    """Joint decoding may re-propose the residue already there; those positions are left out of the record."""
    for state in energies_system.states:
        for residue in state.chains[0].residues:
            residue.name = 'A'
    chain_length = energies_system.states[0].chains[0].length
    # Every masked position comes back as 'A', i.e. unchanged.
    oracle = _StubInverseFoldingOracle(
        {aa: 1 / 20 for aa in _AAS}, generated={index: 'A' for index in range(chain_length)}
    )
    mutator = bg.mutation.InverseFoldingSampling(oracle=oracle, n_mutations=2)

    mutated_system, record = mutator.one_step(energies_system)

    assert record.mutations == []
    assert [s.chains[0].sequence for s in mutated_system.states] == [
        s.chains[0].sequence for s in energies_system.states
    ]


@patch.object(bg.System, 'get_total_energy')
def test_InverseFoldingSampling_n_mutations_is_clamped_to_the_mutable_residues(
    mocked_energy: Mock,
    energies_system: bg.System,
) -> None:
    """Asking for more residues than the chain has mutable ones masks all of them, without repeats."""
    oracle = _StubInverseFoldingOracle({aa: 1 / 20 for aa in _AAS})
    mutator = bg.mutation.InverseFoldingSampling(oracle=oracle, n_mutations=1000)

    mutator.one_step(energies_system)

    indices = oracle.calls[0]['residue_indices']
    chain_id = oracle.calls[0]['chain_id']
    chain = next(c for s in energies_system.states for c in s.chains if c.chain_ID == chain_id)
    assert sorted(indices) == sorted(chain.mutable_residue_indexes)


@patch.object(bg.System, 'get_total_energy')
def test_InverseFoldingSampling_several_residues_are_one_joint_proposal(
    mocked_energy: Mock,
    energies_system: bg.System,
) -> None:
    """n_mutations residues are decoded in a single call, not one call per residue."""
    for state in energies_system.states:
        for residue in state.chains[0].residues:
            residue.name = 'A'
    oracle = _StubInverseFoldingOracle({aa: 1 / 20 for aa in _AAS})
    mutator = bg.mutation.InverseFoldingSampling(oracle=oracle, n_mutations=4)

    _, record = mutator.one_step(energies_system)

    assert len(oracle.calls) == 1
    assert len(oracle.calls[0]['residue_indices']) == 4
    assert len(record.mutations) == 4


@patch.object(bg.System, 'get_total_energy')
def test_InverseFoldingSampling_several_residues_share_one_chain(
    mocked_energy: Mock,
    energies_system: bg.System,
) -> None:
    """The model is asked about one chain at a time, so every residue of a joint proposal comes from it."""
    # the stub proposes 'W' everywhere, so start from 'A' or a residue that is already 'W' is not a change
    for state in energies_system.states:
        for residue in state.chains[0].residues:
            residue.name = 'A'
    oracle = _StubInverseFoldingOracle({aa: 1 / 20 for aa in _AAS})
    mutator = bg.mutation.InverseFoldingSampling(oracle=oracle, n_mutations=3)

    _, record = mutator.one_step(energies_system)

    assert len({mutation.chain_id for mutation in record.mutations}) == 1
    assert {mutation.chain_id for mutation in record.mutations} == {oracle.calls[0]['chain_id']}


def test_InverseFoldingSampling_mutation_bias_stays_positional() -> None:
    """mutation_bias keeps its position, so calls written before t_sampling_proposals existed still work."""
    bias = {aa: 1.0 for aa in _AAS}
    mutator = bg.mutation.InverseFoldingSampling('ESM3', object(), 1, bias)
    assert mutator.mutation_bias == bias
    assert mutator.t_sampling_proposals == bg.mutation.InverseFoldingSampling.DEFAULT_T_SAMPLING_PROPOSALS
    with pytest.raises(TypeError):  # keyword-only, so it can never swallow a positional argument
        bg.mutation.InverseFoldingSampling('ESM3', object(), 1, bias, 0.5)


def test_InverseFoldingSampling_rejects_non_positive_n_mutations() -> None:
    with pytest.raises(ValueError, match='n_mutations must be at least 1'):
        bg.mutation.InverseFoldingSampling(n_mutations=0)


def test_bagel_sets_a_default_boileroom_image_tag() -> None:
    """The pinned boileroom version has no published image, so bagel names a prerelease that does.

    Without this, every modal/apptainer oracle fails to pull. ``setdefault`` keeps an explicit
    ``BOILEROOM_IMAGE_TAG`` winning; see the comment in ``bagel/__init__.py``.
    """
    import os
    import subprocess
    import sys

    assert bg.DEFAULT_BOILEROOM_IMAGE_TAG
    assert os.environ['BOILEROOM_IMAGE_TAG']  # importing bagel applied it (or the caller's own value)

    # A subprocess is the only honest check: the env var is set at import time, once per process.
    script = (
        'import os; os.environ.pop("BOILEROOM_IMAGE_TAG", None); import bagel; '
        'print(os.environ["BOILEROOM_IMAGE_TAG"] == bagel.DEFAULT_BOILEROOM_IMAGE_TAG)'
    )
    unset = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True)
    assert unset.stdout.strip() == 'True'

    script = 'import os; os.environ["BOILEROOM_IMAGE_TAG"] = "explicit"; import bagel; print(os.environ["BOILEROOM_IMAGE_TAG"])'
    explicit = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True)
    assert explicit.stdout.strip() == 'explicit'


def test_default_boileroom_image_tag_is_revisited_when_the_pin_moves() -> None:
    """Tripwire: bagel's tag wins over boileroom's own lookup, so a stale one would be used silently.

    ``DEFAULT_BOILEROOM_IMAGE_TAG`` only exists because the pinned boileroom version has no published
    image. It is set with ``os.environ.setdefault``, which boileroom honours over the tag it derives from
    its version -- so once the pin moves, bagel would keep forcing the old prerelease image rather than
    falling back. Repinning must therefore re-check the tag, or drop it if the new version is published.

    This catches a change of pinned *version*. It cannot catch boileroom publishing an image for the
    version already pinned; that still needs a human to notice and delete the constant.
    """
    import importlib.metadata

    assert importlib.metadata.version('boileroom') == '0.4.5', (
        'The pinned boileroom version changed. Re-check DEFAULT_BOILEROOM_IMAGE_TAG in bagel/__init__.py: '
        'point it at a prerelease published for every image bagel uses (boileroom-esm, boileroom-esmfold2, '
        'boileroom-boltz, boileroom-chai1), or delete it and this test if the new version is published.'
    )
