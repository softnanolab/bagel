import pathlib as pl
import bagel as bg
import numpy as np


# ? Could this not just be a mutation unit test?
def test_tempering_does_not_mutate_immutable_residues(
    esmfold: bg.oracles.folding.ESMFold,
    test_log_path: pl.Path,
    very_high_temp: float,
) -> None:
    mutability = [False, True, False]
    residues = [bg.Residue(name='G', chain_ID='C-A', index=i, mutable=mut) for i, mut in enumerate(mutability)]

    state = bg.State(
        chains=[bg.Chain(residues)],
        energy_terms=[
            bg.energies.PTMEnergy(oracle=esmfold),
            bg.energies.OverallPLDDTEnergy(oracle=esmfold),
            bg.energies.HydrophobicEnergy(oracle=esmfold),
        ],
        name='state_A',
    )

    test_system = bg.System(states=[state], name='test_tempering2')

    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(),
        high_temperature=very_high_temp,  # Ensures any mutation is accepted
        low_temperature=0.001,
        n_steps_high=3,
        n_steps_low=2,
        n_cycles=1,
        preserve_best_system_every_n_steps=None,
        log_frequency=1,
        log_path=test_log_path,
    )

    best_system = minimizer.minimize_system(system=test_system)

    assert best_system.states[0].chains[0].sequence[::2] == 'GG'


def test_tempering_preserve_best_system_every_n_steps(
    esmfold: bg.oracles.folding.ESMFold,
    test_log_path: pl.Path,
    very_high_temp: float,
) -> None:
    np.random.seed(42)

    residues = [bg.Residue(name='G', chain_ID='TEST', index=i, mutable=True) for i in range(10)]

    state = bg.State(
        chains=[bg.Chain(residues)],
        energy_terms=[
            bg.energies.PTMEnergy(oracle=esmfold),
            bg.energies.OverallPLDDTEnergy(oracle=esmfold),
            bg.energies.HydrophobicEnergy(oracle=esmfold),
        ],
        name='state_A',
    )

    test_system = bg.System(states=[state], name='test_tempering2')

    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(),
        high_temperature=very_high_temp,  # Ensures any mutation is accepted
        low_temperature=0.001,
        n_steps_high=3,
        n_steps_low=2,
        n_cycles=3,
        preserve_best_system_every_n_steps=13,
        log_frequency=1,
        log_path=test_log_path,
    )

    best_system = minimizer.minimize_system(test_system)

    from bagel.analysis.analyzer import SimulatedTemperingAnalyzer

    analyzer = SimulatedTemperingAnalyzer(test_log_path / minimizer.experiment_name)

    # Check that step column is 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    assert analyzer.optimization_df['step'].tolist() == list(range(1, 16)), 'Step column does not match expected'

    # Check that accept column is True for steps 3,4,5; 7,8,9; 13,14,15
    for step in [3, 4, 5, 7, 8, 9, 13, 14, 15]:
        assert analyzer.optimization_df.loc[analyzer.optimization_df['step'] == step, 'accept'].values[0] == True, (
            f'Step {step} accept not True'
        )

    # Check that temperature column has the expected pattern: [0.001, 0.001, very_high_temp, very_high_temp, very_high_temp] repeated 3 times
    expected_temps = [0.001, 0.001, very_high_temp, very_high_temp, very_high_temp] * 3
    actual_temps = analyzer.optimization_df['temperature'].iloc[:15].tolist()
    assert all(
        (abs(a - b) < 1e-8 if isinstance(a, float) and isinstance(b, float) else a == b)
        for a, b in zip(actual_temps, expected_temps)
    ), f'Temperature schedule does not match expected: {actual_temps} vs {expected_temps}'

    # Check preserving of the system at step 13
    assert analyzer.current_sequences['state_A'][12] != analyzer.current_sequences['state_A'][13], (
        'Current system at step 13 is the same as from before.'
    )
    assert analyzer.current_sequences['state_A'][13] == analyzer.best_sequences['state_A'][12], (
        'Best system from before was not preserved.'
    )
    assert analyzer.current_sequences['state_A'][13] == analyzer.best_sequences['state_A'][13], (
        'Current and best systems are not the same.'
    )

    # go into the directory with current, and check there's 15 files with .pae and .plddt and .cif
    current_dir = test_log_path / minimizer.experiment_name / 'current' / 'structures'
    assert len(list(current_dir.glob('*.pae'))) == 16, 'There should be 16 .pae files in the current directory.'
    assert len(list(current_dir.glob('*.plddt'))) == 16, 'There should be 16 .plddt files in the current directory.'
    assert len(list(current_dir.glob('*.cif'))) == 16, 'There should be 16 .cif files in the current directory.'


def test_tempering_energy_term_names_in_csv_files(
    esmfold: bg.oracles.folding.ESMFold,
    test_log_path: pl.Path,
    very_high_temp: float,
) -> None:
    """Test that custom energy term names are correctly recorded in energies.csv files."""

    residues = [bg.Residue(name='G', chain_ID='TEST', index=i, mutable=True) for i in range(5)]

    # Create energy terms with custom names
    state = bg.State(
        chains=[bg.Chain(residues)],
        energy_terms=[
            bg.energies.PTMEnergy(name='test', oracle=esmfold, weight=1.0),
            bg.energies.OverallPLDDTEnergy(oracle=esmfold, weight=1.0),
            bg.energies.OverallPLDDTEnergy(name='test', oracle=esmfold, weight=1.0),
        ],
        name='test',
    )

    test_system = bg.System(states=[state], name='test_energy_names')

    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(),
        high_temperature=very_high_temp,  # Ensures any mutation is accepted
        low_temperature=0.001,
        n_steps_high=2,
        n_steps_low=1,
        n_cycles=2,  # Short run for testing
        preserve_best_system_every_n_steps=None,
        log_frequency=1,
        log_path=test_log_path,
    )

    best_system = minimizer.minimize_system(test_system)

    from bagel.analysis.analyzer import SimulatedTemperingAnalyzer

    analyzer = SimulatedTemperingAnalyzer(test_log_path / minimizer.experiment_name)

    # Expected energy term names using the actual state name
    expected_energy_names = [f'{state.name}:pTM_test', f'{state.name}:global_pLDDT', f'{state.name}:global_pLDDT_test']

    # Check that custom energy term names appear in specific columns (2nd, 3rd, 4th) in current energies CSV
    current_energy_columns = analyzer.current_energies_df.columns.tolist()

    for i, expected_name in enumerate(expected_energy_names):
        column_index = i + 1  # 2nd, 3rd, 4th columns (indices 1, 2, 3)
        actual_name = current_energy_columns[column_index]
        assert actual_name == expected_name, (
            f'Column {column_index + 1} in current energies CSV should be "{expected_name}", got "{actual_name}". '
            f'Full columns: {current_energy_columns}'
        )

    # Check that custom energy term names appear in specific columns (2nd, 3rd, 4th) in best energies CSV
    best_energy_columns = analyzer.best_energies_df.columns.tolist()

    for i, expected_name in enumerate(expected_energy_names):
        column_index = i + 1  # 2nd, 3rd, 4th columns (indices 1, 2, 3)
        actual_name = best_energy_columns[column_index]
        assert actual_name == expected_name, (
            f'Column {column_index + 1} in best energies CSV should be "{expected_name}", got "{actual_name}". '
            f'Full columns: {best_energy_columns}'
        )
