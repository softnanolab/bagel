import pathlib as pl
import bagel as bg


# ? Could this not just be a mutation unit test?
def test_tempering_does_not_mutate_immutable_residues(
    folder: bg.oracles.FoldingOracle,
    test_log_path: pl.Path,
    very_high_temp: float,
) -> None:
    mutability = [False, True, False]
    residues = [bg.Residue(name='G', chain_ID='C-A', index=i, mutable=mut) for i, mut in enumerate(mutability)]

    state = bg.State(
        chains=[bg.Chain(residues)],
        energy_terms=[bg.energies.PTMEnergy(), bg.energies.OverallPLDDTEnergy(), bg.energies.HydrophobicEnergy()],
        name='state_A',
    )

    test_system = bg.System(states=[state], name='test_tempering2')

    minimizer = bg.minimizer.SimulatedTempering(
        folder=folder,
        mutator=bg.mutation.Canonical(),
        high_temperature=very_high_temp,  # Ensures any mutation is accepted
        low_temperature=0.001,
        n_steps_high=3,
        n_steps_low=2,
        n_cycles=1,
        preserve_best_system=False,
        log_frequency=1,
        log_path=test_log_path,
    )

    best_system = minimizer.minimize_system(system=test_system)

    assert best_system.states[0].chains[0].sequence[::2] == 'GG'


def test_tempering_does_not_raise_exceptions_with_nominal_inputs(
    simple_state: bg.State, folder: bg.oracles.FoldingOracle, test_log_path: pl.Path
) -> None:
    test_system = bg.System(states=[simple_state], name='test_tempering')

    minimizer = bg.minimizer.SimulatedTempering(
        folder=folder,
        mutator=bg.mutation.Canonical(),
        high_temperature=1,
        low_temperature=0.1,
        n_steps_high=3,
        n_steps_low=2,
        n_cycles=1,
        preserve_best_system=False,
        log_frequency=1,
        log_path=test_log_path,
    )

    best_system = minimizer.minimize_system(test_system)
