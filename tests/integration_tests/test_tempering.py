import pathlib as pl
import bricklane as bl


# ? Could this not just be a mutation unit test?
def test_tempering_does_not_mutate_immutable_residues(
    folder: bl.folding.ESMFolder, test_output_folder: pl.Path
) -> None:
    mutability = [False, True, False]
    residues = [bl.Residue(name='G', chain_ID='C-A', index=i, mutable=mut) for i, mut in enumerate(mutability)]

    state = bl.State(
        chains=[bl.Chain(residues)],
        energy_terms=[bl.energies.PTMEnergy(), bl.energies.OverallPLDDTEnergy(), bl.energies.HydrophobicEnergy()],
        energy_terms_weights=[1.0, 1.0, 5.0],
        state_ID='state_A',
        verbose=True,
    )

    test_system = bl.System(states=[state], output_folder=test_output_folder, name='test_tempering2')

    minimizer = bl.minimizer.SimulatedTempering(
        folder=folder,
        mutator=bl.mutation.Canonical(),
        high_temperature=1000.0,  # Ensures any mutation is accepted
        low_temperature=0.001,
        n_steps_high=3,
        n_steps_low=2,
        n_cycles=1,
        preserve_best_system=False,
        log_frequency=1,
    )

    best_system = minimizer.minimize_system(system=test_system)

    assert best_system.states[0].chains[0].sequence[::2] == 'GG'


def test_tempering_does_not_raise_exceptions_with_nominal_inputs(
    simple_state: bl.State, folder: bl.folding.ESMFolder, test_output_folder: pl.Path
) -> None:
    test_system = bl.System(states=[simple_state], output_folder=test_output_folder, name='test_tempering')

    minimizer = bl.minimizer.SimulatedTempering(
        folder=folder,
        mutator=bl.mutation.Canonical(),
        high_temperature=1,
        low_temperature=0.1,
        n_steps_high=3,
        n_steps_low=2,
        n_cycles=1,
        preserve_best_system=False,
        log_frequency=1,
    )

    best_system = minimizer.minimize_system(test_system)
    best_system.dump_logs(experiment='test_tempering', step=-1)
