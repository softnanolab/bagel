import pathlib as pl
import bagel as bg


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


# def test_tempering_does_not_raise_exceptions_with_nominal_inputs(
#     simple_state: bg.State, test_log_path: pl.Path
# ) -> None:

#     # TODO: create a proper state here that will be used for integration
#     simple_state._energy_terms_value = {}  # clean up before running
#     test_system = bg.System(states=[simple_state], name='test_tempering')

#     minimizer = bg.minimizer.SimulatedTempering(
#         mutator=bg.mutation.Canonical(),
#         high_temperature=1,
#         low_temperature=0.1,
#         n_steps_high=3,
#         n_steps_low=2,
#         n_cycles=1,
#         preserve_best_system_every_n_steps=None,
#         log_frequency=1,
#         log_path=test_log_path,
#     )

#     best_system = minimizer.minimize_system(test_system)
