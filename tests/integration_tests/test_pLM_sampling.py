import pathlib as pl
import bagel as bg


def test_pLM_sampling_does_not_raise_exceptions_with_nominal_inputs(
    plm_only_state: bg.State, test_log_path: pl.Path
) -> None:
    test_system = bg.System(states=[plm_only_state], name='test_pLM')

    # TODO: this is actually a sampler, not a minimizer, need to think about naming
    minimizer = bg.minimizer.MetropolisMinimizer(
        mutator=bg.mutation.Canonical(),
        temperature=1.0,
        n_steps=3,
        log_frequency=1,
        log_path=test_log_path,
    )

    final_system = minimizer.minimize_system(system=test_system)
