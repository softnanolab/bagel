import pathlib as pl
import bagel as bg


def test_annealing_raises_no_errors_for_nominal_inputs(
    simple_state: bg.State, folder: bg.folding.ESMFolder, test_log_path: pl.Path
) -> None:
    test_system = bg.System(states=[simple_state], name='test_annealing')

    minimizer = bg.minimizer.SimulatedAnnealing(
        folder=folder,
        mutator=bg.mutation.Canonical(),
        initial_temperature=1.0,
        final_temperature=0.001,
        n_steps=5,
        log_frequency=1,
        log_path=test_log_path,
    )

    minimizer.minimize_system(test_system)
