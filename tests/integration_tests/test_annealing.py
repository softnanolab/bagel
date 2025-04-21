import pathlib as pl
import bricklane as bl


def test_annealing_raises_no_errors_for_nominal_inputs(
    simple_state: bl.State, folder: bl.folding.ESMFolder, test_log_path: pl.Path
) -> None:
    test_system = bl.System(states=[simple_state], name='test_annealing')

    minimizer = bl.minimizer.SimulatedAnnealing(
        folder=folder,
        mutator=bl.mutation.Canonical(),
        initial_temperature=1.0,
        final_temperature=0.001,
        n_steps=5,
        log_frequency=1,
        log_path=test_log_path,
    )

    minimizer.minimize_system(test_system)
