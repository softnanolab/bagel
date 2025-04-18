import pathlib as pl
import desprot as dp


def test_annealing_raises_no_errors_for_nomial_inputs(
    simple_state: dp.State, folder: dp.folding.ESMFolder, test_output_folder: pl.Path
) -> None:
    test_system = dp.System(states=[simple_state], output_folder=test_output_folder, name='test_annealing')

    minimizer = dp.minimizer.SimulatedAnnealing(
        folder=folder,
        mutator=dp.mutation.Canonical(),
        initial_temperature=1.0,
        final_temperature=0.001,
        n_steps=5,
        log_frequency=1,
    )

    minimizer.minimize_system(test_system)
