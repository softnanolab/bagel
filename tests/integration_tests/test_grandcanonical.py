import pathlib as pl
import desprot as dp
# ? Could these not just be mutation unit tests?


def test_grandcanonical_does_not_change_chain_length_when_mutator_not_allowed_to_remove_or_add(
    simple_state: dp.State, folder: dp.folding.ESMFolder, test_output_folder: pl.Path
) -> None:
    starting_chain_length = len(simple_state.chains[0].residues)
    test_system = dp.System(states=[simple_state], output_folder=test_output_folder, name='test_grandcanonical')

    minimizer = dp.minimizer.SimulatedAnnealing(
        folder=folder,
        mutator=dp.mutation.GrandCanonical(
            chemical_potential=1000.0, move_probabilities={'mutation': 1.0, 'addition': 0.0, 'removal': 0.0}
        ),  # high chemical_potential ensures removal would always be accepted if it was ever chosen.
        initial_temperature=1.0,
        final_temperature=0.001,
        n_steps=5,
        log_frequency=1,
    )

    minimizer.minimize_system(test_system)

    assert len(test_system.states[0].chains[0].residues) == starting_chain_length


def test_grandcanonical_does_not_increase_chain_length_when_mutator_not_allowed_to_add(
    simple_state: dp.State, folder: dp.folding.ESMFolder, test_output_folder: pl.Path
) -> None:
    starting_chain_length = len(simple_state.chains[0].residues)
    test_system = dp.System(states=[simple_state], output_folder=test_output_folder, name='test_grandcanonical')

    minimizer = dp.minimizer.SimulatedAnnealing(
        folder=folder,
        mutator=dp.mutation.GrandCanonical(
            chemical_potential=-1000.0, move_probabilities={'mutation': 0.0, 'addition': 0.0, 'removal': 1.0}
        ),  # very low chemical_potential ensures removal would always be rejected when chosen.
        initial_temperature=1.0,
        final_temperature=0.001,
        n_steps=5,
        log_frequency=1,
    )

    minimizer.minimize_system(test_system)

    assert len(test_system.states[0].chains[0].residues) == starting_chain_length


def test_grandcanonical_does_not_zero_chain_length_when_mutator_only_allowed_to_remove(
    simple_state: dp.State, folder: dp.folding.ESMFolder, test_output_folder: pl.Path
) -> None:
    starting_chain_length = len(simple_state.chains[0].residues)
    test_system = dp.System(states=[simple_state], output_folder=test_output_folder, name='test_grandcanonical')

    minimizer = dp.minimizer.SimulatedAnnealing(
        folder=folder,
        mutator=dp.mutation.GrandCanonical(
            chemical_potential=1000.0, move_probabilities={'mutation': 0.0, 'addition': 0.0, 'removal': 1.0}
        ),
        initial_temperature=1.0,
        final_temperature=0.001,
        n_steps=starting_chain_length + 1,
        log_frequency=1,
    )

    minimizer.minimize_system(test_system)

    assert len(test_system.states[0].chains[0].residues) >= 1
