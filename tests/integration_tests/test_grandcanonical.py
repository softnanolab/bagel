import pathlib as pl
import bagel as bg
# ? Could these not just be mutation unit tests?


def add_chemical_potential_energy(state):
    folding_oracle = None
    for term in state.energy_terms:
        if hasattr(term, 'oracle'):
            folding_oracle = term.oracle
            break
    if folding_oracle is not None and not any(
        isinstance(term, bg.energies.ChemicalPotentialEnergy) for term in state.energy_terms
    ):
        state.energy_terms.append(
            bg.energies.ChemicalPotentialEnergy(
                oracle=folding_oracle, target_size=3, chemical_potential=1.0, weight=1.0
            )
        )


def test_grandcanonical_does_not_change_chain_length_when_mutator_not_allowed_to_remove_or_add(
    real_simple_state: bg.State,
    test_log_path: pl.Path,
    very_high_temp: float,
) -> None:
    add_chemical_potential_energy(real_simple_state)
    starting_chain_length = len(real_simple_state.chains[0].residues)
    test_system = bg.System(states=[real_simple_state], name='test_grandcanonical')

    minimizer = bg.minimizer.SimulatedAnnealing(
        mutator=bg.mutation.GrandCanonical(
            move_probabilities={'substitution': 1.0, 'addition': 0.0, 'removal': 0.0}
        ),  # high temperature ensures removal would always be accepted if it was ever chosen.
        initial_temperature=very_high_temp,
        final_temperature=very_high_temp,
        n_steps=5,
        log_frequency=1,
        log_path=test_log_path,
    )

    minimizer.minimize_system(test_system)

    assert len(test_system.states[0].chains[0].residues) == starting_chain_length


def test_grandcanonical_does_not_increase_chain_length_when_mutator_not_allowed_to_add(
    real_simple_state: bg.State,
    test_log_path: pl.Path,
    very_high_temp: float,
) -> None:
    add_chemical_potential_energy(real_simple_state)
    starting_chain_length = len(real_simple_state.chains[0].residues)
    test_system = bg.System(states=[real_simple_state], name='test_grandcanonical')

    minimizer = bg.minimizer.SimulatedAnnealing(
        mutator=bg.mutation.GrandCanonical(
            move_probabilities={'substitution': 0.0, 'addition': 0.0, 'removal': 1.0}
        ),  # very high T ensures addition would always be accepted when chosen.
        initial_temperature=very_high_temp,
        final_temperature=very_high_temp,
        n_steps=5,
        log_frequency=1,
        log_path=test_log_path,
    )

    minimizer.minimize_system(test_system)

    assert len(test_system.states[0].chains[0].residues) == starting_chain_length


def test_grandcanonical_does_not_zero_chain_length_when_mutator_only_allowed_to_remove(
    real_simple_state: bg.State,
    test_log_path: pl.Path,
    very_high_temp: float,
) -> None:
    add_chemical_potential_energy(real_simple_state)
    starting_chain_length = len(real_simple_state.chains[0].residues)
    test_system = bg.System(states=[real_simple_state], name='test_grandcanonical')

    minimizer = bg.minimizer.SimulatedAnnealing(
        mutator=bg.mutation.GrandCanonical(
            move_probabilities={'substitution': 0.0, 'addition': 0.0, 'removal': 1.0}
        ),  # very high T ensures removal always accepted if possible
        initial_temperature=very_high_temp,
        final_temperature=very_high_temp,
        n_steps=starting_chain_length + 1,
        log_frequency=1,
        log_path=test_log_path,
    )

    minimizer.minimize_system(test_system)

    assert len(test_system.states[0].chains[0].residues) >= 1
