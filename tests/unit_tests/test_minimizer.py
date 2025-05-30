import bagel as bg
import pandas as pd
import numpy as np
import copy
import pytest
from unittest.mock import patch, Mock


# def test_Minimizer_initialise_log_path_method_creates_intended_files(SA_minimizer: brm.SimulatedAnnealing) -> None:
#    assert SA_minimizer.log_path.is_dir(), 'Did not create a results folder.'
#    script_path = SA_minimizer.log_path / 'origional_script.py'
#    assert script_path.is_file(), 'Did not copy python script into results folder'


# def test_Minimizer_one_step_method_accepts_lower_energy_system(
#    SA_minimizer: brm.SimulatedAnnealing,
#    mixed_system: br.System,
# ) -> None:
#    mixed_system.is_calculated = True
#    mixed_system._energy = 1.0
#    lower_energy_system = copy.deepcopy(mixed_system)
#    lower_energy_system._energy = 0.0
#    with patch.object(SA_minimizer.mutator, 'one_step') as mock_one_step_method:
#        mock_one_step_method.return_value = lower_energy_system
#        returned_system, _ = SA_minimizer.minimize_one_step(temperature=0.1, initial_system=mixed_system)
#    assert returned_system == lower_energy_system


# def test_Minimizer_logging_step_method_gives_correct_output_for_first_step(
#    SA_minimizer: brm.SimulatedAnnealing, mixed_system: br.System
# ) -> None:
#    mixed_system.is_calculated = True  # prevents system for initially calculating structure and energies
#    SA_minimizer.logging_step(
#        temperature=0.1, step=0, system=mixed_system, best_system=mixed_system, new_best=True, acceptance_prob=0.5
#    )
#    assert (SA_minimizer.log_path / 'all_systems').is_dir(), 'did not create folder for current system logs'
#    assert (SA_minimizer.log_path / 'best_systems').is_dir(), 'did not create folder for best system logs'
#    assert (SA_minimizer.log_path / 'minimizer_logs.csv').is_file(), 'did not create file for minimizer logs'
#
#    correct_minizer_logs = pd.DataFrame({'step': [0], 'temperature': [0.1], 'acceptance_probability': [0.5]})
#    minimizer_logs = pd.read_csv(SA_minimizer.log_path / 'minimizer_logs.csv')
#    assert all(correct_minizer_logs == minimizer_logs), 'incorrect information in minizer logs file'


# @patch.object(brm.Minimizer, 'minimize_one_step')  # used to prevent unnecessary folding
# def test_SimulatedAnnealing_minimize_system_method_properly_calculates_internal_attributes(
#    mock_minimize_method: Mock, SA_minimizer: brm.SimulatedAnnealing, mixed_system: br.System
# ) -> None:
#    mock_minimize_method.return_value = (mixed_system, 0.5)
#    mixed_system.is_calculated = True  # prevents system for initially calculating structure and energies
#    SA_minimizer.minimize_system(mixed_system)
#    temperatatures_called = [call.args[0] for call in mock_minimize_method.call_args_list]
#    assert np.allclose(temperatatures_called, [0.35, 0.3, 0.25, 0.2, 0.15, 0.1])


# @patch.object(brm.Minimizer, 'minimize_one_step')  # used to prevent unnecessary folding
# def test_SimulatedTempering_minimize_system_method_properly_calculates_internal_attributes(
#    mock_minimize_method: Mock, ST_minimizer: brm.SimulatedTempering, mixed_system: br.System
# ) -> None:
#    mock_minimize_method.return_value = (mixed_system, 0.5)
#    mixed_system.is_calculated = True  # prevents system for initially calculating structure and energies
#    ST_minimizer.minimize_system(mixed_system)
#    temperatatures_called = [call.args[0] for call in mock_minimize_method.call_args_list]
#    assert temperatatures_called == [0.1, 0.1, 0.01] * 2


# @pytest.mark.parametrize('minimizer_name', ['SA_minimizer', 'ST_minimizer'])
# @patch.object(brm.System, 'dump_logs')
# def test_minimizers_log_structures_at_correct_intervals(
#    mock_dump_logs_method: Mock, minimizer_name: str, mixed_system: br.System, request: pytest.FixtureRequest
# ) -> None:
#    minimizer: brm.Minimizer = request.getfixturevalue(minimizer_name)
#    mixed_system.is_calculated = True
#    mixed_system._energy = 1.0
#    lower_energy_system = copy.deepcopy(mixed_system)
#    lower_energy_system._energy = 0.0
#    with patch.object(minimizer.mutator, 'one_step') as mock_one_step_method:  # prevents unnecessary folding
#        mock_one_step_method.return_value = lower_energy_system
#        minimizer.minimize_system(mixed_system)
#    save_structure_calls = [call.kwargs['save_structure'] for call in mock_dump_logs_method.call_args_list]
#    # dump logs method is called for system then best system for each of the 6 minimization steps
#    system_calls = save_structure_calls[::2]
#    best_system_calls = save_structure_calls[1::2]
#    assert system_calls == [True, False, True, False, True, False], 'Incorrectly logging current system structures'
#    assert best_system_calls == [True, False, False, False, False, False], 'Incorrectly logging best system structures'


def test_MonteCarloMinimizer_temperature_input_types():
    """Test that MonteCarloMinimizer can handle different temperature input types."""
    # Create a mock mutator
    mock_mutator = Mock()

    # Test with float input
    n_steps = 5
    float_temp = 1.0
    minimizer_float = bg.minimizer.MonteCarloMinimizer(
        mutator=mock_mutator, temperature=float_temp, n_steps=n_steps, experiment_name='test_float'
    )
    assert np.allclose(minimizer_float.temperature_schedule, np.array([float_temp] * n_steps))

    # Test with list input
    list_temp = [1.0, 2.0, 3.0, 4.0, 5.0]
    minimizer_list = bg.minimizer.MonteCarloMinimizer(
        mutator=mock_mutator, temperature=list_temp, n_steps=n_steps, experiment_name='test_list'
    )
    assert np.allclose(minimizer_list.temperature_schedule, np.array(list_temp))

    # Test with numpy array input
    array_temp = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    minimizer_array = bg.minimizer.MonteCarloMinimizer(
        mutator=mock_mutator, temperature=array_temp, n_steps=n_steps, experiment_name='test_array'
    )
    assert np.allclose(minimizer_array.temperature_schedule, array_temp)

    # Test that invalid input raises ValueError
    with pytest.raises(ValueError):
        bg.minimizer.MonteCarloMinimizer(
            mutator=mock_mutator,
            temperature='invalid',  # type: ignore
            n_steps=n_steps,
            experiment_name='test_invalid',
        )

    # Test that wrong length list raises ValueError
    with pytest.raises(ValueError):
        bg.minimizer.MonteCarloMinimizer(
            mutator=mock_mutator,
            temperature=[1.0, 2.0],  # Wrong length
            n_steps=n_steps,
            experiment_name='test_wrong_length',
        )


def test_SimulatedAnnealing_temperature_schedule():
    """Test that SimulatedAnnealing creates the correct linear temperature schedule."""
    # Create a mock mutator
    mock_mutator = Mock()

    # Test with specific temperature range and steps
    initial_temp = 1.0
    final_temp = 0.1
    n_steps = 10

    minimizer = bg.minimizer.SimulatedAnnealing(
        mutator=mock_mutator,
        initial_temperature=initial_temp,
        final_temperature=final_temp,
        n_steps=n_steps,
        experiment_name='test_annealing',
    )

    # Check that temperature schedule is linear and has correct endpoints
    expected_schedule = np.linspace(initial_temp, final_temp, n_steps)
    assert np.allclose(minimizer.temperature_schedule, expected_schedule), (
        'Temperature schedule is not linear or has incorrect endpoints'
    )

    # Verify first and last temperatures
    assert minimizer.temperature_schedule[0] == initial_temp, 'Initial temperature is incorrect'
    assert minimizer.temperature_schedule[-1] == final_temp, 'Final temperature is incorrect'


def test_SimulatedTempering_temperature_schedule():
    """Test that SimulatedTempering creates the correct cycling temperature schedule."""
    # Create a mock mutator
    mock_mutator = Mock()

    # Test with specific parameters
    high_temp = 1.0
    low_temp = 0.1
    n_steps_high = 3
    n_steps_low = 2
    n_cycles = 2

    minimizer = bg.minimizer.SimulatedTempering(
        mutator=mock_mutator,
        high_temperature=high_temp,
        low_temperature=low_temp,
        n_steps_high=n_steps_high,
        n_steps_low=n_steps_low,
        n_cycles=n_cycles,
        experiment_name='test_tempering',
    )

    # Calculate expected schedule
    cycle_temperatures = [low_temp] * n_steps_low + [high_temp] * n_steps_high
    expected_schedule = np.array(cycle_temperatures * n_cycles)

    # Verify the temperature schedule
    assert np.allclose(minimizer.temperature_schedule, expected_schedule), (
        'Temperature schedule does not match expected cycling pattern'
    )

    # Verify cycle lengths
    total_steps = (n_steps_high + n_steps_low) * n_cycles
    assert len(minimizer.temperature_schedule) == total_steps, 'Total number of steps is incorrect'

    # Verify temperature values
    for i in range(n_cycles):
        cycle_start = i * (n_steps_high + n_steps_low)
        # Check low temperature phase
        assert np.all(minimizer.temperature_schedule[cycle_start : cycle_start + n_steps_low] == low_temp), (
            f'Low temperature phase incorrect in cycle {i}'
        )
        # Check high temperature phase
        assert np.all(
            minimizer.temperature_schedule[cycle_start + n_steps_low : cycle_start + n_steps_low + n_steps_high]
            == high_temp
        ), f'High temperature phase incorrect in cycle {i}'
