import bagel as bg
import numpy as np
import pytest
import pathlib as pl
import tempfile
from unittest.mock import Mock


@pytest.fixture
def temp_log_path():
    """Provides a temporary directory for log paths in unit tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pl.Path(tmpdir)


def test_MonteCarloMinimizer_temperature_input_types(temp_log_path):
    """Test that MonteCarloMinimizer can handle different temperature input types."""
    # Create a mock mutator
    mock_mutator = Mock()

    # Test with float input
    n_steps = 5
    float_temp = 1.0
    minimizer_float = bg.minimizer.MonteCarloMinimizer(
        mutator=mock_mutator,
        temperature=float_temp,
        n_steps=n_steps,
        experiment_name='test_float',
        log_path=temp_log_path,
    )
    assert np.allclose(minimizer_float.temperature_schedule, np.array([float_temp] * n_steps))

    # Test with list input
    list_temp = [1.0, 2.0, 3.0, 4.0, 5.0]
    minimizer_list = bg.minimizer.MonteCarloMinimizer(
        mutator=mock_mutator,
        temperature=list_temp,
        n_steps=n_steps,
        experiment_name='test_list',
        log_path=temp_log_path,
    )
    assert np.allclose(minimizer_list.temperature_schedule, np.array(list_temp))

    # Test with numpy array input
    array_temp = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    minimizer_array = bg.minimizer.MonteCarloMinimizer(
        mutator=mock_mutator,
        temperature=array_temp,
        n_steps=n_steps,
        experiment_name='test_array',
        log_path=temp_log_path,
    )
    assert np.allclose(minimizer_array.temperature_schedule, array_temp)

    # Test that invalid input raises ValueError
    with pytest.raises(ValueError):
        bg.minimizer.MonteCarloMinimizer(
            mutator=mock_mutator,
            temperature='invalid',  # type: ignore
            n_steps=n_steps,
            experiment_name='test_invalid',
            log_path=temp_log_path,
        )

    # Test that wrong length list raises ValueError
    with pytest.raises(ValueError):
        bg.minimizer.MonteCarloMinimizer(
            mutator=mock_mutator,
            temperature=[1.0, 2.0],  # Wrong length
            n_steps=n_steps,
            experiment_name='test_wrong_length',
            log_path=temp_log_path,
        )


def test_SimulatedAnnealing_temperature_schedule(temp_log_path):
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
        log_path=temp_log_path,
    )

    # Check that temperature schedule is linear and has correct endpoints
    expected_schedule = np.linspace(initial_temp, final_temp, n_steps)
    assert np.allclose(minimizer.temperature_schedule, expected_schedule), (
        'Temperature schedule is not linear or has incorrect endpoints'
    )

    # Verify first and last temperatures
    assert minimizer.temperature_schedule[0] == initial_temp, 'Initial temperature is incorrect'
    assert minimizer.temperature_schedule[-1] == final_temp, 'Final temperature is incorrect'


def test_SimulatedTempering_temperature_schedule(temp_log_path):
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
        log_path=temp_log_path,
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
