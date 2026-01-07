"""
Unit tests for the callback system.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

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


@pytest.fixture
def mock_minimizer(temp_log_path):
    """Create a mock minimizer for testing."""
    mock_mutator = Mock()
    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=mock_mutator,
        temperature=1.0,
        n_steps=10,
        experiment_name='test_minimizer',
        log_path=temp_log_path,
    )
    return minimizer


@pytest.fixture
def simple_system(fake_esmfold: bg.oracles.folding.ESMFold) -> bg.System:
    """Create a simple system with one state for testing."""
    sequence = ['A', 'V', 'L', 'G', 'E']
    residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    state = bg.State(
        chains=[bg.Chain(residues)],
        energy_terms=[
            bg.energies.PTMEnergy(oracle=fake_esmfold, weight=1.0),
            bg.energies.OverallPLDDTEnergy(oracle=fake_esmfold, weight=1.0),
        ],
        name='state_A',
    )
    state._energy = -1.5
    state._energy_term_values = {
        state.energy_terms[0].name: -1.0,  # 'pTM'
        state.energy_terms[1].name: -0.5,  # 'global_pLDDT'
    }
    system = bg.System(states=[state], name='simple_system')
    system.total_energy = -1.5
    return system


# ============================================================================
# Test CallbackContext
# ============================================================================


def test_CallbackContext_creation(simple_system, mock_minimizer):
    """Test that CallbackContext can be created with all fields."""
    metrics = {'system_energy': -1.5, 'best_system_energy': -1.5}
    context = bg.callbacks.CallbackContext(
        step=0,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=metrics,
        minimizer=mock_minimizer,
        step_kwargs={'temperature': 1.0, 'accept': True},
    )

    assert context.step == 0
    assert context.system == simple_system
    assert context.best_system == simple_system
    assert context.new_best is False
    assert context.metrics == metrics
    assert context.minimizer == mock_minimizer
    assert context.step_kwargs == {'temperature': 1.0, 'accept': True}


# ============================================================================
# Test Callback (ABC)
# ============================================================================


def test_Callback_default_implementations(simple_system, mock_minimizer):
    """Test that Callback ABC has default no-op implementations."""
    callback = bg.callbacks.Callback()
    metrics = {'system_energy': -1.5}
    context = bg.callbacks.CallbackContext(
        step=0,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=metrics,
        minimizer=mock_minimizer,
        step_kwargs={},
    )

    # Should not raise any exceptions
    callback.on_optimization_start(context)
    callback.on_step_end(context)
    callback.on_optimization_end(context)


# ============================================================================
# Test CallbackManager
# ============================================================================


def test_CallbackManager_initialization():
    """Test CallbackManager initialization."""
    manager = bg.callbacks.CallbackManager()
    assert len(manager.callbacks) == 0

    callback = bg.callbacks.Callback()
    manager = bg.callbacks.CallbackManager([callback])
    assert len(manager.callbacks) == 1


def test_CallbackManager_extract_metrics(simple_system):
    """Test metric extraction from system."""
    manager = bg.callbacks.CallbackManager()
    best_system = simple_system.__copy__()

    metrics = manager.extract_metrics(simple_system, best_system)

    assert 'system_energy' in metrics
    assert 'best_system_energy' in metrics
    assert metrics['system_energy'] == -1.5
    # Should have state-level metrics
    assert 'state_A/state_energy' in metrics
    # Energy term names are 'pTM' and 'global_pLDDT', not class names
    assert any('pTM' in key for key in metrics.keys())


def test_CallbackManager_execution_order(simple_system, mock_minimizer):
    """Test that callbacks execute in order."""
    call_order = []

    class TestCallback(bg.callbacks.Callback):
        def __init__(self, name: str):
            self.name = name

        def on_step_end(self, context: bg.callbacks.CallbackContext) -> None:
            call_order.append(self.name)

    callback1 = TestCallback('callback1')
    callback2 = TestCallback('callback2')
    manager = bg.callbacks.CallbackManager([callback1, callback2])

    metrics = manager.extract_metrics(simple_system, simple_system)
    context = bg.callbacks.CallbackContext(
        step=0,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=metrics,
        minimizer=mock_minimizer,
        step_kwargs={},
    )

    manager.on_step_end(context)
    assert call_order == ['callback1', 'callback2']


def test_CallbackManager_all_callbacks_execute_on_early_stop(simple_system, mock_minimizer):
    """Test that all callbacks execute even if one sets stop flag (Option B)."""
    call_order = []

    class StopCallback(bg.callbacks.Callback):
        def on_step_end(self, context: bg.callbacks.CallbackContext) -> None:
            call_order.append('stop')
            self._should_stop = True

    class ContinueCallback(bg.callbacks.Callback):
        def on_step_end(self, context: bg.callbacks.CallbackContext) -> None:
            call_order.append('continue')

    stop_callback = StopCallback()
    continue_callback = ContinueCallback()
    manager = bg.callbacks.CallbackManager([stop_callback, continue_callback])

    metrics = manager.extract_metrics(simple_system, simple_system)
    context = bg.callbacks.CallbackContext(
        step=0,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=metrics,
        minimizer=mock_minimizer,
        step_kwargs={},
    )

    should_stop = manager.on_step_end(context)
    # Both callbacks should execute
    assert 'stop' in call_order
    assert 'continue' in call_order
    # But should_stop should be True
    assert should_stop is True


def test_CallbackManager_exception_handling(simple_system, mock_minimizer):
    """Test that exceptions in callbacks don't crash the manager."""

    class FailingCallback(bg.callbacks.Callback):
        def on_step_end(self, context: bg.callbacks.CallbackContext) -> None:
            raise ValueError('Test exception')

    class WorkingCallback(bg.callbacks.Callback):
        def __init__(self):
            self.called = False

        def on_step_end(self, context: bg.callbacks.CallbackContext) -> None:
            self.called = True

    failing_callback = FailingCallback()
    working_callback = WorkingCallback()
    manager = bg.callbacks.CallbackManager([failing_callback, working_callback])

    metrics = manager.extract_metrics(simple_system, simple_system)
    context = bg.callbacks.CallbackContext(
        step=0,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=metrics,
        minimizer=mock_minimizer,
        step_kwargs={},
    )

    # Should not raise exception
    manager.on_step_end(context)
    # Working callback should still have been called
    assert working_callback.called is True


# ============================================================================
# Test EarlyStopping
# ============================================================================


def test_EarlyStopping_initialization():
    """Test EarlyStopping initialization."""
    early_stop = bg.callbacks.EarlyStopping(monitor='system_energy', patience=10)
    assert early_stop.monitor == 'system_energy'
    assert early_stop.patience == 10
    assert early_stop.mode == 'min'
    assert early_stop._should_stop is False


def test_EarlyStopping_invalid_mode():
    """Test that EarlyStopping raises error for invalid mode."""
    with pytest.raises(ValueError, match='mode must be "min" or "max"'):
        bg.callbacks.EarlyStopping(monitor='system_energy', patience=10, mode='invalid')


def test_EarlyStopping_patience(simple_system, mock_minimizer):
    """Test EarlyStopping patience mechanism."""
    early_stop = bg.callbacks.EarlyStopping(monitor='system_energy', patience=2, mode='min')

    # First step: improving
    metrics1 = {'system_energy': -2.0}
    context1 = bg.callbacks.CallbackContext(
        step=0,
        system=simple_system,
        best_system=simple_system,
        new_best=True,
        metrics=metrics1,
        minimizer=mock_minimizer,
        step_kwargs={},
    )
    early_stop.on_optimization_start(context1)
    early_stop.on_step_end(context1)
    assert early_stop._best_value == -2.0
    assert early_stop._steps_since_improvement == 0
    assert early_stop._should_stop is False

    # Second step: not improving
    metrics2 = {'system_energy': -1.5}
    context2 = bg.callbacks.CallbackContext(
        step=1,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=metrics2,
        minimizer=mock_minimizer,
        step_kwargs={},
    )
    early_stop.on_step_end(context2)
    assert early_stop._steps_since_improvement == 1
    assert early_stop._should_stop is False

    # Third step: still not improving (patience exceeded)
    early_stop.on_step_end(context2)
    assert early_stop._steps_since_improvement == 2
    assert early_stop._should_stop is True


def test_early_stopping_with_embeddings_similarity(simple_system, mock_minimizer):
    """Test early stopping with state-level metric (embeddings_similarity) using monkeypatching."""
    # Monitor state-level metric: "test_state:embeddings_similarity"
    # Use mode="max" since higher similarity is better
    early_stop = bg.callbacks.EarlyStopping(
        monitor='test_state/embeddings_similarity',
        patience=3,
        mode='max',
        min_delta=0.01,
    )

    # Initialize
    initial_metrics = {'test_state/embeddings_similarity': 0.5}
    initial_context = bg.callbacks.CallbackContext(
        step=0,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=initial_metrics,
        minimizer=mock_minimizer,
        step_kwargs={},
    )
    early_stop.on_optimization_start(initial_context)
    early_stop.on_step_end(initial_context)
    assert early_stop._best_value == 0.5
    assert early_stop._steps_since_improvement == 0
    assert early_stop._should_stop is False

    # Step 1: Improve (0.5 -> 0.6)
    metrics1 = {'test_state/embeddings_similarity': 0.6}
    context1 = bg.callbacks.CallbackContext(
        step=1,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=metrics1,
        minimizer=mock_minimizer,
        step_kwargs={},
    )
    early_stop.on_step_end(context1)
    assert early_stop._best_value == 0.6
    assert early_stop._steps_since_improvement == 0
    assert early_stop._should_stop is False

    # Step 2: Plateau (0.6 -> 0.59, not enough improvement)
    metrics2 = {'test_state/embeddings_similarity': 0.59}
    context2 = bg.callbacks.CallbackContext(
        step=2,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=metrics2,
        minimizer=mock_minimizer,
        step_kwargs={},
    )
    early_stop.on_step_end(context2)
    assert early_stop._best_value == 0.6  # Best unchanged
    assert early_stop._steps_since_improvement == 1
    assert early_stop._should_stop is False

    # Step 3: Still plateau
    early_stop.on_step_end(context2)
    assert early_stop._steps_since_improvement == 2
    assert early_stop._should_stop is False

    # Step 4: Still plateau (patience exceeded)
    early_stop.on_step_end(context2)
    assert early_stop._steps_since_improvement == 3
    assert early_stop._should_stop is True


def test_early_stopping_plateau_detection(simple_system, mock_minimizer):
    """Test that early stopping correctly detects when metric plateaus (no improvement)."""
    early_stop = bg.callbacks.EarlyStopping(monitor='system_energy', patience=2, mode='min')

    # Initialize
    metrics0 = {'system_energy': -2.0}
    context0 = bg.callbacks.CallbackContext(
        step=0,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=metrics0,
        minimizer=mock_minimizer,
        step_kwargs={},
    )
    early_stop.on_optimization_start(context0)
    early_stop.on_step_end(context0)
    assert early_stop._best_value == -2.0
    assert early_stop._steps_since_improvement == 0

    # Simulate plateau: metric stays the same
    plateau_metrics = {'system_energy': -1.8}  # Worse than best, so no improvement
    plateau_context = bg.callbacks.CallbackContext(
        step=1,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=plateau_metrics,
        minimizer=mock_minimizer,
        step_kwargs={},
    )

    # Step 1: No improvement
    early_stop.on_step_end(plateau_context)
    assert early_stop._best_value == -2.0
    assert early_stop._steps_since_improvement == 1
    assert early_stop._should_stop is False

    # Step 2: Still no improvement (patience exceeded)
    early_stop.on_step_end(plateau_context)
    assert early_stop._best_value == -2.0
    assert early_stop._steps_since_improvement == 2
    assert early_stop._should_stop is True


def test_early_stopping_reset_on_improvement(simple_system, mock_minimizer):
    """Test that patience resets when metric improves after a plateau."""
    early_stop = bg.callbacks.EarlyStopping(monitor='system_energy', patience=3, mode='min')

    # Initialize
    metrics0 = {'system_energy': -2.0}
    context0 = bg.callbacks.CallbackContext(
        step=0,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=metrics0,
        minimizer=mock_minimizer,
        step_kwargs={},
    )
    early_stop.on_optimization_start(context0)
    early_stop.on_step_end(context0)
    assert early_stop._best_value == -2.0
    assert early_stop._steps_since_improvement == 0

    # Step 1-2: Plateau (no improvement)
    plateau_metrics = {'system_energy': -1.5}  # Worse than best
    plateau_context = bg.callbacks.CallbackContext(
        step=1,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=plateau_metrics,
        minimizer=mock_minimizer,
        step_kwargs={},
    )
    early_stop.on_step_end(plateau_context)
    assert early_stop._steps_since_improvement == 1
    early_stop.on_step_end(plateau_context)
    assert early_stop._steps_since_improvement == 2
    assert early_stop._should_stop is False

    # Step 3: Improvement! (should reset patience)
    improved_metrics = {'system_energy': -2.5}  # Better than best
    improved_context = bg.callbacks.CallbackContext(
        step=3,
        system=simple_system,
        best_system=simple_system,
        new_best=False,
        metrics=improved_metrics,
        minimizer=mock_minimizer,
        step_kwargs={},
    )
    early_stop.on_step_end(improved_context)
    assert early_stop._best_value == -2.5  # Updated best
    assert early_stop._steps_since_improvement == 0  # Reset!
    assert early_stop._should_stop is False

    # Step 4-5: Plateau again
    early_stop.on_step_end(plateau_context)
    assert early_stop._steps_since_improvement == 1
    early_stop.on_step_end(plateau_context)
    assert early_stop._steps_since_improvement == 2
    assert early_stop._should_stop is False

    # Step 6: Still plateau (patience exceeded)
    early_stop.on_step_end(plateau_context)
    assert early_stop._steps_since_improvement == 3
    assert early_stop._should_stop is True


# ============================================================================
# Test WandBLogger
# ============================================================================


def test_WandBLogger_initialization(monkeypatch):
    """Test WandBLogger initialization."""
    # Unset WANDB_PROJECT to test constructor project
    monkeypatch.delenv('WANDB_PROJECT', raising=False)
    wandb_logger = bg.callbacks.WandBLogger(project='bagel-test')
    assert wandb_logger.project == 'bagel-test'
    assert wandb_logger.config == {}


def test_WandBLogger_with_wandb_mocked(simple_system, mock_minimizer, monkeypatch):
    """Test WandBLogger when wandb is installed and mocked."""
    from unittest.mock import MagicMock, patch

    # Unset WANDB_PROJECT to test constructor project
    monkeypatch.delenv('WANDB_PROJECT', raising=False)

    # Mock wandb module
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_run.name = 'test-run-123'
    mock_wandb.init.return_value = mock_run

    # Patch wandb in the callbacks module
    with patch('bagel.callbacks.wandb', mock_wandb):
        wandb_logger = bg.callbacks.WandBLogger(project='bagel-test', config={'n_steps': 100})

        metrics = {'system_energy': -1.5, 'best_system_energy': -1.5}
        context = bg.callbacks.CallbackContext(
            step=0,
            system=simple_system,
            best_system=simple_system,
            new_best=False,
            metrics=metrics,
            minimizer=mock_minimizer,
            step_kwargs={},
        )

        # Test initialization
        wandb_logger.on_optimization_start(context)
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs['project'] == 'bagel-test'
        # Should use minimizer's experiment_name
        assert call_kwargs['name'] == mock_minimizer.experiment_name
        assert call_kwargs['config']['n_steps'] == 100

        # Test logging
        wandb_logger.on_step_end(context)
        mock_wandb.log.assert_called_once_with(metrics, step=0)

        # Test finish
        wandb_logger.on_optimization_end(context)
        mock_wandb.finish.assert_called_once()


def test_WandBLogger_env_var_override(simple_system, mock_minimizer, monkeypatch):
    """Test that WANDB_PROJECT env var overrides constructor project."""
    from unittest.mock import MagicMock, patch

    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run

    # Set environment variable
    monkeypatch.setenv('WANDB_PROJECT', 'env-project')

    with patch('bagel.callbacks.wandb', mock_wandb):
        wandb_logger = bg.callbacks.WandBLogger(project='constructor-project')

        context = bg.callbacks.CallbackContext(
            step=0,
            system=simple_system,
            best_system=simple_system,
            new_best=False,
            metrics={},
            minimizer=mock_minimizer,
            step_kwargs={},
        )

        wandb_logger.on_optimization_start(context)
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs['project'] == 'env-project'  # Should use env var


def test_WandBLogger_init_failure(simple_system, mock_minimizer, monkeypatch):
    """Test WandBLogger handles wandb.init() failure gracefully."""
    from unittest.mock import MagicMock, patch

    mock_wandb = MagicMock()
    mock_wandb.init.side_effect = Exception('API key invalid')

    with patch('bagel.callbacks.wandb', mock_wandb):
        wandb_logger = bg.callbacks.WandBLogger(project='bagel-test')

        context = bg.callbacks.CallbackContext(
            step=0,
            system=simple_system,
            best_system=simple_system,
            new_best=False,
            metrics={'system_energy': -1.5},
            minimizer=mock_minimizer,
            step_kwargs={},
        )

        # Should not crash
        wandb_logger.on_optimization_start(context)
        assert wandb_logger._run is None

        # Subsequent calls should not crash
        wandb_logger.on_step_end(context)
        wandb_logger.on_optimization_end(context)


# ============================================================================
# Integration Tests
# ============================================================================


def test_callbacks_integrated_into_minimizer(temp_log_path, simple_system):
    """Test that callbacks are integrated into MonteCarloMinimizer."""
    callback_calls = []

    class TestCallback(bg.callbacks.Callback):
        def on_optimization_start(self, context: bg.callbacks.CallbackContext) -> None:
            callback_calls.append('start')

        def on_step_end(self, context: bg.callbacks.CallbackContext) -> None:
            callback_calls.append(f'step_{context.step}')

        def on_optimization_end(self, context: bg.callbacks.CallbackContext) -> None:
            callback_calls.append('end')

    callback = TestCallback()
    mock_mutator = Mock()

    def mock_one_step(system):
        """Mock mutator that returns a copy with calculated energy."""
        copied_system = system.__copy__()
        # Ensure energy is calculated on the copied system
        if copied_system.total_energy is None:
            copied_system.get_total_energy()
        return copied_system, Mock()

    mock_mutator.one_step = mock_one_step

    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=mock_mutator,
        temperature=1.0,
        n_steps=2,
        experiment_name='test',
        log_path=temp_log_path,
        callbacks=[callback],
    )

    minimizer.minimize_system(simple_system)

    # Check that callbacks were called
    assert 'start' in callback_calls
    assert 'step_1' in callback_calls
    assert 'step_2' in callback_calls
    assert 'end' in callback_calls


def test_early_stopping_stops_optimization(temp_log_path, simple_system):
    """Test that early stopping actually stops optimization."""
    step_count = []

    class StepCounter(bg.callbacks.Callback):
        def on_step_end(self, context: bg.callbacks.CallbackContext) -> None:
            step_count.append(context.step)

    early_stop = bg.callbacks.EarlyStopping(monitor='system_energy', patience=1, mode='min')
    step_counter = StepCounter()
    mock_mutator = Mock()

    def mock_one_step(system):
        """Mock mutator that returns a copy with calculated energy."""
        copied_system = system.__copy__()
        # Ensure energy is calculated on the copied system
        if copied_system.total_energy is None:
            copied_system.get_total_energy()
        return copied_system, Mock()

    mock_mutator.one_step = mock_one_step

    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=mock_mutator,
        temperature=1.0,
        n_steps=10,  # Would normally run 10 steps
        experiment_name='test',
        log_path=temp_log_path,
        callbacks=[early_stop, step_counter],
    )

    minimizer.minimize_system(simple_system)

    # Should stop early
    assert len(step_count) < 10


def test_no_callbacks_backward_compatibility(temp_log_path, simple_system):
    """Test that minimizer works with no callbacks (backward compatibility)."""
    mock_mutator = Mock()

    def mock_one_step(system):
        """Mock mutator that returns a copy with calculated energy."""
        copied_system = system.__copy__()
        # Ensure energy is calculated on the copied system
        if copied_system.total_energy is None:
            copied_system.get_total_energy()
        return copied_system, Mock()

    mock_mutator.one_step = mock_one_step

    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=mock_mutator,
        temperature=1.0,
        n_steps=2,
        experiment_name='test',
        log_path=temp_log_path,
        callbacks=None,
    )

    # Should work normally
    result = minimizer.minimize_system(simple_system)
    assert result is not None
