"""
Callback system for Minimizer, following PyTorch Lightning patterns.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any
import logging

from .system import System
from .minimizer import Minimizer

logger = logging.getLogger(__name__)


@dataclass
class CallbackContext:
    """
    Container for all information passed to callbacks during optimization.

    Parameters
    ----------
    step : int
        Current step number (0-indexed).
    system : System
        Current system state after the step.
    best_system : System
        Best system found so far during optimization.
    new_best : bool
        Whether this step found a new best system.
    metrics : dict[str, float]
        Extracted energy metrics including:
        - 'system_energy': Total energy of current system
        - 'best_system_energy': Total energy of best system
        - '{state.name}:{energy_name}': Individual energy term values
        - '{state.name}:state_energy': Total energy per state
    minimizer : Minimizer
        Reference to the minimizer instance running the optimization.
    step_kwargs : dict[str, Any]
        Step-specific keyword arguments (e.g., temperature, accept).
    """

    step: int
    system: System
    best_system: System
    new_best: bool
    metrics: dict[str, float]
    minimizer: Minimizer
    step_kwargs: dict[str, Any]


class Callback(ABC):
    """
    Abstract base class for callbacks in the optimization process.

    Callbacks can be used to monitor, log, or modify the optimization behavior.
    All hook methods have default no-op implementations, so subclasses only need
    to override the methods they care about.

    Examples
    --------
    >>> class MyCallback(Callback):
    ...     def on_step_end(self, context: CallbackContext) -> None:
    ...         print(f"Step {context.step}: energy = {context.metrics['system_energy']}")
    """

    def on_optimization_start(self, context: CallbackContext) -> None:
        """
        Called once before the optimization loop begins.

        Parameters
        ----------
        context : CallbackContext
            Context containing initial system state and minimizer reference.
        """
        pass

    def on_step_end(self, context: CallbackContext) -> None:
        """
        Called after each optimization step.

        This is the main hook for monitoring and logging during optimization.
        All callbacks execute even if early stopping is triggered.

        Parameters
        ----------
        context : CallbackContext
            Context containing current step information, systems, and metrics.
        """
        pass

    def on_optimization_end(self, context: CallbackContext) -> None:
        """
        Called once after the optimization loop completes.

        Parameters
        ----------
        context : CallbackContext
            Context containing final system state and minimizer reference.
        """
        pass


class CallbackManager:
    """
    Manages the execution of callbacks during optimization.

    The CallbackManager coordinates callback execution, extracts metrics from
    systems, and handles early stopping signals.

    Parameters
    ----------
    callbacks : list[Callback] | None, optional
        List of callbacks to execute. If None, an empty list is used.

    Examples
    --------
    >>> from bagel.callbacks import EarlyStopping, WandBLogger
    >>> callbacks = [EarlyStopping(monitor="best_system_energy", patience=100)]
    >>> manager = CallbackManager(callbacks)
    """

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks: list[Callback] = callbacks if callbacks is not None else []
        self._should_stop: bool = False

    def add_callback(self, callback: Callback) -> None:
        """
        Add a callback to the manager.

        Parameters
        ----------
        callback : Callback
            Callback instance to add.
        """
        self.callbacks.append(callback)

    def _extract_metrics(self, system: System, best_system: System) -> dict[str, float]:
        """
        Extract energy metrics from system and best_system.

        Extracts:
        - System-level energies: 'system_energy', 'best_system_energy'
        - State-level energy terms: '{state.name}:{energy_name}'
        - State-level total energies: '{state.name}:state_energy'

        Parameters
        ----------
        system : System
            Current system state.
        best_system : System
            Best system found so far.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to their values.
        """
        metrics: dict[str, float] = {}

        # Ensure energies are calculated
        if system.total_energy is None:
            system.get_total_energy()
        if best_system.total_energy is None:
            best_system.get_total_energy()

        # System-level energies
        metrics['system_energy'] = system.total_energy
        metrics['best_system_energy'] = best_system.total_energy

        # Extract state-level metrics from current system
        for state in system.states:
            # Ensure state energy is calculated
            if state._energy is None:
                state.get_energy()

            # Extract individual energy terms
            for energy_name, energy_value in state._energy_terms_value.items():
                metrics[f'{state.name}:{energy_name}'] = energy_value

            # Extract state total energy
            if state._energy is not None:
                metrics[f'{state.name}:state_energy'] = state._energy

        return metrics

    def on_optimization_start(self, context: CallbackContext) -> None:
        """
        Execute on_optimization_start for all registered callbacks.

        Parameters
        ----------
        context : CallbackContext
            Context containing initial system state.
        """
        self._should_stop = False
        for callback in self.callbacks:
            try:
                callback.on_optimization_start(context)
            except Exception as e:
                logger.error(f'Error in callback {callback.__class__.__name__}.on_optimization_start: {e}', exc_info=True)

    def on_step_end(self, context: CallbackContext) -> bool:
        """
        Execute on_step_end for all registered callbacks.

        All callbacks execute even if one sets the stop flag (Option B from design).
        This ensures logging callbacks can complete their work.

        Parameters
        ----------
        context : CallbackContext
            Context containing current step information.

        Returns
        -------
        bool
            True if optimization should stop early, False otherwise.
        """
        self._should_stop = False
        for callback in self.callbacks:
            try:
                callback.on_step_end(context)
                # Check if this callback set the stop flag
                if hasattr(callback, '_should_stop') and callback._should_stop:
                    self._should_stop = True
            except Exception as e:
                logger.error(f'Error in callback {callback.__class__.__name__}.on_step_end: {e}', exc_info=True)

        return self._should_stop

    def on_optimization_end(self, context: CallbackContext) -> None:
        """
        Execute on_optimization_end for all registered callbacks.

        Parameters
        ----------
        context : CallbackContext
            Context containing final system state.
        """
        for callback in self.callbacks:
            try:
                callback.on_optimization_end(context)
            except Exception as e:
                logger.error(f'Error in callback {callback.__class__.__name__}.on_optimization_end: {e}', exc_info=True)


class EarlyStopping(Callback):
    """
    Callback that monitors a metric and signals early stopping when it stops improving.

    This is a dummy implementation that tracks state but doesn't actually stop
    the optimization (that's handled by CallbackManager). The callback sets
    a `_should_stop` flag when patience is exceeded.

    Parameters
    ----------
    monitor : str
        Metric name to monitor (e.g., "system_energy", "best_system_energy").
    patience : int
        Number of steps with no improvement before stopping.
    min_delta : float, default=0.0
        Minimum change in the monitored metric to be considered an improvement.
    mode : str, default="min"
        Whether lower ("min") or higher ("max") values are better.

    Examples
    --------
    >>> early_stop = EarlyStopping(
    ...     monitor="best_system_energy",
    ...     patience=100,
    ...     mode="min"
    ... )
    """

    def __init__(
        self, monitor: str, patience: int, min_delta: float = 0.0, mode: str = 'min'
    ) -> None:
        if mode not in ('min', 'max'):
            raise ValueError(f'mode must be "min" or "max", got {mode}')

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._should_stop = False
        self._best_value: float | None = None
        self._steps_since_improvement = 0

    def on_optimization_start(self, context: CallbackContext) -> None:
        """Reset tracking state at the start of optimization."""
        self._should_stop = False
        self._best_value = None
        self._steps_since_improvement = 0

    def on_step_end(self, context: CallbackContext) -> None:
        """
        Check if monitored metric has improved and update tracking state.

        Sets `_should_stop` to True when patience is exceeded.
        """
        if self.monitor not in context.metrics:
            logger.warning(
                f'EarlyStopping monitoring "{self.monitor}" but metric not found in context. '
                f'Available metrics: {list(context.metrics.keys())}'
            )
            return

        current_value = context.metrics[self.monitor]

        # Initialize best value on first step
        if self._best_value is None:
            self._best_value = current_value
            self._steps_since_improvement = 0
            return

        # Check if current value is better
        is_better = False
        if self.mode == 'min':
            is_better = current_value < (self._best_value - self.min_delta)
        else:  # mode == 'max'
            is_better = current_value > (self._best_value + self.min_delta)

        if is_better:
            self._best_value = current_value
            self._steps_since_improvement = 0
        else:
            self._steps_since_improvement += 1

        # Check if patience exceeded
        if self._steps_since_improvement >= self.patience:
            self._should_stop = True
            logger.info(
                f'EarlyStopping triggered: {self.monitor} has not improved for {self.patience} steps. '
                f'Best value: {self._best_value}, Current value: {current_value}'
            )


class WandBLogger(Callback):
    """
    Callback for logging metrics to Weights & Biases.

    This is a dummy implementation that stores parameters but doesn't actually
    log to WandB yet. The actual WandB integration will be implemented later.

    Parameters
    ----------
    project : str
        WandB project name.
    name : str | None, optional
        Run name. If None, a default name will be used.
    config : dict[str, Any] | None, optional
        Hyperparameters and configuration to log.

    Examples
    --------
    >>> wandb_logger = WandBLogger(
    ...     project="protein-design",
    ...     name="experiment-1",
    ...     config={"temperature": 1.0, "n_steps": 1000}
    ... )
    """

    def __init__(
        self, project: str, name: str | None = None, config: dict[str, Any] | None = None
    ) -> None:
        self.project = project
        self.name = name
        self.config = config if config is not None else {}

    def on_optimization_start(self, context: CallbackContext) -> None:
        """
        Initialize WandB run (dummy implementation).

        In the full implementation, this would call wandb.init().
        """
        logger.debug(f'WandBLogger: Would initialize run for project "{self.project}", name "{self.name}"')

    def on_step_end(self, context: CallbackContext) -> None:
        """
        Log metrics to WandB (dummy implementation).

        In the full implementation, this would call wandb.log() with all metrics.
        """
        logger.debug(f'WandBLogger: Would log metrics at step {context.step}')

    def on_optimization_end(self, context: CallbackContext) -> None:
        """
        Finish WandB run (dummy implementation).

        In the full implementation, this would call wandb.finish().
        """
        logger.debug(f'WandBLogger: Would finish run')

