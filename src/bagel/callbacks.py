"""
Callback system for Minimizer, following PyTorch Lightning patterns.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from dataclasses import dataclass
from abc import ABC
from typing import Any, TYPE_CHECKING
import logging
import os
import pathlib as pl

from .system import System
from .minimizer import Minimizer
from .oracles.folding import FoldingOracle, FoldingResult

if TYPE_CHECKING:
    import wandb
else:
    try:
        import wandb
    except ImportError:
        wandb = None

logger = logging.getLogger(__name__)


@dataclass
class CallbackContext:
    """
    Container for all information passed to callbacks during optimization.

    Parameters
    ----------
    step : int
        Current step number. Step 0 = before optimization; steps 1..N = after each optimization step (1-indexed).
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
        - '{state.name}/{energy_name}': Individual energy term values
        - '{state.name}/state_energy': Total energy per state
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

    def extract_metrics(self, system: System, best_system: System) -> dict[str, float]:
        """
        Extract energy metrics from system and best_system.

        Extracts:
        - System-level energies: 'system_energy', 'best_system_energy'
        - State-level energy terms: '{state.name}/{energy_name}'
        - State-level total energies: '{state.name}/state_energy'

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

        # Ensure energies are calculated and get system-level energies
        system_energy = system.get_total_energy()
        if system_energy is None:
            raise ValueError('System energy must be calculated. Call get_total_energy() first.')
        metrics['system_energy'] = system_energy

        best_system_energy = best_system.get_total_energy()
        if best_system_energy is None:
            raise ValueError('Best system energy must be calculated. Call get_total_energy() first.')
        metrics['best_system_energy'] = best_system_energy

        # Extract state-level metrics from current system
        for state in system.states:
            # Skip states with no energy terms
            if not state.energy_terms:
                continue

            # Extract individual energy terms using public API
            energy_terms = state.energy_term_values
            for energy_name, energy_value in energy_terms.items():
                metrics[f'{state.name}/{energy_name}'] = energy_value

            # Extract state total energy using property
            state_energy = state.energy
            metrics[f'{state.name}/state_energy'] = state_energy

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
                logger.error(
                    f'Error in callback {callback.__class__.__name__}.on_optimization_start: {e}', exc_info=True
                )

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


class DefaultLogger(Callback):
    """
    Default file-based logger for scalar and sequence data.

    This logger writes:
    - energies.csv with step, per-state energy terms, per-state total energies, and system_energy
    - per-state FASTA files with sequences
    - per-state mask FASTA files with mutability masks

    It does not write structures; use FoldingLogger for structure outputs.

    Output layout
    -------------
    Files are written under the Minimizer.log_path directory in two subtrees:
    - <log_path>/current
    - <log_path>/best

    Parameters
    ----------
    log_interval : int
        Interval (in steps) at which to log the current system.
    """

    def __init__(self, log_interval: int) -> None:
        if log_interval <= 0:
            raise ValueError(f'log_interval must be a positive integer, got {log_interval}')
        self.log_interval = log_interval

    def _ensure_dirs(self, base_path: pl.Path) -> tuple[pl.Path, pl.Path]:
        current_path = base_path / 'current'
        best_path = base_path / 'best'
        current_path.mkdir(parents=True, exist_ok=True)
        best_path.mkdir(parents=True, exist_ok=True)
        return current_path, best_path

    def _dump_system(self, step: int, system: System, path: pl.Path) -> None:
        """
        Core implementation of scalar/sequence logging.
        """
        total_energy = system.get_total_energy()
        if total_energy is None:
            raise ValueError('System energy not calculated. Call get_total_energy() first.')

        path.mkdir(parents=True, exist_ok=True)

        energies: dict[str, int | float] = {'step': step}
        for state in system.states:
            # Skip states with no energy terms
            if not state.energy_terms:
                continue

            energy_terms = state.energy_term_values
            for energy_name, energy_value in energy_terms.items():
                energies[f'{state.name}/{energy_name}'] = energy_value

            state_energy = state.energy
            energies[f'{state.name}/state_energy'] = state_energy

            with open(path / f'{state.name}.fasta', mode='a') as file:
                file.write(f'>{step}\n')
                file.write(f'{":".join(state.total_sequence)}\n')

            mask_per_chain = [
                ''.join(['M' if residue.mutable else 'I' for residue in chain.residues]) for chain in state.chains
            ]
            with open(path / f'{state.name}.mask.fasta', mode='a') as mask_file:
                mask_file.write(f'>{step}\n')
                mask_file.write(f'{":".join(mask_per_chain)}\n')

        energies['system_energy'] = total_energy

        energies_path = path / 'energies.csv'
        write_header = step == 0 and not energies_path.exists()
        with open(energies_path, mode='a') as file:
            if write_header:
                file.write(','.join(energies.keys()) + '\n')
            file.write(','.join(str(energy) for energy in energies.values()) + '\n')

    def on_optimization_start(self, context: CallbackContext) -> None:
        base_path: pl.Path = context.minimizer.log_path
        base_path.mkdir(parents=True, exist_ok=True)

        # Write configuration once at the top level
        context.system.dump_config(base_path)

        current_path, best_path = self._ensure_dirs(base_path)
        # Log initial step 0 without structures to mirror prior behavior
        self._dump_system(step=0, system=context.system, path=current_path)
        self._dump_system(step=0, system=context.best_system, path=best_path)

    def on_step_end(self, context: CallbackContext) -> None:
        base_path: pl.Path = context.minimizer.log_path
        current_path, best_path = self._ensure_dirs(base_path)

        if context.step % self.log_interval == 0:
            self._dump_system(step=context.step, system=context.system, path=current_path)

        if context.new_best:
            self._dump_system(step=context.step, system=context.best_system, path=best_path)


class FoldingLogger(Callback):
    """
    Logger that saves CIF structures and oracle attributes for a specific FoldingOracle instance.

    Parameters
    ----------
    folding_oracle : FoldingOracle
        The FoldingOracle instance to log (e.g. an ESMFold instance).
    log_interval : int
        Interval (in steps) at which to log structures for the current system.
    """

    def __init__(self, folding_oracle: FoldingOracle, log_interval: int) -> None:
        if log_interval <= 0:
            raise ValueError(f'log_interval must be a positive integer, got {log_interval}')
        if not isinstance(folding_oracle, FoldingOracle):
            raise TypeError(f'folding_oracle must be a FoldingOracle instance, got {type(folding_oracle)}')
        self.folding_oracle = folding_oracle
        self.log_interval = log_interval

    def _ensure_folding_dirs(self, base_path: pl.Path) -> tuple[pl.Path, pl.Path]:
        """Ensure current/folding and best/folding directories exist."""
        current_folding_path = base_path / 'current' / 'folding'
        best_folding_path = base_path / 'best' / 'folding'
        current_folding_path.mkdir(parents=True, exist_ok=True)
        best_folding_path.mkdir(parents=True, exist_ok=True)
        return current_folding_path, best_folding_path

    def _dump_folding_results_for_system(self, step: int, system: System, folding_path: pl.Path) -> None:
        """Save CIF files and oracle attributes to the specified folding directory."""
        for state in system.states:
            for oracle, oracle_result in state._oracles_result.items():
                if oracle is not self.folding_oracle:
                    continue
                if not isinstance(oracle_result, FoldingResult):
                    continue

                oracle_name = type(oracle).__name__
                cif_path = folding_path / f'{state.name}_{oracle_name}_{step}.cif'
                oracle_result.to_cif(cif_path)
                oracle_result.save_attributes(folding_path / f'{state.name}_{oracle_name}_{step}')

    def on_optimization_start(self, context: CallbackContext) -> None:
        base_path: pl.Path = context.minimizer.log_path
        current_folding_path, best_folding_path = self._ensure_folding_dirs(base_path)
        # Log step 0 to both current and best
        self._dump_folding_results_for_system(step=0, system=context.system, folding_path=current_folding_path)
        self._dump_folding_results_for_system(step=0, system=context.best_system, folding_path=best_folding_path)

    def on_step_end(self, context: CallbackContext) -> None:
        base_path: pl.Path = context.minimizer.log_path
        current_folding_path, best_folding_path = self._ensure_folding_dirs(base_path)

        if context.step % self.log_interval == 0:
            self._dump_folding_results_for_system(
                step=context.step, system=context.system, folding_path=current_folding_path
            )

        if context.new_best:
            self._dump_folding_results_for_system(
                step=context.step, system=context.best_system, folding_path=best_folding_path
            )


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

    def __init__(self, monitor: str, patience: int, min_delta: float = 0.0, mode: str = 'min') -> None:
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

    Uses environment variables for authentication:
    - WANDB_API_KEY: Your WandB API key (required for online logging)
    - WANDB_ENTITY: Username or team name (optional)
    - WANDB_PROJECT: Project name (optional, overrides constructor project)

    If wandb is not installed, this callback will log warnings but not crash.

    Parameters
    ----------
    project : str
        WandB project name. Can be overridden by WANDB_PROJECT env var.
    config : dict[str, Any] | None, optional
        Hyperparameters and configuration to log.

    Notes
    -----
    The WandB run name is automatically set from the minimizer's experiment_name.
    This ensures consistency between the minimizer's experiment name and WandB run name.

    Examples
    --------
    >>> wandb_logger = WandBLogger(
    ...     project="protein-design",
    ...     config={"temperature": 1.0, "n_steps": 1000}
    ... )
    """

    def __init__(self, project: str, config: dict[str, Any] | None = None) -> None:
        if wandb is None:
            logger.warning(
                'wandb is not installed. WandBLogger will not log metrics. '
                'Install with: pip install wandb or uv sync --extra wandb'
            )

        self.project = os.environ.get('WANDB_PROJECT', project)
        self.config = config if config is not None else {}
        self._run: Any = None

    def _resolve_wandb_dir(self) -> pl.Path:
        """
        Resolve WandB directory following bagel's cache pattern.

        Precedence:
        1) WANDB_DIR environment variable if set
        2) XDG_CACHE_HOME/bagel/.wandb if XDG_CACHE_HOME is defined
        3) ~/.cache/bagel/.wandb otherwise

        Returns
        -------
        pl.Path
            Absolute path to the WandB directory.
        """
        if os.environ.get('WANDB_DIR'):
            resolved = pl.Path(os.environ['WANDB_DIR']).expanduser().resolve()
        else:
            xdg_cache_home = os.getenv('XDG_CACHE_HOME')
            if xdg_cache_home:
                base_cache_dir = pl.Path(xdg_cache_home).expanduser().resolve()
            else:
                base_cache_dir = pl.Path.home() / '.cache'
            resolved = (base_cache_dir / 'bagel' / '.wandb').resolve()

        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def on_optimization_start(self, context: CallbackContext) -> None:
        """
        Initialize WandB run.

        Uses environment variables for authentication.
        """
        if wandb is None:
            return

        try:
            # Merge config with minimizer info if available
            run_config = {**self.config}
            if hasattr(context.minimizer, 'temperature'):
                run_config.setdefault('temperature', context.minimizer.temperature)

            init_kwargs: dict[str, Any] = {
                'project': self.project,
                'config': run_config,
                'dir': str(self._resolve_wandb_dir()),
            }

            # Use minimizer's experiment_name as the WandB run name
            if hasattr(context.minimizer, 'experiment_name'):
                init_kwargs['name'] = context.minimizer.experiment_name

            entity = os.environ.get('WANDB_ENTITY')
            if entity:
                init_kwargs['entity'] = entity

            self._run = wandb.init(**init_kwargs)
            logger.info(f'WandBLogger: Initialized run "{self._run.name}" in project "{self.project}"')
        except Exception as e:
            logger.error(f'WandBLogger: Failed to initialize WandB run: {e}', exc_info=True)
            self._run = None

    def on_step_end(self, context: CallbackContext) -> None:
        """
        Log metrics to WandB.

        Logs all metrics from context.metrics dictionary.
        """
        if wandb is None or self._run is None:
            return

        try:
            wandb.log(context.metrics, step=context.step)
        except Exception as e:
            logger.error(f'WandBLogger: Failed to log metrics: {e}', exc_info=True)

    def on_optimization_end(self, context: CallbackContext) -> None:
        """
        Finish WandB run.
        """
        if wandb is None or self._run is None:
            return

        try:
            wandb.finish()
            logger.info('WandBLogger: Finished run')
        except Exception as e:
            logger.error(f'WandBLogger: Failed to finish WandB run: {e}', exc_info=True)
        finally:
            self._run = None
