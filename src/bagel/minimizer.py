"""
Standard template and objects for energy minimization logic.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

import pathlib as pl
from .system import System
from .mutation import MutationProtocol
from abc import ABC, abstractmethod
from typing import Callable, Any
import numpy as np
import inspect
import csv
import logging
import datetime as dt

logger = logging.getLogger(__name__)

time_stamp: Callable[[], str] = lambda: dt.datetime.now().strftime('%y%m%d_%H%M%S')


class Minimizer(ABC):
    """Base class for energy minimization logic."""

    def __init__(
        self, mutator: MutationProtocol, experiment_name: str, log_frequency: int, log_path: pl.Path | str | None
    ) -> None:
        self.mutator = mutator
        self.experiment_name = experiment_name
        self.log_frequency = log_frequency
        self.log_path: pl.Path = self.initialise_log_path(log_path)

        logger.debug(f'Logging path: {self.log_path}')
        logger.debug(f'Experiment name: {self.experiment_name}')

    def __post_init__(self) -> None:
        pass

    def initialise_log_path(self, log_path: None | str | pl.Path) -> pl.Path:
        """
        Creates folder next to the .py script run named the <self.experiment_name>. Said folder cannot already exist.
        Also copies the .py script run into that folder.
        """
        if isinstance(log_path, pl.Path):
            log_path = log_path / self.experiment_name
        elif isinstance(log_path, str):
            log_path = pl.Path(log_path) / self.experiment_name
        elif log_path is None:
            executed_py_file_path = inspect.stack()[-1][1]
            log_path = pl.Path(executed_py_file_path).resolve().parent / self.experiment_name
        else:
            raise ValueError(f'log_path must be a Path or str, not {type(log_path)}')
        assert isinstance(log_path, pl.Path), f'log_path must be a Path, not {type(log_path)}'
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path

    def dump_logs(self, output_folder: pl.Path, step: int, **kwargs: Any) -> None:
        """Dumps logs to CSV file."""
        output_file = output_folder / f'optimization.log'
        file_exists = output_file.exists()

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                headers = ['step'] + list(kwargs.keys())
                writer.writerow(headers)
            row = [step] + list(kwargs.values())
            writer.writerow(row)

    def log_initial_system(self, system: System, best_system: System) -> None:
        """Logs initial system state."""
        assert isinstance(self.log_path, pl.Path), f'log_path must be a Path, not {type(self.log_path)}'
        system.dump_config(self.log_path)
        self.log_step(-1, system, best_system, False)

    def log_step(self, step: int, system: System, best_system: System, new_best: bool, **kwargs: Any) -> None:
        """Logs step information."""
        real_step = step + 1
        logger.info(f'Step={real_step} - ' + ' - '.join(f'{k}={v}' for k, v in kwargs.items()))
        assert isinstance(self.log_path, pl.Path), f'log_path must be a Path, not {type(self.log_path)}'
        if real_step > 0:
            self.dump_logs(self.log_path, real_step, **kwargs)
        system.dump_logs(real_step, self.log_path / 'current', save_structure=real_step % self.log_frequency == 0)
        best_system.dump_logs(real_step, self.log_path / 'best', save_structure=new_best)

    @abstractmethod
    def minimize_system(self, system: System) -> System:
        """
        Implement a protocol to fully minimize the loss function defined for the system.

        Parameters
        ----------
        system : System
            initial state of the System

        Returns
        -------
        system : System
            System at the end of minimization
        """
        raise NotImplementedError('This method should be implemented by the subclass')


class MonteCarloMinimizer(Minimizer):
    """Base class for Monte Carlo based minimization methods."""

    def __init__(
        self,
        mutator: MutationProtocol,
        temperature: float | list[float] | np.ndarray[Any, np.dtype[np.number]],
        n_steps: int,
        acceptance_criterion: str = 'metropolis',
        experiment_name: str | None = None,
        log_frequency: int = 100,
        preserve_best_system_every_n_steps: int | None = None,
        log_path: pl.Path | str | None = None,
    ) -> None:
        if experiment_name is None:
            experiment_name = f'mc_minimizer_{time_stamp()}'

        self.temperature_schedule: np.ndarray[Any, np.dtype[np.number]]

        if isinstance(temperature, float):
            self.temperature_schedule = np.full(shape=n_steps, fill_value=temperature)
        elif isinstance(temperature, list) or isinstance(temperature, np.ndarray):
            self.temperature_schedule = np.array(temperature)
            if len(self.temperature_schedule) != n_steps:
                raise ValueError(
                    f'temperature must be a float or a list/array of floats with length {n_steps}, not {len(self.temperature_schedule)}'
                )
        else:
            raise ValueError(f'temperature must be a float or a list/array of floats, not {type(temperature)}')

        self.n_steps = n_steps
        self.preserve_best_system_every_n_steps = preserve_best_system_every_n_steps
        self.acceptance_criterion = self._get_acceptance_criterion(acceptance_criterion)
        super().__init__(
            mutator=mutator, experiment_name=experiment_name, log_frequency=log_frequency, log_path=log_path
        )

    def _get_acceptance_criterion(self, name: str) -> Callable[[float, float], float]:
        """
        Get the acceptance criterion function based on name.

        Parameters
        ----------
        name : str
            Name of the acceptance criterion. Currently supported:
            - "metropolis": Metropolis criterion exp(-ΔE/T)

        Returns
        -------
        Callable[[float, float], float]
            Function that takes delta_energy and temperature and returns acceptance probability
        """
        if name == 'metropolis':
            return lambda delta_energy, temperature: float(np.exp(-delta_energy / temperature))
        else:
            raise ValueError(f'Unknown acceptance criterion: {name}')

    def _before_step(self, system: System, step: int) -> System:
        """Hook called before each Monte Carlo step."""
        return system

    def _after_step(self, system: System, best_system: System, step: int) -> System:
        """Hook called after each Monte Carlo step."""
        if self.preserve_best_system_every_n_steps is not None:
            if (step + 1) % self.preserve_best_system_every_n_steps == 0:
                logger.debug(f'Starting new cycle with best system from previous cycle')
                return best_system.__copy__()
        return system

    def minimize_one_step(self, step: int, system: System) -> tuple[System, bool]:
        """Perform one Monte Carlo step."""
        mutated_system, delta_energy = self.mutator.one_step(
            system=system.__copy__(),
            old_system=system,
        )
        acceptance_probability = self.acceptance_criterion(delta_energy, self.temperature_schedule[step])
        logger.debug(f'{delta_energy=}, {acceptance_probability=}')

        if acceptance_probability > np.random.uniform(low=0.0, high=1.0):
            return mutated_system, True
        return system, False

    def minimize_system(self, system: System) -> System:
        """Minimize system using Monte Carlo method."""
        system.get_total_energy()  # update the energy internally
        best_system = system.__copy__()
        assert system.total_energy is not None, 'Cannot start without system having a calculated energy'
        assert best_system.total_energy is not None, (
            'Cannot start without lowest energy system having a calculated energy'
        )

        self.log_initial_system(system, best_system)

        for step in range(self.n_steps):
            new_best = False
            system = self._before_step(system, step)
            system, accept = self.minimize_one_step(step, system)
            system = self._after_step(system, best_system, step)

            assert system.total_energy is not None, 'Cannot evolve system if current energy not available'

            if system.total_energy < best_system.total_energy:
                new_best = True
                best_system = system.__copy__()

            self.log_step(
                step, system, best_system, new_best, temperature=self.temperature_schedule[step], accept=accept
            )

        assert best_system.total_energy is not None, f'Best energy {best_system.total_energy} cannot be None!'
        return best_system


class SimulatedAnnealing(MonteCarloMinimizer):
    """Simulated annealing with linearly decreasing temperature schedule."""

    def __init__(
        self,
        mutator: MutationProtocol,
        initial_temperature: float,
        final_temperature: float,
        n_steps: int,
        acceptance_criterion: str = 'metropolis',
        experiment_name: str | None = None,
        log_frequency: int = 100,
        preserve_best_system_every_n_steps: int | None = None,
        log_path: pl.Path | str | None = None,
    ) -> None:
        if experiment_name is None:
            experiment_name = f'simulated_annealing_{time_stamp()}'
        super().__init__(
            mutator=mutator,
            temperature=initial_temperature,  # This will be overwritten
            n_steps=n_steps,
            acceptance_criterion=acceptance_criterion,
            experiment_name=experiment_name,
            log_frequency=log_frequency,
            preserve_best_system_every_n_steps=preserve_best_system_every_n_steps,
            log_path=log_path,
        )

        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature_schedule = np.linspace(
            start=self.initial_temperature, stop=self.final_temperature, num=self.n_steps
        )


class SimulatedTempering(MonteCarloMinimizer):
    """Simulated tempering with cycling temperature schedule."""

    def __init__(
        self,
        mutator: MutationProtocol,
        high_temperature: float,
        low_temperature: float,
        n_steps_high: int,
        n_steps_low: int,
        n_cycles: int,
        acceptance_criterion: str = 'metropolis',
        experiment_name: str | None = None,
        log_frequency: int = 100,
        preserve_best_system_every_n_steps: int | None = None,
        log_path: pl.Path | str | None = None,
    ) -> None:
        if experiment_name is None:
            experiment_name = f'simulated_tempering_{time_stamp()}'
        total_n_steps = (n_steps_low + n_steps_high) * n_cycles
        super().__init__(
            mutator=mutator,
            temperature=low_temperature,  # This will be overwritten
            n_steps=total_n_steps,
            acceptance_criterion=acceptance_criterion,
            experiment_name=experiment_name,
            log_frequency=log_frequency,
            preserve_best_system_every_n_steps=preserve_best_system_every_n_steps,
            log_path=log_path,
        )

        self.high_temperature = high_temperature
        self.low_temperature = low_temperature
        self.n_steps_high = n_steps_high
        self.n_steps_low = n_steps_low
        self.n_cycles = n_cycles

        cycle_temperatures = np.concatenate(
            [
                np.full(shape=self.n_steps_low, fill_value=self.low_temperature),
                np.full(shape=self.n_steps_high, fill_value=self.high_temperature),
            ]
        )
        self.temperature_schedule = np.tile(cycle_temperatures, reps=self.n_cycles)
