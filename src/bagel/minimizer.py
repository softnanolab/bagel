"""
Standard template and objects for energy minimization logic.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Saffar, Stefano Angioletti-Uberti
"""

import pathlib as pl
from .system import System

# from .folding import FoldingAlgorithm
from .mutation import MutationProtocol
from abc import ABC, abstractmethod
from typing import Callable, Any, NoReturn
from dataclasses import dataclass, field
import numpy as np

import inspect
import csv
import shutil
import logging

logger = logging.getLogger(__name__)

import datetime as dt

time_stamp: Callable[[], str] = lambda: dt.datetime.now().strftime('%y%m%d_%H%M%S')


class Minimizer(ABC):
    """Standard template for energy minimisation logic."""

    mutator: MutationProtocol
    experiment_name: str
    log_frequency: int
    log_path: pl.Path | str | None = None

    def __post_init__(self) -> None:
        self.log_path: pl.Path = self.initialise_log_path(self.log_path)
        logger.debug(f'Logging path: {self.log_path}')
        logger.debug(f'Experiment name: {self.experiment_name}')

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

    def minimize_one_step(self, temperature: float, system: System) -> tuple[System, bool]:
        mutated_system, delta_energy = self.mutator.one_step(
            system=system.__copy__(),
            old_system=system,
        )
        acceptance_probability = np.exp(-delta_energy / temperature)
        logger.debug(f'{delta_energy=}, {acceptance_probability=}')
        accept = False
        if acceptance_probability > np.random.uniform(low=0.0, high=1.0):
            accept = True
            return mutated_system, accept
        return system, accept

    def initialise_log_path(self, log_path: None | str | pl.Path) -> pl.Path:
        """
        Creates folder next to the .py script run named the <self.experiment_name>. Said folder cannot already exist.
        Also copies the .py script run into that folder.
        """
        if isinstance(log_path, pl.Path):
            log_path = log_path / self.experiment_name
            log_path.mkdir(parents=True, exist_ok=True)
            return log_path
        elif isinstance(log_path, str):
            log_path = pl.Path(log_path) / self.experiment_name
            log_path.mkdir(parents=True, exist_ok=True)
            return log_path
        elif log_path is None:
            executed_py_file_path = inspect.stack()[-1][1]
            log_path = pl.Path(executed_py_file_path).resolve().parent / self.experiment_name
            # TODO: let's not use this right now, as we are not fully implementing this
            # assert not log_path.is_dir(), f'file already exists at {log_path}. Select new experiment_name'
            # log_path.mkdir()
            # logger.debug(f'Saving all log files into {log_path}:')
            # shutil.copy(executed_py_file_path, log_path / 'origional_script.py')
            return log_path
        else:
            raise ValueError(f'log_path must be a Path or str, not {type(log_path)}')

    def dump_logs(self, output_folder: pl.Path, step: int, **kwargs: Any) -> None:
        output_file = output_folder / f'optimization.log'

        # Check if file exists to determine if we need to write headers
        file_exists = output_file.exists()

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write headers if file is new
            if not file_exists:
                headers = ['step'] + list(kwargs.keys())
                writer.writerow(headers)

            # Write the data row
            row = [step] + list(kwargs.values())
            writer.writerow(row)

    def logging_step(self, step: int, system: System, best_system: System, new_best: bool, **kwargs: Any) -> None:
        """Dumps logs for the minimizer, current and best systems as well as printing summary results to the terminal"""
        real_step = step + 1
        logger.info(f'Step={real_step} - ' + ' - '.join(f'{k}={v}' for k, v in kwargs.items()))
        assert isinstance(self.log_path, pl.Path), f'log_path must be a Path, not {type(self.log_path)}'
        # special logging for the zeroth step
        if step == -1:
            system.dump_config(self.log_path)
        else:
            self.dump_logs(self.log_path, real_step, **kwargs)
        system.dump_logs(real_step, self.log_path / 'current', save_structure=real_step % self.log_frequency == 0)
        best_system.dump_logs(real_step, self.log_path / 'best', save_structure=new_best)


@dataclass
class FlexibleMinimizer(Minimizer):
    mutator: MutationProtocol
    temperature_schedule: list[float]
    n_steps: int
    log_frequency: int = 100
    preserve_best_system_every_n_steps: int | None = None
    experiment_name: str = field(default_factory=lambda: f'flexible_minimiser_{time_stamp()}')
    log_path: pl.Path | str | None = None

    def minimize_system(self, system: System) -> System:
        # raise NotImplementedError('Flexible minimizer DOES NOT work in new implementation with Oracles, yet')

        system.get_total_energy()  # update the energy internally
        best_system = system.__copy__()
        assert system.total_energy is not None, 'Cannot start without system having a calculated energy'
        assert best_system.total_energy is not None, (
            'Cannot start without lowest energy system having a calculated energy'
        )
        self.logging_step(-1, system, best_system, False)
        for step in range(self.n_steps):
            new_best = False
            temperature = self.temperature_schedule[step]
            system, accept = self.minimize_one_step(temperature, system)
            assert system.total_energy is not None, 'Cannot evolve system if current energy not available'
            if system.total_energy < best_system.total_energy:
                new_best = True
                best_system = system.__copy__()  # This automatically records the energy in best_system.total_energy
            self.logging_step(step, system, best_system, new_best, temperature=temperature, accept=accept)

            if self.preserve_best_system_every_n_steps is not None:
                if (step + 1) % self.preserve_best_system_every_n_steps == 0:
                    logger.debug(f'Starting new cycle with best system from previous cycle')
                    system = best_system.__copy__()  # begin next cycle using best system from previous cycle

        assert best_system.total_energy is not None, f'Best energy {best_system.total_energy} cannot be None!'
        return best_system


@dataclass
class MonteCarloSampler(FlexibleMinimizer):
    def __init__(
        self,
        mutator: MutationProtocol,
        temperature: float,
        n_steps: int,
        log_frequency: int = 100,
        experiment_name: str = field(default_factory=lambda: f'MC_sampler_{time_stamp()}'),
        log_path: pl.Path | str | None = None,
    ) -> None:
        super().__init__(
            mutator=mutator,
            temperature_schedule=[temperature] * n_steps,
            n_steps=n_steps,
            log_frequency=log_frequency,
            preserve_best_system_every_n_steps=None,
            experiment_name=experiment_name,
            log_path=log_path,
        )


@dataclass
class SimulatedAnnealing(FlexibleMinimizer):
    def __init__(
        self,
        mutator: MutationProtocol,
        initial_temperature: float,
        final_temperature: float,
        n_steps: int,
        log_frequency: int = 100,
        experiment_name: str = field(default_factory=lambda: f'simulated_annealing_{time_stamp()}'),
        log_path: pl.Path | str | None = None,
    ) -> None:
        super().__init__(
            mutator=mutator,
            temperature_schedule=list(np.linspace(start=initial_temperature, stop=final_temperature, num=n_steps)),
            n_steps=n_steps,
            log_frequency=log_frequency,
            preserve_best_system_every_n_steps=None,
            experiment_name=experiment_name,
            log_path=log_path,
        )
        self.acceptance: list[bool] = []
        super().__post_init__()

    def dump_logs(self, output_folder: pl.Path, step: int, **kwargs: Any) -> None:
        self.acceptance.append(kwargs['accept'])
        kwargs['acceptance_rate'] = sum(self.acceptance) / len(self.acceptance)
        super().dump_logs(output_folder, step, **kwargs)


@dataclass
class SimulatedTempering(FlexibleMinimizer):
    def __init__(
        self,
        mutator: MutationProtocol,
        high_temperature: float,
        low_temperature: float,
        n_steps_high: int,
        n_steps_low: int,
        n_cycles: int,
        preserve_best_system_every_n_steps: bool | None = None,
        log_frequency: int = 100,
        experiment_name: str = field(default_factory=lambda: f'simulated_annealing_{time_stamp()}'),
        log_path: pl.Path | str | None = None,
    ) -> None:
        ## Create the temperature schedule
        # Cycle through low and high temperatures
        temperature_schedule = np.concatenate(
            [
                np.full(shape=n_steps_low, fill_value=low_temperature),
                np.full(shape=n_steps_high, fill_value=high_temperature),
            ]
        )
        temperature_schedule = np.tile(temperature_schedule, reps=n_cycles)
        n_steps = len(temperature_schedule)

        super().__init__(
            mutator=mutator,
            temperature_schedule=temperature_schedule,
            n_steps=n_steps,
            log_frequency=log_frequency,
            preserve_best_system_every_n_steps=preserve_best_system_every_n_steps,
            experiment_name=experiment_name,
            log_path=log_path,
        )
        self.acceptance: list[bool] = []
        super().__post_init__()

    def dump_logs(self, output_folder: pl.Path, step: int, **kwargs: Any) -> None:
        """
        Dump the logs for the simulated tempering. Before sending it off to the superclass,
        we can compute additional information, such as the cumulative acceptance rate.
        """
        # TODO: this could be done a bit elegantly, if we keep track of self.step as well
        self.acceptance.append(kwargs['accept'])
        kwargs['acceptance_rate'] = sum(self.acceptance) / len(self.acceptance)
        super().dump_logs(output_folder, step, **kwargs)


#! PROPOSE TO REMOVE, AS SOON AS IMPLEMENTATION ABOVE IS DONE
# @dataclass
# class SimulatedAnnealing(Minimizer):
#    mutator: MutationProtocol
#    initial_temperature: float
#    final_temperature: float
#    n_steps: int
#    log_frequency: int = 100
#    # ? move this into MutationProtocol (as that is the design)?
#    experiment_name: str = field(default_factory=lambda: f'simulated_annealing_{time_stamp()}')
#    log_path: pl.Path | str | None = None
#
#    def __post_init__(self) -> None:
#        super().__post_init__()
#        self.temperatures = np.linspace(start=self.initial_temperature, stop=self.final_temperature, num=self.n_steps)
#        self.acceptance: list[bool] = []
#
#    def dump_logs(self, output_folder: pl.Path, step: int, **kwargs: Any) -> None:
#        self.acceptance.append(kwargs['accept'])
#        kwargs['acceptance_rate'] = sum(self.acceptance) / len(self.acceptance)
#        super().dump_logs(output_folder, step, **kwargs)
#
#    def minimize_system(self, system: System) -> System:
#        system.get_total_energy() # update the energy internally
#        best_system = system.__copy__()
#        assert best_system.total_energy is not None, 'Cannot start without best system has a calculated energy'
#        self.logging_step(-1, system, best_system, False)
#        for step in range(self.n_steps):
#            new_best = False
#            system, accept = self.minimize_one_step(self.temperatures[step], system)
#            assert system.total_energy is not None, 'Cannot minimize system before energy is calculated'
#            if system.total_energy < best_system.total_energy:
#                best_system = system.__copy__()  # This automatically records the energy in best_system.total_energy
#                new_best = True
#            self.logging_step(step, system, best_system, new_best, temperature=self.temperatures[step], accept=accept)
#
#        assert best_system.total_energy is not None, f'{best_system=} energy cannot be None!'
#        return best_system


# @dataclass
# class SimulatedTempering(Minimizer):
#    mutator: MutationProtocol
#    high_temperature: float
#    low_temperature: float
#    n_steps_high: int
#    n_steps_low: int
#    n_cycles: int
#    preserve_best_system: bool = False
#    log_frequency: int = 1000
#    experiment_name: str = field(default_factory=lambda: f'simulated_tempering_{time_stamp()}')
#    log_path: pl.Path | str | None = None
#
#    """
#    Cycling between low and high temperatures.
#    """
#
#    def __post_init__(self) -> None:
#        super().__post_init__()
#        cycle_temperatures = np.concat(
#            [
#                np.full(shape=self.n_steps_low, fill_value=self.low_temperature),
#                np.full(shape=self.n_steps_high, fill_value=self.high_temperature),
#            ]
#        )
#        self.temperatures = np.tile(cycle_temperatures, reps=self.n_cycles)
#        self.n_steps_cycle = self.n_steps_high + self.n_steps_low
#        self.acceptance: list[bool] = []
#
#    def dump_logs(self, output_folder: pl.Path, step: int, **kwargs: Any) -> None:
#        """
#        Dump the logs for the simulated tempering. Before sending it off to the superclass,
#        we can compute additional information, such as the cumulative acceptance rate.
#        """
#        # TODO: this could be done a bit elegantly, if we keep track of self.step as well
#        self.acceptance.append(kwargs['accept'])
#        kwargs['acceptance_rate'] = sum(self.acceptance) / len(self.acceptance)
#        super().dump_logs(output_folder, step, **kwargs)
#
#    def minimize_system(self, system: System) -> System:
#        system.get_total_energy() # update the energy internally
#        best_system = system.__copy__()
#        assert best_system.total_energy is not None, 'Cannot start without best system has a calculated energy'
#        self.logging_step(-1, system, best_system, False)
#        for step, temperature in enumerate(self.temperatures):
#            new_best = False
#            system, accept = self.minimize_one_step(temperature, system)
#            assert system.total_energy is not None, 'Cannot minimize system before energy is calculated'
#
#            if system.total_energy < best_system.total_energy:
#                best_system = system.__copy__()  # This automatically records the energy in best_system.total_energy
#                new_best = True
#
#            self.logging_step(step, system, best_system, new_best, temperature=temperature, accept=accept)
#
#            if self.preserve_best_system and (step + 1) % self.n_steps_cycle == 0:
#                logger.debug(f'Starting new cycle with best system from previous cycle')
#                system = best_system.__copy__()  # begin next cycle using best system from previous cycle
#
#        assert best_system.total_energy is not None, f'{best_system=} energy cannot be None!'
#        return best_system
