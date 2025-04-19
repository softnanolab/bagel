"""Standard template and objects for energy minimisation logic."""

import pathlib as pl
from .system import System
from .folding import FoldingAlgorithm
from .mutation import MutationProtocol
from abc import ABC, abstractmethod
from typing import Callable, Any
from dataclasses import dataclass, field
import numpy as np

import csv
import os
import logging

logger = logging.getLogger(__name__)

import datetime as dt

time_stamp: Callable[[], str] = lambda: dt.datetime.now().strftime('%y%m%d_%H%M%S')


class Minimizer(ABC):
    """Standard template for energy minimisation logic."""

    mutator: MutationProtocol
    folder: FoldingAlgorithm
    experiment_name: str
    log_frequency: int

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

    def minimize_one_step(self, temperature: float, system: System) -> System:
        mutated_system, delta_energy, delta_chemical = self.mutator.one_step(
            folding_algorithm=self.folder,
            system=system.__copy__(),
            old_system=system,
        )
        acceptance_probability = np.exp(-(delta_energy + delta_chemical) / temperature)
        logger.debug(f'{delta_energy=}, {delta_chemical=}, {acceptance_probability=}')
        if acceptance_probability > np.random.uniform(low=0.0, high=1.0):
            return mutated_system
        return system

    @abstractmethod
    def dump_logs(self, output_folder: pl.Path, experiment_name: str, step: int, **kwargs: Any) -> None:
        output_file = output_folder / f'{experiment_name}.log'

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

    def logging_step(self, step: int, system: System, best_system: System, **kwargs: Any) -> None:
        # TODO: this should be more elegant in terms of the output folder
        if step % self.log_frequency == 0:
            logger.info(f'Step={step} - ' + ' - '.join(f'{k}={v}' for k, v in kwargs.items()))
            system.dump_logs(self.experiment_name, step)
            best_system.dump_logs(f'{self.experiment_name}_BEST', step)
            assert system.output_folder is not None, 'System output folder is not set'
            self.dump_logs(system.output_folder, self.experiment_name, step, **kwargs)


@dataclass
class SimulatedAnnealing(Minimizer):
    folder: FoldingAlgorithm
    mutator: MutationProtocol
    initial_temperature: float
    final_temperature: float
    n_steps: int
    log_frequency: int = 100
    # ? move this into MutationProtocol (as that is the design)?
    experiment_name: str = field(default_factory=lambda: f'simulated_annealing_{time_stamp()}')

    def __post_init__(self) -> None:
        self.temperatures = np.linspace(start=self.initial_temperature, stop=self.final_temperature, num=self.n_steps)

    def dump_logs(self, output_folder: pl.Path, experiment_name: str, step: int, **kwargs: Any) -> None:
        super().dump_logs(output_folder, experiment_name, step, **kwargs)

    def minimize_system(self, system: System) -> System:
        best_system = system.__copy__()
        best_system.get_total_energy(self.folder)  # update the energy internally
        assert best_system.total_energy is not None, 'Cannot preserve best system before energy is calculated'
        for step in range(self.n_steps):
            system = self.minimize_one_step(self.temperatures[step], system)
            assert system.total_energy is not None, 'Cannot minimize system before energy is calculated'
            if system.total_energy < best_system.total_energy:
                best_system = system.__copy__()  # This automatically records the energy in best_system.total_energy
            self.logging_step(step, system, best_system, temperature=self.temperatures[step])

        assert best_system.total_energy is not None, f'{best_system=} energy cannot be None!'
        return best_system


@dataclass
class SimulatedTempering(Minimizer):
    folder: FoldingAlgorithm
    mutator: MutationProtocol
    high_temperature: float
    low_temperature: float
    n_steps_high: int
    n_steps_low: int
    n_cycles: int
    preserve_best_system: bool = False
    log_frequency: int = 1000
    experiment_name: str = field(default_factory=lambda: f'simulated_tempering_{time_stamp()}')
    """
    We start with a high temperature and then move to a low temperature.
    """

    def __post_init__(self) -> None:
        cycle_temperatures = np.concat(
            [
                np.full(shape=self.n_steps_low, fill_value=self.low_temperature),
                np.full(shape=self.n_steps_high, fill_value=self.high_temperature),
            ]
        )
        self.temperatures = np.tile(cycle_temperatures, reps=self.n_cycles)
        self.n_steps_cycle = self.n_steps_high + self.n_steps_low
        self.acceptance: list[bool] = []

    def dump_logs(self, output_folder: pl.Path, experiment_name: str, step: int, **kwargs: Any) -> None:
        """
        Dump the logs for the simulated tempering. Before sending it off to the superclass,
        we can compute additional information, such as the cumulative acceptance rate.
        """
        # TODO: this could be done a bit elegantly, if we keep track of self.step as well
        self.acceptance.append(kwargs['accept'])
        kwargs['acceptance_rate'] = sum(self.acceptance) / len(self.acceptance)
        super().dump_logs(output_folder, experiment_name, step, **kwargs)

    def minimize_system(self, system: System) -> System:
        best_system = system.__copy__()
        best_system.get_total_energy(self.folder)  # update the energy internally
        assert best_system.total_energy is not None, 'Cannot preserve best system before energy is calculated'
        for step, temperature in enumerate(self.temperatures):
            system = self.minimize_one_step(temperature, system)
            assert system.total_energy is not None, 'Cannot minimize system before energy is calculated'

            accept = False
            if system.total_energy < best_system.total_energy:
                accept = True
                best_system = system.__copy__()  # This automatically records the energy in best_system.total_energy

            self.logging_step(step, system, best_system, temperature=temperature, accept=accept)

            if self.preserve_best_system and step % self.n_steps_cycle == 0:
                logger.debug(f'Starting new cycle with best system from previous cycle')
                system = best_system.__copy__()  # begin next cycle using best system from previous cycle

        assert best_system.total_energy is not None, f'{best_system=} energy cannot be None!'
        return best_system


@dataclass
class FlexibleMinimiser(Minimizer):
    folder_schedule: Callable[[int], FoldingAlgorithm]
    mutator: MutationProtocol
    temperature_schedule: Callable[[int], float]
    n_steps: int
    log_frequency: int = 100
    experiment_name: str = field(default_factory=lambda: f'flexible_minimiser_{time_stamp()}')

    def minimize_system(self, system: System) -> System:
        best_system = system
        best_energy = system.get_total_energy(folding_algorithm=self.folder_schedule(0))
        for step in range(self.n_steps):
            temperature = self.temperature_schedule(step)
            self.folder = self.folder_schedule(step)
            system = self.minimize_one_step(temperature, system)
            assert system.total_energy is not None, 'Cannot minimize system before energy is calculated'
            if system.total_energy < best_energy:
                best_system = system.__copy__()  # This automatically records the energy in best_system.total_energy
            self.logging_step(step, system, best_system, temperature=temperature)

        assert best_system.total_energy is not None, f'Best energy {best_energy} cannot be None!'
        return best_system
