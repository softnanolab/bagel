"""Top level object defining the input for a protein design pipeline"""

from biotite.structure.io.pdbx import CIFFile, set_structure
from collections import OrderedDict
from .state import State
from .chain import Chain, Residue
from typing import Any
from dataclasses import dataclass
from .folding import FoldingAlgorithm
from .constants import aa_dict
from copy import deepcopy
import pathlib as pl
import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


@dataclass
class System:
    """Top level object defining the input for a protein design pipeline. In practice, a system will be a collection of
    states, each representing a (potentially different) collection of chains."""

    states: list[State]
    name: str | None = None
    total_energy: float | None = None
    # ! Removing because this below is never used
    # _old_energy: float | None = None

    def __copy__(self) -> 'System':
        """Copy the system object, setting the energy to None"""
        return System(
            states=deepcopy(self.states),
            total_energy=self.total_energy,
            name=self.name,
        )

    def get_total_energy(self, folding_algorithm: FoldingAlgorithm) -> float:
        if self.total_energy is None:
            self.total_energy = np.mean([state.get_energy(folding_algorithm) for state in self.states])  # type: ignore
        return self.total_energy  # type: ignore

    def dump_logs(self, step: int, path: pl.Path, save_structure: bool = True) -> None:
        r"""
        Saves logging information for the system under the given directory path. This folder contains:

        - a CSV file named 'energies.csv'. Columns include 'step', '\<state.name\>_\<energy.name\>' for all energies,
          '\<state.name\>_chemical_potential_energy', '\<state.name\>_energy' and 'system_energy'. Note the final
          column is the sum of the mean weighted energies and the chemical potential energies of each state.
        - a FASTA file for all sequences named '\<state.name\>.fasta'. Each header is the sequence's step and each
          sequence is a string of amino acid letters with : seperating each chain.
        - a further directory named 'structures' containing all CIF files. Files are named '\<state.name>_\<step>.cif'
          for all states.

        Expects the energies of the system to already be calculated.

        Parameters
        ----------
        step : int
            The index of the current optimisation step.
        path: pl.Path
            The directory in which the log files will be saved into.
        save_structure: bool, default=True
            Whether to save the CIF file of each state.
        """
        assert self.total_energy is not None, 'System energy not calculated. Call get_total_energy() first.'

        structure_path = path / 'structures'
        if step == 0:
            structure_path.mkdir(parents=True)

        assert path.exists(), 'Path does not exist. Please create the directory first.'
        assert structure_path.exists(), 'Structure path does not exist. Please create the directory first.'

        energies: dict[str, int | float] = {'step': step}  #  order of insertion consistent in every dump_logs call
        for state in self.states:
            for energy in state.energy_terms:
                energies[f'{state.name}:{energy.name}'] = energy.value
            assert state._energy is not None, 'State energy not calculated. Call get_energy() first.'
            energies[f'{state.name}:state_energy'] = state._energy  # HACK

            with open(path / f'{state.name}.fasta', mode='a') as file:
                file.write(f'>{step}\n')
                file.write(f'{":".join(state.total_sequence)}\n')

            if save_structure:
                file = CIFFile()
                set_structure(file, state._structure)  # HACK
                file.write(structure_path / f'{state.name}_{step}.cif')  # type: ignore

        energies['system_energy'] = self.total_energy

        energies_path = path / 'energies.csv'
        with open(energies_path, mode='a') as file:
            if step == 0:
                file.write(','.join(energies.keys()) + '\n')
            file.write(','.join([str(energy) for energy in energies.values()]) + '\n')

    def dump_config(self, path: pl.Path) -> None:
        """
        Saves information about how each energy term was configured in a csv file named "config.csv". Columns include
        'state_name', 'energy_name', and 'weight'.

        Parameters
        ----------
        path: pl.Path
            The directory in which the config.csv file will be created.
        """
        assert path.exists(), 'Path does not exist. Please create the directory first.'
        with open(path / 'config.csv', mode='w') as file:
            file.write('state,energy,weight\n')
            for state in self.states:
                for i, term in enumerate(state.energy_terms):
                    file.write(f'{state.name},{term.name},{state.energy_terms_weights[i]}\n')

    def add_chain(self, sequence: str, mutability: list[int], chain_ID: str, state_index: list[int]) -> None:
        """
        Add a chain to the state.

        Parameters
        ----------
        sequence : str
            amino acid sequence of the chain
        mutability : list[int]
            list of 0s and 1s indicating if the residue is mutable or not
        chain_index : int
            index of the chain (global, same for all states the chain is part of)
        state_index : int
            index of the state in the system
        """
        assert len(sequence) == len(mutability), 'sequence and mutability lists must be of the same length'
        new_chain = Chain(residues=[])  # ? , mutability=[]
        # First generate the chain one residue at a time
        for i in range(len(sequence)):
            assert sequence[i] in aa_dict.keys(), 'sequence contains invalid amino acid'
            assert mutability[i] in [0, 1], 'mutability list must contain only 0 or 1'
            mutable = True if mutability[i] == 1 else False
            residue = Residue(name=sequence[i], chain_ID=chain_ID, index=i, mutable=mutable)
            new_chain.residues.append(residue)
        # Add the chain to the states it is part of
        for st_idx in state_index:
            self.states[st_idx].chains.append(new_chain)

    def chemical_potential_contribution(self) -> float:
        return sum([state.get_chemical_potential_contribution() for state in self.states])
