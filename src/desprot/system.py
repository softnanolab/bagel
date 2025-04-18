"""Top level object defining the input for a protein design pipeline"""

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
    output_folder: pl.Path | None = None
    # ! Removing because this below is never used
    # _old_energy: float | None = None

    def __post_init__(self) -> None:
        if self.output_folder is None:
            self.output_folder = pl.Path(__file__).resolve().parent.parent / 'data' / f'{self.name}'
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def __copy__(self) -> 'System':
        """Copy the system object, setting the energy to None"""
        return System(states=deepcopy(self.states), total_energy=self.total_energy)

    def get_total_energy(self, folding_algorithm: FoldingAlgorithm) -> float:
        if self.total_energy is None:
            self.total_energy = np.mean([state.get_energy(folding_algorithm) for state in self.states])  # type: ignore
        return self.total_energy  # type: ignore

    def dump_logs(self, experiment: str, step: int) -> None:
        r"""
        Saves logging information for the system into a folder named "logs step \<step\>". This folder contains:
        - a FASTA file named "sequences". Each header is the corresponding chain ID.
        - a PDB file for the structure of each state. Each file is named "structure \<state_ID\>".
        - a CSV file named "energies". Columns include 'state_ID', 'energy_name', and 'energy_value'.
        - a TXT file named "total_weighted_energy" that only includes a single floating point number.
        Expects the energies of the system to already be calculated.

        Parameters
        ----------
        step : int
            The index of the current optimisation step.
        file_path: str | None, default=None
            The directory in which the log folder will be created. By default, this is set to the current working
            directory (the path the console is at when it runs the file).
        """
        # TODO: this might be better to just take a filepath as input and the naming being taken care of in Minimiser
        # but leave it like this for now
        assert self.total_energy is not None, 'Cannot dump logs before system energies are calculated'
        assert self.output_folder is not None, 'Cannot dump logs before output folder is set'
        current_output_folder = self.output_folder / experiment
        current_output_folder.mkdir(parents=True, exist_ok=True)

        sequences: dict[str, str] = {}
        energies: OrderedDict[str, float] = OrderedDict({'step': step})
        for state in self.states:
            assert state.to_cif(current_output_folder / 'CIF' / f'{state.state_ID}_{step}.cif'), (
                f'Structure file for {state.state_ID} was not created'
            )
            sequences[state.state_ID] = ':'.join(state.total_sequence)

            for energy in state.energy_terms:
                # "state_ID:energy_name" - energy_value
                energies[f'{state.state_ID}:{energy.name}'] = energy.value

        for state in self.states:
            assert state._energy is not None, 'Cannot dump logs before state energies are calculated'
            energies[f'{state.state_ID}:total_energy'] = state._energy  # HACK: as this is not 'public' effectivelly
        energies['total_energy'] = self.total_energy

        # make output file if it doesn't exist
        energy_file = current_output_folder / 'energies.csv'
        if not energy_file.exists():
            with open(energy_file, 'w') as f:  # write headers
                for i, item in enumerate(energies.keys()):
                    if i < len(energies.keys()) - 1:
                        f.write(f'{item},')
                    else:
                        f.write(f'{item}')
                f.write('\n')

        # write Step and also name
        with open(energy_file, 'a') as f:
            values = list(energies.values())
            for i, value in enumerate(values):
                if i < len(values) - 1:
                    f.write(f'{value},')
                else:
                    f.write(f'{value}')
            f.write('\n')

        # open the fasta file and write the sequences
        for state_ID, sequence in sequences.items():
            with open(current_output_folder / f'{state_ID}.fasta', 'a') as f:
                f.write(f'>{step}\n{sequence}\n')

    def dump_config(self, experiment: str) -> None:
        """
        Saves information about how each energy term was configured in a csv file named "config". Columns include
        'state_ID', 'energy_name', and 'weight'.

        Parameters
        ----------
        file_path: str | None, default=None
            The directory in which the config file will be created. By default, this is set to the current working
            directory (the path the console is at when it runs the file).
        """
        assert self.output_folder is not None, 'Cannot dump config before output folder is set'
        # TODO: this might be better to just take a filepath as input and the naming being taken care of in Minimiser
        # but leave it like this for now
        energy_terms: list[dict[str, str | float]] = []
        for state in self.states:
            for energy in state.energy_terms:
                # energy_terms.append({"state_ID": state.state_ID, "energy_name": energy.name, "weight": energy.weight})
                energy_terms.append({'state_ID': state.state_ID, 'energy_name': energy.name})

        experiment_folder = self.output_folder / experiment
        experiment_folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(energy_terms).to_csv(experiment_folder / 'config.csv')

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
