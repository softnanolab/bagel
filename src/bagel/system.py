"""
Top-level object defining the overall protein design task, including all the States.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from . import __version__ as bagel_version
from .state import State
from .chain import Chain, Residue
from dataclasses import dataclass

from .oracles.folding import FoldingOracle, FoldingResult
from .constants import aa_dict
from copy import deepcopy
import pathlib as pl
import numpy as np

import logging

logger = logging.getLogger(__name__)


@dataclass
class System:
    """Top level object defining the input for a protein design pipeline. In practice, a system will be a collection of
    states, each representing a (potentially different) collection of chains."""

    states: list[State]
    name: str | None = None
    total_energy: float | None = None

    def __copy__(self) -> 'System':
        """Copy the system object, setting the energy to None"""
        return deepcopy(self)

    def get_total_energy(self) -> float:
        if self.total_energy is None:
            self.total_energy = np.sum([state.get_energy() for state in self.states])
        return self.total_energy

    def dump_logs(self, step: int, path: pl.Path, save_structure: bool = True) -> None:
        r"""
        Saves logging information for the system under the given directory path. This folder contains:

        - a CSV file named 'energies.csv'. Columns include 'step', '\<state.name\>_\<energy.name\>' for all energies,
          '\<state.name\>_energy' and 'system_energy'. Note the final
          column is the sum of the mean weighted energies of each state.
        - a FASTA file for all sequences named '\<state.name\>.fasta'. Each header is the sequence's step and each
        sequence is a string of amino acid letters with : seperating each chain.
        - a FASTA file of per-residue mutability masks named '\<state.name\>.mask.fasta'. Each header is the
          sequence's step and each sequence is a string with 'M' for mutable and 'I' for immutable residues (default),
          with : separating chains in the same order as the sequence FASTA.
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
            for energy_name, energy_value in state._energy_terms_value.items():
                energies[f'{state.name}:{energy_name}'] = energy_value
            assert state._energy is not None, 'State energy not calculated. Call get_energy() first.'
            energies[f'{state.name}:state_energy'] = state._energy  # HACK

            with open(path / f'{state.name}.fasta', mode='a') as file:
                file.write(f'>{step}\n')
                file.write(f'{":".join(state.total_sequence)}\n')

            mask_per_chain = [
                ''.join(['M' if residue.mutable else 'I' for residue in chain.residues]) for chain in state.chains
            ]
            with open(path / f'{state.name}.mask.fasta', mode='a') as mask_file:
                mask_file.write(f'>{step}\n')
                mask_file.write(f'{":".join(mask_per_chain)}\n')

            if save_structure:
                for oracle, oracle_result in state._oracles_result.items():
                    if isinstance(oracle, FoldingOracle) and isinstance(oracle_result, FoldingResult):
                        oracle_name = type(oracle).__name__
                        state.to_cif(oracle, structure_path / f'{state.name}_{oracle_name}_{step}.cif')
                        oracle_result.save_attributes(structure_path / f'{state.name}_{oracle_name}_{step}')
                    else:
                        logger.debug(
                            f'Skipping {oracle.__class__.__name__} for CIF export, as it is not a FoldingOracle'
                        )

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
        # Write version.txt
        if bagel_version:
            with open(path / 'version.txt', mode='w') as vfile:
                vfile.write(f'{bagel_version}\n')
        # Write config.csv
        with open(path / 'config.csv', mode='w') as file:
            file.write('state,energy,weight\n')
            for state in self.states:
                for i, term in enumerate(state.energy_terms):
                    file.write(f'{state.name},{term.name},{term.weight}\n')

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
