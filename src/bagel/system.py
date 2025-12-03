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
from .oracles.base import OraclesResultDict
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
            state_energies = []
            for state in self.states:
                if not state.energy_terms:
                    logger.warning(
                        f"State '{state.name}' has no energy terms. Skipping from system total energy calculation."
                    )
                    continue
                state_energies.append(state.energy)

            if not state_energies:
                raise ValueError('System has no states with energy terms defined. Cannot compute system total energy.')

            self.total_energy = np.sum(state_energies)
        return self.total_energy

    def reset(self) -> None:
        """
        Clear energy caches so system knows it must recalculate.

        Note: Cache invalidation is now automatic when chains or energy_terms change,
        so explicit reset() is optional but available for explicit clearing.
        """
        self.total_energy = None
        for state in self.states:
            state._energy = None
            state._energy_term_values = {}
            state._oracles_result = OraclesResultDict()
            state._cache_key = None

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
