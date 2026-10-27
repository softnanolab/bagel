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


def validate_unique_chain_ids(states: list[State]) -> None:
    """
    Check that every chain_ID identifies exactly one Chain.

    A chain may belong to several States, but then it must be the very same Chain object in each of them, so that a
    mutation is seen by all of them. Two different Chain objects with the same chain_ID are refused if either of them
    has mutable residues, since a mutation could not then be told which chain it applies to. Immutable chains may share
    a chain_ID (e.g. different targets in different States). Within one State every chain_ID must be unique anyway,
    because the folded structure is split into chains by chain_ID.

    Raises
    ------
    ValueError
        If a chain_ID is used by more than one Chain.
    """
    owner: dict[str, tuple[Chain, str]] = {}
    for state in states:
        ids_in_state: set[str] = set()
        for chain in state.chains:
            if chain.chain_ID in ids_in_state:
                raise ValueError(
                    f"State '{state.name}' contains the chain_ID '{chain.chain_ID}' more than once. Every chain in a "
                    f'State needs its own unique chain_ID.'
                )
            ids_in_state.add(chain.chain_ID)
            other = owner.get(chain.chain_ID)
            if other is not None and other[0] is not chain and (chain.mutable_residues or other[0].mutable_residues):
                raise ValueError(
                    f"The chain_ID '{chain.chain_ID}' identifies two different Chains (in States "
                    f"'{other[1]}' and '{state.name}') and at least one of them has mutable residues. This is not "
                    f'possible: BAGEL identifies a chain by its chain_ID, so give every mutable chain its own chain_ID. '
                    f'To use the same chain in several States, add the very same Chain object to each of them instead '
                    f'of building a copy. (Chains without mutable residues may share a chain_ID.)'
                )
            owner.setdefault(chain.chain_ID, (chain, state.name))


@dataclass
class System:
    """Top level object defining the input for a protein design pipeline. In practice, a system will be a collection of
    states, each representing a (potentially different) collection of chains."""

    states: list[State]
    name: str | None = None
    total_energy: float | None = None

    def __post_init__(self) -> None:
        validate_unique_chain_ids(self.states)

    def __copy__(self) -> 'System':
        """Copy the system object, setting the energy to None"""
        return deepcopy(self)

    def get_total_energy(self) -> float:
        # Always recompute by accessing state.energy, which triggers
        # automatic cache invalidation if chains have changed.
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

        total_energy = float(np.sum(state_energies))
        self.total_energy = total_energy
        return total_energy

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
        validate_unique_chain_ids(self.states)
