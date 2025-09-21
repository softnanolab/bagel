"""
Objects defining the different protocols for mutating the Chains.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

import numpy as np

# from .folding import FoldingAlgorithm
from .chain import Chain
from .system import System
from .constants import mutation_bias_no_cystein
from dataclasses import dataclass, field
from typing import Dict, Tuple
from abc import ABC, abstractmethod
from .oracles.base import OraclesResultDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MutationProtocol(ABC):
    mutation_bias: Dict[str, float] = field(default_factory=lambda: mutation_bias_no_cystein)
    n_mutations: int = 1
    exclude_self: bool = True

    @abstractmethod
    def one_step(
        self,
        system: System,
        old_system: System,
    ) -> tuple[System, float]:
        """
        Abstract method for performing a single mutation step.

        Parameters
        ----------
        system : System
            The system to be mutated
        old_system : System
            The original system before mutation, used for comparison

        Returns
        -------
        System
            The mutated system
        float
            The energy difference between the mutated and original system
        float
            The difference in chemical potential contribution (related to size) between the mutated and original system
        """
        pass

    def choose_chain(self, system: System) -> Chain:
        """
        Choose one of the chains in the whole System that needs to be mutated. This is done by selecting a chain
        proportionally to the number of mutable aminoacid it has compared to the total number of mutable aminoacids
        available within the whole system. Because the number of mutable aminoacids can change during the simulation,
        probability must be recalculated at each step (i.e. after each mutation).
        """
        unique_chain_list: list[Chain] = []
        for state in system.states:
            for chain in state.chains:
                # Using a set ensures that each chain is only counted once
                if chain not in unique_chain_list:
                    unique_chain_list.append(chain)

        n_total_mutables = sum([len(chain.mutable_residues) for chain in unique_chain_list])
        probability = np.zeros(len(unique_chain_list))
        for i, chain in enumerate(unique_chain_list):
            n_mutables = len(chain.mutable_residues)
            probability[i] = n_mutables / n_total_mutables
        # Step 2:
        # the chain is mutated according to the protocol chosen. Side note: a chain can be part of multiple states, and
        # mutations need to be made so that they are consistent across all states. This is taken care of by the fact
        # that the same object is used.
        return np.random.choice(unique_chain_list, p=probability)  # type: ignore

    def mutate_random_residue(self, chain: Chain) -> None:
        # Choose a residue to mutate
        index = np.random.choice(chain.mutable_residue_indexes)
        # Choose a new aminoacid
        current_aa = chain.residues[index].name
        aa_keys = list(self.mutation_bias.keys())
        probs = np.array([self.mutation_bias[a] for a in aa_keys], dtype=float)
        if self.exclude_self:  # exclude the current amino acid from the probability distribution
            mask = np.array([a != current_aa for a in aa_keys], dtype=bool)
            probs = probs * mask
            total = probs.sum()
            if total <= 0:
                raise ValueError(
                    f'No valid mutation targets after excluding current AA={current_aa}. '
                    'Check mutation_bias provides non-zero probability to at least one alternative.'
                )
            probs = probs / total
        amino_acid = np.random.choice(aa_keys, p=probs)
        chain.mutate_residue(index=index, amino_acid=amino_acid)

    def reset_system(self, system: System) -> System:
        system.total_energy = None
        for state in system.states:
            state._energy_terms_value = {}
            state._oracles_result = OraclesResultDict()
        return system


class Canonical(MutationProtocol):
    """
    Canonical mutation protocol, making a substitution at random n residues.
    It cannot add or remove residues.

    Parameters
    ----------
    n_mutations : int, optional
        Number of mutations to perform in each step.
    mutation_bias : Dict[str, float], optional
        Bias for the substitution. The keys are the amino acids, the values are the probabilities.
    """

    def __init__(
        self,
        n_mutations: int = 1,
        mutation_bias: Dict[str, float] = mutation_bias_no_cystein,
        exclude_self: bool = True,
    ):
        self.n_mutations = n_mutations
        self.mutation_bias = mutation_bias
        self.exclude_self = exclude_self

    def one_step(
        self,
        system: System,
        old_system: System,
    ) -> tuple[System, float]:
        for _ in range(self.n_mutations):
            chain = self.choose_chain(system)
            self.mutate_random_residue(chain=chain)
        self.reset_system(system=system)  # Reset the system so it knows it must recalculate fold and energy
        delta_energy = system.get_total_energy() - old_system.get_total_energy()
        return system, delta_energy


class GrandCanonical(MutationProtocol):
    """
    Grand canonical mutation protocol, making a substitution, addition or removal at random n residues.

    Parameters
    ----------
    n_mutations : int, optional
        Number of mutations to perform in each step.
    mutation_bias : Dict[str, float], optional
        Bias for the substitution. The keys are the amino acids, the values are the probabilities.
    move_probabilities : dict[str, float], optional
        Probabilities for the different moves. The keys are 'substitution', 'addition', and 'removal', and the values
        are the probabilities of performing the corresponding move. These should be normalized to sum to 1.
    """

    def __init__(
        self,
        n_mutations: int = 1,
        mutation_bias: Dict[str, float] = mutation_bias_no_cystein,
        move_probabilities: dict[str, float] = {
            'substitution': 0.5,
            'addition': 0.25,
            'removal': 0.25,
        },
        exclude_self: bool = True,
    ):
        self.n_mutations = n_mutations
        self.mutation_bias = mutation_bias
        self.move_probabilities = move_probabilities
        self.exclude_self = exclude_self
        # Check that no probabilities are negative
        if any([prob < 0 for prob in self.move_probabilities.values()]):
            raise ValueError('Probabilities must be positive')

        # Check that the sum of the probabilities is 1 otherwise normalize
        if sum(self.move_probabilities.values()) != 1.0:
            self.move_probabilities = {
                move: prob / sum(self.move_probabilities.values()) for move, prob in self.move_probabilities.items()
            }
            logger.warning('Recalculated move probabilties to ensure they sum to 1')
            logger.info(self.move_probabilities)

    def remove_random_residue(self, chain: Chain, system: System) -> None:
        # First of all, only try this if it does not bring chains to 0 length
        if chain.length > 1:
            # Choose a residue to remove
            index = np.random.choice(chain.mutable_residue_indexes)
            chain_ID = chain.chain_ID
            # Sanity check
            assert chain_ID == chain.residues[index].chain_ID
            # Remove the residue from the chain it is part of
            chain.remove_residue(index=index)
            # Remove the residue from energy terms of all the states in the system
            for state in system.states:
                state.remove_residue_from_all_energy_terms(chain_ID=chain_ID, residue_index=index)

    def add_random_residue(self, chain: Chain, system: System) -> None:
        # Choose where to add the residue
        index = np.random.choice(range(chain.length + 1))
        chain_ID = chain.residues[0].chain_ID
        # Choose a new aminoacid
        amino_acid = np.random.choice(list(self.mutation_bias.keys()), p=list(self.mutation_bias.values()))
        chain.add_residue(index=index, amino_acid=amino_acid)
        # Now you need to decide which energy terms you want to associate to this residue. You do it based on its
        # neighbours. You look within the same chain and the same state and you add the residue to the same energy terms
        # the neighbours are part of. You actually look left and right, and randomly decide between the two. If the
        # residue is at the beginning or at the end of the chain, you just look at one of them.
        for state in system.states:
            state.add_residue_to_all_energy_terms(chain_ID=chain_ID, residue_index=index)

    def one_step(
        self,
        system: System,
        old_system: System,
    ) -> tuple[System, float]:
        for _ in range(self.n_mutations):
            chain = self.choose_chain(system)
            # Now pick a move to make among removal, addition, or mutation
            assert self.move_probabilities.keys() == {'substitution', 'addition', 'removal'}, (
                'Move probabilities must be mutation, addition and removal'
            )
            move = np.random.choice(
                list(self.move_probabilities.keys()),
                p=list(self.move_probabilities.values()),
            )
            if move == 'substitution':
                self.mutate_random_residue(chain=chain)
            elif move == 'addition':
                self.add_random_residue(chain=chain, system=system)
            elif move == 'removal':
                self.remove_random_residue(chain=chain, system=system)

        self.reset_system(system=system)  # Reset the system so it knows it must recalculate fold and energy
        delta_energy = system.get_total_energy() - old_system.get_total_energy()

        return system, delta_energy
