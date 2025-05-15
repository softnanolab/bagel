"""
Objects defining the different protocols for mutating the Chains.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Saffar, Stefano Angioletti-Uberti
"""

import numpy as np
#from .folding import FoldingAlgorithm
from .chain import Chain
from .system import System
from .constants import mutation_bias_no_cystein
from dataclasses import dataclass, field
from typing import Dict, Tuple
from abc import ABC, abstractmethod


@dataclass
class MutationProtocol(ABC):
    mutation_bias: Dict[str, float] = field(default_factory=lambda: mutation_bias_no_cystein)
    n_mutations: int = 1

    @abstractmethod
    def one_step(
        self, 
        #folding_algorithm: FoldingAlgorithm, 
        system: System, old_system: System
    ) -> Tuple[System, float, float]:
        """
        Makes one mutation and returns the new system, the energy difference and the difference in size
        compared to the old system.
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
        amino_acid = np.random.choice(list(self.mutation_bias.keys()), p=list(self.mutation_bias.values()))
        chain.mutate_residue(index=index, amino_acid=amino_acid)

    def reset_system(self, system: System) -> System:
        print('Resetting system RESET ALL ORACLES OUTPUTS')
        system.total_energy = None
        for state in system.states:
            state._energy_terms_value = {}
            state._oracles_output = {}
        return system


class Canonical(MutationProtocol):
    # TODO default factory fix
    def __init__(
        self,
        n_mutations: int = 1,
        mutation_bias: Dict[str, float] = mutation_bias_no_cystein,
    ):
        self.n_mutations = n_mutations
        self.mutation_bias = mutation_bias

    def one_step(
        self, 
        #folding_algorithm: FoldingAlgorithm, 
        system: System, old_system: System
    ) -> Tuple[System, float, float]:
        for i in range(self.n_mutations):
            chain = self.choose_chain(system)
            self.mutate_random_residue(chain=chain)
        # print( "Canonical mutation")
        # for i in range( len( system.states ) ):
        #    for j in range( len( system.states[i].chains )):
        #        print( f"NEW state[{i}].chains[{j}] {system.states[i].chains[j].sequence}")
        #        print( f"OLD state[{i}].chains[{j}] {old_system.states[i].chains[j].sequence}")
        self.reset_system(system=system)  # Reset the system so it knows it must recalculate fold and energy
        # print( f"HERE {system.states[0]._folding_metrics}" )
        delta_energy = system.get_total_energy() - old_system.get_total_energy()
        return system, delta_energy


class GrandCanonical(MutationProtocol):
    # TODO default factory fix
    def __init__(
        self,
        n_mutations: int = 1,
        mutation_bias: Dict[str, float] = mutation_bias_no_cystein,
        move_probabilities: dict[str, float] = {
            'mutation': 0.5,
            'addition': 0.25,
            'removal': 0.25,
        },
    ):
        self.n_mutations = n_mutations
        self.mutation_bias = mutation_bias
        self.move_probabilities = move_probabilities
        # Check that no probabilities are negative
        if any([prob < 0 for prob in self.move_probabilities.values()]):
            raise ValueError('Probabilities must be positive')

        # Check that the sum of the probabilities is 1 otherwise normalize
        if sum(self.move_probabilities.values()) != 1.0:
            self.move_probabilities = {
                move: prob / sum(self.move_probabilities.values()) for move, prob in self.move_probabilities.items()
            }
            print('Recalcalculated move probabilties to ensure they sum to 1')
            print(self.move_probabilities)

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
        #folding_algorithm: FoldingAlgorithm, 
        system: System, old_system: System
    ) -> Tuple[System, float, float]:
        for i in range(self.n_mutations):
            chain = self.choose_chain(system)
            # Now pick a move to make among removal, addition, or mutation
            # print('Current probabilties')
            print(self.move_probabilities)
            assert self.move_probabilities.keys() == {'mutation', 'addition', 'removal'}, (
                'Move probabilities must be mutation, addition and removal'
            )
            move = np.random.choice(
                list(self.move_probabilities.keys()),
                p=list(self.move_probabilities.values()),
            )
            if move == 'mutation':
                # print( "mutation")
                self.mutate_random_residue(chain=chain)
            elif move == 'addition':
                # print( "addition")
                self.add_random_residue(chain=chain, system=system)
            elif move == 'removal':
                # print( "removal")
                self.remove_random_residue(chain=chain, system=system)

        self.reset_system(system=system)  # Reset the system so it knows it must recalculate fold and energy
        # print(f'HERE {system.states[0]._folding_metrics}')
        delta_energy = system.get_total_energy() - old_system.get_total_energy()

        return system, delta_energy


@dataclass
class Genetic(MutationProtocol):
    name: str = 'GeneticAlgorithm'

    def mate_chains(self, system: System) -> None:
        raise NotImplementedError
