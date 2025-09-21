"""
Standard object to encode the tertiary structure, losses, and folding logic for a chain or complex of chains.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from .chain import Chain
from .oracles import Oracle, FoldingOracle, OraclesResultDict
from .energies import EnergyTerm
from typing import Optional
from pathlib import Path
from biotite.structure.io.pdbx import CIFFile, set_structure
from dataclasses import dataclass, field
from typing import List, Any
from copy import deepcopy
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class State:
    """
    A State is a multimeric collection of :class:`.Chain` objects with associated :class:`.EnergyTerm` objects.
    Chains can be independent of other States, or be shared between multiple States.

    Parameters
    ----------
    name : str
        Unique identifier for this State.
    chains : List[:class:`.Chain`]
        List of single monomeric Chains in this State.
    energy_terms : List[:class:`.EnergyTerm`]
        Collection of EnergyTerms that define the State.

    Attributes
    ----------
    _energy : Optional[float]
        Cached total (weighted) energy value for the State.
    _oracles_result : dict[Oracle, OracleResult]
        Results of different oracles, e.g., folding, embedding, etc.
    _energy_terms_value : dict[(str, float)]
        Cached (unweighted)values of individual :class:`.EnergyTerm` objects.
    """

    name: str
    chains: List[Chain]
    energy_terms: List[EnergyTerm]
    _energy: Optional[float] = field(default=None, init=False)
    _oracles_result: OraclesResultDict = field(default_factory=lambda: OraclesResultDict(), init=False)
    _energy_terms_value: dict[(str, float)] = field(default_factory=lambda: {}, init=False)

    def __post_init__(self) -> None:
        """Sanity check."""
        return

    def __copy__(self) -> Any:
        """Copy the state object, setting the structure and energy to None."""
        return deepcopy(self)

    @property
    def oracles_list(self) -> list[Oracle]:
        return list(set([term.oracle for term in self.energy_terms]))

    @property
    def total_sequence(self) -> List[str]:
        return [chain.sequence for chain in self.chains]

    def get_energy(self) -> float:
        """Calculate energy of state using energy terms ."""
        if self._energy_terms_value == {}:  # If energies not yet calculated
            # Check if the output of the oracle is already calculated, otherwise calculate it
            for oracle in self.oracles_list:
                if oracle not in self._oracles_result:
                    self._oracles_result[oracle] = oracle.predict(chains=self.chains)

        # Check that all energy term names are unique
        energy_term_names = [term.name for term in self.energy_terms]
        assert len(energy_term_names) == len(set(energy_term_names)), (
            f"Energy term names must be unique. Found duplicates: {energy_term_names}. Please rename using 'name'."
        )

        total_energy = 0.0
        for term in self.energy_terms:
            unweighted_energy, weighted_energy = term.compute(oracles_result=self._oracles_result)
            total_energy += weighted_energy
            self._energy_terms_value[term.name] = unweighted_energy
            logger.debug(f'Energy term {term.name} has value {unweighted_energy}')

        self._energy = total_energy

        logger.debug(f'**Weighted** energy for state {self.name} is {self._energy}')

        return self._energy

    def to_cif(self, oracle: FoldingOracle, filepath: Path) -> bool:
        """
        Write the state to a CIF file of a specific FoldingOracle.

        Parameters
        ----------
        filepath : Path
            Path to the file to write the CIF structure to.

        Returns
        -------
        bool
            True if the file was written successfully, False otherwise.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        structure_file = CIFFile()
        set_structure(structure_file, self._oracles_result.get_structure(oracle))
        logger.debug(f'Writing CIF structure of {self.name} from {type(oracle).__name__} to {filepath}')
        structure_file.write(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f'Structure file {filepath} was not created')
        else:
            return True

    def total_residues(self) -> int:
        return sum([len(chain.residues) for chain in self.chains])

    def remove_residue_from_all_energy_terms(self, chain_ID: str, residue_index: int) -> None:
        """Remove the residue from the energy terms associated to it in the current state."""
        for term in self.energy_terms:
            # The order of these two operations is important. FIRST you remove the residue, THEN you shift the indices
            # If you do the opposite, you will remove the wrong residue
            term.remove_residue(chain_ID, residue_index)
            # ensuring residue indexes in energy terms are updated to reflect a change in chain length
            term.shift_residues_indices_after_removal(chain_ID, residue_index)

    def add_residue_to_all_energy_terms(self, chain_ID: str, residue_index: int) -> None:
        """
        You look within the same chain and the same state and you add the residue to the same energy terms the
        neighbours are part of. You actually look left and right, and randomly decide between the two. If the residue is
        at the beginning or at the end of the chain, you just look at one of them. You do it for all
        terms that are inheritable.
        """

        # Get the chain that needs to be checked to inherit the energy terms from the neighbours
        chains = self.chains
        chain = None
        for i in range(len(chains)):
            if chains[i].chain_ID == chain_ID:
                chain = chains[i]
                break

        if chain is None:
            # This is ok, it can happen if a residue is added to a chain that is not in one of the states
            return

        # Remember the following selection is done AFTER the residue has been added to the Chain object via chain.add_residue
        left_residue = chain.residues[residue_index - 1] if residue_index > 0 else None
        right_residue = chain.residues[residue_index + 1] if residue_index < len(chain.residues) - 1 else None
        # Now choose randomly between the left and the right residue, if they exist
        assert left_residue is not None or right_residue is not None, (
            'This should not be possible unless a whole chain has disappeared but was still picked for mutation'
        )
        if left_residue is None:
            parent_residue = right_residue
        elif right_residue is None:
            parent_residue = left_residue
        else:
            parent_residue = np.random.choice([left_residue, right_residue])  # type: ignore

        assert parent_residue is not None, 'The parent residue is None, should not happen!'
        # Now add the residue to the energy terms associated to the parent residue
        for term in self.energy_terms:
            # The order of these two operations is important.
            # **Opposite** to what you do when you remove, you FIRST shift indices, and only THEN add one in the 'hole'
            # created.

            # MUST be called BEFORE add_residue. In this way, the residue of the parent index in the
            # residue_group attributed of the energy term is correct and updated to the same value of residue.index
            term.shift_residues_indices_before_addition(chain_ID, residue_index)
            # Add the residue to the energy term if the parent residue is part of it and the term is inheritable
            # The function automatically checks if the parent is also in it, or not.
            if term.inheritable:
                # Just a double check here before proceeding
                parent_index = parent_residue.index
                assert parent_residue.chain_ID == chain_ID, (
                    'The parent residue is not in the same chain, should not happen!'
                )
                term.add_residue(chain_ID, residue_index, parent_index)
