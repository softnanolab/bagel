"""
Standard object to encode the tertiary structure, losses, and folding logic for a chain or complex of chains.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Saffar, Stefano Angioletti-Uberti
"""

from .chain import Chain
from .folding import FoldingAlgorithm, FoldingMetrics
from .energies import EnergyTerm
from typing import Optional
from pathlib import Path
from biotite.structure.io.pdbx import CIFFile, set_structure
from dataclasses import dataclass, field
from biotite.structure import AtomArray
from typing import List, Any
from copy import deepcopy
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class State:
    """
    A State is a multimeric collection of :class:`.Chain` objects with associated :class:`.EnergyTerm` objects and chemical potential.
    Chains can be independent of other States, or be shared between multiple States.

    Parameters
    ----------
    name : str
        Unique identifier for this State.
    chains : List[:class:`.Chain`]
        List of single monomeric Chains in this State.
    energy_terms : List[:class:`.EnergyTerm`]
        Collection of EnergyTerms that define the State.
    energy_terms_weights : List[float] # TODO: this will be moved into EnergyTerms
        Weights for each EnergyTerm.
    chemical_potential : Optional[float], optional
        Chemical potential value for this state. Default is None.

    Attributes
    ----------
    _energy : Optional[float]
        Cached total (weighted) energy value for the State.
    _structure : AtomArray
        Atomic structure representation.
    _folding_metrics : Optional[FoldingMetrics]
        Metrics from the :class:`.FoldingAlgorithm` such as pLDDT, or PAE.
    _energy_terms_value : dict[(str, float)]
        Cached (unweighted)values of individual :class:`.EnergyTerm` objects.
    """

    name: str
    chains: List[Chain]  # This is a list of single monomeric chains
    energy_terms: List[EnergyTerm]
    energy_terms_weights: List[float]
    chemical_potential: Optional[float] = None
    _energy: Optional[float] = field(default=None, init=False)
    _structure: AtomArray = field(default=None, init=False)
    _folding_metrics: Optional[FoldingMetrics] = field(default=None, init=False)
    _energy_terms_value: dict[(str, float)] = field(default_factory=lambda: {}, init=False)

    def __post_init__(self) -> None:
        """Sanity check."""
        assert len(self.energy_terms_weights) == len(self.energy_terms), 'wrong number of energy term weights supplied'

    def __copy__(self) -> Any:
        """Copy the state object, setting the structure and energy to None."""
        return deepcopy(self)

    @property
    def total_sequence(self) -> List[str]:
        return [chain.sequence for chain in self.chains]

    def fold(self, folding_algorithm: FoldingAlgorithm) -> tuple[AtomArray, FoldingMetrics]:
        """Predict new structure of state. Stores structure and folding metrics as private attributes."""
        assert self._structure is None, 'State already has a structure'
        self._structure, self._folding_metrics = folding_algorithm.fold(chains=self.chains)
        return self._structure, self._folding_metrics

    def get_energy(self, folding_algorithm: FoldingAlgorithm) -> float:
        """Calculate energy of state using energy terms (not including chemical potential)."""
        if self._energy_terms_value == {}:  # If energies not yet calculated
            if self._structure is None or self._folding_metrics is None:
                self._structure, self._folding_metrics = self.fold(folding_algorithm=folding_algorithm)

            for term in self.energy_terms:
                energy = term.compute(self._structure, self._folding_metrics)
                self._energy_terms_value[term.name] = energy
                logger.debug(f'Energy term {term.name} has value {energy}')

        total_energy = sum(
            [energy * weight for energy, weight in zip(self._energy_terms_value.values(), self.energy_terms_weights)]
        )
        self._energy = total_energy

        logger.debug(f'**Weighted** energy for state {self.name} is {self._energy}')

        return self._energy

    def to_cif(self, filepath: Path) -> bool:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        structure_file = CIFFile()
        set_structure(structure_file, self._structure)
        logger.debug(f'Writing CIF structure of {self.name} to {filepath}')
        structure_file.write(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f'Structure file {filepath} was not created')
        else:
            return True

    def total_residues(self) -> int:
        return sum([len(chain.residues) for chain in self.chains])

    def get_chemical_potential_contribution(self) -> float:
        """Calculate the size contribution to the grand canonical energy of the state."""
        return 0.0 if self.chemical_potential is None else self.chemical_potential * self.total_residues()

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
            if term.inheritable:  # type: ignore
                # Just a double check here before proceeding
                parent_index = parent_residue.index
                assert parent_residue.chain_ID == chain_ID, (
                    'The parent residue is not in the same chain, should not happen!'
                )
                term.add_residue(chain_ID, residue_index, parent_index)
