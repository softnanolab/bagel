"""standard object to encode the tertiary structure, losses, and folding logic for a chain or complex of chains."""

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
    chains: List[Chain]  # This is a list of single monomeric chains
    energy_terms: List[EnergyTerm]
    energy_terms_weights: List[float]
    state_ID: str
    chemical_potential: Optional[float] = None
    _energy: Optional[float] = field(default=None, init=False)
    _structure: AtomArray = field(default=None, init=False)
    _folding_metrics: Optional[FoldingMetrics] = field(default=None, init=False)
    _energy_terms_value: dict[(str, float)] = field(default_factory=lambda: {}, init=False)
    verbose: bool = False

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
        """predict new structure of state. Stores structure and folding metrics as private attributes."""
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
                if self.verbose:
                    print(f'Energy term {term.name} has value {energy}')

        total_energy = sum(
            [energy * weight for energy, weight in zip(self._energy_terms_value.values(), self.energy_terms_weights)]
        )
        self._energy = total_energy / sum(self.energy_terms_weights)  # returns average so value ~ between 0 and 1

        if self.verbose:
            print(f'**Weighted** energy for state {self.state_ID} is {self._energy}')

        return self._energy

    def to_cif(self, filepath: Path) -> bool:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        structure_file = CIFFile()
        set_structure(structure_file, self._structure)
        logger.debug(f'Writing CIF structure of {self.state_ID} to {filepath}')
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
            term.remove_residue(chain_ID, residue_index)

    def add_residue_to_all_energy_terms(self, chain_ID: str, residue_index: int) -> None:
        """
        You look within the same chain and the same state and you add the residue to the same energy terms the
        neighbours are part of. You actually look left and right, and randomly decide between the two. If the residue is
        at the beginning or at the end of the chain, you just look at one of them You do it for all terms except the
        TemplateMatchingEnergyTerm as this never makes sense.
        """
        chains = self.chains
        for i in range(len(chains)):
            if chains[i].chain_ID == chain_ID:
                chain = chains[i]

        left_residue = chain.residues[residue_index - 1] if residue_index > 0 else None
        right_residue = chain.residues[residue_index + 1] if residue_index < len(chain.residues) else None
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
            parent_index = parent_residue.index
            # Just a double check here before proceeding
            assert parent_residue.chain_ID == chain_ID, (
                'The parent residue is not in the same chain, should not happen!'
            )
            # Add the residue to the energy term if the parent residue is part of it and the term is inheritable
            # The function automatically checks if the parent is also in it, or not
            if term.inheritable:  # type: ignore
                term.add_residue(chain_ID, residue_index, parent_index)
