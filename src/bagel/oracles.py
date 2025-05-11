"""
Standard object to encode the tertiary structure, losses, and folding logic for a chain or complex of chains.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Saffar, Stefano Angioletti-Uberti
"""

from abc import abstractmethod
from .chain import Chain
from .folding import FoldingAlgorithm, FoldingMetrics
from .protein_LM import proteinLanguageModel
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
from torch import Tensor

logger = logging.getLogger(__name__)

@dataclass
class Oracle:
    """
    An Oracle is any algorithm that, given a State as input, can return the energy of that specific state.
    It is a wrapper around the :class:`.FoldingAlgorithm` class, which is used to predict the structure of the state.

    Parameters
    ----------
    name : str
        Unique identifier for this Oracle.
    """
    name: str

    def __post_init__(self) -> None:
        """Sanity check."""
        assert len(self.energy_terms_weights) == len(self.energy_terms), 'wrong number of energy term weights supplied'

    def __copy__(self) -> Any:
        """Copy the state object, setting the structure and energy to None."""
        return deepcopy(self)

    @abstractmethod    
    def make_prediction(self, state) -> None:
        pass

class FoldingOracle(Oracle):
    """
    A FoldingOracle is a specific type of Oracle that uses a folding algorithm to predict the structure of a state.

    Parameters        

    """
    def __init__(self, name: str, folding_algorithm: FoldingAlgorithm ) -> None:
        super().__init__(name)
        self.folding_algorithm = folding_algorithm

    def make_prediction(self, state):
        assert isinstance(self.folding_algorithm, FoldingAlgorithm)
        structure, folding_metrics = self.fold( state )
        return ( structure, folding_metrics )

    def fold(self, state ) -> tuple[AtomArray, FoldingMetrics]:
        """Predict new structure of state. Stores structure and folding metrics as private attributes."""
        assert state._structure is None, 'State already has a structure'
        structure, folding_metrics = self.folding_algorithm.fold( chains = state.chains )
        return structure, folding_metrics
    
class EmbeddingsOracle(Oracle):
    """
    A ESM2 is a specific type of Oracle that uses the ESM2 protein Language Model to predict the residues' embeddings 
    Parameters        

    """
    def __init__(self, name: str, protein_language_model : proteinLanguageModel ) -> None:
        super().__init__(name)
        self.pLM = protein_language_model 
    
    def make_prediction(self, state):
        assert isinstance( self.pLM, proteinLanguageModel )
        embeddings = self.pLM.calculate_embeddings( state )
        return embeddings
