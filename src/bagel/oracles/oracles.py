"""
Oracles are algorithms that, given a State as input, can return a prediction.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Saffar, Stefano Angioletti-Uberti
"""

from abc import abstractmethod
from dataclasses import dataclass
from pydantic import BaseModel
from ..chain import Chain
from .embedding.base import ProteinLanguageModel
from biotite.structure import AtomArray
from typing import List, Any
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

@dataclass
class Oracle():
    """
    An Oracle is any algorithm that, given a State as input, can return a prediction.
    """
    name: str

    def __post_init__(self) -> None:
        """Sanity check."""
        return

    def __copy__(self) -> Any:
        """Copy the state object, setting the structure and energy to None."""
        return deepcopy(self)

    @abstractmethod    
    def make_prediction(self, state) -> None:
        pass

########################################################
# FoldingOracle
########################################################

class FoldingMetrics(BaseModel):
    """
    Stores statistics from the folding algorithm.
    """

    pass

    class Config:
        arbitrary_types_allowed = True  # This is needed for numpy array support

class FoldingOracle(Oracle):
    """
    A FoldingOracle is a specific type of Oracle that uses a folding algorithm to predict the 3D structure of a State.
    """
    _structure: AtomArray
    _folding_metrics: FoldingMetrics

    def make_prediction(self, state: "State"):
        """
        Predict new structure of state. 
        Stores structure and folding metrics as private attributes.
        """
        assert self._structure is None, 'State already has a structure'
        assert self._folding_metrics is None, 'State already has folding metrics'
        self._structure, self._folding_metrics = self.fold(chains = state.chains)
        return self._structure, self._folding_metrics

    @abstractmethod
    def fold(self, chains: List[Chain]) -> tuple[AtomArray, FoldingMetrics]:
        raise NotImplementedError('This method should be implemented by the folding algorithm')
    

########################################################
# EmbeddingOracle
########################################################

class EmbeddingOracle(Oracle):
    """
    An EmbeddingOracle is a specific type of Oracle that uses a sequence-based model to predict the residues' embeddings.
    """
    
    def __init__(self, name: str, protein_language_model : ProteinLanguageModel ) -> None:
        super().__init__(name)
        self.pLM = protein_language_model 
    
    def make_prediction(self, state: "State"):
        embeddings = self.pLM.calculate_embeddings( state )
        return embeddings
