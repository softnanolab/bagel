from abc import ABC, dataclass, abstractmethod
from typing import List
from pydantic import BaseModel
from torch import nn, Tensor
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProteinLanguageModel(ABC):
    """
    Template for an object that predicts residues embeddings from the aminoacid sequence."""
    """
    An Embedding is a N_residues x N_features matrix that contains the embeddings of the residues in the state.
    """
    name: str
    model : nn
    
    @abstractmethod
    def _load(self) -> None:
        """
        Loads the folding algorithm.
        """
        pass

    @abstractmethod
    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Pre-process the sequence to be passed to the model for calculating the embeddings.
        """
        pass

    @abstractmethod 
    def calculate_embeddings(self, state) -> None:
        """
        Calculate the embeddings of the residues in the state.
        """
        pass
