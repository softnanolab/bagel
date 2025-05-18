from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..chain import Chain
from ..state import State
import logging

logger = logging.getLogger(__name__)

@dataclass
class LanguageModel(ABC):
    """
    Template for an object that predicts residues embeddings from the aminoacid sequence."""
    """
    An Embedding is a N_residues x N_features matrix that contains the embeddings of the residues in the state.
    """

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
    def _post_process(self, chains: list[Chain]) -> list[str]:
        """
        Takes the output from the oracle and post-process it to make it in the right format expected, if needed.
        For example, a protein language model might return a tensor of shape (N_residues, N_features), but we 
        want to have a list of 1D tensors of shape (N_features,). 
        """
        pass

    @abstractmethod 
    def calculate_embeddings(self, state: State ) -> None:
        """
        Calculate the embeddings of the residues in the state.
        """
        pass
