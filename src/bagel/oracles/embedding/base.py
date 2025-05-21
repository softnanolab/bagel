"""
EmbeddingOracles are algorithms that, given a State as input, return latent embeddings.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Saffar, Stefano Angioletti-Uberti
"""
from abc import abstractmethod
from pydantic import BaseModel
from ..base import Oracle
from ...chain import Chain


class EmbeddingResults(BaseModel):
    """
    Stores statistics from the embedding algorithm.
    """

    pass

class EmbeddingOracle(Oracle):
    """
    An EmbeddingOracle is a specific type of Oracle that uses a sequence-based model to predict the residues' embeddings.
    """
    
    def make_prediction(self, state: "State"):
        results = self.embed(state)
        return results
    
    @abstractmethod
    def embed(self, state: "State") -> EmbeddingResults:
        raise NotImplementedError('This method should be implemented by the embedding algorithm')


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