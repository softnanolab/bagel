"""
EmbeddingOracles are algorithms that, given a State as input, return latent embeddings.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from abc import abstractmethod
from ..base import Oracle, OracleResult
from ...chain import Chain
import numpy as np
import numpy.typing as npt
from typing import Type, Any


class EmbeddingResult(OracleResult):
    """
    Stores statistics from the embedding algorithm.
    """

    input_chains: list[Chain]
    embeddings: npt.NDArray[np.float64]

    class Config:
        arbitrary_types_allowed = True  # This is needed for numpy array support


class EmbeddingOracle(Oracle):
    """
    An EmbeddingOracle is a specific type of Oracle that uses a sequence-based model to predict the residues' embeddings.
    """

    result_class: Type[EmbeddingResult] = EmbeddingResult  # holds class, not instance

    def predict(self, chains: list[Chain]) -> EmbeddingResult:
        return self.embed(chains=chains)

    @abstractmethod
    def embed(self, chains: list[Chain]) -> EmbeddingResult:
        raise NotImplementedError('This method should be implemented by the embedding algorithm')

    @abstractmethod
    def _pre_process(self, chains: list[Chain]) -> Any:
        """
        Pre-process the sequence to be passed to the model for calculating the embeddings.
        """
        pass

    @abstractmethod
    def _post_process(self, output: Any) -> EmbeddingResult:
        """
        Takes the output from the oracle and post-process it to make it in the right format expected, if needed.
        For example, a protein language model might return a tensor of shape (N_residues, N_features), but we
        want to have a list of 1D tensors of shape (N_features,).
        """
        pass
