"""
EmbeddingOracles are algorithms that, given a State as input, return latent embeddings.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from abc import abstractmethod
from pathlib import Path

from ..base import Oracle, OracleResult
from ...chain import Chain
import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict
from typing import Any


class EmbeddingResult(OracleResult):
    """
    Stores statistics from the embedding algorithm.
    """

    input_chains: list[Chain]
    embeddings: npt.NDArray[np.float64]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def save_attributes(self, filepath: Path) -> None:
        np.savetxt(filepath.with_suffix('.embeddings'), self.embeddings, fmt='%.6f', header='embeddings')


def single_sample_embeddings(values: Any, model_name: str) -> npt.NDArray[np.float64]:
    """Extract the residue embeddings from BAGEL's supported one-sample batch."""
    embeddings = np.asarray(values, dtype=np.float64)
    if embeddings.ndim != 3 or embeddings.shape[0] != 1:
        raise ValueError(
            f'{model_name} embeddings must have shape (1, residues, features); got {embeddings.shape}. '
            'BAGEL does not support embedding batches.'
        )
    return np.asarray(embeddings[0], dtype=np.float64)


class EmbeddingOracle(Oracle):
    """
    An EmbeddingOracle is a specific type of Oracle that uses a sequence-based model to predict the residues' embeddings.
    """

    result_class: type[EmbeddingResult]

    def predict(self, chains: list[Chain]) -> EmbeddingResult:
        return self.embed(chains=chains)

    @abstractmethod
    def embed(self, chains: list[Chain]) -> EmbeddingResult:
        raise NotImplementedError('This method should be implemented by the embedding algorithm')

    def _pre_process(self, chains: list[Chain]) -> Any:
        """Join chains into BoilerRoom's string multimer representation."""
        return [':'.join(chain.sequence for chain in chains)]

    @abstractmethod
    def _post_process(self, output: Any, chains: list[Chain]) -> EmbeddingResult:
        """
        Takes the output from the oracle and post-process it to make it in the right format expected, if needed.
        For example, a protein language model might return a tensor of shape (N_residues, N_features), but we
        want to have a list of 1D tensors of shape (N_features,).
        """
        raise NotImplementedError
