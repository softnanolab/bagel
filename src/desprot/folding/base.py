from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel
from biotite.structure import AtomArray
from ..chain import Chain


class FoldingMetrics(BaseModel):
    """
    Stores statistics from the folding algorithm.
    """

    pass

    class Config:
        arbitrary_types_allowed = True  # This is needed for numpy array support


class FoldingAlgorithm(ABC):
    """Template for an object that predicts structure of proteins from sequence."""

    @abstractmethod
    def _load(self) -> None:
        """
        Loads the folding algorithm.
        """
        pass

    @abstractmethod
    def fold(self, chains: List[Chain]) -> tuple[AtomArray, FoldingMetrics]:
        """
        Predicts structure of given protein.
        Returns structure and statistics from the folding algorithm.

        Args:
            sequence: List[str]
                List of strings, each representing a protein sequence chain.

        Returns:
            tuple[AtomArray, FoldingMetrics]
            Structure and statistics from the folding algorithm.
        """
        pass
