"""
FoldingOracles are algorithms that, given a State as input, return a 3D structure and statistics from the folding algorithm.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Saffar, Stefano Angioletti-Uberti
"""

from abc import abstractmethod
from pydantic import BaseModel
from ...chain import Chain
from ..base import Oracle
from biotite.structure import AtomArray
from typing import List


class FoldingResults(BaseModel):
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
    _folding_results: FoldingResults

    def make_prediction(self, state: 'State'):
        """
        Predict new structure of state.
        Stores structure and folding metrics as private attributes.
        """
        assert self._structure is None, 'State already has a structure'
        assert self._folding_results is None, 'State already has folding results'
        self._structure, self._folding_results = self.fold(chains=state.chains)
        return self._structure, self._folding_results

    @abstractmethod
    def fold(self, chains: List[Chain]) -> tuple[AtomArray, FoldingResults]:
        raise NotImplementedError('This method should be implemented by the folding algorithm')
