"""
Oracles are algorithms that, given a State as input, can return a prediction.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from abc import ABC, abstractmethod
from typing import Any, Type
import pathlib as pl
import logging
from pydantic import BaseModel
from ..chain import Chain


logger = logging.getLogger(__name__)


class OracleResult(BaseModel):
    """
    Results from an Oracle.
    """

    input_chains: list[Chain]

    @abstractmethod
    def save_attributes(self, filepath: pl.Path) -> None:
        pass


class Oracle(ABC):
    """
    An Oracle is any algorithm that, given a State as input, can return a prediction.
    """

    result_class: Type[OracleResult] = OracleResult  # holds class, not instance

    def __post_init__(self) -> None:
        """Sanity check."""
        return

    def __copy__(self) -> Any:
        """Return a reference of the oracle."""
        return self

    def __deepcopy__(self, memo: dict) -> Any:
        """
        Return a reference of the oracle.

        This is to avoid copying the same oracle multiple times, when it is not necessary to do so.
        Deepcopy would break, as the same oracle is referenced multiple times for both a single State and a System.
        Long-term, this design pattern should be changed to something more robust, as this can easily break.
        """
        return self

    @abstractmethod
    def predict(self, chains: list[Chain]) -> OracleResult:
        pass


from .folding import FoldingResult, FoldingOracle
from .embedding import EmbeddingResult, EmbeddingOracle
from biotite.structure import AtomArray
import numpy as np
import numpy.typing as npt


class OraclesResultDict(dict[Oracle, OracleResult]):
    def __setitem__(self, oracle: Oracle, result: OracleResult) -> None:
        # Type check before setting
        if not isinstance(result, oracle.result_class):
            raise TypeError(f'Expected {oracle.result_class.__name__}, got {type(result).__name__}')
        # Call the parent class's __setitem__ to do the actual setting
        super().__setitem__(oracle, result)

    def __getitem__(self, oracle: Oracle) -> OracleResult:
        # Get the result using the parent class's __getitem__
        result = super().__getitem__(oracle)
        # Double-check type when retrieving
        assert isinstance(result, oracle.result_class), f'Result type mismatch'
        return result

    def get_structure(self, oracle: Oracle) -> AtomArray:
        assert isinstance(oracle, FoldingOracle), 'Oracle must be a FoldingOracle'
        result = self[oracle]
        assert isinstance(result, FoldingResult), 'Result must be a FoldingResult'
        return result.structure

    def get_embeddings(self, oracle: Oracle) -> npt.NDArray[np.float64]:
        assert isinstance(oracle, EmbeddingOracle), 'Oracle must be a EmbeddingOracle'
        result = self[oracle]
        assert isinstance(result, EmbeddingResult), 'Result must be a EmbeddingResult'
        return result.embeddings

    def get_input_chains(self, oracle: Oracle) -> list[Chain]:
        # TODO: if we have multiple oracles, input_chains will be redundant across all of the oracles
        # it might make sense - at some point - to have a single input_chains for all oracles
        # which would, however, forbid - for instance - the use of different chains for different oracles
        result = self[oracle]
        return result.input_chains
