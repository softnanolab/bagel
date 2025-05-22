"""
Oracles are algorithms that, given a State as input, can return a prediction.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Saffar, Stefano Angioletti-Uberti
"""

from abc import ABC, abstractmethod
from typing import Any
from copy import deepcopy
import logging
from pydantic import BaseModel
from ..chain import Chain

logger = logging.getLogger(__name__)


class OracleResult(BaseModel):
    """
    Results from an Oracle.
    """

    input_chains: list[Chain]


class Oracle(ABC):
    """
    An Oracle is any algorithm that, given a State as input, can return a prediction.
    """

    result_class: OracleResult

    def __post_init__(self) -> None:
        """Sanity check."""
        return

    def __copy__(self) -> Any:
        return deepcopy(self)

    @abstractmethod
    def predict(self, chains: list[Chain]) -> OracleResult:
        pass
