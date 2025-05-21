"""
Oracles are algorithms that, given a State as input, can return a prediction.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Saffar, Stefano Angioletti-Uberti
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
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
        return deepcopy(self)

    @abstractmethod    
    def make_prediction(self, state) -> None:
        pass