"""
FoldingOracles are algorithms that, given a State as input, return a 3D structure and statistics from the folding algorithm.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from abc import abstractmethod
from ...chain import Chain
from ..base import Oracle, OracleResult
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, set_structure
from typing import Type
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FoldingResult(OracleResult):
    """
    Stores statistics from the folding algorithm.
    """

    input_chains: list[Chain]
    structure: AtomArray

    class Config:
        arbitrary_types_allowed = True  # This is needed for numpy array support

    def to_cif(self, filepath: Path) -> bool:
        """
        Write the structure to a CIF file.

        Parameters
        ----------
        filepath : Path
            Path to the file to write the CIF structure to.

        Returns
        -------
        bool
            True if the file was written successfully, False otherwise.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        structure_file = CIFFile()
        set_structure(structure_file, self.structure)
        logger.debug(f'Writing CIF structure to {filepath}')
        structure_file.write(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f'Structure file {filepath} was not created')
        else:
            return True


class FoldingOracle(Oracle):
    """
    A FoldingOracle is a specific type of Oracle that uses a folding algorithm to predict the 3D structure of a State.
    """

    result_class: Type[FoldingResult] = FoldingResult  # holds class, not instance

    def predict(self, chains: list[Chain]) -> FoldingResult:
        """
        Predict new structure of chains.
        """
        return self.fold(chains=chains)

    @abstractmethod
    def fold(self, chains: list[Chain]) -> FoldingResult:
        raise NotImplementedError('This method should be implemented by the folding algorithm')
