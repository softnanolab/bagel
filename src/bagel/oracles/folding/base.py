"""
FoldingOracles are algorithms that, given a State as input, return a 3D structure and statistics from the folding algorithm.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from abc import abstractmethod
from types import SimpleNamespace
from typing import Any

from ...chain import Chain
from ..base import Oracle, OracleResult
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, set_structure
import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict, field_validator
from pathlib import Path
import logging

from .msa import MSAInputs, load_msas, msas_for_chains
from .utils import (
    prepare_single_structure,
    single_sample_matrix,
    single_sample_scalar,
    single_sample_vector,
    validate_array_range,
)

logger = logging.getLogger(__name__)


class FoldingResult(OracleResult):
    """
    Stores statistics from the folding algorithm.
    """

    input_chains: list[Chain]
    structure: AtomArray

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
            True if the file was written successfully.

        Raises
        ------
        FileNotFoundError
            If the structure file was not created after writing.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        structure_file = CIFFile()
        set_structure(structure_file, self.structure)
        logger.debug(f'Writing CIF structure to {filepath}')
        structure_file.write(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f'Structure file {filepath} was not created')
        return True


class ConfidenceFoldingResult(FoldingResult):
    """Folding result with the confidence fields shared by structure models."""

    local_plddt: npt.NDArray[np.float64]
    ptm: npt.NDArray[np.float64]
    pae: npt.NDArray[np.float64]

    @field_validator('local_plddt')
    @classmethod
    def validate_local_plddt(cls, value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(value, 'local_plddt')

    @field_validator('ptm')
    @classmethod
    def validate_ptm(cls, value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(value, 'ptm')

    def save_attributes(self, filepath: Path) -> None:
        np.savetxt(filepath.with_suffix('.plddt'), self.local_plddt[0], fmt='%.6f', header='plddt')
        np.savetxt(filepath.with_suffix('.pae'), self.pae[0], fmt='%.6f', header='pae')


class FoldingOracle(Oracle):
    """
    A FoldingOracle is a specific type of Oracle that uses a folding algorithm to predict the 3D structure of a State.
    """

    result_class: type[FoldingResult]

    def predict(self, chains: list[Chain]) -> FoldingResult:
        """
        Predict new structure of chains.
        """
        return self.fold(chains=chains)

    @abstractmethod
    def fold(self, chains: list[Chain]) -> FoldingResult:
        raise NotImplementedError('This method should be implemented by the folding algorithm')


class ConfidenceFoldingOracle(FoldingOracle):
    """Shared BoilerRoom adapter for single-sample confidence folding models."""

    result_class: type[ConfidenceFoldingResult]
    model_name: str
    model: Any
    required_fields = ('plddt', 'pae', 'ptm')
    #: Whether the model can take a user-supplied MSA (``msa=`` in the constructor).
    supports_user_msa = False

    def __init__(
        self,
        backend: str = 'modal',
        device: str | None = None,
        config: dict[str, Any] | None = None,
        msa: MSAInputs | None = None,
    ) -> None:
        if msa and not self.supports_user_msa:
            raise ValueError(f'{self.model_name} does not support a user-supplied MSA')
        self.backend = backend
        self.device = device
        # Read once here: folding then only forwards the stored text, it never reads a file or searches for an MSA.
        self.msas = load_msas(msa)
        self._load(config)

    @abstractmethod
    def _load(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the model-specific BoilerRoom wrapper."""
        raise NotImplementedError

    def _pre_process(self, chains: list[Chain]) -> Any:
        """Join chains into BoilerRoom's string multimer representation."""
        return [':'.join(chain.sequence for chain in chains)]

    def fold(self, chains: list[Chain]) -> ConfidenceFoldingResult:
        options: dict[str, Any] = {'include_fields': list(self.required_fields)}
        msa = msas_for_chains(self.msas, chains)
        if msa is not None:
            options['msa'] = msa
        output = self.model.fold(self._pre_process(chains), options=options)
        return self._reduce_output(output, chains)

    def _reduce_output(self, output: Any, chains: list[Chain]) -> ConfidenceFoldingResult:
        return self.result_class(
            input_chains=chains,
            structure=prepare_single_structure(output.atom_array, chains, self.model_name),
            local_plddt=single_sample_vector(output.plddt, 'plddt', self.model_name),
            ptm=single_sample_scalar(output.ptm, 'ptm', self.model_name),
            pae=single_sample_matrix(output.pae, 'pae', self.model_name),
        )


class RankedSamplesFoldingOracle(ConfidenceFoldingOracle):
    """
    Confidence folding oracle for models that return several ranked samples per fold.

    BoilerRoom returns every sample (diffusion samples, AlphaFold models, ...); BAGEL uses a single structure per fold,
    so the top-ranked sample is selected and the rest are dropped.
    """

    def _select_sample(self, output: Any) -> int:
        """Index of the top-ranked sample: the one with the lowest ``sample_ranks`` if reported, else the first."""
        ranks = getattr(output, 'sample_ranks', None)
        if ranks is None:
            return 0  # BoilerRoom lists samples best first when it reports no explicit rank
        return int(np.argmin(ranks))

    def _reduce_output(self, output: Any, chains: list[Chain]) -> ConfidenceFoldingResult:
        atom_arrays = output.atom_array
        if atom_arrays is None or len(atom_arrays) == 0:
            raise ValueError(f'{self.model_name} output does not contain any atom_array')
        n_samples = len(atom_arrays)
        for field in self.required_fields:
            values = getattr(output, field, None)
            if values is not None and len(values) != n_samples:
                raise ValueError(f'{self.model_name} {field} has {len(values)} samples; expected {n_samples}')
        best = self._select_sample(output)
        if not 0 <= best < n_samples:
            raise ValueError(f'{self.model_name} selected sample {best}, but the output has {n_samples} samples')
        selected = SimpleNamespace(
            atom_array=atom_arrays[best : best + 1],
            **{
                field: None if getattr(output, field, None) is None else getattr(output, field)[best : best + 1]
                for field in self.required_fields
            },
        )
        return super()._reduce_output(selected, chains)
