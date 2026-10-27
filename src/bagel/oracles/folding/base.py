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
    #: Number of structures (diffusion samples, ...) per prediction. The energy of a State is the mean over all of them.
    #: With 1 (the default) a prediction is a single structure, exactly as before.
    n_samples: int = 1

    def predict(self, chains: list[Chain]) -> FoldingResult:
        """
        Predict new structure of chains. With several samples this is the best one.
        """
        return self.fold(chains=chains)

    def predict_samples(self, chains: list[Chain]) -> list[FoldingResult]:
        """
        Predict ``n_samples`` structures of chains from a single run of the model, best first.

        Oracles that return one structure per prediction give a list with that structure.
        """
        return [self.predict(chains=chains)]

    @abstractmethod
    def fold(self, chains: list[Chain]) -> FoldingResult:
        raise NotImplementedError('This method should be implemented by the folding algorithm')


class ConfidenceFoldingOracle(FoldingOracle):
    """Shared BoilerRoom adapter for single-sample confidence folding models."""

    result_class: type[ConfidenceFoldingResult]
    model_name: str
    model: Any
    required_fields = ('plddt', 'pae', 'ptm')
    #: The most samples one prediction can hold (None: no limit of its own). 1 for deterministic models.
    max_samples: int | None = 1
    #: Whether the model can take a user-supplied MSA (``msa=`` in the constructor).
    supports_user_msa = False
    #: Whether a fold whose chains have no MSA must still say so (an all-None ``msa`` list) when the oracle holds MSAs.
    #: True for models that would otherwise search an MSA because ``use_msa`` was switched on for the given MSAs.
    explicit_empty_msa = False
    #: Whether chains of equal sequence share one MSA in the model, so they must be given the same MSA entry.
    identical_chains_share_msa = False

    def __init__(
        self,
        backend: str = 'modal',
        device: str | None = None,
        config: dict[str, Any] | None = None,
        msa: MSAInputs | None = None,
        n_samples: int = 1,
    ) -> None:
        if msa and not self.supports_user_msa:
            raise ValueError(f'{self.model_name} does not support a user-supplied MSA')
        if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
            raise ValueError(f'n_samples must be a positive integer, got {n_samples!r}')
        if self.max_samples is not None and n_samples > self.max_samples:
            raise ValueError(
                f'{self.model_name} gives at most {self.max_samples} sample(s) per prediction'
                + (' (it is deterministic)' if self.max_samples == 1 else '')
                + f', got n_samples={n_samples}'
            )
        self.n_samples = n_samples
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

    def _require_samples(self, config: dict[str, Any], key: str, default: int) -> None:
        """
        Make sure the model is configured to generate at least ``n_samples`` samples.

        ``key`` is the model's own sample-count option and ``default`` its value when absent from ``config``. It is raised
        to ``n_samples`` if needed; a value given by the user that is too small is refused.
        """
        available = config.get(key, default)
        if available >= self.n_samples:
            return
        if key in config:
            raise ValueError(
                f'{self.model_name} config {key}={available} generates fewer samples than n_samples={self.n_samples}'
            )
        config[key] = self.n_samples

    def _fold_options(self, chains: list[Chain]) -> dict[str, Any]:
        """Options of the model call."""
        options: dict[str, Any] = {'include_fields': list(self.required_fields)}
        msa = msas_for_chains(
            self.msas,
            chains,
            keep_empty=self.explicit_empty_msa,
            identical_chains_share=self.identical_chains_share_msa,
        )
        if msa is not None:
            options['msa'] = msa
        return options

    def fold(self, chains: list[Chain]) -> ConfidenceFoldingResult:
        """Predict the structure of chains; with several samples, the best one."""
        return self.fold_samples(chains)[0]

    def fold_samples(self, chains: list[Chain]) -> list[ConfidenceFoldingResult]:
        """Predict ``n_samples`` structures of chains with one call of the model, best first."""
        output = self.model.fold(self._pre_process(chains), options=self._fold_options(chains))
        return self._reduce_samples(output, chains)

    def predict_samples(self, chains: list[Chain]) -> list[FoldingResult]:
        return list(self.fold_samples(chains))

    def _reduce_samples(self, output: Any, chains: list[Chain]) -> list[ConfidenceFoldingResult]:
        """The ``n_samples`` results of the model output, best first."""
        return [self._reduce_output(output, chains)]

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
    Confidence folding oracle for models that return several samples per fold.

    BoilerRoom returns every sample (diffusion samples, AlphaFold models, ...). The ``n_samples`` best ones are kept
    and the rest dropped, so that with ``n_samples=1`` (the default) the prediction is the top-ranked structure.

    Samples are ranked by the model's own ranking when BoilerRoom reports it (``sample_ranks``, or a list already
    sorted best first); models that do not rank their samples are ranked by pTM, ties broken by mean pLDDT.
    """

    max_samples: int | None = None
    #: Rank by pTM (then mean pLDDT) instead of trusting the order of the model output.
    rank_by_ptm = False

    def _ranked_indices(self, output: Any, n_available: int) -> list[int]:
        """Indices of the samples of ``output``, best first."""
        if self.rank_by_ptm:
            ptm, plddt = getattr(output, 'ptm', None), getattr(output, 'plddt', None)
            well_formed = (
                ptm is not None
                and plddt is not None
                and len(ptm) == len(plddt) == n_available
                and all(v is not None and np.size(v) == 1 for v in ptm)
                and all(v is not None and np.size(v) > 0 for v in plddt)
            )
            if well_formed:
                scores = [
                    (-float(np.asarray(ptm[i]).reshape(-1)[0]), -float(np.mean(plddt[i])), i)
                    for i in range(n_available)
                ]
                return [index for *_, index in sorted(scores)]
            return list(range(n_available))  # malformed values are reported when the samples are reduced
        ranks = getattr(output, 'sample_ranks', None)
        if ranks is not None:
            return [int(i) for i in np.argsort(ranks, kind='stable')]
        return list(range(n_available))  # BoilerRoom lists samples best first when it reports no explicit rank

    def _reduce_samples(self, output: Any, chains: list[Chain]) -> list[ConfidenceFoldingResult]:
        atom_arrays = output.atom_array
        if atom_arrays is None or len(atom_arrays) == 0:
            raise ValueError(f'{self.model_name} output does not contain any atom_array')
        n_available = len(atom_arrays)
        for field in self.required_fields:
            values = getattr(output, field, None)
            if values is not None and len(values) != n_available:
                raise ValueError(
                    f'{self.model_name} {field} must contain exactly one sample per structure; '
                    f'got {len(values)} for {n_available} structure(s)'
                )
        if n_available < self.n_samples:
            raise ValueError(
                f'{self.model_name} returned {n_available} sample(s), fewer than n_samples={self.n_samples}'
            )
        results: list[ConfidenceFoldingResult] = []
        for index in self._ranked_indices(output, n_available)[: self.n_samples]:
            selected = SimpleNamespace(
                atom_array=atom_arrays[index : index + 1],
                **{
                    field: None if getattr(output, field, None) is None else getattr(output, field)[index : index + 1]
                    for field in self.required_fields
                },
            )
            results.append(ConfidenceFoldingOracle._reduce_output(self, selected, chains))
        return results

    def _reduce_output(self, output: Any, chains: list[Chain]) -> ConfidenceFoldingResult:
        return self._reduce_samples(output, chains)[0]
