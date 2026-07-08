"""
ESMFold2 oracle for protein structure prediction using EvolutionaryScale's ESMFold2.

The heavy ``boileroom`` model wrapper is imported lazily inside ``_load`` so that
importing this module (and constructing mocked oracles in tests) does not require
the ``boileroom.models.esmfold2`` package to be installed. The ESMFold2 wrapper
lives in boileroom >= 0.3.x (``boileroom.models.esmfold2.esmfold2``); older pinned
versions do not ship it.
"""

import pathlib as pl
import logging
from typing import TYPE_CHECKING, Any, Type

import numpy as np
import numpy.typing as npt

from ...chain import Chain
from .utils import reindex_chains
from pydantic import field_validator
from .base import FoldingOracle, FoldingResult

from biotite.structure import AtomArray

if TYPE_CHECKING:
    from boileroom.models.esmfold2.types import ESMFold2Output  # type: ignore

logger = logging.getLogger(__name__)


def validate_array_range(
    array: npt.NDArray[np.float64], field_name: str, min_val: float = 0, max_val: float = 1
) -> npt.NDArray[np.float64]:
    """Validate that an array's values fall within a specified range."""
    if not isinstance(array, np.ndarray):
        raise ValueError(f'{field_name} must be a numpy array')
    if not np.all((array >= min_val) & (array <= max_val)):
        raise ValueError(f'All values in {field_name} must be between {min_val} and {max_val}')
    return array


class ESMFold2Result(FoldingResult):
    """Stores statistics from the ESMFold2 structure prediction.

    Attributes
    ----------
    input_chains : list[Chain]
        The input chains that were folded.
    structure : AtomArray
        The predicted 3D structure.
    local_plddt : npt.NDArray[np.float64]
        Per-residue pLDDT confidence scores (0 to 1).
    ptm : npt.NDArray[np.float64]
        Predicted Template Modelling score (0 to 1).
    pae : npt.NDArray[np.float64]
        Predicted Aligned Error matrix.
    """

    input_chains: list[Chain]
    structure: AtomArray
    local_plddt: npt.NDArray[np.float64]
    ptm: npt.NDArray[np.float64]
    pae: npt.NDArray[np.float64]

    @field_validator('local_plddt')
    def validate_local_plddt(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'local_plddt', 0, 1)

    @field_validator('ptm')
    def validate_ptm(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'ptm', 0, 1)

    def save_attributes(self, filepath: pl.Path) -> None:
        np.savetxt(filepath.with_suffix('.plddt'), self.local_plddt[0], fmt='%.6f', header='plddt')
        np.savetxt(filepath.with_suffix('.pae'), self.pae[0], fmt='%.6f', header='pae')


class ESMFold2(FoldingOracle):
    """Oracle that uses ESMFold2 to predict protein structures from sequence.

    ESMFold2 is a diffusion-based all-atom structure prediction model from
    EvolutionaryScale. The boileroom wrapper runs it **locally** on the selected
    backend (Modal GPU or Apptainer): model weights are loaded from Hugging Face
    (``biohub/ESMFold2``) and inference runs in-process. No inference API or API
    token is involved.

    Multimers are handled **natively** as separate chains — ESMFold2 does not use
    a glycine linker or a positional-id skip. ``_pre_process`` serializes the
    chains as a ``':'``-delimited string, which the boileroom wrapper parses back
    into distinct per-chain entities.

    Parameters
    ----------
    use_modal : bool
        Whether to run the boileroom wrapper via the Modal backend (otherwise
        Apptainer).
    config : dict
        Configuration options forwarded to the boileroom ESMFold2 wrapper.
        Supported keys include:
        - model_name: ESMFold2 weights to load (e.g. "biohub/ESMFold2")
        - num_sampling_steps: Diffusion sampling steps
        - num_loops: Refinement loops
    """

    result_class: Type[ESMFold2Result] = ESMFold2Result

    def __init__(
        self,
        use_modal: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        if config is None:
            config = {}
        self.use_modal = use_modal
        self.default_config: dict[str, Any] = {
            'model_name': 'esmfold2-fast-2026-05',
            'num_sampling_steps': 100,
            'num_loops': 20,
        }
        self._load(config)

    def _load(self, config: dict[str, Any] | None = None) -> None:
        # Imported here (not at module scope) so the module stays importable
        # without the boileroom ESMFold2 wrapper present, and so mocked oracles in
        # tests can patch out _load entirely.
        from boileroom.models.esmfold2.esmfold2 import ESMFold2 as ESMFold2Boiler  # type: ignore

        if config is None:
            config = {}
        merged_config = {**self.default_config, **config}
        backend = 'modal' if self.use_modal else 'apptainer'
        self.model = ESMFold2Boiler(backend=backend, config=merged_config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """Join chains with ':' for multimers."""
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def fold(self, chains: list[Chain]) -> ESMFold2Result:
        """Fold a list of chains using ESMFold2.

        Parameters
        ----------
        chains : list[Chain]
            List of protein chains to fold.

        Returns
        -------
        ESMFold2Result
            Folding result with structure and confidence metrics.
        """
        output = self.model.fold(self._pre_process(chains))
        return self._reduce_output(output, chains)

    def _reduce_output(self, output: 'ESMFold2Output', chains: list[Chain]) -> ESMFold2Result:
        """Convert ESMFold2Output to ESMFold2Result.

        Parameters
        ----------
        output : ESMFold2Output
            Raw output from the boileroom wrapper.
        chains : list[Chain]
            Original input chains for chain reindexing.

        Returns
        -------
        ESMFold2Result
            Reduced result with relevant metrics.
        """
        # boileroom >= 0.3.x returns atom_array as a list (one AtomArray per input).
        atoms = output.atom_array[0] if isinstance(output.atom_array, (list, tuple)) else output.atom_array
        atoms = reindex_chains(atoms, [chain.chain_ID for chain in chains])

        # Extract pLDDT - ESMFold2 returns per-residue pLDDT directly
        local_plddt = np.array([0.0])  # default
        if output.plddt is not None and len(output.plddt) > 0 and output.plddt[0] is not None:
            local_plddt = output.plddt[0]
            if local_plddt.ndim == 1:
                local_plddt = local_plddt[None, :]  # add batch dim

        # Extract PTM
        ptm = np.array([[0.0]])
        if output.ptm is not None and len(output.ptm) > 0 and output.ptm[0] is not None:
            ptm = output.ptm[0]
            if ptm.ndim == 1:
                ptm = ptm[None, :]

        # Extract PAE
        pae = np.zeros((1, 0, 0))
        if output.pae is not None:
            pae = output.pae
            if pae.ndim == 2:
                pae = pae[None, :, :]

        return self.result_class(
            input_chains=chains,
            structure=atoms,
            local_plddt=local_plddt,
            ptm=ptm,
            pae=pae,
        )
