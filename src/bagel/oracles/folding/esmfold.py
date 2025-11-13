"""
standard template and objects for structure prediction
"""

import os
import pathlib as pl
import numpy as np
import numpy.typing as npt
from ...chain import Chain
from ...constants import atom_order
from .utils import reindex_chains
from pydantic import field_validator
from .base import FoldingOracle, FoldingResult
from typing import List, Any, Type
from boileroom import app  # type: ignore
from boileroom.models.esm.esmfold import ESMFoldOutput  # type: ignore
from boileroom.models.esm.esmfold import ESMFold as ESMFoldBoiler
from modal import App

from biotite.structure import AtomArray
import logging

logger = logging.getLogger(__name__)


def validate_array_range(
    array: npt.NDArray[np.float64], field_name: str, min_val: float = 0, max_val: float = 1
) -> npt.NDArray[np.float64]:
    """
    Validates that an array is a numpy array and its values fall within the specified range.

    Args:
        array: Array to validate
        field_name: Name of the field for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f'{field_name} must be a numpy array')
    if not np.all((array >= min_val) & (array <= max_val)):
        raise ValueError(f'All values in {field_name} must be between {min_val} and {max_val}')
    return array


class ESMFoldResult(FoldingResult):
    """
    Stores statistics from the ESMFold folding algorithm.
    """

    input_chains: list[Chain]
    structure: AtomArray  # structure of the predicted model
    local_plddt: npt.NDArray[np.float64]  # local ( per residue ) predicted LDDT score (0 to 1)
    ptm: npt.NDArray[np.float64]  # (global) predicted template modelling score (0 to 1)
    pae: npt.NDArray[np.float64]  # pairwise predicted alignment error

    # for ptm: see Zhang Y and Skolnick J (2004). "Scoring function for automated assessment of
    # protein structure template quality". Proteins. 57 (4): 702–710. doi:10.1002/prot.20264

    @classmethod
    def validate_score_array(cls, array: npt.NDArray[np.float64], field_name: str) -> npt.NDArray[np.float64]:
        if not isinstance(array, np.ndarray):
            raise ValueError(f'{field_name} must be a numpy array')
        if not np.all((array >= 0) & (array <= 1)):
            raise ValueError(f'All values in {field_name} must be between 0 and 1')
        return array

    @field_validator('local_plddt')
    def validate_local_plddt(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'local_plddt', 0, 1)

    @field_validator('ptm')
    def validate_ptm(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'ptm', 0, 1)

    def save_attributes(self, filepath: pl.Path) -> None:
        np.savetxt(filepath.with_suffix('.plddt'), self.local_plddt[0], fmt='%.6f', header='plddt')
        np.savetxt(filepath.with_suffix('.pae'), self.pae[0], fmt='%.6f', header='pae')


class ESMFold(FoldingOracle):
    """
    Object that uses ESMFold to predict structure of proteins from sequence.

    WIP: For now we will be using ModalFold to do this reliably without much env issues.
    """

    result_class: Type[ESMFoldResult] = ESMFoldResult

    def __init__(self, use_modal: bool = False, config: dict[str, Any] = {}, modal_app_context: App | None = None):
        """
        NOTE this can only be called once. Attempting to initialise this object multiple times in one process creates
        breaking exceptions.
        """
        self.use_modal = use_modal
        self.modal_app_context = modal_app_context
        self.default_config = {
            'output_pdb': False,
            'output_cif': False,
            'output_atomarray': True,
            'glycine_linker': '',
            'position_ids_skip': 512,
        }
        self._load(config)

        if self.use_modal and self.modal_app_context is None:
            # Register the cleanup function to be called at exit, so no
            # ephermal app is left running when the object is destroyed
            import atexit

            atexit.register(self.__del__)

    def __del__(self) -> None:
        """Cleanup the app context when the object is destroyed or at exit"""
        if self.use_modal and hasattr(self, 'modal_app_context') and self.modal_app_context is not None:  # type: ignore
            self.modal_app_context.__exit__(None, None, None)  # type: ignore
            self.modal_app_context = None

    def _load(self, config: dict[str, Any] = {}) -> None:
        if self.use_modal and self.modal_app_context is None:
            self.modal_app_context = app.run()
            self.modal_app_context.__enter__()  # type: ignore
        config = {**self.default_config, **config}
        self.model = ESMFoldBoiler(config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Pre-process the sequence to be passed to the model for folding.
        Here, we assume, that we are using HuggingFace's implementation of ESMFold.
        Therefore, individual chains are separated by a ":" character.
        """
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def fold(self, chains: List[Chain]) -> ESMFoldResult:
        """
        Fold a list of chains using ESMFold.
        """
        if self.use_modal:
            return self._reduce_output(self._remote_fold(self._pre_process(chains)), chains)
        else:
            logger.debug('Given that use_modal is False, trying to fold with ESMFold locally...')
            assert os.environ.get('MODEL_DIR'), 'MODEL_DIR must be set when using ESMFold locally'
            return self._reduce_output(self._local_fold(self._pre_process(chains)), chains)

    def _remote_fold(self, sequence: List[str]) -> ESMFoldOutput:
        return self.model.fold.remote(sequence)

    def _local_fold(self, sequence: List[str]) -> ESMFoldOutput:
        # assert that transformers is installed
        try:
            import transformers
        except ImportError:
            raise ImportError(
                'transformers is not installed. Please install it to use ESMFold locally. See README.md for installation instructions.'
            )
        return self.model.fold.local(sequence)

    def _reduce_output(self, output: ESMFoldOutput, chains: List[Chain]) -> ESMFoldResult:
        """
        Reduce ESMFoldOutput (from boileroom.esmfold) to a ESMFoldResult object.
        In principle, any other metric from ESMFoldOutput can be passed down into the ESMFoldResult object.
        For instance, one could pass the distogram_logits to create an EnergyTerm related to that.
        """
        atoms = output.atom_array
        atoms = reindex_chains(atoms, [chain.chain_ID for chain in chains])
        results = self.result_class(
            input_chains=chains,
            structure=atoms,
            local_plddt=output.plddt[..., atom_order['CA']],  # we only get CA atoms' plddt
            ptm=output.ptm,
            pae=output.predicted_aligned_error,
        )
        return results
