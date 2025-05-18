"""
standard template and objects for structure prediction
"""

import numpy as np
import numpy.typing as npt
from ..chain import Chain
from .utils import reindex_chains
from pydantic import field_validator
from ..oracles import FoldingOracle, FoldingMetrics
from typing import List, Any
from modalfold import app  # type: ignore
from modalfold.esmfold import ESMFold, ESMFoldOutput  # type: ignore


# TODO: add proper types to next modalfold version
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


class ESMFoldingMetrics(FoldingMetrics):
    """
    Stores statistics from the ESMFold folding algorithm.

    TODO: Thing whether this should be part of desprot, or modalfold.
    There might be some clear standards that we would like to enforce in modalfold,
    but at the same time give user the flexibility to use the output as they want.
    To be discussed.
    """

    local_plddt: npt.NDArray[np.float64]  # global template modelling score (0 to 1)
    ptm: npt.NDArray[np.float64]  # global predicted local distance difference test score (0 to 1)
    pae: npt.NDArray[np.float64]  # pairwise predicted alignment error

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


class ESMFolder(FoldingOracle):
    """
    Object that uses ESMFold to predict structure of proteins from sequence.

    WIP: For now we will be using ModalFold to do this reliably without much env issues.
    """

    def __init__(self, use_modal: bool = False, config: dict[str, Any] = {}):
        """
        NOTE this can only be called once. Attempting to initialise this object multiple times in one process creates
        breaking exceptions.
        """
        self.use_modal = use_modal
        self.default_config = {
            'output_pdb': False,
            'output_cif': False,
            'output_atomarray': True,
            'glycine_linker': '',
            'position_ids_skip': 512,
        }
        self._load(config)

        if self.use_modal:
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
        if self.use_modal:
            self.modal_app_context = app.run()
            self.modal_app_context.__enter__()  # type: ignore
        config = {**self.default_config, **config}
        self.model = ESMFold(config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Pre-process the sequence to be passed to the model for folding.
        Here, we assume, that we are using HuggingFace's implementation of ESMFold.
        Therefore, individual chains are separated by a ":" character.
        """
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def fold(self, chains: List[Chain]) -> tuple[AtomArray, ESMFoldingMetrics]:
        """
        Fold a list of chains using ESMFold.
        """
        if self.use_modal:
            return self._reduce_output(self._remote_fold(self._pre_process(chains)), chains)
        else:
            logger.info('Given that use_modal is False, trying to fold with ESMFold locally...')
            logger.info('Assuming that all packages are available locally...')
            # TODO: Hugging Face Cache might need to be set here properly to make it work
            return self._reduce_output(self._local_fold(self._pre_process(chains)), chains)

    def _remote_fold(self, sequence: List[str]) -> ESMFoldOutput:
        return self.model.fold.remote(sequence)

    def _local_fold(self, sequence: List[str]) -> ESMFoldOutput:
        return self.model.fold.local(sequence)

    def _reduce_output(self, output: ESMFoldOutput, chains: List[Chain]) -> tuple[AtomArray, ESMFoldingMetrics]:
        """
        Reduce ESMFoldOutput (from ModalFold) to a ESMFoldingMetrics object
        """
        atoms = output.atom_array
        metrics = ESMFoldingMetrics(
            local_plddt=output.plddt,
            ptm=output.ptm,
            pae=output.predicted_aligned_error,
        )
        atoms = reindex_chains(atoms, [chain.chain_ID for chain in chains])

        return atoms, metrics
