"""
standard template and objects for structure prediction
"""

import numpy as np
import numpy.typing as npt
from ...chain import Chain
from .base import EmbeddingResults, EmbeddingOracle
from typing import List, Any
from modalfold import app
from modalfold.esm2 import ESM2, ESM2Output

import logging

logger = logging.getLogger(__name__)


class ESM2Results(EmbeddingResults):
    """
    Stores statistics from ESM-2.
    """

    embedding: npt.NDArray[np.float64]


class ESM2(EmbeddingOracle):
    def __init__(self, use_modal: bool = False, config: dict[str, Any] = {}) -> None:
        """
        NOTE this can only be called once. Attempting to initialise this object multiple times in one process creates
        breaking exceptions.
        Initialise the ESM2 model.
        WIP: For now we will be using ModalFold to do this reliably without much env issues.
        """
        self.use_modal = use_modal
        self.default_config = {
            'output_hidden_states': False,
            'model_name': 'esm2_t33_650M_UR50D',
        }
        if self.use_modal:
            # Register the cleanup function to be called at exit, so no
            # ephermal app is left running when the object is destroyed
            import atexit

            atexit.register(self.__del__)

        self.model = self._load(config)

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
        return ESM2(config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Multimers are pre-processed by joining the sequences with a ":" character.
        """
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def embed(self, chains: list[Chain]) -> ESM2Results:
        """
        Calculate the embeddings of the residues in the chains.
        """
        chains = self._pre_process(chains)

        if ':' in chains[0]:
            raise NotImplementedError('ESM-2 does not support multimers as of modalfold v0.0.13')

        if self.use_modal:
            return self._post_process(self._remote_embed(self._pre_process(chains)))
        else:
            logger.info('Given that use_modal is False, trying to embed with ESM-2 locally...')
            # TODO: Hugging Face Cache might need to be set here properly to make it work
            return self._post_process(self._local_embed(self._pre_process(chains)))

    def _remote_embed(self, sequence: List[str]) -> ESM2Output:
        return self.model.embed.remote(sequence)

    def _local_embed(self, sequence: List[str]) -> ESM2Output:
        return self.model.embed.local(sequence)

    def _post_process(self, output: ESM2Output) -> np.ndarray:
        embedding = output.embeddings[0, 1:-1, :]  # remove first and last token embeddings (not a residue)
        assert len(embedding.shape) == 2, (
            f'Embeddings is expected to be a 2D tensor, not shape: {embedding.shape}. '
            'The ESM2 Oracle does not support batches.'
        )
        return embedding
