"""
standard template and objects for structure prediction
"""

import os
import pathlib as pl
import numpy as np
import numpy.typing as npt
from ...chain import Chain
from .base import EmbeddingResult, EmbeddingOracle
from typing import List, Any
from boileroom import app  # type: ignore
from boileroom.models.esm.esm2 import ESM2Output  # type: ignore
from boileroom.models.esm.esm2 import ESM2 as ESM2Boiler
from modal import App
import logging

logger = logging.getLogger(__name__)


class ESM2Result(EmbeddingResult):
    """
    Stores statistics from ESM-2.
    """

    input_chains: list[Chain]
    embeddings: npt.NDArray[np.float64]

    @classmethod
    def save_attributes(cls, filepath: pl.Path) -> None:
        np.savetxt(filepath.with_suffix('.embeddings'), cls.embeddings, fmt='%.6f', header='embeddings')


class ESM2(EmbeddingOracle):
    """
    Object that uses ESM-2 to predict the embeddings of the residues in the chains.
    """

    result_class = ESM2Result

    def __init__(
        self, use_modal: bool = False, config: dict[str, Any] = {}, modal_app_context: App | None = None
    ) -> None:
        """
        NOTE this can only be called once. Attempting to initialise this object multiple times in one process creates
        breaking exceptions.
        Initialise the ESM2 model.
        WIP: For now we will be using ModalFold to do this reliably without much env issues.
        """
        self.use_modal = use_modal
        self.modal_app_context = modal_app_context
        self.default_config = {
            'output_hidden_states': False,
            'model_name': 'esm2_t33_650M_UR50D',
        }
        if self.use_modal and self.modal_app_context is None:
            # Register the cleanup function to be called at exit, so no
            # ephermal app is left running when the object is destroyed
            import atexit

            atexit.register(self.__del__)

        self._load(config)

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
        self.model = ESM2Boiler(config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """
        Multimers are pre-processed by joining the sequences with a ":" character.
        """
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def embed(self, chains: list[Chain]) -> ESM2Result:
        """
        Calculate the embeddings of the residues in the chains.
        """
        self.input_chains = chains
        processed_chains = self._pre_process(chains)

        if self.use_modal:
            return self._post_process(self._remote_embed(processed_chains))
        else:
            logger.debug('Given that use_modal is False, trying to embed with ESM-2 locally...')
            assert os.environ.get('MODEL_DIR'), 'MODEL_DIR must be set when using ESM-2 locally'
            return self._post_process(self._local_embed(processed_chains))

    def _remote_embed(self, sequence: List[str]) -> ESM2Output:
        return self.model.embed.remote(sequence)

    def _local_embed(self, sequence: List[str]) -> ESM2Output:
        # assert that transformers is installed
        try:
            import transformers
        except ImportError:
            raise ImportError(
                'transformers is not installed. Please install it to use ESM-2 locally. See README.md for installation instructions.'
            )
        return self.model.embed.local(sequence)

    def _post_process(self, output: ESM2Output) -> ESM2Result:
        #! TODO This will need to be reverted back once change in boileroom is done
        # embeddings = output.embeddings[0, 1:-1, :]  # remove first and last token embeddings (not a residue)
        embeddings = output.embeddings[0, :, :]  # remove first and last token embeddings (not a residue)
        assert len(embeddings.shape) == 2, (
            f'Embeddings is expected to be a 2D tensor, not shape: {embeddings.shape}. '
            'The ESM2 Oracle does not support batches.'
        )
        return self.result_class(input_chains=self.input_chains, embeddings=embeddings)
