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
from boileroom.models.esm.types import ESM2Output  # type: ignore
from boileroom.models.esm.esm2 import ESM2 as ESM2Boiler
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

    def __init__(self, backend: str = "modal", config: dict[str, Any] = {}) -> None:
        """
        Initialize ESM2 oracle.

        Parameters
        ----------
        backend : str
            Backend to use. Supported values: "modal", "local"
        config : dict[str, Any]
            Configuration dictionary passed to the model
        """
        self.backend = backend
        self.default_config = {
            'output_hidden_states': False,
            'model_name': 'esm2_t33_650M_UR50D',
        }
        self._load(config)

    def _load(self, config: dict[str, Any] = {}) -> None:
        config = {**self.default_config, **config}
        self.model = ESM2Boiler(backend=self.backend, config=config)

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

        if self.backend == "local":
            logger.debug('Embedding with ESM-2 locally...')
            assert os.environ.get('MODEL_DIR'), 'MODEL_DIR must be set when using ESM-2 locally'
        
        output = self.model.embed(processed_chains)
        return self._post_process(output)

    def _post_process(self, output: ESM2Output) -> ESM2Result:
        #! TODO This will need to be reverted back once change in boileroom is done
        # embeddings = output.embeddings[0, 1:-1, :]  # remove first and last token embeddings (not a residue)
        embeddings = output.embeddings[0, :, :]  # remove first and last token embeddings (not a residue)
        assert len(embeddings.shape) == 2, (
            f'Embeddings is expected to be a 2D tensor, not shape: {embeddings.shape}. '
            'The ESM2 Oracle does not support batches.'
        )
        return self.result_class(input_chains=self.input_chains, embeddings=embeddings)
