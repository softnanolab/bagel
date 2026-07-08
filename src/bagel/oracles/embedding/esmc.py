"""
ESMC oracle for computing protein embeddings using EvolutionaryScale's ESM-C models.

The heavy ``boileroom`` model wrappers are imported lazily inside ``_load`` so
that importing this module (and constructing mocked oracles in tests) does not
require the ``boileroom.models.esm3`` package to be installed. The ESM-C wrapper
lives in boileroom >= 0.3.x (``boileroom.models.esm3.esmc``); older pinned
versions do not ship it.
"""

import pathlib as pl
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ...chain import Chain
from .base import EmbeddingResult, EmbeddingOracle

if TYPE_CHECKING:
    from boileroom.models.esm3.types import ESMCOutput  # type: ignore

logger = logging.getLogger(__name__)


class ESMCResult(EmbeddingResult):
    """Stores embedding results from ESM-C."""

    input_chains: list[Chain]
    embeddings: npt.NDArray[np.float64]

    @classmethod
    def save_attributes(cls, filepath: pl.Path) -> None:
        np.savetxt(filepath.with_suffix('.embeddings'), cls.embeddings, fmt='%.6f', header='embeddings')


class ESMC(EmbeddingOracle):
    """Oracle that uses ESM-C to compute per-residue embeddings.

    ESM-C (ESM Cambrian) is a protein language model from EvolutionaryScale
    trained on billions of protein sequences. It supports 300M, 600M, and
    6B parameter variants.

    Parameters
    ----------
    use_modal : bool
        Whether to run inference remotely via Modal.
    config : dict
        Configuration options. Supported keys:
        - model_name: ESM-C model variant (default: "esmc_600m")
    """

    result_class = ESMCResult

    def __init__(
        self,
        use_modal: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        if config is None:
            config = {}
        self.use_modal = use_modal
        self.default_config: dict[str, Any] = {
            'model_name': 'esmc_600m',
        }
        self._load(config)

    def _load(self, config: dict[str, Any] | None = None) -> None:
        # Imported here (not at module scope) so the module stays importable
        # without the boileroom ESM-C wrapper present, and so mocked oracles in
        # tests can patch out _load entirely.
        from boileroom.models.esm3.esmc import ESMC as ESMCBoiler  # type: ignore

        if config is None:
            config = {}
        merged_config = {**self.default_config, **config}
        backend = 'modal' if self.use_modal else 'apptainer'
        self.model = ESMCBoiler(backend=backend, config=merged_config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """Join chains with ':' separator for multimers."""
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def embed(self, chains: list[Chain]) -> ESMCResult:
        """Compute ESM-C embeddings for the residues in the given chains.

        Parameters
        ----------
        chains : list[Chain]
            List of protein chains to embed.

        Returns
        -------
        ESMCResult
            Embedding result with per-residue embeddings.
        """
        self.input_chains = chains
        processed_chains = self._pre_process(chains)
        output = self.model.embed(processed_chains)
        return self._post_process(output)

    def _post_process(self, output: 'ESMCOutput') -> ESMCResult:
        embeddings = output.embeddings[0, :, :]
        assert len(embeddings.shape) == 2, (
            f'Embeddings is expected to be a 2D tensor, not shape: {embeddings.shape}. '
            'The ESMC Oracle does not support batches.'
        )
        return self.result_class(input_chains=self.input_chains, embeddings=embeddings)
