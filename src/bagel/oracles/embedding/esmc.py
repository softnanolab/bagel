"""ESM-C embedding oracle."""

import logging
from typing import TYPE_CHECKING, Any

from ...chain import Chain
from .base import EmbeddingResult, EmbeddingOracle, single_sample_embeddings

if TYPE_CHECKING:
    from boileroom.models.esm3.types import ESMCOutput

logger = logging.getLogger(__name__)


class ESMCResult(EmbeddingResult):
    """Stores embedding results from ESM-C."""


class ESMC(EmbeddingOracle):
    """Oracle that uses ESM-C to compute per-residue embeddings.

    ESM-C (ESM Cambrian) is a protein language model from EvolutionaryScale
    trained on billions of protein sequences. BoilerRoom 0.4.1 supports the
    300M and 600M variants.

    Parameters
    ----------
    backend : str
        BoilerRoom backend, normally ``"modal"`` or ``"apptainer"``.
    device : str | None
        Optional device passed to BoilerRoom.
    config : dict
        Configuration options. Supported keys:
        - model_name: ESM-C model variant (default: "esmc_600m")

    Notes
    -----
    Multimers are encoded **jointly**, not per chain. ``_pre_process`` joins the
    chains with ``':'`` into a single string, which boileroom turns into one
    ``SEQ1|SEQ2`` sequence (``'|'`` is ESM-C's chain-break token) and runs through
    a **single forward pass**. Self-attention therefore spans all chains, so a
    residue's embedding reflects the other chains and differs from the embedding
    it would get in isolation.

    This also makes the embeddings **order-sensitive**: they are *not* permutation
    invariant across chain order. ESM-C uses rotary (RoPE) positions assigned over
    the whole concatenated stream with no per-chain reset, so cross-chain relative
    positions — and hence attention — change if the chain order changes. Concretely,
    ``embed([chain_A, chain_B]) != permute(embed([chain_B, chain_A]))``. Treat the
    order of ``chains`` as meaningful and stable. To obtain isolated per-chain
    embeddings instead, embed each chain in a separate call.
    """

    result_class = ESMCResult

    def __init__(
        self,
        backend: str = 'modal',
        device: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.device = device
        self._load(config)

    def _load(self, config: dict[str, Any] | None = None) -> None:
        # Imported here (not at module scope) so the module stays importable
        # without the boileroom ESM-C wrapper present, and so mocked oracles in
        # tests can patch out _load entirely.
        from boileroom.models.esm3.esmc import ESMC as ESMCBoiler

        config = {'model_name': 'esmc_600m', **(config or {})}
        self.model = ESMCBoiler(backend=self.backend, device=self.device, config=config)

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
        processed_chains = self._pre_process(chains)
        output = self.model.embed(processed_chains)
        return self._post_process(output, chains)

    def _post_process(self, output: 'ESMCOutput', chains: list[Chain]) -> ESMCResult:
        return self.result_class(
            input_chains=chains,
            embeddings=single_sample_embeddings(output.embeddings, 'ESMC'),
        )
