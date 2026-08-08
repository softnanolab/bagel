"""ESM-2 embedding oracle."""

from ...chain import Chain
from .base import EmbeddingResult, EmbeddingOracle, single_sample_embeddings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from boileroom.models.esm.types import ESM2Output


class ESM2Result(EmbeddingResult):
    """Residue embeddings from ESM-2."""


class ESM2(EmbeddingOracle):
    """
    Object that uses ESM-2 to predict the embeddings of the residues in the chains.
    """

    result_class = ESM2Result

    def __init__(
        self,
        backend: str = 'modal',
        device: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize ESM2 oracle.

        Parameters
        ----------
        backend : str
            Backend to use. Supported values: "modal", "apptainer", "apptainer:<image-tag>".
        config : dict[str, Any]
            Configuration dictionary passed to the model
        """
        self.backend = backend
        self.device = device
        self._load(config)

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.esm.esm2 import ESM2 as ESM2Boiler

        self.model = ESM2Boiler(backend=self.backend, device=self.device, config=config)

    def embed(self, chains: list[Chain]) -> ESM2Result:
        """
        Calculate the embeddings of the residues in the chains.
        """
        processed_chains = self._pre_process(chains)
        output = self.model.embed(processed_chains)
        return self._post_process(output, chains)

    def _post_process(self, output: 'ESM2Output', chains: list[Chain]) -> ESM2Result:
        return self.result_class(
            input_chains=chains,
            embeddings=single_sample_embeddings(output.embeddings, 'ESM2'),
        )
