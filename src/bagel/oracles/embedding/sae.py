"""Sparse-autoencoder (SAE) feature oracle.

Wraps BoilerRoom's ``SAE`` model, which maps a sequence to sparse-autoencoder
features of the ESM-C representation space (see the Biohub ESM-C paper). Each
protein is summarized by a single **feature vector**: the max activation of every
SAE feature across its residues. That vector is what :class:`~bagel.energies.SAEnergy`
turns into an energy.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict

from ...chain import Chain
from .base import EmbeddingOracle, EmbeddingResult

if TYPE_CHECKING:
    from boileroom.models.sae.types import SAEFeaturesOutput

logger = logging.getLogger(__name__)


class SAEResult(EmbeddingResult):
    """Stores sparse-autoencoder features for a set of chains.

    Attributes
    ----------
    features : np.ndarray
        Per-protein SAE feature vector of shape ``(num_features,)``, obtained by
        max-pooling each feature across the residues.
    embeddings : np.ndarray | None
        Optional dense per-residue SAE activations of shape
        ``(residues, num_features)``. ``None`` unless requested.
    layer : int
        ESM-C transformer layer the SAE was applied to.
    sae_model : str
        Identifier of the SAE weights used.
    """

    # SAE features are a per-protein vector; per-residue activations are optional,
    # so relax the base requirement that ``embeddings`` is always present.
    embeddings: npt.NDArray[np.float64] | None = None  # type: ignore[assignment]
    features: npt.NDArray[np.float64]
    layer: int = -1
    sae_model: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def save_attributes(self, filepath: Path) -> None:
        np.savetxt(filepath.with_suffix('.features'), self.features, fmt='%.6f', header='sae_features')


class SAE(EmbeddingOracle):
    """Oracle that exposes ESM-C sparse-autoencoder features for a sequence.

    The oracle runs BoilerRoom's ``SAE`` model, which internally runs ESM-C to get
    per-residue hidden states and projects them through a trained sparse
    autoencoder into a high-dimensional, sparse feature space. The pooled
    per-protein feature vector is stored on :class:`SAEResult.features` and is the
    quantity :class:`~bagel.energies.SAEnergy` acts on.

    Parameters
    ----------
    backend : str
        BoilerRoom backend, normally ``"modal"`` or ``"apptainer"``.
    device : str | None
        Optional device passed to BoilerRoom.
    config : dict | None
        Configuration options forwarded to BoilerRoom's ``SAE``. By default this
        uses BoilerRoom's default SAE: the ESMC-6B, layer-60, ``k64``,
        ``codebook16384`` model served through the Biohub **Forge** API (set
        ``forge_token`` here or the ``ESM_API_KEY`` env var). Pass
        ``feature_source="local"`` (plus ``esmc_model_name`` / ``sae_repo_id`` /
        ``sae_layer``) to run the 300M / 600M SAEs locally instead. Other keys:
        ``num_features``, ``k``, ``normalize_features``, ``include_per_residue``.

    Notes
    -----
    Multimers are encoded **jointly**: ``_pre_process`` joins chains with ``':'``
    into a single string that ESM-C runs through one forward pass, so a residue's
    features reflect the other chains (and are order-sensitive), exactly as for the
    :class:`~bagel.oracles.embedding.esmc.ESMC` oracle.
    """

    result_class = SAEResult

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
        # without the boileroom SAE wrapper present, and so mocked oracles in
        # tests can patch out _load entirely.
        from boileroom.models.sae.sae import SAE as SAEBoiler

        # Pass config straight through so BoilerRoom's SAE defaults apply
        # (ESMC-6B / layer 60 via Forge). Use feature_source="local" for the
        # 300M / 600M SAEs.
        self.model = SAEBoiler(backend=self.backend, device=self.device, config=config or {})

    def embed(self, chains: list[Chain]) -> SAEResult:
        """Compute SAE features for the residues in the given chains.

        Parameters
        ----------
        chains : list[Chain]
            Protein chains to featurize.

        Returns
        -------
        SAEResult
            Result with the pooled per-protein feature vector in ``features``.
        """
        processed_chains = self._pre_process(chains)
        output = self.model.embed(processed_chains)
        return self._post_process(output, chains)

    def _post_process(self, output: 'SAEFeaturesOutput', chains: list[Chain]) -> SAEResult:
        pooled = np.asarray(output.pooled_features, dtype=np.float64)
        if pooled.ndim != 2 or pooled.shape[0] != 1:
            raise ValueError(
                f'SAE pooled_features must have shape (1, num_features); got {pooled.shape}. '
                'BAGEL does not support embedding batches.'
            )
        features = pooled[0]
        per_residue = getattr(output, 'features', None)
        embeddings = None
        if per_residue is not None:
            per_residue = np.asarray(per_residue, dtype=np.float64)
            # Drop the leading batch axis for the single sample.
            embeddings = per_residue[0] if per_residue.ndim == 3 else per_residue
        return self.result_class(
            input_chains=chains,
            embeddings=embeddings,
            features=features,
            layer=int(getattr(output, 'layer', -1)),
            sae_model=str(getattr(output, 'sae_model', '')),
        )
