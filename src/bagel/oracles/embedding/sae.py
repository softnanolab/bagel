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

# BoilerRoom's default SAE model (ESMC-6B / layer 60 / k64 / codebook 16384), served
# through the Forge API. Duplicated here (rather than importing SAECore, which pulls
# torch) so the oracle can report its configured model id without heavy imports.
DEFAULT_FORGE_SAE_MODEL = 'esmc-6b-2024-12-sae-layer60-k64-codebook16384'


def _resolve_sae_model_id(config: dict[str, Any]) -> str:
    """Identifier of the SAE model a given config selects.

    Mirrors BoilerRoom's default resolution: the Forge ``forge_sae_model`` for the
    (default) forge source, or the local ``sae_repo_id`` when ``feature_source`` is
    ``'local'``. Returns an empty string when a local repo id is not specified.
    """
    source = str(config.get('feature_source', 'forge'))
    if source == 'local':
        return str(config.get('sae_repo_id', ''))
    return str(config.get('forge_sae_model', DEFAULT_FORGE_SAE_MODEL))


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
    # Per-residue identity of every embedding/activation row, as reported by
    # BoilerRoom: ``chain_index`` is the 0-based chain ordinal (in input order) and
    # ``residue_index`` is the 0-based index of the residue within its chain. These
    # let residue-selective terms (e.g. ResidueSAEnergy) cross-check the row->residue
    # mapping they reconstruct from ``input_chains``.
    chain_index: npt.NDArray[np.int_] | None = None
    residue_index: npt.NDArray[np.int_] | None = None

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

        cfg = config or {}
        # Identifier of the SAE model this oracle is configured to use, resolved
        # from the config the same way BoilerRoom resolves its defaults (ESMC-6B /
        # layer 60 via Forge). Exposed so energy terms can require a specific SAE
        # model. Kept torch-free: we do not import SAECore just to read its default.
        self.sae_model_id = _resolve_sae_model_id(cfg)

        # Pass config straight through so BoilerRoom's SAE defaults apply
        # (ESMC-6B / layer 60 via Forge). Use feature_source="local" for the
        # 300M / 600M SAEs.
        self.model = SAEBoiler(backend=self.backend, device=self.device, config=cfg)

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
        chain_index = self._single_sample_indices(getattr(output, 'chain_index', None))
        residue_index = self._single_sample_indices(getattr(output, 'residue_index', None))
        return self.result_class(
            input_chains=chains,
            embeddings=embeddings,
            features=features,
            layer=int(getattr(output, 'layer', -1)),
            sae_model=str(getattr(output, 'sae_model', '')),
            chain_index=chain_index,
            residue_index=residue_index,
        )

    @staticmethod
    def _single_sample_indices(values: Any) -> npt.NDArray[np.int_] | None:
        """Extract a 1-D per-residue index array for BAGEL's single-sample batch.

        BoilerRoom returns ``chain_index`` / ``residue_index`` as ``(1, residues)``
        (padded with ``-1``) or ``(residues,)``. Returns the unpadded 1-D array, or
        ``None`` if the field is absent.
        """
        if values is None:
            return None
        arr = np.asarray(values)
        if arr.ndim == 2:
            if arr.shape[0] != 1:
                raise ValueError(
                    f'SAE chain/residue index must have shape (1, residues); got {arr.shape}. '
                    'BAGEL does not support embedding batches.'
                )
            arr = arr[0]
        arr = np.asarray(arr, dtype=int)
        return arr[arr != -1]
