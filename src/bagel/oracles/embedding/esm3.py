"""
ESM3 oracle for multi-track predictions via boileroom's ESM3 wrapper.

ESM3 is an all-to-all masked model, so beyond sequence embeddings it can predict
per-residue tracks (SASA, secondary structure, function, residue annotations)
from sequence alone. This oracle requests those tracks from boileroom and decodes
the ones with a clean scalar/label interpretation (SASA -> Angstrom^2 expected
value; secondary structure -> SS8 letters); function/residue-annotation logits are
surfaced raw (decoding those to labels needs the SDK's large vocabularies).

The boileroom wrapper is imported lazily inside ``_load`` so this module stays
importable on the pinned boileroom (which predates the ESM3 track outputs); the
track outputs require boileroom >= the release that ships
``boileroom.models.esm3`` track logits.

Decoding constants below are copied from EvolutionaryScale's ``esm`` SDK
(``esm.utils.constants.esm3``); ``esm`` is not a bagel dependency, so they are
inlined with their provenance rather than imported.
"""

import pathlib as pl
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from ...chain import Chain
from .base import EmbeddingResult, EmbeddingOracle

if TYPE_CHECKING:
    from boileroom.models.esm3.types import ESM3Output  # type: ignore

logger = logging.getLogger(__name__)


# --- Decoding constants (from esm.utils.constants.esm3) -----------------------
# SASA discretization boundaries used by the SDK's SASADiscretizingTokenizer. The
# bins are [0, b0], [b0, b1], ..., [b_last, 2*b_last]; a residue's SASA logits are a
# distribution over these bins (after the tokenizer's special tokens).
_SASA_BOUNDARIES = [0.8, 4.0, 9.6, 16.4, 24.5, 32.9, 42.0, 51.5, 61.2, 70.9, 81.6, 93.3, 107.2, 125.4, 151.4]
_SASA_BIN_EDGES = np.array([0.0, *_SASA_BOUNDARIES, _SASA_BOUNDARIES[-1] * 2.0], dtype=np.float64)
# Representative Angstrom^2 value per bin (bin midpoints); length 16.
_SASA_BIN_MIDPOINTS = (_SASA_BIN_EDGES[:-1] + _SASA_BIN_EDGES[1:]) / 2.0
# SS8 alphabet (esm SSE_8CLASS_VOCAB); the SDK head's non-special classes, in order.
_SS8_VOCAB = 'GHITEBSC'


def _softmax(logits: npt.NDArray[np.float64], axis: int = -1) -> npt.NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def decode_sasa(sasa_logits: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Decode SASA logits to per-residue SASA (Angstrom^2) as an expected value.

    Parameters
    ----------
    sasa_logits : ndarray
        ``(..., residues, vocab)`` logits over the SASA token vocabulary. Only the
        trailing ``len(_SASA_BIN_MIDPOINTS)`` bin logits are used (leading special
        tokens, if present, are ignored).

    Returns
    -------
    ndarray
        ``(..., residues)`` expected SASA in Angstrom^2: ``softmax(bins) . midpoints``.
    """
    n_bins = _SASA_BIN_MIDPOINTS.shape[0]
    bin_logits = np.asarray(sasa_logits, dtype=np.float64)[..., -n_bins:]
    probabilities = _softmax(bin_logits, axis=-1)
    return probabilities @ _SASA_BIN_MIDPOINTS


def decode_secondary_structure(ss_logits: npt.NDArray[np.float64]) -> Any:
    """Decode SS8 logits to per-residue secondary-structure letters (argmax).

    Uses the last ``len(_SS8_VOCAB)`` classes (``GHITEBSC``). For a single sequence
    (2D ``(residues, vocab)`` input) returns an SS8 string; for batched input returns
    an array of single-character labels.
    """
    n_classes = len(_SS8_VOCAB)
    indices = np.argmax(np.asarray(ss_logits, dtype=np.float64)[..., -n_classes:], axis=-1)
    letters = np.array(list(_SS8_VOCAB))[indices]
    if letters.ndim == 1:
        return ''.join(letters.tolist())
    return letters


class ESM3Result(EmbeddingResult):
    """Embedding + decoded multi-track results from ESM3.

    ``sasa`` (Angstrom^2) and ``secondary_structure`` (SS8 string) are decoded;
    ``function_logits`` and ``residue_annotation_logits`` are raw per-residue logits.
    All track fields are ``None`` unless the corresponding track was requested.
    """

    input_chains: list[Chain]
    embeddings: npt.NDArray[np.float64]
    sasa: npt.NDArray[np.float64] | None = None
    secondary_structure: str | None = None
    function_logits: npt.NDArray[np.float64] | None = None
    residue_annotation_logits: npt.NDArray[np.float64] | None = None

    @classmethod
    def save_attributes(cls, filepath: pl.Path) -> None:
        np.savetxt(filepath.with_suffix('.embeddings'), cls.embeddings, fmt='%.6f', header='embeddings')


class ESM3(EmbeddingOracle):
    """Oracle that uses ESM3 to compute embeddings and per-residue track predictions.

    Parameters
    ----------
    use_modal : bool
        Whether to run the boileroom wrapper via Modal (otherwise Apptainer).
    config : dict
        Configuration forwarded to the boileroom ESM3 wrapper (e.g. ``model_name``).
    tracks : list[str]
        Extra tracks to request/decode, any of: ``"sasa"``, ``"secondary_structure"``,
        ``"function"``, ``"residue_annotations"``. Empty by default (embeddings only).
        The structure/folding track is not available through this oracle.
    """

    result_class = ESM3Result

    # bagel track name -> (boileroom include_fields key, result attribute, decoder tag)
    _TRACKS: dict[str, tuple[str, str, str | None]] = {
        'sasa': ('sasa_logits', 'sasa', 'sasa'),
        'secondary_structure': ('secondary_structure_logits', 'secondary_structure', 'ss'),
        'function': ('function_logits', 'function_logits', None),
        'residue_annotations': ('residue_annotation_logits', 'residue_annotation_logits', None),
    }

    def __init__(
        self,
        use_modal: bool = False,
        config: dict[str, Any] | None = None,
        tracks: list[str] | None = None,
    ) -> None:
        if config is None:
            config = {}
        self.use_modal = use_modal
        self.tracks = list(tracks or [])
        unknown = [track for track in self.tracks if track not in self._TRACKS]
        if unknown:
            raise ValueError(f'Unknown ESM3 tracks: {unknown}. Valid tracks: {sorted(self._TRACKS)}')
        self.default_config: dict[str, Any] = {'model_name': 'esm3_sm_open_v1'}
        self._load(config)

    def _load(self, config: dict[str, Any] | None = None) -> None:
        # Lazy import: keeps this module importable without the boileroom ESM3
        # wrapper present, and lets tests patch _load out.
        from boileroom.models.esm3.esm3 import ESM3 as ESM3Boiler  # type: ignore

        if config is None:
            config = {}
        merged_config = {**self.default_config, **config}
        backend = 'modal' if self.use_modal else 'apptainer'
        self.model = ESM3Boiler(backend=backend, config=merged_config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """Join chains with ':' for multimers (encoded jointly by ESM3)."""
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def embed(self, chains: list[Chain]) -> ESM3Result:
        """Compute ESM3 embeddings and any requested/decoded tracks for the chains."""
        self.input_chains = chains
        include_fields = [self._TRACKS[track][0] for track in self.tracks]
        options = {'include_fields': include_fields} if include_fields else None
        output = self.model.embed(self._pre_process(chains), options=options)
        return self._post_process(output)

    def _post_process(self, output: 'ESM3Output') -> ESM3Result:
        embeddings = output.embeddings[0, :, :]
        assert len(embeddings.shape) == 2, (
            f'Embeddings is expected to be a 2D tensor, not shape: {embeddings.shape}. '
            'The ESM3 Oracle does not support batches.'
        )
        result_kwargs: dict[str, Any] = {'input_chains': self.input_chains, 'embeddings': embeddings}
        for track in self.tracks:
            field, attribute, decoder = self._TRACKS[track]
            logits = getattr(output, field, None)
            if logits is None:
                continue
            per_residue = np.asarray(logits)[0]  # drop batch -> (residues, vocab)
            if decoder == 'sasa':
                result_kwargs['sasa'] = decode_sasa(per_residue)
            elif decoder == 'ss':
                result_kwargs['secondary_structure'] = decode_secondary_structure(per_residue)
            else:
                result_kwargs[attribute] = per_residue
        return self.result_class(**result_kwargs)
