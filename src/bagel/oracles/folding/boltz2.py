"""Boltz-2 structure-prediction oracle."""

from typing import Any

from .base import ConfidenceFoldingResult, RankedSamplesFoldingOracle


class Boltz2Result(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from Boltz-2."""


class Boltz2(RankedSamplesFoldingOracle):
    """Predict protein structures with Boltz-2 through BoilerRoom.

    ``n_samples`` diffusion samples are generated per prediction (``diffusion_samples``) and ranked by pTM.

    **MSAs.** Each chain gets one source, in this order (boileroom, which does the work, documents it in full):

    1. an alignment you provide with ``msa={chain_ID: A3M text or A3M file}``: the alignment of the sequence in its
       first row. It is used while the chain has that sequence, never fetched again or overwritten, and the server is
       not called for it. If the chain is mutated it no longer applies (no error) and the new sequence is treated like
       any other; it applies again if the sequence returns to the original;
    2. no MSA server (``config={'use_msa_server': False}``): the chain is folded alone and the cache is never read;
    3. the cache, with the server on: the MSA of this sequence from an earlier fold;
    4. the server: the MSA of the chain on its own (unpaired) is fetched and cached by sequence, so it is not asked for
       again unless the sequence changes.

    ``use_msa_server`` is True by default (boileroom's default), so a chain without an alignment, such as a designed
    binder, gets its MSA from the server; set ``config={'use_msa_server': False}`` to fold it alone.
    """

    rank_by_ptm = True
    supports_user_msa = True
    strict_msa_sequence = False

    result_class: type[Boltz2Result] = Boltz2Result
    model_name = 'Boltz2'

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.boltz.boltz2 import Boltz2 as Boltz2Boiler

        model_config = dict(config or {})
        self._require_samples(model_config, 'diffusion_samples', default=1)
        # the model gets exactly the config the user gave unless more than one sample is needed
        passed = model_config if (config is not None or self.n_samples > 1) else None
        self.model = Boltz2Boiler(backend=self.backend, device=self.device, config=passed)
