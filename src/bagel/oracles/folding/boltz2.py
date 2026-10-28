"""Boltz-2 structure-prediction oracle."""

from typing import Any

from .base import ConfidenceFoldingResult, RankedSamplesFoldingOracle


class Boltz2Result(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from Boltz-2."""


class Boltz2(RankedSamplesFoldingOracle):
    """Predict protein structures with Boltz-2 through BoilerRoom.

    ``n_samples`` diffusion samples are generated per prediction (``diffusion_samples``) and ranked by pTM.

    BoilerRoom's Boltz-2 MSA cache is switched off (``msa_cache_enabled=False``, unless you set it in ``config``). The
    cache is keyed by the sequence alone and is consulted even in no-MSA mode (``use_msa_server=False``), so a stale
    MSA could silently be used when none should be; and a fold that mixes cached and new chains, which is every step
    of a design loop (fixed target, new binder), can crash. With the cache off, ``use_msa_server=False`` really means
    no MSA, and ``use_msa_server=True`` fetches the MSAs of every chain at every fold.
    """

    rank_by_ptm = True

    result_class: type[Boltz2Result] = Boltz2Result
    model_name = 'Boltz2'

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.boltz.boltz2 import Boltz2 as Boltz2Boiler

        model_config = dict(config or {})
        self._require_samples(model_config, 'diffusion_samples', default=1)
        model_config.setdefault('msa_cache_enabled', False)
        self.model = Boltz2Boiler(backend=self.backend, device=self.device, config=model_config)
