"""Chai-1 structure-prediction oracle."""

from typing import Any

from .base import ConfidenceFoldingResult, RankedSamplesFoldingOracle


class Chai1Result(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from Chai-1."""


class Chai1(RankedSamplesFoldingOracle):
    """Predict protein structures with Chai-1 through BoilerRoom.

    ``n_samples`` diffusion samples are generated per prediction (``num_diffn_samples``, 1 unless asked otherwise) and
    ranked by pTM.
    """

    rank_by_ptm = True

    result_class: type[Chai1Result] = Chai1Result
    model_name = 'Chai1'

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.chai.chai1 import Chai1 as Chai1Boiler

        model_config = dict(config or {})
        model_config.setdefault('num_diffn_samples', self.n_samples)
        self._require_samples(model_config, 'num_diffn_samples', default=self.n_samples)
        self.model = Chai1Boiler(backend=self.backend, device=self.device, config=model_config)
