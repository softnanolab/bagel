"""Boltz-2 structure-prediction oracle."""

from typing import Any

from .base import ConfidenceFoldingOracle, ConfidenceFoldingResult


class Boltz2Result(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from Boltz-2."""


class Boltz2(ConfidenceFoldingOracle):
    """Predict protein structures with Boltz-2 through BoilerRoom."""

    result_class: type[Boltz2Result] = Boltz2Result
    model_name = 'Boltz2'

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.boltz.boltz2 import Boltz2 as Boltz2Boiler

        self.model = Boltz2Boiler(backend=self.backend, device=self.device, config=config)
