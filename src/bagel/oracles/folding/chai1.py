"""Chai-1 structure-prediction oracle."""

from typing import Any

from .base import ConfidenceFoldingOracle, ConfidenceFoldingResult


class Chai1Result(ConfidenceFoldingResult):
    """Structure and normalized confidence metrics from Chai-1."""


class Chai1(ConfidenceFoldingOracle):
    """Predict protein structures with Chai-1 through BoilerRoom."""

    result_class: type[Chai1Result] = Chai1Result
    model_name = 'Chai1'

    def _load(self, config: dict[str, Any] | None = None) -> None:
        from boileroom.models.chai.chai1 import Chai1 as Chai1Boiler

        self.model = Chai1Boiler(backend=self.backend, device=self.device, config=config)
