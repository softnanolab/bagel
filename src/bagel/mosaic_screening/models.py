"""
Which mosaic model stands in for which bagel folding oracle, and with which (cheap) settings.

The oracle that evaluates the *real* energy in bagel and the model that screens candidates in mosaic are independent:
a :class:`MosaicModelSpec` is given per oracle (or taken from the defaults below), so one can, for instance, accept moves
with an ``ESMFold2`` oracle at its default settings while screening with mosaic's ESMFold2 at 14 sampling steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..oracles.base import Oracle


@dataclass(frozen=True)
class MosaicModelSpec:
    """
    Settings of the mosaic structure model used to screen candidate mutations.

    Parameters
    ----------
    model : {'esmfold2', 'boltz2'}
        Mosaic structure model.
    recycling_steps : int
        Trunk recycling steps of the mosaic forward pass.
    sampling_steps : int
        Diffusion sampling steps of the mosaic forward pass. Fewer steps are cheaper and give a noisier structure;
        screening only needs the ranking of candidates, not a final structure.
    checkpoint : {'full', 'fast'}
        ESMFold2 only: mosaic's ``ESMFold2Full`` (the checkpoint bagel's ``ESMFold2`` oracle loads by default) or
        ``ESMFold2Fast``. Ignored for Boltz2.
    """

    model: str
    recycling_steps: int
    sampling_steps: int
    checkpoint: str = 'full'

    def __post_init__(self) -> None:
        if self.model not in ('esmfold2', 'boltz2'):
            raise ValueError(f"model must be 'esmfold2' or 'boltz2', got {self.model!r}")
        if self.recycling_steps < 0 or self.sampling_steps < 1:
            raise ValueError('recycling_steps must be >= 0 and sampling_steps >= 1.')
        if self.checkpoint not in ('full', 'fast'):
            raise ValueError("checkpoint must be 'full' or 'fast'")

    @classmethod
    def esmfold2(
        cls, recycling_steps: int = 3, sampling_steps: int = 14, checkpoint: str = 'full'
    ) -> 'MosaicModelSpec':
        """ESMFold2; the defaults are the ones the screening was validated with."""
        return cls('esmfold2', recycling_steps, sampling_steps, checkpoint)

    @classmethod
    def boltz2(cls, recycling_steps: int = 1, sampling_steps: int = 25) -> 'MosaicModelSpec':
        """Boltz-2 with mosaic's own cheap defaults (1 recycling step, 25 sampling steps, empty MSA)."""
        return cls('boltz2', recycling_steps, sampling_steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def label(self) -> str:
        extra = f'/{self.checkpoint}' if self.model == 'esmfold2' else ''
        return f'{self.model}{extra} (recycling={self.recycling_steps}, sampling={self.sampling_steps})'


def default_spec_for(oracle: Oracle) -> MosaicModelSpec | None:
    """Default mosaic model for a bagel oracle, or None if mosaic has no counterpart (e.g. ESMFold, Chai-1, embeddings)."""
    # Compared by class name so that importing this module does not import the oracle backends.
    name = type(oracle).__name__
    if name == 'ESMFold2':
        return MosaicModelSpec.esmfold2()
    if name == 'Boltz2':
        return MosaicModelSpec.boltz2()
    return None
