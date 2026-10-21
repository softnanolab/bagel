"""
Screening of candidate mutations with mosaic (JAX) forward passes. See :class:`MosaicScreening`.

Importing this package does not import JAX or mosaic: they live in a separate environment (a Modal class, or a
subprocess/Apptainer image) because their torch version conflicts with the one bagel's boileroom pins.
"""

from .backends import ModalMosaicBackend, MosaicBackend, SubprocessMosaicBackend
from .errors import MosaicBackendError, MosaicIncompatibleError
from .models import MosaicModelSpec, default_spec_for
from .plan import SUPPORTED_TERMS, build_plans
from .protocol import MosaicScreening

__all__ = [
    'ModalMosaicBackend',
    'MosaicBackend',
    'MosaicBackendError',
    'MosaicIncompatibleError',
    'MosaicModelSpec',
    'MosaicScreening',
    'SUPPORTED_TERMS',
    'SubprocessMosaicBackend',
    'build_plans',
    'default_spec_for',
]
