"""Exceptions of the mosaic screening protocol."""


class MosaicIncompatibleError(ValueError):
    """Raised when a bagel setup cannot be handled by :class:`~bagel.mosaic_screening.MosaicScreening`.

    It is raised *before* any call to mosaic, with a message that says what is incompatible and what to change.
    """


class MosaicBackendError(RuntimeError):
    """Raised when the mosaic backend cannot be reached or returns an invalid answer."""
