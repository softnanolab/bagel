from .base import FoldingResult, FoldingOracle
from .esmfold import ESMFold, ESMFoldResult

# Protenix imports are kept separate so that a missing or broken boileroom
# Protenix installation does not prevent the rest of BAGEL from loading.
try:
    from .protenix import ProtenixOracle, ProtenixFoldResult
except Exception as e:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        f"Could not import ProtenixOracle (Protenix support disabled): {e}"
    )
    ProtenixOracle = None  # type: ignore[assignment,misc]
    ProtenixFoldResult = None  # type: ignore[assignment,misc]

__all__ = [
    'FoldingOracle',
    'FoldingResult',
    'ESMFold',
    'ESMFoldResult',
    'ProtenixOracle',
    'ProtenixFoldResult',
]
