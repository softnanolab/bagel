from .base import Oracle, OracleResult, OraclesResultDict
from .embedding import EmbeddingOracle, ESM2, ESM2Result
from .folding import FoldingOracle, ESMFold, ESMFoldResult

# Protenix imports are conditional -- they may not be available if boileroom's
# Protenix wrapper has import issues (e.g. Modal version incompatibility).
try:
    from .folding import ProtenixOracle, ProtenixFoldResult
    if ProtenixOracle is None:
        raise ImportError("ProtenixOracle was not loaded (see earlier warning)")
except Exception as _e:
    import logging as _logging
    _logging.getLogger(__name__).warning(f"ProtenixOracle not available: {_e}")
    # Fall back to direct import attempt
    try:
        from .folding.protenix import ProtenixOracle, ProtenixFoldResult  # type: ignore[assignment]
    except Exception:
        ProtenixOracle = None  # type: ignore[assignment,misc]
        ProtenixFoldResult = None  # type: ignore[assignment,misc]

__all__ = [
    'Oracle',
    'OracleResult',
    'OraclesResultDict',
    'ESM2',
    'ESM2Result',
    'ESMFold',
    'ESMFoldResult',
    'EmbeddingOracle',
    'FoldingOracle',
    'ProtenixOracle',
    'ProtenixFoldResult',
]
