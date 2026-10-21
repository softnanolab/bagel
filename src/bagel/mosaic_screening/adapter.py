"""
Adapter that lets bagel's own energy-term code run on a mosaic prediction.

Mosaic returns, for a complex, per-residue pLDDT (0-1), the pairwise PAE (Angstrom) and the pLDDT/PAE-derived pTM. The
adapter wraps them in the oracle's own ``result_class`` (with one placeholder atom per residue carrying the chain ID and
residue index) and calls ``term.compute`` on it. So the pseudo-energy is computed by the very code that computes the real
energy; nothing is re-implemented, and the residue groups are applied by bagel's own masking code.

Only terms that read nothing but ``local_plddt``, ``pae`` and ``ptm`` can be evaluated this way (see
:data:`bagel.mosaic_screening.plan.SUPPORTED_TERMS`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from biotite.structure import AtomArray
import numpy as np
import numpy.typing as npt

from ..chain import Chain
from ..energies import EnergyTerm
from ..oracles.base import Oracle, OracleResult, OraclesResultDict


@dataclass
class MosaicPrediction:
    """Confidence outputs of one mosaic forward pass, in mosaic token order (designed chain first)."""

    plddt: npt.NDArray[np.float64]  # [n_tokens], 0-1
    pae: npt.NDArray[np.float64]  # [n_tokens, n_tokens], Angstrom
    ptm: float  # 0-1


def placeholder_structure(chains: list[Chain]) -> AtomArray:
    """One CA atom per residue, with the chain ID and residue index bagel uses to build residue masks."""
    n = sum(c.length for c in chains)
    atoms = AtomArray(n)
    atoms.chain_id[:] = np.concatenate([np.full(c.length, c.chain_ID) for c in chains])
    atoms.res_id[:] = np.concatenate([[r.index for r in c.residues] for c in chains])
    atoms.res_name[:] = np.concatenate([[r.three_letter_name for r in c.residues] for c in chains])
    atoms.atom_name[:] = 'CA'
    atoms.element[:] = 'C'
    return atoms


def build_result(oracle: Oracle, chains: list[Chain], prediction: MosaicPrediction) -> OracleResult:
    """Wrap a mosaic prediction in ``oracle.result_class``; ``chains`` must be in mosaic token order."""
    n = sum(c.length for c in chains)
    if prediction.plddt.shape != (n,) or prediction.pae.shape != (n, n):
        raise ValueError(
            f'Mosaic returned plddt {prediction.plddt.shape} and pae {prediction.pae.shape} for {n} residues.'
        )
    result_class: Any = oracle.result_class  # a ConfidenceFoldingResult subclass: plddt, pae and ptm are its fields
    result: OracleResult = result_class(
        input_chains=chains,
        structure=placeholder_structure(chains),
        local_plddt=np.clip(prediction.plddt, 0.0, 1.0)[None, :],
        ptm=np.clip(np.asarray(prediction.ptm, dtype=np.float64), 0.0, 1.0).reshape(1, 1),
        pae=prediction.pae[None, :, :],
    )
    return result


def pseudo_energy(
    terms: list[EnergyTerm], chains: list[Chain], prediction: MosaicPrediction
) -> tuple[float, dict[str, float]]:
    """
    Weighted sum of ``terms`` evaluated on a mosaic prediction, and the unweighted value of each term.

    The terms may use different oracle objects (their results are all built from the same prediction).
    """
    results = OraclesResultDict()
    for term in terms:
        if term.oracle not in results:
            results[term.oracle] = build_result(term.oracle, chains, prediction)
    total = 0.0
    per_term: dict[str, float] = {}
    for term in terms:
        unweighted, weighted = term.compute(results)
        total += float(np.asarray(weighted).reshape(-1)[0])
        per_term[term.name] = float(np.asarray(unweighted).reshape(-1)[0])
    return total, per_term
