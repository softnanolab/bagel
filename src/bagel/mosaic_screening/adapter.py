"""
Adapter that lets bagel's own energy-term code run on a mosaic prediction.

Mosaic returns, for a complex, per-residue pLDDT (0-1), the pairwise PAE (Angstrom) and the pLDDT/PAE-derived pTM. The
adapter wraps them in the oracle's own ``result_class`` (with one placeholder atom per residue carrying the chain ID and
residue index) and calls ``term.compute`` on it. So the pseudo-energy is computed by the very code that computes the real
energy; nothing is re-implemented, and the residue groups are applied by bagel's own masking code.

Terms that read only ``local_plddt``, ``pae`` and ``ptm`` need nothing else; terms that read atom coordinates get a structure
rebuilt from mosaic's atom37 output (see :data:`bagel.mosaic_screening.plan.SUPPORTED_TERMS`).
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


#: Order of the 37 heavy-atom slots of mosaic's ``atom37_coords`` (AlphaFold ``atom_types``).
ATOM37_NAMES: tuple[str, ...] = (
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1',
    'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD',
    'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2',
    'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT',
)  # fmt: skip
#: Side-chain heavy atoms of each residue in the order of standard PDB files (what the oracles' structures use); the backbone
#: always comes first as N, CA, C, O.
_SIDE_CHAIN_ORDER = {
    'ALA': 'CB', 'ARG': 'CB CG CD NE CZ NH1 NH2', 'ASN': 'CB CG OD1 ND2', 'ASP': 'CB CG OD1 OD2', 'CYS': 'CB SG',
    'GLN': 'CB CG CD OE1 NE2', 'GLU': 'CB CG CD OE1 OE2', 'GLY': '', 'HIS': 'CB CG ND1 CD2 CE1 NE2',
    'ILE': 'CB CG1 CG2 CD1', 'LEU': 'CB CG CD1 CD2', 'LYS': 'CB CG CD CE NZ', 'MET': 'CB CG SD CE',
    'PHE': 'CB CG CD1 CD2 CE1 CE2 CZ', 'PRO': 'CB CG CD', 'SER': 'CB OG', 'THR': 'CB OG1 CG2',
    'TRP': 'CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2', 'TYR': 'CB CG CD1 CD2 CE1 CE2 CZ OH', 'VAL': 'CB CG1 CG2',
}  # fmt: skip


@dataclass
class MosaicPrediction:
    """
    Outputs of one mosaic forward pass, in mosaic token order (designed chain first).

    ``atom37_coords`` / ``atom37_mask`` (all-atom heavy-atom coordinates in the 37-slot layout) are only present when the
    backend was asked for the structure, which it is when a supported energy term needs coordinates.
    """

    plddt: npt.NDArray[np.float64]  # [n_tokens], 0-1
    pae: npt.NDArray[np.float64]  # [n_tokens, n_tokens], Angstrom
    ptm: float  # 0-1
    atom37_coords: npt.NDArray[np.float64] | None = None  # [n_tokens, 37, 3]
    atom37_mask: npt.NDArray[np.float64] | None = None  # [n_tokens, 37]


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


def structure_from_atom37(
    chains: list[Chain], coords: npt.NDArray[np.float64], mask: npt.NDArray[np.float64]
) -> AtomArray:
    """
    All-atom structure with the annotations and atom order of the structures bagel gets from its oracles: heavy atoms only,
    per residue backbone first (N, CA, C, O) and then the side chain in standard PDB order; no OXT. Chain IDs and residue indices
    are bagel's (``Residue.chain_ID`` and ``Residue.index``), so residue groups select the right residues.
    """
    residues = [r for c in chains for r in c.residues]
    if coords.shape != (len(residues), 37, 3) or mask.shape != (len(residues), 37):
        raise ValueError(f'atom37 arrays {coords.shape}, {mask.shape} do not match {len(residues)} residues')
    slots = []
    for i, residue in enumerate(residues):
        order = ('N CA C O ' + _SIDE_CHAIN_ORDER[residue.three_letter_name]).split()
        rank = {name: r for r, name in enumerate(order)}
        slots.append(
            sorted(
                (k for k in range(36) if mask[i, k] > 0.5),  # slot 36 is OXT
                key=lambda k: (rank.get(ATOM37_NAMES[k], len(order)), k),
            )
        )
    atoms = AtomArray(sum(len(s) for s in slots))
    row = 0
    for i, (residue, ks) in enumerate(zip(residues, slots)):
        for k in ks:
            atoms.coord[row] = coords[i, k]
            atoms.chain_id[row] = residue.chain_ID
            atoms.res_id[row] = residue.index
            atoms.res_name[row] = residue.three_letter_name
            atoms.atom_name[row] = ATOM37_NAMES[k]
            atoms.element[row] = ATOM37_NAMES[k][0]
            row += 1
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
        structure=(
            structure_from_atom37(chains, prediction.atom37_coords, prediction.atom37_mask)
            if prediction.atom37_coords is not None and prediction.atom37_mask is not None
            else placeholder_structure(chains)
        ),
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
