"""
Post-hoc screening of acid-labile designs produced by `histidine_ph_switch.py`.

Folds each candidate once and reports the structural quantities that decide
whether the pH switch is real, then ranks the survivors.

WHY THIS IS A FILTER AND NOT AN ENERGY TERM
-------------------------------------------
Two of the three measurements here (buried histidine count, histidine-carboxylate
contacts) are already energy terms in the design script, and are recomputed here
only so you can read them off a finished design.

The third, the charge-mimic comparison, is deliberately NOT an energy term. It
folds a copy of the sequence with every histidine replaced by a cationic residue
and reports how far the predicted confidence falls. As an optimisation target
that comparison is a bad idea:

  - ESMFold has no electrostatics. It is a sequence-statistics model with no
    desolvation term and no charge-charge term. "Does the arginine variant fold
    worse" is therefore not an electrostatics question; it asks whether that
    sequence looks less like a natural folded protein to a language model. The
    two overlap loosely, because buried arginine genuinely is rare in natural
    proteins, but the score never tells you which one you measured.
  - Arginine and lysine are both longer than histidine and put the charge
    further from the backbone, and neither is a planar aromatic cation with the
    charge delocalised over two ring nitrogens. A confidence drop may just be a
    packing clash.
  - Optimising against it rewards marginal stability, since one cheap way to
    make a variant fold badly is to make the parent barely fold at all.
  - It doubles oracle cost per Monte Carlo step.

Run once per finished candidate, none of that matters much, and a large drop is
still weak evidence that the fold does not tolerate charge at those positions.
Treat it as a tie-breaker between designs that already pass the structural
filters, never as a primary criterion.

For an actual predicted transition midpoint, run PROPKA or a Poisson-Boltzmann
calculation on the structures this script writes. Those compute the desolvation
and charge-charge terms properly. Nothing here does.

USAGE
-----
    python scripts/ph_switch/filter_designs.py designs.fasta --out results.csv

`designs.fasta` may be FASTA, or a plain text file with one sequence per line.
Set BAGEL_BACKEND=apptainer to run the oracle locally instead of on Modal.
"""

from __future__ import annotations

import os
import copy
import argparse
import pathlib as pl
from typing import Any, Iterable

import numpy as np
import pandas as pd

import bagel as bg

from histidine_ph_switch import _relative_residue_sasa, HistidineCarboxylateContactEnergy


IMIDAZOLE_NITROGENS = HistidineCarboxylateContactEnergy.IMIDAZOLE_NITROGENS
CARBOXYLATE_OXYGENS = HistidineCarboxylateContactEnergy.CARBOXYLATE_OXYGENS


# ----------------------------------------------------------------------------
# Structural measurements, taken straight off a predicted structure.
# ----------------------------------------------------------------------------
def buried_histidines(structure: Any, sasa_cutoff: float = 0.15) -> tuple[int, float]:
    """
    Count histidines whose relative SASA falls below `sasa_cutoff`.

    Returns (hard_count, soft_sum). `hard_count` is the plain number below the
    threshold, which is what you quote. `soft_sum` is the graded burial score the
    design script optimises, summing 1 - min(rel_sasa / cutoff, 1) over every
    histidine; it distinguishes a design with three marginally buried histidines
    from one with three deeply buried ones.
    """
    if len(structure) == 0:
        return 0, 0.0
    _, _, res_names, relative = _relative_residue_sasa(structure)
    his = res_names == 'HIS'
    if not np.any(his):
        return 0, 0.0
    rel_his = relative[his]
    hard = int(np.count_nonzero(rel_his < sasa_cutoff))
    soft = float(np.sum(1.0 - np.minimum(rel_his / sasa_cutoff, 1.0)))
    return hard, soft


def histidine_carboxylate_contacts(structure: Any, distance_cutoff: float = 4.5) -> int:
    """
    Count imidazole nitrogens within `distance_cutoff` of an Asp/Glu carboxylate oxygen.

    Any non-zero count is a reason to look at the design by hand. These dyads
    raise the histidine pKa and stabilise the protonated form, which pushes the
    transition the wrong way and can invert the switch outright.
    """
    if len(structure) == 0:
        return 0
    his_mask = (structure.res_name == 'HIS') & np.isin(structure.atom_name, IMIDAZOLE_NITROGENS)
    acid_mask = np.isin(structure.res_name, ('ASP', 'GLU')) & np.isin(structure.atom_name, CARBOXYLATE_OXYGENS)
    if not np.any(his_mask) or not np.any(acid_mask):
        return 0
    d = np.linalg.norm(structure.coord[his_mask][:, None, :] - structure.coord[acid_mask][None, :, :], axis=-1)
    return int(np.count_nonzero(d < distance_cutoff))


# ----------------------------------------------------------------------------
# The charge-mimic comparison. Tie-breaker only; see the module docstring.
# ----------------------------------------------------------------------------
def charge_mimic_ptm(oracle: Any, chains: list[Any], mimic_residue: str) -> float | None:
    """
    Fold a copy of `chains` with every histidine replaced by `mimic_residue`.

    Returns the mimic's pTM, or None when the sequence has no histidine and the
    comparison is undefined.
    """
    mimic_chains = copy.deepcopy(chains)
    n_substituted = 0
    for chain in mimic_chains:
        for residue in chain.residues:
            if residue.name == 'H':
                residue.name = mimic_residue
                n_substituted += 1
    if n_substituted == 0:
        return None
    result = oracle.predict(chains=mimic_chains)
    return float(np.asarray(result.ptm).reshape(-1)[0])


# ----------------------------------------------------------------------------
# Input handling.
# ----------------------------------------------------------------------------
def read_sequences(path: pl.Path) -> list[tuple[str, str]]:
    """Read (name, sequence) pairs from a FASTA file or a one-sequence-per-line file."""
    text = path.read_text().strip()
    if not text:
        raise ValueError(f'{path} is empty')

    records: list[tuple[str, str]] = []
    if text.lstrip().startswith('>'):
        name, chunks = None, []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if name is not None:
                    records.append((name, ''.join(chunks)))
                name, chunks = line[1:].strip() or f'design_{len(records)}', []
            else:
                chunks.append(line.upper())
        if name is not None:
            records.append((name, ''.join(chunks)))
    else:
        for i, line in enumerate(text.splitlines()):
            line = line.strip().upper()
            if line:
                records.append((f'design_{i}', line))

    valid = set(bg.constants.aa_dict.keys())
    for name, seq in records:
        bad = sorted(set(seq) - valid)
        if bad:
            raise ValueError(f"Sequence '{name}' contains non-standard residues: {bad}")
    return records


# ----------------------------------------------------------------------------
# Screening.
# ----------------------------------------------------------------------------
def screen(
    records: Iterable[tuple[str, str]],
    sasa_cutoff: float = 0.15,
    distance_cutoff: float = 4.5,
    mimic_residues: tuple[str, ...] = ('R', 'K'),
    structure_dir: pl.Path | None = None,
) -> pd.DataFrame:
    """Fold each design, measure it, and return one row per design."""
    backend = os.getenv('BAGEL_BACKEND', 'modal')
    print(f'Backend: {backend}')
    oracle = bg.oracles.ESMFold(backend=backend)

    rows = []
    for name, sequence in records:
        residues = [bg.Residue(name=aa, chain_ID='A', index=i) for i, aa in enumerate(sequence)]
        chains = [bg.Chain(residues)]

        result = oracle.predict(chains=chains)
        structure = result.structure
        ptm = float(np.asarray(result.ptm).reshape(-1)[0])
        plddt = float(np.mean(result.local_plddt[0]))

        hard, soft = buried_histidines(structure, sasa_cutoff=sasa_cutoff)
        contacts = histidine_carboxylate_contacts(structure, distance_cutoff=distance_cutoff)

        row: dict[str, Any] = {
            'name': name,
            'length': len(sequence),
            'n_his': sequence.count('H'),
            'ptm': ptm,
            'plddt': plddt,
            'buried_his': hard,
            'burial_score': soft,
            'his_carboxylate_contacts': contacts,
            'sequence': sequence,
        }

        # Tie-breaker only. A large positive drop means the predicted fold does
        # not survive a permanent positive charge at the histidine positions.
        for mimic in mimic_residues:
            mimic_ptm = charge_mimic_ptm(oracle, chains, mimic)
            row[f'ptm_drop_H{mimic}'] = None if mimic_ptm is None else ptm - mimic_ptm

        rows.append(row)
        print(
            f'{name}: pTM={ptm:.3f} pLDDT={plddt:.3f} buried_His={hard} contacts={contacts}'
        )

        if structure_dir is not None:
            structure_dir.mkdir(parents=True, exist_ok=True)
            result.to_cif(structure_dir / f'{name}.cif')

    return pd.DataFrame(rows)


def rank(
    df: pd.DataFrame,
    min_ptm: float = 0.70,
    min_plddt: float = 0.75,
    min_buried_his: int = 4,
) -> pd.DataFrame:
    """
    Apply the hard filters, then order what is left.

    The hard filters are structural and are the ones to trust: a design must be
    confidently folded, must bury enough histidines to shift the folding free
    energy appreciably, and must have no histidine-carboxylate dyad.

    Ordering within the survivors is by burial score first, then by the
    arginine-mimic pTM drop. The mimic only ever breaks ties between designs that
    already passed, which is the most weight that comparison can carry.
    """
    df = df.copy()
    df['passes'] = (
        (df['ptm'] >= min_ptm)
        & (df['plddt'] >= min_plddt)
        & (df['buried_his'] >= min_buried_his)
        & (df['his_carboxylate_contacts'] == 0)
    )
    sort_cols = ['passes', 'burial_score']
    ascending = [False, False]
    if 'ptm_drop_HR' in df.columns and df['ptm_drop_HR'].notna().any():
        sort_cols.append('ptm_drop_HR')
        ascending.append(False)
    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sequences', type=pl.Path, help='FASTA, or one sequence per line')
    parser.add_argument('--out', type=pl.Path, default=pl.Path('ph_switch_screen.csv'))
    parser.add_argument('--structures', type=pl.Path, default=None, help='directory for predicted CIF files')
    parser.add_argument('--sasa-cutoff', type=float, default=0.15)
    parser.add_argument('--distance-cutoff', type=float, default=4.5)
    parser.add_argument('--min-ptm', type=float, default=0.70)
    parser.add_argument('--min-plddt', type=float, default=0.75)
    parser.add_argument('--min-buried-his', type=int, default=4)
    parser.add_argument(
        '--no-mimic',
        action='store_true',
        help='skip the charge-mimic folds, halving oracle cost',
    )
    args = parser.parse_args()

    records = read_sequences(args.sequences)
    print(f'Screening {len(records)} design(s)')

    df = screen(
        records,
        sasa_cutoff=args.sasa_cutoff,
        distance_cutoff=args.distance_cutoff,
        mimic_residues=() if args.no_mimic else ('R', 'K'),
        structure_dir=args.structures,
    )
    ranked = rank(
        df,
        min_ptm=args.min_ptm,
        min_plddt=args.min_plddt,
        min_buried_his=args.min_buried_his,
    )
    ranked.to_csv(args.out, index=False)

    display_cols = [c for c in ranked.columns if c != 'sequence']
    print()
    print(ranked[display_cols].to_string(index=False))
    print()
    print(f"{int(ranked['passes'].sum())} of {len(ranked)} design(s) passed the hard filters")
    print(f'Wrote {args.out}')
    print('Next step for a predicted midpoint: run PROPKA or Poisson-Boltzmann on the structures.')


if __name__ == '__main__':
    main()
