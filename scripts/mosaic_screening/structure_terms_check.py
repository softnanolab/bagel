"""
Per-term agreement between the structure-based energy terms evaluated on mosaic's all-atom output and on bagel's own ESMFold2
oracle, for random single mutants of a binder (needs the mosaic app on Modal deployed).

    BENCH_GPU=A100-80GB uv run python scripts/mosaic_screening/structure_terms_check.py --target ubq --mutants 24
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'mosaic_proposal'))
from common import AMINO_ACIDS, TARGETS, initial_binder, make_oracle  # noqa: E402

import bagel as bg  # noqa: E402
from bagel.mosaic_screening import ModalMosaicBackend, build_plans  # noqa: E402
from bagel.mosaic_screening.adapter import pseudo_energy  # noqa: E402
from bagel.mosaic_screening.protocol import spearman  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='ubq', choices=list(TARGETS))
    parser.add_argument('--mutants', type=int, default=24)
    parser.add_argument('--length', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='scripts/mosaic_screening/results/structure_terms_check.json')
    args = parser.parse_args()
    np.random.seed(args.seed)
    oracle = make_oracle()
    E = bg.energies
    base = initial_binder(args.seed)[: args.length]
    binder = bg.Chain([bg.Residue(a, 'B', i, mutable=True) for i, a in enumerate(base)])
    target = bg.Chain([bg.Residue(a, 'T', i, mutable=False) for i, a in enumerate(TARGETS[args.target])])
    terms = [
        E.PTMEnergy(oracle=oracle),
        E.OverallPLDDTEnergy(oracle=oracle),
        E.SurfaceAreaEnergy(oracle=oracle),
        E.SurfaceAreaEnergy(oracle=oracle, residues=binder.residues, name='binder'),
        E.HydrophobicEnergy(oracle=oracle),
        E.HydrophobicEnergy(oracle=oracle, residues=binder.residues, mode='surface', name='binder_surface'),
        E.HydropathyEnergy(oracle=oracle),
        E.GlobularEnergy(oracle=oracle),
        E.BuriedHistidineEnergy(oracle=oracle),
        E.HistidineCarboxylateContactEnergy(oracle=oracle),
    ]
    system = bg.System(states=[bg.State(name='s', chains=[binder, target], energy_terms=terms)])
    plan = build_plans(system)[0]
    assert plan is not None and all(s.used for s in plan.statuses), [s for s in plan.statuses if not s.used]
    group = plan.groups[0]
    sequences = [base]
    for _ in range(args.mutants):
        i = np.random.randint(len(base))
        sequences.append(base[:i] + np.random.choice([a for a in AMINO_ACIDS.replace('C', '') if a != base[i]]) + base[i + 1 :])
    predictions = []
    for start in range(0, len(sequences), 9):
        chunk = sequences[start : start + 9]
        padded = chunk + [chunk[-1]] * (9 - len(chunk))
        predictions += ModalMosaicBackend().predict(group.spec, plan.target_sequences, padded, structure=True)[: len(chunk)]
    mosaic_values: dict[str, list[float]] = {t.name: [] for t in terms}
    bagel_values: dict[str, list[float]] = {t.name: [] for t in terms}
    for sequence, prediction in zip(sequences, predictions):
        trial = system.__copy__()
        trial.states[0].chains[0].residues[:] = [bg.Residue(a, 'B', i, mutable=True) for i, a in enumerate(sequence)]
        _, parts = pseudo_energy(plan.terms_of(trial.states[0], group), plan.ordered_chains(trial.states[0]), prediction)
        real = trial.states[0].energy_term_values
        for name in mosaic_values:
            mosaic_values[name].append(parts[name])
            bagel_values[name].append(real[name])
    summary = {}
    for name in mosaic_values:
        m, b = np.array(mosaic_values[name]), np.array(bagel_values[name])
        summary[name] = {
            'spearman': spearman(list(m), list(b)),
            'mean_mosaic': float(m.mean()),
            'mean_bagel': float(b.mean()),
            'sd_mosaic': float(m.std()),
            'sd_bagel': float(b.std()),
        }
        print(f'{name:40s} rho={summary[name]["spearman"]:6.2f}  mosaic {m.mean():9.4f}+-{m.std():.4f}  bagel {b.mean():9.4f}+-{b.std():.4f}', flush=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'summary': summary, 'mosaic': mosaic_values, 'bagel': bagel_values}, indent=1))


if __name__ == '__main__':
    main()
