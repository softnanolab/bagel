"""
Check of MosaicScreening against the real thing on Modal (needs the app of src/bagel/mosaic_screening/modal_service.py to be
deployed and the Modal profile that owns it to be active).

1. versions of the mosaic worker;
2. parity: for K random single mutants of a random binder, the pseudo-energy computed from mosaic (via the adapter, with
   bagel's own term code) against the energy of bagel's ESMFold2 oracle: Spearman correlation of the energies and of the
   changes with respect to the starting sequence, plus timings;
3. a short optimisation with MosaicScreening (calibration on) as a MutationProtocol in a MonteCarloMinimizer.

    uv run python scripts/mosaic_screening/modal_check.py --target ubq --mutants 24 --steps 10
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'mosaic_proposal'))
from common import AMINO_ACIDS, TARGETS, build_system, initial_binder, make_oracle  # noqa: E402

import bagel as bg  # noqa: E402
from bagel.mosaic_screening import ModalMosaicBackend, MosaicScreening, build_plans  # noqa: E402
from bagel.mosaic_screening.adapter import pseudo_energy  # noqa: E402
from bagel.mosaic_screening.protocol import spearman  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='ubq', choices=list(TARGETS))
    parser.add_argument('--mutants', type=int, default=24)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default='scripts/mosaic_screening/results/modal_check.json')
    args = parser.parse_args()
    np.random.seed(args.seed)

    backend = ModalMosaicBackend()
    t0 = time.time()
    print('versions:', backend.versions(), f'({time.time() - t0:.0f} s incl. container start)', flush=True)

    oracle = make_oracle()
    base_seq = initial_binder(args.seed)[:40]
    system = build_system(oracle, base_seq, args.target)
    plan = build_plans(system)[0]
    assert plan is not None
    group = plan.groups[0]

    # ---- parity -------------------------------------------------------------------------------------------------
    sequences = [base_seq]
    for _ in range(args.mutants):
        i = np.random.randint(len(base_seq))
        aa = np.random.choice([a for a in AMINO_ACIDS.replace('C', '') if a != base_seq[i]])
        sequences.append(base_seq[:i] + aa + base_seq[i + 1 :])
    batch = sequences + [sequences[-1]] * max(0, 9 - len(sequences))
    t0 = time.time()
    predictions = []
    for start in range(0, len(sequences), 9):
        chunk = sequences[start : start + 9]
        chunk = chunk + [chunk[-1]] * (9 - len(chunk))
        predictions += backend.predict(group.spec, plan.target_sequences, chunk)[: len(sequences[start : start + 9])]
    t_mosaic = time.time() - t0
    pseudo, real = [], []
    t0 = time.time()
    for seq, prediction in zip(sequences, predictions):
        trial = system.__copy__()
        trial.states[0].chains[0].residues[:] = [bg.Residue(a, 'B', i, mutable=True) for i, a in enumerate(seq)]
        chains = plan.ordered_chains(trial.states[0])
        value, _ = pseudo_energy(plan.terms_of(trial.states[0], group), chains, prediction)
        pseudo.append(value)
        real.append(trial.get_total_energy())
    t_bagel = time.time() - t0
    d_pseudo = [p - pseudo[0] for p in pseudo[1:]]
    d_real = [r - real[0] for r in real[1:]]
    result = {
        'target': args.target,
        'n_sequences': len(sequences),
        'spearman_energy': spearman(pseudo, real),
        'spearman_delta': spearman(d_pseudo, d_real),
        'seconds_mosaic_incl_compile': t_mosaic,
        'seconds_bagel_folds': t_bagel,
        'pseudo': pseudo,
        'real': real,
    }
    print(json.dumps({k: v for k, v in result.items() if k not in ('pseudo', 'real')}, indent=1), flush=True)

    # ---- short optimisation -------------------------------------------------------------------------------------
    system = build_system(oracle, initial_binder(args.seed + 1)[:40], args.target)
    protocol = MosaicScreening(backend=backend, calibration_steps=2, calibration_mutants=8)
    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=protocol,
        temperature=0.05,
        n_steps=args.steps,
        log_path=pathlib.Path('scripts/mosaic_screening/results'),
        experiment_name=f'modal_check_{args.target}_{int(time.time())}',
        callbacks=[],
    )
    t0 = time.time()
    e0 = system.get_total_energy()
    best = minimizer.minimize_system(system)
    result.update(
        {
            'run_seconds': time.time() - t0,
            'energy_start': e0,
            'energy_best': best.get_total_energy(),
            'stats': protocol.stats,
            'calibration': protocol.calibration,
            'log_path': str(minimizer.log_path),
        }
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ('pseudo', 'real')}, indent=1), flush=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=float))
    print('wrote', out)


if __name__ == '__main__':
    main()
