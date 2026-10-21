"""
Equal wall-clock run, one arm: random proposals ('canonical') or mosaic-screened proposals ('screened'), bagel's ESMFold2 on the GPU
given by BENCH_GPU (A100-80GB in the published benchmark), until a wall-clock budget is used up. Every step is stamped with the time since
the start, so that best energy against time, accepted moves, etc. can be computed afterwards (see scripts/mosaic_proposal/analyze_timed.py).

    BENCH_GPU=A100-80GB uv run python scripts/mosaic_screening/run_timed.py --arm screened --target ubq --index 0 --out results/x

The benchmark of docs/mosaic_proposal.md (section 11h) was run with the earlier prototype of the screened arm; this driver uses
bagel.mosaic_screening.MosaicScreening with the calibration off (calibration_steps=0), which proposes the same way.
"""

import argparse
import json
import pathlib as pl
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(pl.Path(__file__).resolve().parents[1] / 'mosaic_proposal'))
from common import TARGETS, build_system, initial_binder, make_oracle  # noqa: E402

import bagel as bg  # noqa: E402
from bagel.callbacks import Callback  # noqa: E402
from bagel.mosaic_screening import MosaicScreening  # noqa: E402

TARGET_IDS = {'bcma': 0, 'ubq': 1, 'gb1': 2}


def start_sequence(target: str, index: int) -> str:
    return initial_binder(700 + 10 * TARGET_IDS[target] + index)


class Recorder(bg.mutation.MutationProtocol):
    """Wraps a mutation protocol and records, for every proposal, its true energy change and timing."""

    def __init__(self, inner: bg.mutation.MutationProtocol) -> None:
        self.inner = inner
        self.rows: list[dict[str, Any]] = []
        self.t_start = time.time()

    def set_log_path(self, path: pl.Path) -> None:
        self.inner.set_log_path(path)

    def one_step(self, system, temperature=None):  # noqa: ANN001
        t0 = time.time()
        mutated, record = self.inner.one_step(system)
        t_propose = time.time() - t0
        e_old = system.get_total_energy()
        t1 = time.time()
        e_new = mutated.get_total_energy()
        info = dict(getattr(self.inner, 'last_info', {}) or {})
        self.rows.append(
            {
                'step': len(self.rows) + 1,
                'e_current': e_old,
                'e_proposed': e_new,
                'delta': e_new - e_old,
                'mutation': [(m.residue_index, m.old_amino_acid, m.new_amino_acid) for m in record.mutations],
                't_propose': t_propose,
                't_fold': time.time() - t1,
                't_elapsed': time.time() - self.t_start,
                **info,
            }
        )
        return mutated, record


class TimeBudget(Callback):
    """Stops the minimisation at the first step that ends after `seconds` of wall-clock time since the first step began."""

    def __init__(self, recorder: Recorder, seconds: float) -> None:
        self.recorder, self.seconds, self._should_stop = recorder, seconds, False

    def on_step_end(self, context) -> None:  # noqa: ANN001
        self._should_stop = (time.time() - self.recorder.t_start) >= self.seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['canonical', 'screened'], required=True)
    parser.add_argument('--target', choices=list(TARGETS)[1:], required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--budget_s', type=float, default=2400.0)
    parser.add_argument('--n_candidates', type=int, default=8)
    parser.add_argument('--out', type=pl.Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seed = (5000 if args.arm == 'canonical' else 6000) + 100 * TARGET_IDS[args.target] + args.index
    np.random.seed(seed)

    oracle = make_oracle()
    start = start_sequence(args.target, args.index)
    system = build_system(oracle, start, args.target)
    system.get_total_energy()  # fold the start sequence now, so that the clock below measures only the optimisation
    mutator: bg.mutation.MutationProtocol
    if args.arm == 'canonical':
        mutator = bg.mutation.Canonical(n_mutations=1)
    else:
        mutator = MosaicScreening(n_candidates=args.n_candidates, calibration_steps=0)
        mutator.one_step(system)  # warm the mosaic container (compilation) before the clock starts
    recorder = Recorder(mutator)
    cycle = np.concatenate([np.full(50, 0.1), np.full(200, 0.02)])
    temps = np.tile(cycle, 20)  # 5000 steps: more than any run can reach in the budget
    name = f'{args.arm}_{args.target}_{args.index}'
    minimizer = bg.minimizer.MonteCarloMinimizer(
        mutator=recorder,
        temperature=temps,
        n_steps=len(temps),
        experiment_name=name,
        log_path=args.out,
        callbacks=[bg.callbacks.DefaultLogger(log_interval=1), TimeBudget(recorder, args.budget_s)],
    )
    recorder.t_start = time.time()  # the clock starts here
    best = minimizer.minimize_system(system)
    summary = {
        'arm': args.arm,
        'target': args.target,
        'index': args.index,
        'seed': seed,
        'start_sequence': start,
        'budget_s': args.budget_s,
        'best_energy': best.get_total_energy(),
        'best_sequence': best.states[0].chains[0].sequence,
        'wall_time_s': time.time() - recorder.t_start,
        'n_steps': len(recorder.rows),
        'steps': recorder.rows,
    }
    (args.out / f'{name}.json').write_text(json.dumps(summary, indent=1, default=float))
    print(f'DONE {name} steps={summary["n_steps"]} best={summary["best_energy"]:.4f} wall={summary["wall_time_s"]:.0f}s')


if __name__ == '__main__':
    main()
