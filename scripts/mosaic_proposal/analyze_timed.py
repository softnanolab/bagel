"""
Equal wall-clock comparison: random proposals vs proposals screened with mosaic forward passes, bagel's ESMFold2 on an A100-80GB for both,
40 minutes per run, 3 targets x 8 start sequences (paired).

PRE-SPECIFIED before any result was seen:
  PRIMARY comparison : screened vs random after the full 40-minute budget
  PRIMARY metric     : energy of the best sequence at 40 minutes, re-folded 8 times with bagel (unbiased against the oracle noise)
  PRIMARY test       : two-sided paired sign-flip permutation test over the 24 pairs (same target, same start sequence), alpha = 0.05,
                       with a paired bootstrap 95% confidence interval and the Wilcoxon test.
Descriptive (no multiplicity correction): the same at 10 and 20 minutes; reported (optimiser) energies; accepted moves and accepted improving moves
per hour; energy gained per hour; equal-number-of-steps comparison inside the same runs; compute-time accounting.
"""

import glob
import json
import os
import pathlib as pl

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

R = pl.Path(os.environ.get('TIMED_RESULTS', pl.Path(__file__).parent / 'results' / 'timed'))
TARGETS = ['bcma', 'ubq', 'gb1']
N_IDX = 8
CHECK = (600, 1200, 2400)
T4, A100 = 0.000164, 0.000694


def fasta_best(path: pl.Path) -> dict[int, str]:
    out = {}
    for e in path.read_text().split('>')[1:]:
        l = e.strip().split('\n')
        if len(l) >= 2:
            out[int(l[0])] = l[1].split(':')[0]
    return out


def load_refold() -> dict[str, float]:
    out = {}
    for f in glob.glob(str(R / 'refold_out_*.json')):
        for k, v in json.load(open(f)).items():
            out[k] = v['mean']
    return out


def signflip(d: np.ndarray, n_mc: int = 200000, seed: int = 0) -> float:
    rng = np.random.RandomState(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_mc, len(d)))
    return float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def boot(d: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    m = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(name: str, a: np.ndarray, b: np.ndarray, la: str, lb: str, tg: list[str] | None = None, lower_is_better: bool = True) -> None:
    d = a - b
    lo, hi = boot(d)
    w = stats.wilcoxon(d, alternative='two-sided') if np.any(d != 0) else None
    print(f'\n{name}\n  n pairs = {len(d)}; {la} mean {a.mean():+.3f}, {lb} mean {b.mean():+.3f}')
    print(f'  {la} - {lb}: mean {d.mean():+.3f}, 95% bootstrap CI [{lo:+.3f}, {hi:+.3f}], paired d {d.mean() / d.std(ddof=1):+.2f}; {la} lower in {int((d < 0).sum())}/{len(d)} pairs')
    print(f'  sign-flip permutation p = {signflip(d):.4f}' + (f'; Wilcoxon p = {w.pvalue:.4f}' if w else '') + f'; smallest difference detectable with 80% power = {2.8 * d.std(ddof=1) / np.sqrt(len(d)):.2f}')
    if tg is not None:
        for t in TARGETS:
            m = np.array([x == t for x in tg])
            print(f'    {t:5s} n={m.sum()}: mean difference {d[m].mean():+.3f} ({la} lower in {int((d[m] < 0).sum())}/{m.sum()})')


def main() -> None:
    keys = [(t, i) for t in TARGETS for i in range(N_IDX)]
    tg = [t for t, _ in keys]
    runs = {(a, t, i): json.load(open(R / f'{a}_{t}_{i}' / f'{a}_{t}_{i}.json')) for a in ('canonical', 'screened') for t, i in keys}
    refold = load_refold()

    def n_at(r: dict, T: float) -> int:
        return sum(1 for s in r['steps'] if s['t_elapsed'] <= T)

    def best_energy_at(r: dict, T: float) -> float:
        st, n = r['steps'], n_at(r, T)
        e = [s['e_current'] for s in st] + [st[-1]['e_proposed']]
        return float(min(e[: n + 1]))

    def refolded_at(a: str, t: str, i: int, T: float) -> float:
        best = fasta_best(R / f'{a}_{t}_{i}' / f'{a}_{t}_{i}' / 'best' / 'binder_vs_TNF.fasta')
        n = n_at(runs[(a, t, i)], T)
        return refold[f'{t}|{best[max(s for s in best if s <= n)]}']

    print('=' * 100 + '\nEQUAL WALL-CLOCK: random vs mosaic-screened proposals, bagel on an A100-80GB, 40 min per run, 3 targets x 8 starts\n' + '=' * 100)
    for T in (2400, 1200, 600):
        can = np.array([refolded_at('canonical', t, i, T) for t, i in keys])
        scr = np.array([refolded_at('screened', t, i, T) for t, i in keys])
        report(('PRIMARY: ' if T == 2400 else '') + f're-folded energy of the best sequence after {T // 60} min: screened vs random', scr, can, 'screened', 'random', tg)
    print('\n' + '-' * 100 + '\nReported best energy (noisy oracle, optimistic)')
    for T in (2400, 1200, 600):
        can = np.array([best_energy_at(runs[('canonical', t, i)], T) for t, i in keys])
        scr = np.array([best_energy_at(runs[('screened', t, i)], T) for t, i in keys])
        report(f'reported best energy after {T // 60} min: screened vs random', scr, can, 'screened', 'random')

    print('\n' + '-' * 100 + '\nSteps, accepted moves and compute per run (means over the 24 runs of each arm)')
    for a in ('canonical', 'screened'):
        for lo, hi, label in ((0, 2400, 'whole run'), (0, 600, 'first 10 min'), (1800, 2400, 'last 10 min')):
            steps, acc, imp, gain = [], [], [], []
            for t, i in keys:
                st = runs[(a, t, i)]['steps']
                ec = [s['e_current'] for s in st] + [st[-1]['e_proposed']]
                idx = [k for k, s in enumerate(st) if lo < s['t_elapsed'] <= hi]
                if not idx:
                    continue
                dt = (hi - lo) / 3600.0
                steps.append(len(idx) / dt)
                acc.append(sum(ec[k + 1] != ec[k] for k in idx) / dt)
                imp.append(sum(ec[k + 1] < ec[k] for k in idx) / dt)
                gain.append(max(0.0, min(ec[: idx[0] + 1]) - min(ec[: idx[-1] + 2])) / dt)
            print(f'  {a:10s} {label:13s}: steps/hour {np.mean(steps):7.0f}   accepted moves/hour {np.mean(acc):6.1f}   accepted IMPROVING moves/hour {np.mean(imp):6.1f}   best-energy gain/hour {np.mean(gain):5.2f}')
    n_can = np.array([len(runs[('canonical', t, i)]['steps']) for t, i in keys]); n_scr = np.array([len(runs[('screened', t, i)]['steps']) for t, i in keys])
    print(f'  total steps in 40 min: random {n_can.mean():.0f} (range {n_can.min()}-{n_can.max()}), screened {n_scr.mean():.0f} (range {n_scr.min()}-{n_scr.max()})')

    print('\n' + '-' * 100 + '\nSame number of STEPS inside the same pairs (reported best energy of the random run at the number of steps the screened run completed in 40 min)')
    a = np.array([best_energy_at(runs[('screened', t, i)], 2400) for t, i in keys])
    b = []
    for (t, i), n in zip(keys, n_scr):
        st = runs[('canonical', t, i)]['steps']; e = [s['e_current'] for s in st] + [st[-1]['e_proposed']]
        b.append(float(min(e[: min(int(n), len(st)) + 1])))
    report('reported best energy at equal number of steps: screened vs random', a, np.array(b), 'screened', 'random')

    print('\n' + '-' * 100 + '\nCompute accounting per run (compute time only; idle time of warm containers not counted; A100 $0.000694/s)')
    for a in ('canonical', 'screened'):
        tf = np.mean([sum(s['t_fold'] for s in runs[(a, t, i)]['steps']) for t, i in keys]); tp = np.mean([sum(s['t_propose'] for s in runs[(a, t, i)]['steps']) for t, i in keys])
        print(f'  {a:10s} bagel fold time {tf:6.0f} s, mosaic time {tp:6.0f} s -> compute cost ${(tf + tp) * A100:.2f} per run')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    grid = np.arange(60, 2401, 60)
    for ax, t in zip(axes, TARGETS):
        for a, c in (('canonical', 'tab:gray'), ('screened', 'tab:red')):
            cu = np.array([[best_energy_at(runs[(a, t, i)], T) for T in grid] for i in range(N_IDX)])
            ax.plot(grid / 60, cu.mean(0), color=c, lw=2, label=a)
            ax.fill_between(grid / 60, cu.mean(0) - cu.std(0, ddof=1) / np.sqrt(N_IDX), cu.mean(0) + cu.std(0, ddof=1) / np.sqrt(N_IDX), color=c, alpha=0.2)
        ax.set_title(t); ax.set_xlabel('wall-clock time (min)'); ax.set_ylabel('best reported energy (mean ± SEM, n=8)'); ax.legend()
    fig.tight_layout(); fig.savefig(R / 'curves_vs_time.png', dpi=140)
    print(f'\nplot: {R / "curves_vs_time.png"}')


if __name__ == '__main__':
    main()
