# Mosaic forward-pass screening: benchmark results

Results behind `bagel.mosaic_screening` (see `mosaic_screening.md`). The earlier gradient-based experiment is in
`experiment_with_gradient-based_choice_failure.md`.

## 11f. Broader benchmark: proposals screened with mosaic forward passes (2026-10-16)

Question: can mosaic's forward pass (not its gradient, see 11e) make bagel's Monte Carlo find lower energies? `MosaicScreenedProposal`: each
step samples 8 random single substitutions not yet tried from the current sequence, scores the 8 mutated sequences and the current one with
mosaic's loss in one batched call (A100), proposes the candidate with the lowest loss; bagel then folds and accepts or rejects as usual.
Design: 3 targets (BCMA 54 residues, ubiquitin 76 [PDB 1UBQ], protein G B1 domain 56 [PDB 1PGA]) x 10 start sequences x {random proposals,
screened proposals} = 60 runs, 500 steps, the same energy as before (pTM, pLDDT of the complex, cross PAE, ipSAE-min), temperature 0.1 for
50 steps then 0.02 for 200, twice; the same start sequence in both arms (paired), different RNG seeds. Plus a surrogate-only arm (30 runs):
the same Metropolis loop with mosaic's loss as the energy, no bagel folds, the best sequences at steps 100/250/500 re-folded afterwards.
Primary metric and test were fixed before the results: energy of the step-500 best sequence re-folded 8 times, two-sided paired sign-flip
permutation test over the 30 pairs, alpha 0.05. Scripts: `run_bench.py`, `run_surrogate.py`, `reeval2.py`, `analyze_bench2.py`,
`launch_bench2.sh`, `finish_bench2.sh`; output `results/bench2/final_analysis.txt`, `results/bench2/curves.png`.

| comparison (re-folded energy, lower is better) | mean difference [95% bootstrap CI] | pairs favouring the first | p |
|---|---|---|---|
| **PRIMARY** step 500: screened - random (-2.11 vs -0.53) | **-1.58 [-1.93, -1.21]**, paired d = -1.54 | 28/30 | < 0.0001 (sign-flip, 200,000 draws); Wilcoxon < 0.0001 |
| per target: BCMA / ubiquitin / protein G | -1.65 / -1.22 / -1.88 | 10/10, 9/10, 9/10 | |
| step 250: screened - random | -1.53 [-1.89, -1.15] | 28/30 | < 0.0001 |
| step 100: screened - random | -1.39 [-1.75, -1.05] | 29/30 | < 0.0001 |
| step 500: surrogate-only - random | -0.10 [-0.34, +0.13] | 14/30 | 0.40 |
| step 500: surrogate-only - screened | +1.48 [+1.01, +1.92] | 4/30 | < 0.0001 |

Cost (compute time only; the idle time of warm containers is not counted, my estimate of the whole campaign is about $85-130): per run,
random $0.66 (4033 s on a T4); screened $1.80 (T4 4077 s plus 1629 s of A100), 1.4 times the wall-clock time; surrogate-only $0.27.
Equal-cost comparison (reported best energy, not re-folded, at the cost a random run spends in 500 steps, $0.66): screened -1.68 vs random
-0.78, difference -0.90 [-1.23, -0.57], screened lower in 25/30 pairs. This ignores idle A100 time; if the A100 is not shared by several runs the
advantage shrinks.

What the data show. (1) Screening with mosaic's forward pass gives much lower energies than random proposals on all three targets, with a
large effect compared with the run-to-run spread; unlike the gradient (11e), the forward pass is accurate: Spearman(predicted, true energy
change) = +0.65 over the 15,000 screened proposals. (2) The advantage is created in the first ~100 steps (gap -1.39 at step 100, -1.53 at 250,
-1.58 at 500): from steps 100 to 500 the two arms improve about equally. In the first 100 steps the screened arm gained 2.0 energy units on
average against 0.5; accepted improving moves are about equally frequent (26 vs 24 per run) but twice as large (0.10 vs 0.055); the improving
tail of the proposals is heavier (5th percentile of the energy change -0.16 vs -0.08, 1st -0.34 vs -0.17) and so is the bad tail (95th
percentile +0.99 vs +0.61). Averaged over all steps the proposals look the same (fraction improving 0.104 vs 0.093, mean change +0.55 vs +0.54,
acceptance 0.159 vs 0.163), so average proposal statistics hide the effect. (3) Optimising the surrogate alone does not help: the sequences
re-folded with bagel are no better than random-proposal ones (p = 0.40), and the surrogate is optimistic about its own best sequences (-1.00
against -0.63 when re-folded; Spearman across runs +0.78). (4) The pilot's proline concern did not materialise: proline is 4.3% of the final
screened sequences against 3.7% for random; hydrophobic and charged fractions are the same.
Limits: one energy function, three small targets (54-76 residues), random start sequences, 500 steps, one batch size (8), temperatures and the number
of candidates not tuned, equal-cost comparison based on the optimiser-reported energy (noisy) and on a compute-only cost model. The behaviour
from already optimised start sequences, longer runs and larger targets is not tested here (the offline screening result of 11e suggests the
forward pass also ranks mutations of optimised sequences well). The surrogate-only arm, the phase analysis and the composition analysis were not
pre-specified.

## 11g. The same benchmark with hotspot-focused terms (2026-10-19 to 2026-10-20)

Identical to 11f (3 targets x 10 start sequences x {random, screened by mosaic}, plus the surrogate-only arm, 500 steps, the same temperatures, the
same starts) except for the energy: pTM (w 1) and pLDDT of the complex (w 1) as before, but PAE (w 2) and ipSAE-min (w 2) are computed between
the binder and a hotspot patch of the target only (`BENCH_TERMS=hotspot`). The patches come from a fixed geometric rule applied to the PDB
structures (`compute_hotspots.py`, `hotspots.json`): the exposed hydrophobic residue with the largest exposed hydrophobic area within 10 A,
plus the 8 exposed residues closest to it. BCMA (4ZFO): F14 D15 S16 L17 L18 H19 A20 C21 (contains D15-S16-L17); ubiquitin (1UBQ): L8 P38 D39 R42 V70
L71 R72 L73; protein G B1 (1PGA): M1 T2 E19 V21 D22 A23 A24 T25. These are not validated epitopes. No term pulls the binder to the hotspot; the energy
only rewards a confident relative placement. The surrogate terms were checked against bagel on the three start sequences (values agree).
Script outputs: `results/bench3/final_analysis.txt`, `results/bench3/curves.png`. Same pre-specified primary test as 11f.

| comparison (re-folded energy, lower is better) | mean difference [95% CI] | pairs favouring the first | p |
|---|---|---|---|
| **PRIMARY** step 500: screened - random (-1.74 vs -0.76) | **-0.99 [-1.35, -0.63]**, paired d = -0.97 | 26/30 | < 0.0001 |
| per target: BCMA / ubiquitin / protein G | -0.74 / -1.19 / -1.04 | 8/10, 9/10, 9/10 | |
| step 250: screened - random | -0.97 [-1.29, -0.65] | 26/30 | < 0.0001 |
| step 100: screened - random | -0.72 [-0.95, -0.50] | 28/30 | < 0.0001 |
| step 500: surrogate-only - random | +0.37 [+0.16, +0.58] | 7/30 | 0.002 |
| step 500: surrogate-only - screened | +1.36 [+1.04, +1.68] | 4/30 | < 0.0001 |

Equal cost (reported energy, compute-only cost model of 11f): at $0.33 screened - random -0.20 [-0.46, +0.06], 19/30, p = 0.16; at $0.66 -0.48 [-0.83, -0.15],
21/30, p = 0.009. Per-run compute cost as in 11f (random $0.66, screened $1.82, surrogate-only $0.27). Diagnostics: Spearman(mosaic prediction, true
change) over the screened proposals +0.59; averaged over all steps the proposals look like random ones (fraction improving 0.100 vs 0.095); the surrogate
is optimistic about its own best sequences (-0.87 against -0.38 when re-folded; Spearman across runs +0.38).
Comparison with 11f (all-residue energy, different energy values, so only effect sizes are comparable, and the difference between the two benchmarks was
not tested): the screened-minus-random gap is smaller here (-0.99 against -1.58; the confidence intervals overlap slightly) but clearly present on all three
targets; the surrogate-only arm, which matched random before (-0.10, p = 0.40), is now worse than random (+0.37, p = 0.002), with a much weaker agreement
between surrogate and re-folded energies (Spearman +0.38 against +0.78), so mosaic's loss with the hotspot masks is a less reliable stand-in for bagel's energy.
Not checked: whether the optimised binders actually contact the hotspot patch (this would need the folded structures of the final sequences).


## 11h. Equal wall-clock comparison (2026-10-21)

Question: at equal wall-clock time, does proposing with mosaic screening beat random proposals, when both use bagel's ESMFold2 on an
A100-80GB? 3 targets (BCMA, ubiquitin, GB1) x 8 start sequences x 2 arms = 48 runs of 40 minutes, 24 pairs (same target and start sequence).
Screened arm: 8 random candidates scored with mosaic (A100-80GB), lowest pseudo-energy proposed (`selection_temperature = 0`), calibration off.
The same cyclic temperature schedule per step (0.1 for 50 steps, 0.02 for 200) in both arms.

**Pre-specified** primary metric: energy of the best sequence at 40 minutes, re-folded 8 times with bagel (removes the optimism of picking the
best of a noisy oracle); test: two-sided paired sign-flip permutation, alpha = 0.05. Everything at 10 and 20 minutes is descriptive.

| time | screened - random (re-folded best energy) | 95% bootstrap CI | p (sign-flip) | screened lower in |
|---|---|---|---|---|
| 40 min (primary) | -0.58 | [-1.12, -0.01] | 0.060 | 16/24 |
| 20 min | -0.85 | [-1.29, -0.40] | 0.002 | 18/24 |
| 10 min | -0.70 | [-1.13, -0.29] | 0.004 | 16/24 |

The primary test does **not** reach alpha = 0.05; the smallest difference this design could detect with 80% power was 0.82. By target at 40 minutes
the mean differences were -0.31 (BCMA, lower in 4/8), -0.76 (ubiquitin, 7/8) and -0.67 (GB1, 5/8).

Steps tried and acceptance (mean over the 24 runs of each arm; accepted = the energy of the current system changed after the step, which
includes accepted uphill moves):

| arm | at | steps tried | accepted | acceptance rate |
|---|---|---|---|---|
| random | 10 min | 407 (328-512) | 83 | 20.8% |
| random | 20 min | 813 (656-1026) | 110 | 13.9% |
| random | 40 min | 1621 (1292-2053) | 136 | 8.6% |
| screened | 10 min | 94 (15-143) | 35 | 41.7% |
| screened | 20 min | 205 (92-321) | 47 | 23.1% |
| screened | 40 min | 495 (324-891) | 74 | 15.6% |

(rates are the mean of the per-run rates; pooled over runs they are 20.4/13.5/8.4% and 37.4/22.7/14.9%.) A screened step takes about three times
as long as a random one (a mosaic call plus the fold), so it makes ~3x fewer steps, but accepts roughly twice as large a fraction of them. In
the first 10 minutes the screened arm gains energy faster (11.9 vs 8.0 per hour) but in the last 10 minutes more slowly (0.49 vs 1.00), so the
advantage shrinks with time. At an equal number of steps the screened arm is clearly better (-1.23, lower in 22/24 pairs), which is not the
comparison that matters in practice. Compute cost per run is the same (about $1.67 for both arms on A100 prices at the time).

Reading: screening helps most for short budgets and for ubiquitin/GB1; at 40 minutes the advantage is borderline. Screening only for the
first N steps and then switching to random sampling (`screening_steps`) is a natural option but has not been tested.
Analysis script: `scripts/mosaic_proposal/analyze_timed.py`; driver for new runs: `scripts/mosaic_screening/run_timed.py`.
