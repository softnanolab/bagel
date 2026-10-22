# Experiment: choosing mutations from mosaic gradients (failed)

This records the experiment in which the substitution was proposed from the gradient of a mosaic loss. It did not work (section 11e
explains why: a mutation is a discontinuous jump, so the first-order estimate does not predict it). The follow-up that did work, screening
candidates with mosaic *forward passes*, is in `mosaic_proposal.md` (sections 11f and 11g) and is implemented in
`bagel.mosaic_screening`. The code described here was a prototype that is not part of the repository.

# Mosaic gradient proposals in BAGEL: design, protocol and implementation notes

Status: prototype on branch `claude/bagel-mosaic-integration-2a4ab5`. Everything below was written against bagel 0.2.11 and the
mosaic checkout at `~/Claude/mosaic` (commit `1eb4af6`, 2026-10-06). Numbers in "Results" are from one start sequence or three
seeds and are indicative only.

## 1. Idea in one paragraph

BAGEL optimises in sequence space with a discrete Monte Carlo loop and scores every candidate with real oracle calls and its
own energy terms. Mosaic optimises in logit space with gradients through JAX re-implementations of the same kind of models.
We keep BAGEL as the host. Mosaic is used only to **propose** the next substitution: one forward and backward pass of a
differentiable surrogate loss, taken at the one-hot encoding of the current sequence, gives a first-order estimate of the loss
change of every single substitution; a (position, amino acid) pair is sampled from those estimates. BAGEL then folds the mutated
sequence with its own oracle, evaluates its full energy and accepts or rejects (Metropolis) exactly as for any other mutation
protocol. A poor surrogate costs efficiency, not correctness of the energy being optimised. The target use is **optimisation,
not exact sampling** (the proposals are asymmetric and no Hastings correction is applied; the same is true of
`InverseFoldingSampling`).

## 2. Why this direction, and not mosaic as host

* BAGEL energies run on biotite `AtomArray`s and numpy (SASA, shape complementarity, RMSD, salt bridges ...) and cannot be traced
  by JAX, so BAGEL terms cannot be wrapped as mosaic losses. Only terms that are functions of the model outputs (pLDDT, PAE,
  pTM, ipSAE, ...) can be re-implemented.
* Mosaic's soft sequence has to be decoded to a hard sequence at the end; here the sequence is always hard.
* **Environment conflict (tested with `uv lock`)**: pinned boileroom 0.4.5 requires `torch>=2.6,<2.7`, mosaic pins
  `torch==2.11.0`. They cannot share a Python environment, so mosaic runs as a separate service (a Modal class).
  Mosaic also only declares Linux x86_64 as a supported environment and needs a GPU.

## 3. Architecture

```
bagel process (any machine)                          Modal (A100-80GB), image built from the mosaic checkout
------------------------------------------           -----------------------------------------------------
MonteCarloMinimizer                                   class MosaicGradient
  mutator = MosaicGradientProposal  --- problem JSON + sequence ---->   builds features once per problem
     build_problem(System) -> MosaicProblem                              (ESMFold2Full, binder_features)
     (translated terms + residue masks)                                  builds weighted loss from masks
  <------------- value, per-term values, d loss / d one-hot [L,20] ----  jitted value_and_grad (cached)
  sample (position, aa) from the gradient
  fold with BAGEL oracle (ESMFold2 on Modal, vanilla)
  full BAGEL energy -> Metropolis accept/reject
```

Files (all under `src/bagel/mosaic_proposal/`):

| file | runs in | role |
|---|---|---|
| `spec.py` | bagel env (no jax) | `build_problem(System)` -> `MosaicProblem` (JSON-serialisable): term translation, residue masks, gates |
| `proposal.py` | bagel env (no jax) | `MosaicGradientProposal` (a `MutationProtocol`), sampling maths, `ModalGradientClient` |
| `mosaic_losses.py` | mosaic env | masked JAX losses mirroring the BAGEL terms; `build_weighted_loss` |
| `modal_service.py` | Modal | `MosaicGradient` Modal class, image definition, `gradient()` method |

Tests: `tests/unit_tests/test_mosaic_proposal.py` (10 tests, fake client, no GPU: run with `pytest ... --oracles skip`).
Scripts: `scripts/mosaic_proposal/` (`common.py`, `feasibility.py`, `run_arm.py`, `analyze.py`).

## 4. Mapping between BAGEL terms and mosaic losses

Only terms listed here influence the proposal. Others are reported in `problem.skipped` (with the reason) and are still part of the
BAGEL energy that accepts or rejects the move. A term whose oracle is not a BAGEL `ESMFold2` is also skipped. If **no** term is
translatable `build_problem` raises (there is no gradient to propose from).

| BAGEL term | mosaic-side loss (`mosaic_losses.py`) | residue groups |
|---|---|---|
| `PTMEnergy` | `FullComplexPTM`: `-max_i pTM_i` from the PAE logits (`predicted_tm_score`) | whole complex (as in BAGEL) |
| `PLDDTEnergy` | `MaskedPLDDT`: `-mean(plddt[mask])`, pLDDT in [0,1] | respected (group -> mask) |
| `OverallPLDDTEnergy` | `MaskedPLDDT` with all residues | all |
| `PAEEnergy` | `MaskedPAE`: `mean(PAE[pair mask]) / 30`, diagonal removed, `cross_term_only` respected | both groups respected |
| `ipSAEEnergy` | `MaskedIPSAE`: JAX port of `_directional_ipsae`, `pae_cutoff` and `direction` (`1to2`,`2to1`,`min`,`max`,`mean`) respected | both groups respected |

Mosaic's own terms (`BinderTargetPAE`, `PLDDTLoss`, `BinderTargetIPSAE`, ...) only know "the whole binder" / "the whole target"
(plus `epitope_idx`/`paratope_idx` for contacts), which is why masked versions were written here instead of reusing them.
Residue-subset evaluation is therefore possible, but needs custom `LossTerm`s, which is what `mosaic_losses.py` provides.
All terms return an energy to minimise with the same sign and normalisation as BAGEL; the loss is `sum_i weight_i * term_i`.

Not translated (candidates for later): `HydrophobicEnergy`, `SeparationEnergy`, `GlobularEnergy`, `SecondaryStructureEnergy`,
`SurfaceAreaEnergy`, anything needing SASA/atoms from a biotite structure. Several could be written on mosaic's atom37
coordinates; `ipSAE`'s hard PAE cutoff gives a zero gradient while no residue pair is below the cutoff (true at a random start).

Residue-group translation: a BAGEL group is `(chain_ids, residue indices)`. Mosaic token order is
`[design chain, other chains in state order]`, so each index becomes `offset[chain] + residue.index`. The chain order passed to
mosaic differs from the State's chain order if the designed chain is not first; masks are built in mosaic order so this is harmless.

## 5. The proposal algorithm

For the current design-chain sequence `s` (length `L`), mosaic returns `g[i, a] = d loss / d x[i, a]` at `x = onehot(s)`
(token order `ARNDCQEGHILKMFPSTWYV`, i.e. `mosaic.common.TOKENS`).

1. First-order change of substituting amino acid `a` at position `i`: `delta[i, a] = g[i, a] - g[i, s_i]`.
2. Candidates: mutable positions of the design chain, amino acids with `mutation_bias > 0` (cysteine excluded by default),
   not the current amino acid, and not in the exclusion set of section 6.
3. `z = delta / std(delta[candidates])`; sample `(i, a)` with probability `softmax(-z / T_prop)` over all candidates jointly.
   `T_prop` (`proposal_temperature`, default 0.3) is dimensionless, so it does not depend on the term weights.
   Measured on three random 60-mer starts (feasibility step 2): `T_prop = 0.3` gives `P(top1) = 0.49-0.65`, effective number of
   candidates 2-4; `T_prop = 1.0` gives effective number 28-244.
4. `n_mutations > 1`: several distinct positions sampled from the single gradient at the current sequence (no re-evaluation).
5. Optional exploration: with probability `exploration_probability` the step is delegated to another protocol
   (`Canonical` by default) without calling mosaic. The "mixed" benchmark arm is `exploration_probability = 0.5`.

The mutated system is returned as an ordinary `MutationRecord` (`Mutation(move_type='substitution', ...)`), so callbacks, replay and
loggers work unchanged. Diagnostics of the last mosaic-guided step are in `mutator.last_info`
(`predicted_delta`, `surrogate_value`, `surrogate_terms`).

## 6. The fix: memory of rejected proposals (v2)

**Problem found during the benchmark.** The gradient at a given sequence is deterministic (fixed PRNG key in the service). After a
rejected move the minimizer keeps the same sequence, so the next call samples from exactly the same distribution. At low `T_prop`
that distribution is concentrated on very few moves, so the same rejected mutation is proposed again about half the time
(`P(top1)` 0.49-0.65 above), and each repeat costs a fold (plus a mosaic call). Symptoms in the first run: acceptance as low as
0.07 and best energies not improving for hundreds of steps in the mosaic and mixed arms.

**Fix (`remember_rejected=True`, the default).** `MosaicGradientProposal` keeps, for the most recent 8 sequences, the gradient
returned by mosaic and the set of `(position, amino acid)` moves already proposed from that sequence.

* On each step the sequence is looked up; on a hit the cached gradient is reused (no mosaic call) and the already-proposed moves are
  excluded from the candidate set (`sample_substitution(..., exclude=tried)`).
* The sampled move is added to the set. Only done for `n_mutations == 1` (for several simultaneous mutations the joint move is not
  tracked).
* If every candidate has been proposed, the set is cleared and sampling starts over.
* `remember_rejected=False` reproduces the memoryless v1 behaviour (CLI: `run_arm.py --memoryless`).
* Test: `test_rejected_moves_are_not_proposed_again_and_the_gradient_is_cached`.

This is a form of tabu search at the level of one sequence. It keeps proposals deterministic given the gradient but stops wasting
folds. Alternatives not tried: stochastic gradients (vary the PRNG key per call, so the surrogate itself changes), a higher `T_prop`,
or best-of-B (evaluate B top candidates and return the best, which costs B folds per step).

## 7. Gates (what is refused, and where)

| condition | behaviour |
|---|---|
| `exploration=GrandCanonical(...)` passed to the constructor | `MosaicIncompatibleError` immediately |
| a chain length differs from the one the problem was built for (what GrandCanonical changes) | `MosaicIncompatibleError` in `one_step`, **before any call to mosaic** (tested: call count stays 1) |
| more than one State | `MosaicIncompatibleError` (single State only for now) |
| more than one chain with mutable residues | `MosaicIncompatibleError` |
| no translatable energy term | `MosaicIncompatibleError` |
| `proposal_temperature <= 0`, `exploration_probability` outside [0,1] | `ValueError` |

Fixed shapes: the compiled JAX loss is specific to the sequence lengths, so any length-changing move is incompatible by design.

## 8. Mosaic service (Modal)

* Image: `python:3.12-slim-bookworm` + `uv sync --group jax-cuda` on a copy of the mosaic checkout (env var `MOSAIC_REPO`),
  plus `modal` installed in the mosaic venv. (The default `debian_slim` base failed on stale apt mirrors.)
* Weights: `biohub/ESMFold2` (Full) downloaded at first use into the Modal volume `bagel-mosaic-proposal-cache`, using the
  workspace secret `huggingface-secret`. Same checkpoint family as bagel's default ESMFold2 oracle.
* Model settings: `num_loops=3` (as boileroom's default), `num_sampling_steps=14` (boileroom default is 50; fewer is enough for a
  proposal), no MSA. Overridable through `problem_options`.
* GPU `A100-80GB` (`MOSAIC_GPU`), up to `MOSAIC_MAX_CONTAINERS` containers, 15 min scale-down. Cold start + compile of a new problem:
  about 5 min; warm call about 5 s for a 60-mer binder + 157-residue target. Compilation happens once per distinct problem
  (the JSON of the problem is the cache key).
* Deploy: `MOSAIC_REPO=~/Claude/mosaic modal deploy src/bagel/mosaic_proposal/modal_service.py`; client:
  `ModalGradientClient(app_name='bagel-mosaic-proposal')`.
* Bug fixed on the way: the jitted `value_and_grad` must be created once per container, not inside the request (otherwise every
  call recompiles; warm calls took 47 s instead of 5 s).

## 9. Environment gotchas found

1. **BAGEL's own ESMFold2 oracle did not start**: pinned boileroom 0.4.5 requests Docker image `jakublala/boileroom-esmfold2:0.4.5`,
   which is not on Docker Hub. Set `BOILEROOM_IMAGE_TAG=0.4.5-alpha.12` (newest tag at the time) for every run. Should be fixed
   by repinning boileroom or publishing the tag.
2. **vanilla vs exact**: `optimization='vanilla'` (boileroom default) runs the stock image (torch 2.11, esm 3.4.1.post1, T4 by
   default) with **no flash-attn, TransformerEngine or xformers** (probed on Modal: `find_spec` is None for all three). `exact`/`fast`
   use a separate kit image that compiles flash-attn etc. (A100/H100 only, tens of minutes to build on 48 CPUs, ~27 GB of weights).
   Earlier project notes found that an *older* no-flash image (torch 2.6, esm 3.3.0) gave clearly degraded predictions. Re-checked on
   the current stock image: min ipSAE 0.40 for p53/MDM2 (1YCR; earlier values 0.33-0.35 with flash-attn or the Biohub API, 0.15 on
   the old no-flash image), 0.89 for barnase/barstar (1BRS), 0.00 for p53 vs a shuffled MDM2. So the current vanilla image looked
   healthy on these three cases; this is supporting evidence, not proof.
3. `uv sync` of bagel on macOS works; mosaic cannot be installed on macOS (CUDA-only wheels).

## 10. Benchmark protocol (1TNF monomer binder)

* Target: chain A of 1TNF (157 aa, RCSB FASTA), fixed, immutable. Binder: 60 residues, all mutable, random start without cysteine
  (`np.random.RandomState(1000 + seed)`, identical for every arm of a seed).
* Energy (every term has a mosaic counterpart, all residues): `PTMEnergy` (w 1), `OverallPLDDTEnergy` (w 1),
  `PAEEnergy` binder x target (w 2, cross term), `ipSAEEnergy` binder x target (w 2, direction `min`).
* Oracle: bagel `ESMFold2(backend='modal', config={'optimization': 'vanilla'})`.
* Minimizer: `SimulatedAnnealing`, T 0.05 -> 0.005 linearly over 500 steps (my choice, not tuned), Metropolis, one mutation per step,
  `EarlyStopping(monitor='best_system_energy', patience=250)`.
* Arms (3 seeds each): `canonical` (standard bagel `Canonical`), `mosaic` (every proposal from the gradient, `T_prop` 0.3),
  `mixed` (each step a `Canonical` move with probability 0.5, otherwise a gradient proposal).
  The runner wraps the mutator in a `Recorder` that stores, for every step, the proposal, its true energy change, timings and, for
  gradient proposals, the predicted change.
* Budget fairness: every step is one new fold, so arms are compared at equal oracle calls; wall-clock per step differs (mosaic call).
* Run: `scripts/mosaic_proposal/run_arm.py --arm {canonical,mosaic,mixed} --seed S --out DIR` (env `BOILEROOM_IMAGE_TAG=0.4.5-alpha.12`).
  Analyse: `analyze.py results/bench` (best-energy curves, acceptance, predicted-vs-true correlation of the gradient proposals).
* Feasibility (`feasibility.py`): for one start sequence compares, over about 60 single mutants, the first-order gradient prediction,
  the exact change of the mosaic surrogate and the true change of the bagel energy.

## 11. Results so far

**Feasibility (seed 0 start, 59 mutants):**

| comparison with true bagel dE | Spearman (random mutants, n=39) | Spearman (all, n=59) |
|---|---|---|
| exact mosaic surrogate change | +0.53 | +0.51 |
| first-order gradient prediction | +0.35 | +0.29 |

The 20 mutations ranked best by the gradient had mean true dE +0.006 and 45% improved the energy, against +0.040 and 33% for random
mutations. Term values at the start agree closely (pTM 0.685 vs 0.681, pLDDT 0.666 vs 0.669, PAE/30 0.792 vs 0.815, ipSAE 0 vs 0).
The first-order prediction underestimates the size of the change by an order of magnitude (about 0.004 vs 0.1), so only its
ranking is usable, which is why the sampler standardises the deltas.

**Benchmark, final (3 seeds x 3 protocols, up to 500 steps, patience 250; mosaic and mixed = v2 with memory of rejected moves).**
Run directory: `scripts/mosaic_proposal/results/bench/` (per-run JSON with every proposal, `best_energy.png`). The first, memoryless
version of the mosaic and mixed arms was stopped at about step 250-290 (partial logs in `results/bench/v1_memoryless_killed/`) and is
not part of these numbers.

Best system energy (lower is better; start energies about +0.3 to +0.5):

| protocol | seed 0 | seed 1 | seed 2 | mean | steps run (s0/s1/s2) |
|---|---|---|---|---|---|
| random (`Canonical`) | -0.939 | -2.483 | -2.450 | -1.957 | 500 / 500 / 500 |
| mosaic only | -2.694 | -1.972 | -2.509 | -2.392 | 500 / 310 (early stop) / 500 |
| mixed 50/50 | -1.892 | -3.122 | -1.697 | -2.237 | 500 / 500 / 454 (early stop) |

Best energy at fixed step (mean over seeds): step 100: random -0.81, mosaic -1.32, mixed -1.13; step 250: random -1.47, mosaic -1.54,
mixed -1.91. Acceptance 0.02-0.13 for every protocol; wall time 27-33 s per step (the gradient adds about 1-5 s on average because the
cache removes the call after a rejection).

What the data does and does not support:

* The gradient protocols found lower energies on average (mosaic -2.39, mixed -2.24 against random -1.96) and were ahead early
  (step 100). But seed-to-seed spread within one protocol is about 1.5 units (random: -0.94 to -2.48), larger than the mean differences,
  and there are 3 seeds. Per seed: mosaic beat random on seed 0 (by 1.8), tied on seed 2 (-2.51 vs -2.45) and lost on seed 1 (-1.97 vs
  -2.48); mixed beat random on seeds 0 and 1 and lost on seed 2 (-1.70 vs -2.45). **This is not statistically distinguishable from
  no effect**; at most it is a hint of faster early progress.
* Early stopping hit two gradient runs (mosaic seed 1 at step 310, mixed seed 2 at step 454) but no random run, so the gradient
  protocols sometimes had fewer steps. Random seed 1 improved late (best at step 389).
* **Within the runs the gradient did not look informative (but see section 11b: this diagnostic is contaminated by oracle noise).** Over all gradient proposals the Spearman correlation between predicted
  and true energy change was -0.05 (mosaic, n=1310) and -0.35 (mixed, n=692; negative, i.e. the wrong sign), and the mean true energy
  change of gradient proposals (+1.02 mosaic, +0.92 mixed) was not lower than that of random proposals (+0.79 in the random protocol,
  +0.87 for the random half of mixed). This contrasts with the feasibility test at a random start (top-ranked mutations clearly less
  harmful than random ones). Possible reasons, none tested: the surrogate is less reliable on sequences that have already been
  optimised (feasibility was only measured at the start); the memory of rejected moves forces later proposals down the ranking;
  heavy-tailed energy changes dominate the means. So any advantage seen may come from the early phase, from luck with 3 seeds, or from
  something other than the predicted ranking (for example a different, more concentrated distribution of mutations).
* `T_prop = 0.3`, the annealing schedule, the term weights and the number of steps were not tuned.

## 11b. The folding oracle is noisy (found 2026-10-12): consequences for every number above

BAGEL's ESMFold2 oracle is stochastic: folding the **same sequence** several times in a row gives different energies.
Six consecutive folds of two stored best sequences: canonical lineage 0 (stored best -0.939): 0.255, 0.196, 0.255, -0.629, 0.194, 0.193;
mosaic lineage 1 (stored best -1.972): -0.259, -0.079, -1.161, -1.560, -0.960, -1.409. Re-folding the start sequences of the phase-2
runs gave energies higher than the stored phase-1 best by +0.32 on average (sd 0.36, n=22 at the time of writing; up to +1.2).
Replicates that start in a fresh Modal container agree closely with each other (same random state at the first fold), which is why
the three replicates of a lineage look alike at step 0 but not like the stored value.

Consequences:

1. **Winner's curse.** The best energy an optimiser reports is the minimum of many noisy evaluations and is biased low. Metropolis
   acceptance also compares noisy energies, so a state whose energy was a lucky low draw is hard to leave (low acceptance, long
   plateaus): this probably explains much of the stalling seen in all protocols, not only the gradient ones.
2. **The within-run diagnostic of section 11 is contaminated.** The "true" energy change of a proposal contains the difference between
   a lucky current energy and a fresh draw, which shifts almost all proposals upward (mean change about +0.8 for every protocol) and
   can make predicted and true changes uncorrelated or anticorrelated. That diagnostic should not be read as evidence that the gradient
   carries no information; the clean test is the feasibility experiment repeated with several folds per mutant.
3. **Comparisons between protocols** should use the energy of the final sequences re-folded k times (`reeval.py`, k=8) as well as the
   optimiser-reported best energy. Both are reported in the final analysis.
4. Possible remedies, not implemented: re-evaluate the current state at every step (or when comparing), average several folds per
   candidate (costly), or fix the diffusion seed in the oracle so that the energy is a deterministic function of the sequence.

## 11c. Final results of the extended benchmark (2026-10-12)

Full output: `scripts/mosaic_proposal/results/final_analysis.txt`; plot `results/phase2/curves.png`; per-run data in `results/bench` (phase 1),
`results/phase2` and `results/phase2b` (phase 2). Phase 2 was interrupted when the Claude session ended (all 27 runs killed at step
196-270 of 500) and resumed from the last saved sequence of each run (`run_arm.py --resume_from`, same annealing schedule); the
current state is re-folded on resume, which perturbs all three protocols in the same way. `phase2_io.py` stitches the two segments.

Design. Phase 1: 3 seeds per protocol, 500 steps from random starts. Phase 2: each phase-1 run's best sequence is the start of 3 new
seeds, 500 further steps, no early stopping: 9 runs per protocol (3 lineages x 3 seeds), so 1000 steps in total per lineage. Each
protocol continues from its own lineages, so start energies differ between protocols. Runs from the same lineage are not independent;
tests are done on lineage means (n=3 per protocol), where the smallest attainable permutation p-value is 0.1.

Part 1 (500 steps from random starts, n=3 per protocol):

| | reported best (3 seeds) | mean | re-folded 8x (unbiased) mean |
|---|---|---|---|
| random | -0.94 / -2.48 / -2.45 | -1.96 | -1.61 |
| mosaic | -2.69 / -1.97 / -2.51 | -2.39 | -2.06 |
| mixed | -1.89 / -3.12 / -1.70 | -2.24 | -2.01 |

Kruskal-Wallis p = 0.56 (reported) and 0.73 (re-folded); every pairwise permutation p >= 0.5. No difference can be claimed; with 3 runs
per protocol significance was not attainable (smallest p 0.1).

Part 2 (next 500 steps, 9 runs per protocol, 3 lineages):

| | reported best after 1000 steps | re-folded 8x | change in phase 2 (from the run's own re-folded start) |
|---|---|---|---|
| random | -2.33 | -2.03 | -0.77 |
| mosaic | -2.82 | -2.65 | -0.75 |
| mixed | -2.34 | -2.01 | -0.39 |

Lineage-level tests: Kruskal-Wallis p = 0.56 (reported and re-folded energies) and 0.49 (change in phase 2). Mosaic minus random,
re-folded: -0.62, 95% bootstrap CI [-1.48, +0.04], permutation p = 0.3; mixed minus random: -0.02, CI [-1.11, +1.16], p = 1.0.
Mosaic minus mixed, change in phase 2: -0.37, CI [-0.81, +0.08], p = 0.3. **No difference is statistically significant.**
Observations that are not proven effects: (i) mosaic ends with the lowest energy by both measures, but mostly because its lineages
started lower (mean stored start -2.39 against -1.96 for random); the change achieved in phase 2 is the same as random's (-0.75 vs -0.77);
(ii) mixed improved least in phase 2; (iii) the gradient protocols were ahead early in phase 1 (step 100 means -1.32 mosaic, -1.13 mixed,
-0.81 random, p >= 0.6). The change in phase 2 depends strongly on the start energy (r = 0.6-0.84; lineages that start higher gain more),
a confounder the design cannot remove because the protocols have different starts.
Winner's curse: re-folded start energies are higher than the stored phase-1 best by +0.40 (random), +0.33 (mosaic), +0.29 (mixed); the
reported phase-2 best is optimistic by 0.30 (random), 0.17 (mosaic), 0.33 (mixed) relative to the re-folded mean.

Sample size (my estimate, not computed from these data with a power program): detecting a difference of 0.5 energy units between two
protocols with a run-to-run sd of about 0.8 at 80% power would need on the order of 40 independent runs per protocol. Independent
means different start sequences and seeds, not repeated seeds from the same start.

## 11d. BCMA binder benchmark: 10 mosaic runs vs 10 random runs (2026-10-12)

Setup: 60-residue binder against human BCMA (TNFRSF17, UniProt Q02223, 54 residues, PDB 4ZFO chain A; note 1XU1 is TACI, not BCMA).
Energy: pTM (w 1), pLDDT of the complex (w 1), binder-target cross PAE (w 2), ipSAE-min (w 2). Algorithms: `Canonical` (uniform random
substitutions) vs `MosaicGradientProposal` (every proposal from the gradient, `T_prop` 0.3, memory of rejected moves). Metropolis with
temperature 0.1 for 50 steps then 0.02 for 200 steps, four times, 1000 steps, no early stopping. Run i starts from the same random
sequence in both algorithms (paired design) and every run has its own RNG seed. Scripts: `run_bcma.py`, `analyze_bcma.py`,
`launch_bcma.sh`, `finish_bcma.sh`; output `results/bcma/final_analysis.txt`, `results/bcma/curves.png`.
Primary metric and test were fixed before the results: mean energy of the final best sequence re-folded 8 times (unbiased against the oracle
noise), two-sided Mann-Whitney U, alpha 0.05.

| metric (lower is better) | random (n=10) | mosaic (n=10) | mosaic - random [95% bootstrap CI] | p |
|---|---|---|---|---|
| **PRIMARY**: re-folded final best sequence | -1.18 (sd 0.57) | -1.21 (sd 0.71) | -0.02 [-0.56, +0.49] | 0.91 (Mann-Whitney), 0.92 (paired Wilcoxon) |
| reported best energy at step 1000 | -1.36 | -1.41 | -0.05 [-0.55, +0.44] | 0.91 |
| best energy up to step 500 | -0.66 | -1.07 | -0.41 [-0.95, +0.12] | 0.16 unpaired; 0.049 paired (mosaic lower in 8/10 pairs) |
| best energy up to step 250 | -0.27 | -0.63 | -0.36 [-0.87, +0.12] | 0.21 unpaired; 0.13 paired |
| best energy up to step 100 | +0.24 | -0.28 | -0.52 [-0.92, -0.12] | 0.045 unpaired; 0.004 paired (mosaic lower in 9/10 pairs) |

Conclusions. (1) At the end of 1000 steps the algorithms are indistinguishable on the primary metric (difference -0.02; the CI excludes
advantages larger than about 0.5 energy units for either algorithm; probability that a mosaic run beats a random run 0.48). (2) The
only effect that stands out is **speed early in the run**: mosaic is lower at step 100 in 9 of 10 pairs (paired p = 0.004, which survives a
Bonferroni correction over four checkpoints; the unpaired p = 0.045 does not). By step 500 the advantage is no longer robust after
correction and by step 1000 it is gone: the random runs catch up. (3) The two lowest final sequences came from mosaic runs
(re-folded -2.47 and -2.40, against -2.09 for the best random run), but the chance that the top two of 20 are both from one algorithm
is 24% under no difference, so this is not evidence. (4) Cost: 7.4 s per step for mosaic against 7.1 s for random (4% slower), acceptance
0.086 against 0.102. (5) Optimism of the reported best energy relative to the re-folded mean: -0.18 (random), -0.21 (mosaic).
Diagnostic that does not support the gradient ranking: over 10,000 mosaic proposals the Spearman correlation between predicted and
true energy change is -0.28 (wrong sign; contaminated by the oracle noise and by the memory of rejected moves, see 11b) and gradient proposals
were on average not less harmful than random ones (mean true change +0.92 against +0.78). Why mosaic is faster at the start is not
established; one untested hypothesis is that the gradient favours a different mutation distribution (for example residues that raise
pLDDT quickly) rather than ranking mutations accurately. Limits: one target, one energy function, no tuning of `T_prop` or the temperatures.
With sd of about 0.6-0.7 and n=10 per group, an unpaired test has 80% power only for differences of about 0.8 energy units or more.

## 11e. Why the gradient proposals are not predictive (investigation of the negative correlation, 2026-10-15)

Observation (BCMA benchmark): over 10,000 mosaic proposals, Spearman(predicted change, true energy change) = -0.28; within one starting
state the median is -0.20. Question: implementation bug or property of the method? Scripts: `gradcheck.py`, `gradcheck_analyze.py`,
`finite_diff.py` (the service gained a `soft_value` method for the last one); data in `results/gradcheck/`.

**No implementation bug found; three independent checks:**
1. *The surrogate agrees with bagel.* For 360 single mutants (60 each of 6 final best sequences: 20 best-predicted, 20 random, 20
   worst-predicted by the gradient), the exact change of the mosaic loss (re-evaluated at the mutated sequence) correlates with the true
   bagel energy change (each side averaged over 3-8 folds) with Spearman +0.94 pooled (+0.73 to +0.96 per sequence). Token order, masks,
   loss signs and the bagel side are therefore consistent.
2. *The gradient is correct for what it differentiates.* Finite differences on one sequence (mosaic_9, 12 mutations): moving a fraction
   eps of a position's one-hot entry from the current residue to the new one changes the loss by about eps x (g_new - g_current) for
   eps < 0.5 (for example slope -0.05, observed -0.007 to -0.012 at eps = 0.3).
3. *The loss value from the gradient call and from the soft-input call agree* (-2.4074 vs -2.4079).

**What the gradient does not see.** At eps just above 0.5 the most likely residue flips; mosaic then rebuilds the binder atom layout and
the ESMC language-model features for the new residue (both are detached/stop-gradded by design, see `refresh_binder_geometry` and
`recompute_binder_lm_hidden` in mosaic) and the loss jumps by +2 to +3 for most mutations (for example F15K: first-order prediction
-0.05, actual at eps = 1 +2.98; conservative changes such as S->T give +0.19). The jump is 40-300 times the first-order prediction and
mostly harmful. A real mutation is therefore a discontinuous step, not a small move along the gradient. Consequently the first-order
prediction is uncorrelated with the exact change of mosaic's own loss (Spearman +0.05 pooled) and with the true change (+0.05).
Mean true change by group: best-predicted +1.34, random +1.22, worst-predicted +1.48 (at these optimised sequences 95-100% of single
mutants are harmful). So the ranking carries no usable information at optimised sequences; it did at a random start (section 11:
top-ranked mutants clearly less harmful), where the sequence is far from a good minimum.

**The negative sign is not a property of the gradient.** In the controlled test the correlation is about zero with a spread between
sequences (-0.21 to +0.34, n=60 each) and +0.14 inside the best-predicted tail. In the runs, part of the negative pooled correlation is a
confound between states: the current energy correlates positively with the predicted change (+0.20) and negatively with the true change
(-0.51), because the oracle noise makes a state with a lucky low energy look hard to improve. A residual within-state median of -0.20
is not explained (hypotheses, not tested: the memory of rejected moves forces later proposals down the ranking; the 33% share of
proposals concentrated on the 5 most-proposed positions).

**What does work: the mosaic forward pass as a cheap screen.** Screening 20 random mutants of an optimised sequence and keeping the one
with the lowest *exact* surrogate change gives a mean true change of +0.12, against +1.22 for a random mutant and +0.90 when the
lowest first-order prediction is kept; the 7 improving mutants (of 360) sit at the 5th percentile of the exact-surrogate ranking (0.44
for the first-order ranking, where 0.5 = no information). This is a small sample (6 sequences, 7 improving mutants) and was not run as
an optimisation. A proposal built on forward evaluations of a batch of candidate mutants (no gradient) is the obvious next experiment.


## 12. Known limitations and next steps

* One State, one designed chain, ESMFold2 only (other oracles/models need their own mosaic wrapper and image).
* Surrogate/oracle mismatch: the mosaic JAX ESMFold2 is a re-implementation with fewer sampling steps; agreement is moderate
  (Spearman ~0.5 on mutation effects at one sequence), measured at a random start only. Re-measure after the sequence has improved.
* Only 3 seeds per arm, no tuning of the annealing schedule, `T_prop` or term weights. Differences between arms are smaller than
  seed-to-seed variation (section 11). Next test: more seeds (10+), re-measure the feasibility correlation at optimised sequences, and
  compare `T_prop` values and best-of-B proposals.
* Not implemented: Hastings correction, best-of-B candidates (several folds per step, batched through one oracle call), stochastic
  surrogate (changing PRNG key), multi-state gradient sums, more term translations (SASA/atom based terms on atom37),
  adaptive `T_prop`.
* Possible follow-ups: log every proposal in `DefaultLogger` instead of the benchmark-only `Recorder`; add the mosaic service to
  CI as an optional marker (needs Modal credentials); repin boileroom so the image tag problem disappears.
