# Proposal: a bagel mutation protocol that screens candidate mutations with the official mosaic forward pass

Status: **proposal, nothing implemented from it yet.** Written 2026-10-21 against bagel 0.2.11 and the official
`escalante-bio/mosaic` at commit `1eb4af6` (its current `main`; no commits after it at the time of writing).
Evidence quoted here comes from the experiments in `docs/mosaic_proposal.md` (sections 11e-11g). Items I did not verify are marked
**[not verified]**. Decisions I need from you are collected in section 11.

---

## 0. Summary

**What the protocol does.** At each Monte Carlo step it samples `B` random single-residue substitutions, scores each mutated
sequence (and the current one) with a *pseudo-energy* computed on a mosaic structure model with **forward passes only**,
proposes the candidate with the lowest pseudo-energy, and then bagel folds it with its own oracle, computes the real energy, and accepts
or rejects as usual. Mosaic only decides *which* mutation to try; bagel's energy still decides acceptance.

**Why the forward pass.** The pseudo-energy of a *mutated sequence* includes everything that changes when a residue is replaced (the atom layout and
the language-model features are rebuilt for the new residue). In the experiments of this session it tracked the real bagel energy change closely
(Spearman +0.94 over 360 single mutants of optimised sequences, section 11e of `docs/mosaic_proposal.md`), and screening candidates with it beat random
proposals on all three targets in two independent benchmarks (section 11f: -1.58 [-1.93, -1.21], 28/30 pairs, p < 0.0001; section 11g with hotspot terms:
-0.99 [-1.35, -0.63], 26/30).

**Decisions that shape everything else (section 11):**
1. Whether the pseudo-energy is built by *translating bagel terms to JAX terms with masks* (what you specified) or by *wrapping the mosaic
   model as a bagel folding oracle* so bagel's own term code runs unchanged on mosaic's output (section 4). I think the second is simpler and
   covers far more terms, because the pseudo-energy is only evaluated, never optimised through, so its terms do not need to be written in a JAX-friendly form. I describe both and recommend a hybrid.
2. Whether v1 must support more than one *designed* chain per state. The official mosaic wrappers only inject a designed sequence into the
   **first** chain (section 6.3), so this is the main structural limitation.
3. Which mosaic models v1 must support. All evidence so far is for one model pair (ESMFold2 in both bagel and mosaic).

**What the evidence does not show** (so the protocol should not be sold beyond it): one energy family (pTM, pLDDT, PAE, ipSAE), three small
targets (54-76 residues), random starting sequences, 500 steps, `B = 8`, one model pair. The advantage was created in the first ~100 steps and
stayed constant afterwards; at equal compute cost it was significant in one benchmark and borderline in the other (section 9).

**Open question that decides whether this is worth building (measured 2026-10-21).** The timing of a bagel fold depends mostly on the GPU, not on
the model: the same 114-residue complex takes 6.43 s on a T4 and 1.16 s on an A100-80GB with bagel's own oracle, while a mosaic forward pass on the
same A100 takes 0.56-0.68 s per sequence (0.31-0.37 s per sequence in a batch of 8). Diffusion steps matter little (50 -> 14 steps: about -10%).
All cost comparisons in section 11f/11g of `docs/mosaic_proposal.md` used a T4 for bagel. With bagel on an A100 a plain step takes about 1.2 s and costs
about $0.0008 (cheaper per fold than the T4, $0.00105), while a screened step with 8 candidates takes about 3.7 s and costs about 3 times more, so at equal
wall-clock or equal cost plain bagel could run about 3 times more steps. **Whether screening still wins at equal time/cost on a fast bagel oracle has not been tested.**

---

## 1. Evidence base and what each finding implies for the design

| Finding (source) | Implication for the design |
|---|---|
| The forward pass on the mutated sequence tracks the real energy change (rho +0.94 over 360 mutants of optimised sequences, 11e). | Score candidate sequences with forward passes of the mosaic model. |
| Same-GPU timing (A100-80GB, 114 residues): bagel fold 1.03-1.16 s; mosaic forward 0.56-0.68 s per call, 0.31-0.37 s per sequence in a batch of 8; on a T4 the bagel fold takes 5.9-6.4 s. | The earlier "12-23 times faster" was mostly GPU. Real same-GPU advantage: about 1.7-1.8x per call, about 3.1-3.3x per sequence when batched. The remaining factor is not decomposed (candidates: precomputed features, no coordinate return, JIT, batching). |
| Screening with 8 candidates lowered the re-folded energy at 500 steps by 1.58 (all-residue terms) and 0.99 (hotspot terms) on 3 targets (11f, 11g). | The idea works in the tested regime. |
| The gain forms in the first ~100 steps, then both arms improve about equally (11f, 11g). | Consider a *screening schedule* (screen the first N steps, then plain random). **Untested hypothesis**, cheap to test with the existing harness. |
| A screened step costs 1.4x the wall-clock and 2.7x the compute of a plain step (11f). At equal compute cost the advantage was -0.90 (p < 0.0001) with all-residue terms but only -0.20 (p = 0.16, at the cost of 250 random steps) and -0.48 (p = 0.009, 500 steps) with hotspot terms. Idle GPU time was not counted. | Expose `n_candidates`, report cost telemetry, do not claim a cost advantage in general. |
| Surrogate-only Monte Carlo matched random with all-residue terms (p = 0.40) but was **worse** than random with hotspot terms (+0.37, p = 0.002); surrogate vs bagel agreement across runs fell from rho 0.78 to 0.38 (11f, 11g). | Fidelity of the pseudo-energy to the bagel energy matters, and it varies with the energy definition. Add a calibration check (section 8). |
| The bagel oracle is noisy: re-folding the same sequence changes the energy (up to ~1.2 for one sequence, 11b). | Calibration must average several folds per sequence, otherwise the agreement is underestimated. |
| Terms without a mosaic counterpart are simply absent from the pseudo-energy (your rule). | A *coverage* number (share of total weight covered) must be reported, because the omitted terms are optimised *against* the screening (section 9, risk R3). |

---

## 2. Scope and non-goals

In scope: substitution-only mutation protocols; one or several States; one or several chains per State; the official mosaic at a pinned commit;
a separate-process GPU service (mosaic and bagel's pinned boileroom cannot share a Python environment, shown with `uv lock` earlier: boileroom 0.4.5
needs `torch<2.7`, mosaic pins `torch==2.11.0`).

Out of scope for v1 (proposed): length-changing moves (`GrandCanonical`); non-protein chains; using mosaic's own extra loss
terms that bagel does not have (for example pseudo-likelihoods, inverse folding) as an additional *prior* on proposals. That last item is an
interesting extension, listed in section 10.

---

## 3. Mapping bagel EnergyTerms to official mosaic terms (deliverable 1)

Source for bagel: the 24 `EnergyTerm` subclasses of `src/bagel/energies.py` (each `compute()` was read in full). Source for mosaic: every `LossTerm`
of the official repo. "Equivalent" means numerically the same quantity (checked where stated); "approximate" means a related but different quantity;
"none" means no counterpart. Conventions: all values are to be minimised in both libraries; mosaic's `plddt` is on a 0-1 scale (as bagel's `local_plddt`)
and `pae` is in Angstrom.

### 3.1 bagel term -> mosaic

| # | bagel term | what it computes | closest official mosaic term | equivalence | masking needed in the bagel layer | what the pseudo-energy needs from the model |
|---|---|---|---|---|---|---|
| 1 | `PTMEnergy` | -pTM of the whole complex | no `LossTerm`; the helper function `predicted_tm_score(logits, bins)` with no pair mask is the same quantity (`BinderPTMLoss`, `IPTMLoss`, `pTMEnergy` are different quantities) | **equivalent via the helper** (values agreed in this session: 0.685 vs 0.681) | none | `pae_logits` |
| 2 | `PLDDTEnergy(residues)` | -mean pLDDT over a residue group | `PLDDTLoss` (mean over the whole binder only) | equivalent only if the group is exactly the binder; otherwise needs a masked mean | residue mask over tokens | `plddt` |
| 3 | `OverallPLDDTEnergy` | -mean pLDDT over all residues | none (`PLDDTLoss` is binder-only) | trivial (masked mean over all tokens) | all-ones mask | `plddt` |
| 4 | `PAEEnergy` | mean PAE of pairs (group 1 x group 2, both orders) divided by 30; optional union block | `BinderTargetPAE`, `TargetBinderPAE`, `WithinBinderPAE` | equivalent up to the factor 30 and the symmetrisation, **only** when the groups are the whole binder and whole target | pair mask, diagonal excluded | `pae` |
| 5 | `LISEnergy` | -mean of (cutoff - pae)/cutoff over cross pairs with pae < cutoff | none | none (simple to compute from `pae`) | pair mask | `pae` |
| 6 | `ipSAEEnergy` | directional ipSAE between two groups, `direction` in {1to2, 2to1, min, max, mean} | `BinderTargetIPSAE`, `TargetBinderIPSAE`, `IPSAE_min` | equivalent for whole binder x whole target (cutoff 10 and the d0 formula are identical; mosaic's max reduction is the default); general groups need masks | pair mask for both directions | `pae` |
| 7 | `SurfaceAreaEnergy` | mean atom SASA of selected atoms / max SASA, SASA of the whole complex | none | none | atom mask | all heavy-atom coordinates |
| 8 | `HydrophobicEnergy` | fraction of hydrophobic atoms (optionally weighted by SASA) | none | none | atom mask | all heavy-atom coordinates |
| 9 | `HydropathyEnergy` | mean Kyte-Doolittle index (optionally SASA-weighted) | none | none | residue mask | all heavy-atom coordinates (SASA) |
| 10 | `SaltBridgeEnergy` | count of charged side-chain atom pairs within a cutoff between two groups | none | none | two residue groups | side-chain atom coordinates |
| 11 | `RingSymmetryEnergy` | spread of distances between group centroids | none | none | N residue groups | backbone coordinates |
| 12 | `SeparationEnergy` | distance between two groups (centroid or minimum CB distance) | none (mosaic's contact terms use the distogram, a different quantity) | none | two residue groups | backbone / CB coordinates |
| 13 | `FlexEvoBindEnergy` | mean minimum all-atom distance between two groups, optional pLDDT weight | none | none | two residue groups | all-atom coordinates (+ `plddt`) |
| 14 | `GlobularEnergy` | standard deviation of backbone-atom distances to the centroid | nearest: `ActualRadiusOfGyration` (binder only, N atoms only, elu against a target radius) | approximate, **not the same formula** | residue mask | backbone coordinates |
| 15 | `TemplateMatchEnergy` | RMSD to a template after superposition (atom-level correspondence) | none | none | residue group, template atoms | atom coordinates with matching atom lists |
| 16 | `SecondaryStructureEnergy` | fraction of residues whose P-SEA label differs from the target | nearest: `HelixLoss` (distogram helix propensity) | approximate, different quantity | residue group | CA coordinates |
| 17 | `ShapeComplementarityEnergy` | probe-excluded surface dot complementarity at the interface | none | none | two residue groups | all-atom coordinates |
| 18 | `BuriedHistidineEnergy` | saturating reward for buried histidines | none | none | optional residue group | all-atom coordinates |
| 19 | `HistidineCarboxylateContactEnergy` | counts His N / Asp-Glu O atom pairs in contact, whole complex | none | none | none (term has no groups) | side-chain atom coordinates |
| 20 | `BinderRMSDEnergy` | RMSD of CA atoms to a *reference State's* structure | none | none; also needs **another State's** fold result | residue group | two structures (cross-state) |
| 21 | `ChemicalPotentialEnergy` | penalty on total residue count | none | **constant for substitution-only moves** (adds exactly 0 to the change in energy, so omitting it is exact) | none | sequence length only |
| 22 | `EmbeddingsSimilarityEnergy` | 1 - mean cosine similarity of per-residue embeddings to a reference | none (mosaic's language-model losses are pseudo-likelihoods, not embeddings) | none | residue group, order-matched rows | embedding oracle output |
| 23 | `SAEnergy` | weighted sum of sparse-autoencoder features (ESMC-6B layer 60 only) | none | none | none | embedding oracle output |
| 24 | `ResidueSAEnergy` | pooled per-residue SAE activations | none | none | residue group | embedding oracle output |

Summary: 6 bagel terms (1-6) can be written as JAX terms from the confidence outputs alone and need masking for general residue groups
(official mosaic supports whole-binder/whole-target only for these; the exceptions are `BinderTargetContact` and a few likelihood losses, section 3.2);
13 need atomic coordinates (7-19, plus 20 which also needs another State); 4 need a language-model/embedding oracle or a sequence-length term (21-24).

### 3.2 mosaic term -> bagel (terms with no bagel counterpart)

| mosaic term (official) | what it is | bagel counterpart |
|---|---|---|
| `WithinBinderContact`, `ESMFoldInterContact`, `BinderTargetContact` (has `paratope_idx`, `epitope_idx`) | distogram contact log-likelihoods | none |
| `HelixLoss`, `ESMFoldGlobularity`, `DistogramRadiusOfGyration`, `MAERadiusOfGyration`, `DistogramCE` | distogram-based shape priors | none (see rows 14 and 16 for the nearest bagel terms) |
| `IPTMLoss`, `BinderTargetIPTM`, `DistogramIPTMProxy`, `BinderPTMLoss`, `pTMEnergy` (cross-chain proxy) | iPTM family and binder-only pTM | none (closest: `ipSAEEnergy`, `LISEnergy`) |
| `ESMCPseudoLikelihood`, `ESMCPseudoPerplexity` (has `design_idx`), `ESM2PseudoLikelihood`, `AbLangPseudoLikelihood`, `Ablang2PseudoLikelihood` | language-model pseudo-likelihoods | none |
| `StabilityModel`, `TrigramLL`, `UnigramExcess`, `BigramExcess` | sequence-only priors | none |
| `ProteinMPNNLoss`, `InverseFoldingSequenceRecovery`, `AllResiduePLLLoss`, `FixedStructureInverseFoldingLL`, `ProteinaInverseFoldingRecovery` (has `hotspots`) | inverse-folding likelihoods on the predicted structure | none (bagel's `InverseFoldingSampling` uses ESM3 as a proposal, not an energy) |
| `FixedPositionsPenalty`, `NoCys`, `SoftClip`, `ClippedLoss`, `SetPositions`, `RandomChoice`, `ClippedGradient`, `NormedGradient` | wrappers and penalties for the relaxed optimisation | not energy terms |
| model wrappers `Boltz2Loss`, `ESMFold2Loss`, `MultiSample*Loss`, `AlphaFoldLoss`, ... | run a model and call an inner loss | not energy terms |

**Rule I propose:** mosaic-only terms are *not* added to the pseudo-energy (it must mirror the bagel energy). Offering them as an optional, explicit
"proposal prior" is a later extension (section 10).

---

## 4. The central design question: how to build the pseudo-energy

### Option A (what you specified): translate bagel terms into JAX terms with masks
For each translatable bagel term, a JAX `LossTerm` evaluated on mosaic's `StructureModelOutput`, with bagel's residue groups turned into token masks.
This is what the prototype already does for terms 1-6 (`mosaic_losses.py`).

### Option B: wrap the mosaic model as a bagel `FoldingOracle` and reuse bagel's own terms
The official `predict()` already returns, for a hard one-hot sequence, a structure (all heavy atoms of standard residues through the `atom37`
view), `plddt`, `pae`, and the `StructureModelOutput` (which also has `pae_logits`). An adapter would build a bagel `ConfidenceFoldingResult` from
that (AtomArray from `atom37_coords/mask`, `local_plddt`, `pae`, `ptm` from the helper) and bagel's **own** `EnergyTerm.compute()` runs on it
unchanged.

| | Option A: translated JAX terms | Option B: adapter oracle |
|---|---|---|
| Terms covered | 1-6 only (the confidence-only terms) | 1-19 (all structure-based terms, including SASA, hydrophobic, separation, salt bridge, shape complementarity, ...); 20-24 remain unavailable |
| Residue groups, chain order, multi-chain | must be re-implemented as masks (and kept consistent with bagel's `get_residue_mask`) | handled by bagel's own code, because the AtomArray uses bagel's chain IDs and `Residue.index` |
| Risk of divergence between the two implementations | real (every term is written twice; this session already found ipSAE details that must match) | none by construction (same code), apart from the model's numbers |
| Terms must be written in JAX | yes (only needed to run them on the GPU) | no |
| Where the energy runs | on the GPU, only scalars come back | on the host CPU in numpy/biotite; returns coordinates + a PAE matrix per candidate (for 136 residues that is about 18,000 numbers plus ~1,000 atoms) |
| Speed | fastest | extra CPU time per candidate (SASA-type terms: tens of ms, **not measured**); for `B = 8` this is likely small against the 3.9 s GPU batch **[not verified]** |
| Hard parts | maintaining masks and per-term parity tests | atom-list differences between mosaic's heavy-atom output and the real oracle's output (hydrogens, terminal atoms): matters for `TemplateMatchEnergy` and anything that counts atoms **[not verified]**; building an AtomArray from `atom37` |

**My recommendation: hybrid, selectable per term.** Use Option B as the general mechanism (it gives the widest coverage with the least duplicated code
and the fewest parity risks) and keep Option A only for terms where profiling shows the host-side evaluation is a bottleneck. In practice the config would
have `term_backend='adapter' | 'translated' | 'auto'`. This is a design choice for you (decision D1); the rest of this document works with either.

Note on your rule 2 ("a term without an equivalent is simply absent"): under Option B almost no structure-based term is absent, so the rule would mostly
apply to terms 20-24. If you prefer to keep strictly "mosaic terms only", Option A is the faithful reading, and the section 3.1 table is the complete list of what it covers.

---

## 5. The protocol (sketch of behaviour, not code)

`MosaicScreenedProposal(mosaic=<MosaicBackend>, models=<per-state model specs>, n_candidates=8, ...)` as a `MutationProtocol`:

1. Choose the chain with bagel's own `MutationProtocol.choose_chain` (probability proportional to its number of mutable residues), so chain-selection statistics are unchanged.
2. Sample `n_candidates` distinct single substitutions uniformly over the mutable positions of that chain and the amino acids allowed by `mutation_bias`,
   excluding moves already proposed from the current sequence (a tabu memory; needed because scores are deterministic and the same candidate would otherwise be proposed again).
3. For **every State that contains the chosen chain**: score the current sequence and the candidate sequences with that State's pseudo-energy (one batched mosaic call per State, constant batch size).
   States that do not contain the chain are unaffected by the move and need no call.
4. Score of a candidate = sum over the affected States of (pseudo-energy of the mutated State - pseudo-energy of the current State). This mirrors `System.get_total_energy()`, which sums State energies.
5. Propose the candidate with the lowest score; return the mutated `System` and a `MutationRecord` like any other protocol (callbacks and replay keep working).
6. Cache scores per (State, sequence-of-its-chains) so that steps after a rejection need no new mosaic call until the sampled candidates are all new.
7. Telemetry per step: predicted change, number of mosaic calls, mosaic seconds, State count (for the cost reporting that the evidence says we need).

Choices that should be configurable: `n_candidates` (default 8; the evidence covers only 8), `screening_schedule` (always; first N steps only; every k-th step), the model spec per bagel oracle (section 7), `unmapped_terms='warn'|'error'`, `min_weight_coverage`, `calibration='off'|'warn'|'error'`.

---

## 6. Masking, multiple chains and multiple States

### 6.1 Token order and the index map
Mosaic scores one complex as a flat token sequence with the designed chain first and the other chains after it (every official model wrapper assumes this).
For each State the translation layer builds, once:

- `chain_order = [design chain, *other chains of the State in State order]`;
- `offset[chain_ID]` = number of tokens before that chain;
- the map `(chain_ID, Residue.index) -> token position = offset[chain_ID] + position of the residue in the chain`.

A bagel `EnergyTerm`'s `residue_groups` (arrays of chain IDs and residue indices) become boolean token masks through this map. Two things must be guarded:
(a) `Residue.index` must equal the residue's list position in its chain (true unless residues were added or removed, which `GrandCanonical` does; checked at construction, error otherwise);
(b) bagel's own `get_residue_mask` aligns masks with the confidence arrays by chain order of first appearance in the structure; the adapter must therefore emit chains in the same order the State's oracle would
(the order of `state.chains`), or the masks built by bagel's term code would silently misalign **[this is the single easiest place to introduce a wrong-residue bug; a dedicated test is required]**.

### 6.2 Multiple chains per State
Fixed chains (targets, partners) are unproblematic: they are passed to mosaic as fixed chains with their real sequences and all of them are scored together.
Fixed residues *inside* the designed chain are handled as in the prototype: the whole chain is passed as a one-hot of its current sequence, and only mutable positions are ever proposed.

### 6.3 More than one designed chain in a State (main limitation)
In the official models the designed sequence is written into the first `N` tokens only (ESMFold2: `losses/esmfold2.py:48`, `set_binder_sequence` and the geometry/language-model refresh,
whose docstring says "assumes a fully-designed, contiguous, first binder"; the same prefix convention holds for Boltz2, AF2, Protenix, OF3, OpenDDE and Promera). Nothing asserts on violations; a second designed chain would simply be treated as fixed.
Handling considered (**not in v1, see section 13: v1 allows one designed chain per State**): when a move targets chain `c`, `c` becomes the first (designed) chain and every other chain, including other designed chains, is passed as a fixed chain with its current sequence.
Consequence: the feature pack depends on the current sequences of the fixed chains, so it must be rebuilt whenever one of them changes (cache keyed by `(State, design chain, sequences of the other chains)`).
**Measured (2026-10-21, official model code, `probe_feature_shapes` in `modal_service.py`):** the atom axis of the input features of a design pack
depends on the composition of a *fixed* chain: with a 60-residue designed chain, a fixed chain of 20 Gly gives 928 atoms, 20 Ala 960, a mixed
20-residue chain 1024, 20 Trp 1120. A designed chain, in contrast, always has a constant size (14 atoms per residue placeholder; two designed chains of 60 and
20 residues give 1120 atoms whatever their sequences). So every time a fixed chain mutates to a different total atom count, the JAX function sees a new input
shape and is recompiled (about 20 s measured for a warm 114-residue problem; 100 s or more with container start). Compiled shapes are cached, so the cost is paid once per distinct
atom count, but it would make a run with a changing fixed chain jittery. Options: accept and cache; pad the fixed chain to a constant atom budget (needs a change to mosaic, the featurisation
is in `_featurize`/`refresh_binder_geometry`); or restrict v1. The cost of the featurisation itself (CPU) is still **[not verified]**.
v1 restricts a State to a single designed chain and raises a clear error for more (guardrail G4). Decision D2 taken (section 13).

### 6.4 Multiple States
Each State is its own mosaic "problem" (its own complex, features and terms, hence its own compiled function). Costs scale with the number of affected States times `n_candidates`,
and each new problem triggers a compilation (about 2-5 minutes in this session for one problem). The service must keep several compiled problems resident (GPU memory and warm-container cost grow with the State count).
A chain shared by several States is handled by step 3-4 of section 5. Terms that couple States (`BinderRMSDEnergy`) are not translatable (row 20) and are flagged.

---

## 7. Different oracles for bagel and for mosaic (your item 5)

Principle: the protocol takes an explicit **model spec for mosaic** per bagel folding oracle used in the State, independent of the oracle that evaluates the real energy.

- Default mapping by class (to be confirmed by tests): bagel `ESMFold2` -> mosaic ESMFold2 Full (same checkpoint family as boileroom's `biohub/ESMFold2`, which is what the evidence used); bagel `Boltz2` -> mosaic Boltz2; bagel `ESMFold`, `Chai1`, ESM-C/ESM2/ESM3/SAE oracles -> **no mosaic counterpart**.
- The user may override the model per State (for example evaluate with Boltz2 in bagel while screening with ESMFold2 in mosaic). Settings such as recycling steps and diffusion sampling steps are part of the spec; the prototype used 3 and 14 (bagel's default is 3 and 50).
- A State with terms on several bagel oracles needs one mosaic spec per oracle that has a counterpart; terms on an oracle with none (embedding-based terms) are unmapped.
- Different model families have different calibrations of pLDDT and PAE; the weights of the bagel terms were chosen for the bagel oracle. **Nothing in this session tests a mismatched pair**, so the screening gain for such pairs is unknown (risk R4); the calibration step (section 8) is the protection.
- Whether mosaic's pLDDT is on a 0-1 scale for every model is documented in code only for some models (confirmed here for ESMFold2); a per-model unit test against the bagel oracle is required before a model is declared supported.

---

## 8. Guardrails (your item 6)

All checks live in one validation function that runs at construction and again at the first step (and the cheap ones every step). Failures raise
`MosaicIncompatibleError` (a `ValueError` subclass) with a message that says what is incompatible, why, and what to change. "Exit" is implemented as an exception, so callers and the `Minimizer` stop immediately; I do not propose `sys.exit`.

| # | Condition | Checked when | Action | Message content |
|---|---|---|---|---|
| G1 | `GrandCanonical` (or any protocol that adds/removes residues) combined with this protocol, as a wrapper, an exploration protocol, or in the same minimizer | construction | **error** | the compiled mosaic functions have fixed shapes; a length change would need a new feature pack and recompilation (minutes) for every move; use substitution-only sampling |
| G2 | a chain length differs from the one the problem was built for | every step, before any mosaic call | **error** | same explanation, names the chain and the two lengths |
| G3 | `Residue.index` != list position in a chain (residues were added or removed) | construction | **error** | masks would address the wrong tokens |
| G4 | a State has more designed chains than the backend supports (or the fixed-chain rebuild is disabled) | construction | **error** | official mosaic injects a designed sequence only into the first chain; lists the chains |
| G5 | a non-protein chain, or a residue letter outside the 20 amino acids | construction | **error** | mosaic models used here support standard amino acids only |
| G6 | a chain with a template, or a State whose oracle needs one | construction | **error** | the ESMFold2 wrapper in mosaic does not support templates |
| G7 | no term of a State has a mosaic counterpart | construction | **error** | nothing to build a pseudo-energy from |
| G8 | some terms have no counterpart | construction | **message, not an error**: the terms are left out of the pseudo-energy and listed in the log and in `Mosaic_pseudo_energy.txt` (section 13.2) | list of used and unused terms with the reason, and the coverage by number of terms and by weight |
| G9 | (no threshold) | - | the coverage is only reported, as decided | - |
| G10 | a term that needs another State's result (`BinderRMSDEnergy`) | construction | treated as unmapped | explains the cross-state dependency |
| G11 | a term whose bagel oracle has no mosaic spec (and no default) | construction | unmapped; error if it is the only oracle of the State | tells which spec to supply |
| G12 | mosaic backend not reachable, or its reported `mosaic` commit differs from the pinned one | construction | **error** | prints both commits |
| G13 | calibration agreement (Spearman between pseudo-energy and bagel energy over K sampled mutants, bagel energies averaged over several folds) below a threshold | first step if `calibration` is on | warn or error | the measured value and the threshold. **The threshold is not known**: the benchmarks gave 0.59-0.65 for the screened proposals and 0.38-0.78 for surrogate-vs-re-folded across runs; I would not hard-code a number before more targets are measured |
| G14 | more than one designed State chain with `n_candidates` x States above a cost limit | construction | warn | estimated seconds and dollars per step (from the measured costs) |

---

## 9. Risks, things that may not make sense, and open technical points

**R1. Evidence is narrow.** One energy family, 3 small targets, random starts, 500 steps, `B = 8`, one model pair. Multi-State, multi-chain, other models and larger complexes are untested. The protocol should ship as experimental with the calibration and telemetry on by default.

**R2. Cost.** Measured with bagel on a T4: 1.4x wall clock and 2.7x compute per run (compute-only model); at equal cost the advantage was clear for all-residue
terms (-0.90) but not significant at the cost of 250 random steps for hotspot terms (-0.20, p = 0.16). **With bagel on an A100 the picture is different (section 0):**
a bagel fold takes about 1.2 s and is cheaper per fold than on a T4, whereas scoring 8 candidates costs about 2.5 s of A100 time, so a screened step is about 3 times slower and
more expensive than a plain step. The equal-step advantage is real, but equal-time/equal-cost with a fast bagel oracle is untested. A warm A100 held by a single run is billed while idle.
A screening schedule (first N steps only) might remove most of the cost because the gain appears early, **but this is a hypothesis, not a result**.

**R3. Omitted terms are optimised against.** If a heavy term (say a hydrophobic-surface term) is absent from the pseudo-energy, screening picks mutations that are good for the covered terms and may systematically worsen the omitted one. Option B largely removes this, Option A does not. Hence G9 (coverage) and the calibration.

**R4. Different model pairs are unvalidated.** Weights and units are tuned for the bagel oracle; the agreement of two different models can be much lower than for the same family. In the hotspot benchmark, where the surrogate agreed less with the real energy (+0.38 across runs instead of +0.78), the surrogate-only arm became worse than random.

**R5. Several designed chains in one State.** Official mosaic does not support it directly (section 6.3). The workaround needs a feature-pack rebuild whenever another designed chain changes; cost unknown.

**R6. Reproducibility of "official mosaic".** The official repository has **no lock file** (`uv.lock` is absent and ignored) and its twelve git dependencies (`joltz`, `esmjfold2`, `boltz`, ...) are not pinned to commits, so two image builds on different days can differ. Pinning the mosaic commit is not enough. Proposal: resolve once on Linux, commit a lock/constraints file into the bagel service directory and build with it frozen; the service reports the mosaic commit and a hash of the lock at start (G12). Also, mosaic supports only Linux x86_64 and needs a GPU, so it cannot run in a macOS development environment.

**R7. Official vs your fork.** The official version has no residue selection in its loss terms; your fork added it on 2026-10-12. Under this proposal bagel provides the masks itself (Option A) or does not need them (Option B), so the official version is sufficient. The prototype never used your selection feature, and I have not compared it with my masked terms.

**R8. Noise.** The bagel oracle is noisy (11b) while the mosaic forward pass is deterministic for a fixed diffusion key, so the screening ranks candidates by one fixed noise realisation. The evidence says this is good enough (rho about +0.6 in the loop), but it is a property to keep in mind, and calibration needs several folds per sequence.

**R9. Compile time and warm-up.** Each distinct (problem, batch size) is compiled once per container (2-5 minutes observed for one problem). Many States or varying batch sizes multiply this; the batch size must be constant.

**R10. Bias in the proposal distribution.** Choosing the best of `B` random candidates is not a symmetric proposal; with the Metropolis test on the real energy this is an optimiser, not an exact sampler (same status as the existing `InverseFoldingSampling`).

**R11. Adapter details that could bite (Option B).** Atom lists of mosaic's heavy-atom view vs the real oracle output; element annotation (`to_structure` sets the element from the first letter of the atom name, which bagel's shape-complementarity term uses for van der Waals radii); residue numbering (`reindex_residues` in bagel overwrites `res_id` with `Residue.index`); chain-ID mapping. All **[not verified]**.

**R12. What does not need to be solved.** Because the pseudo-energy is only evaluated, hard cutoffs, argmax, top-k and other non-smooth terms are no problem.

---

## 10. Possible extensions (not in v1)
- Screening schedule and adaptive `n_candidates`.
- Using the surrogate to *pre-filter* more candidates than `B` at a cheaper model setting.
- An optional "proposal prior" from mosaic-only terms (pseudo-likelihoods, inverse folding) with a weight, clearly separated from the pseudo-energy.
- Reusing the surrogate to skip a bagel fold when it is confidently worse (changes the sampling; needs its own evidence).

---

## 11. Questions and decisions for you

- **D1.** Option A (translated JAX terms with masks), Option B (adapter oracle, bagel terms unchanged), or the hybrid I recommend? Is covering the coordinate-based terms (SASA, hydrophobic, separation, ...) a goal for v1, or should v1 stay with the six confidence-based terms?
- **D2.** Must v1 support more than one *designed* chain per State? If yes I propose the "mutated chain becomes first, others fixed" scheme and a feasibility measurement of the pack rebuild before committing to it.
- **D3.** Which mosaic models must v1 support? The evidence is for ESMFold2 only; Boltz2 is the obvious second candidate but I have not run any Boltz2 path in this session.
- **D4.** Compute backend: Modal only (as prototyped), or also a local GPU / SLURM-Apptainer path like boileroom's? This decides how the service is packaged.
- **D5.** Is it acceptable to ship a pinned lock file and a Dockerfile/Modal image definition for the mosaic service inside the bagel repository (needed for R6)?
- **D6.** Unmapped terms: warn and continue (my default) or error? What coverage threshold would you consider too low (for example 50% of the total weight)?
- **D8.** Do you want the calibration step on by default? It costs several bagel folds per sampled mutant (for K = 24 mutants and 3 folds each, about 72 folds at the start of a run).

## 12. Proposed implementation plan (after the decisions)
1. **Service (mosaic side).** Pinned official commit and lock; handshake (commit + lock hash); a forward-only batched scoring call that accepts a problem description (chains, design chain, models, optional terms) and sequences; fixed batch size; optional return of structure and confidence arrays (needed for Option B); several compiled problems resident; cost telemetry.
2. **Translation layer (pure Python, no GPU, unit-tested).** Term catalogue implementing section 3 (status, reason, weights); index map and masks (section 6); State/chain remapping; guardrail validation (section 8) with one test per row G1-G14.
3. **Parity tests.** Each translated term against the bagel term on synthetic `plddt`/`pae`/`pae_logits` arrays (CPU, no model); the adapter against bagel on synthetic AtomArrays; the masks against `get_residue_mask`.
4. **Protocol.** `MosaicScreenedProposal` as in section 5, with a fake backend for unit tests (extending the tests already written for the prototype).
5. **Validation.** Calibration utility; re-run the 3-target benchmark harness of section 11f with the new protocol and require that it reproduces the old result within its confidence interval; a multi-State and a multi-chain integration test on a small system; cost reporting.
6. **Documentation.** The mapping tables of section 3 as a maintained doc generated from the catalogue, so it cannot drift from the code.


---

## 13. Decisions taken and research results (2026-10-21)

### 13.1 Decisions
| | Decision |
|---|---|
| D1 | **Adapter first** (bagel's own term code runs on a result object built from mosaic's output). Translated JAX terms only if measurements show the adapter overhead matters; the JAX terms already written serve as a cross-check. |
| D2 | **One designed chain per State in v1.** More than one raises an error (G4). The rotation scheme is out of v1; the measured recompilation on a changing fixed chain (section 6.3) is the reason it would need extra work. |
| D3 | ~~All oracles shared by bagel and mosaic~~ **Superseded in section 14:** v1 supports the folding oracles **ESMFold2** and **Boltz2** only; no embedding oracles (ESM-C, ESM2) in v1. |
| D4 | Deploy on **Modal**, on a **local GPU**, and on **SLURM/PBS** (HX1 at Imperial is PBS Pro, 13.4). |
| D5 | Pin the mosaic version for reproducibility; a fork is acceptable. Publishing a built image is acceptable if the licences allow it (13.5). |
| D6 | Terms without a mosaic equivalent are left out of the pseudo-energy. A message and the file `Mosaic_pseudo_energy.txt` (in the run's log folder) list the used and unused terms and two coverage numbers: **terms used / terms total**, and **sum of the weights used / sum of all weights** (absolute values). |
| D8 | A calibration check runs **by default** and can be removed or tuned; if the rank correlation is below **0.25** the protocol flags it and switches to pure random sampling (open questions in 13.6). |
| - | The earlier prototype code on this branch that is not part of the new protocol is to be **deleted**, after the equal wall-clock comparison has finished (those runs still import it). |
| - | The "screen only the first N steps" option is **deferred**; it will be examined with the data of the equal wall-clock comparison. |

### 13.2 `Mosaic_pseudo_energy.txt` (proposed content)
Written once at construction into the log folder (next to `config.csv`) and echoed to the log: for each State, a table of its energy terms with columns *term, weight, used (yes/no), mosaic model/term or adapter used, reason if unused*; then
`coverage by number of terms = used/total` and `coverage by weight = sum |w_used| / sum |w_all|`, per State and for the whole System; and the list of chains, the designed chain, and the mosaic model spec for each oracle.

### 13.3 Oracle equivalence found by reading the code (nothing run numerically)
**Boltz2.** bagel/boileroom: `boltz==2.2.1` (`boltz2_conf.ckpt`), 3 recycling steps, **200** diffusion steps, 1 sample, bf16 mixed precision with cuequivariance kernels, a **ColabFold MSA for every chain, including the designed one**, default GPU T4; outputs `plddt` (0-1), `pae` (Angstrom), `ptm`.
Official mosaic: the same checkpoint through a JAX translation, defaults **1** recycling step and **25** diffusion steps, fp32, the designed chain gets an *empty* MSA (the profile is overwritten), no `ptm` output (compute it with the helper `predicted_tm_score`). Verdict: can be matched with changes (settings, MSA policy); the samples are stochastic with different random generators, so only agreement in distribution is expected. The mosaic `boltz` is a fork whose version relative to 2.2.1 is not confirmed.
**ESM2.** boileroom: Hugging Face `EsmModel`, `esm2_t33_650M` by default, embeddings = `last_hidden_state` (after the final layer norm), CLS/EOS removed, chains concatenated without a separator and shifted by 512 positions (rotary patch), fp32. Mosaic: `esm2quinox` (post-final-norm hidden state through `apply_trunk`), **no public embedding function** (it has to be written; the mask-token rescale of the pseudo-likelihood code must not be applied), **no multi-chain handling**.
**ESM-C.** boileroom: `esmc_600m` forced by bagel, embeddings from the `esm` SDK (post-norm not confirmed), chains joined with `|`, bf16 compute. Mosaic: `esmc_300m/600m/6B` through `load_esmc` (default `compute_dtype="bfloat16"`, use `float32` for comparisons), post-norm hidden state, **no multi-chain handling**. The **SAE has no mosaic counterpart** (searched), so `SAEnergy` and `ResidueSAEnergy` stay unmapped.
**Consequence for v1:** embedding terms can be matched for States where the embedding oracle sees a **single chain**. For several chains bagel's own chain handling (position offset for ESM2, `|` for ESM-C) has no mosaic counterpart; replicating it in mosaic would need changes inside `esm2quinox`/ESM-C code (open question Q-E).
Numerical tests required before declaring any of these equivalent: Boltz2 pLDDT/PAE agreement in distribution at matched settings, with and without MSA; ESM2/ESM-C per-residue cosine and maximum absolute difference between boileroom and mosaic embeddings for the same monomers (and fp32 vs bf16); `EmbeddingsSimilarityEnergy` values with reference embeddings from the same pipeline.

### 13.4 HX1 (Imperial) from the official RCS documentation
Scheduler **PBS Pro**; GPU queue `a100`; 15 GPU nodes with 4 x **A100 80GB SXM** (same GPU as on Modal), 72 cores and 1 TB RAM per node; limits documented: 72 h walltime, 12 GPUs per user; login host with IPv6 only and no key-based SSH; home quota 1 TB without backup. **Not documented (ask RCS):** Apptainer, outbound internet from compute nodes, NVIDIA driver/CUDA versions, JAX, node-to-node networking and long-running servers, MIG. The policy page says HX1 is intended for multi-node jobs, double-precision GPU jobs, or AI jobs needing more than 48 GB of GPU memory, with other GPU work directed to CX3; whether a co-located bagel oracle plus mosaic scorer (two models on one GPU) fits that policy should be confirmed with RCS. Source: https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/hx1/ and the cluster-specification page.
Implication for the design: since internet access is uncertain, weights and the container image must be **pre-staged** (the HX1 page documents sftp/rsync to the GPFS home), the scorer must start without network access (offline Hugging Face mode, a local cache directory) and JAX must not preallocate the whole GPU (`XLA_PYTHON_CLIENT_MEM_FRACTION`) when sharing it with the torch-based oracle.

### 13.5 Licences (from files read in the clones; not legal advice)
All 12 git-sourced dependencies and mosaic itself carry a licence file: MIT (mosaic, joltz, boltz, boltzgen, joltzgen, jpromera, jopenfold3), Apache-2.0 (protenij, esmjfold2, jopendde, OpenDDE), BSD-3-Clause (jablang); `jproteina-complexa` is multi-licence (code Apache-2.0, weights under the NVIDIA Open Model License, datasets CC BY 4.0). The PyPI direct dependencies are MIT/BSD/Apache-2.0/PSF-style, plus `gemmi` under MPL-2.0 (file-level copyleft); `esm` is MIT as read on PyPI and in its repository, with a request to follow the Biohub Acceptable Use Policy (whether that binds users was not determined). The mosaic package itself ships ProteinMPNN weights (MIT), AbMPNN weights, trigram data and stability data (CC BY 4.0, attribution required). Model weights are not in the image (downloaded at run time); the Hugging Face cards read were MIT for ESMFold2/ESM-C/Boltz-2 (ESM-C weights also carry a tag `license:other` whose meaning was not determined).
No blocker was found for publishing an image with the code. What was **not** audited: the base OS image, CUDA/cuDNN and other NVIDIA wheels, transitive dependencies, and the locked versions.
Proposed way to proceed: (1) resolve and commit a lock/constraints file on Linux; (2) build the image from it; (3) run a licence scanner (for example `pip-licenses`) on the built image; (4) ship a `THIRD_PARTY_LICENSES` directory with every LICENSE/NOTICE file and keep the `*.dist-info/licenses`; (5) keep the CC BY 4.0 attributions of the bundled data; (6) only then choose a registry (Docker Hub like boileroom, or the GitHub container registry); (7) if the image will be redistributed beyond the group, have the institution confirm the Acceptable Use Policy question and the scan result.

### 13.6 Open questions (also asked in the chat)
- **Q-A** Calibration: are the M mutants folded *in addition to* the proposal during the first S steps (unbiased sample, M extra folds per step), or is the correlation measured only on the proposals actually made (free but selected by the predictor)?
- **Q-B** Fallback rule: point estimate below 0.25, or the upper end of the 95% interval below 0.25? (With n pairs the uncertainty is about 1/sqrt(n-3).)
- **Q-C** Boltz2: should the mosaic side use mosaic's cheap defaults (1 recycling step, 25 diffusion steps, empty binder MSA) and rely on the calibration, or match bagel's settings (3 recycling steps, 200 steps)?
- **Q-D** HX1: the answers to the undocumented points (Apptainer, internet, drivers) have to come from RCS or you.
- **Q-E** Embedding terms with several chains per State: restrict v1 to embedding oracles that see a single chain, or invest in replicating boileroom's chain handling in mosaic?
- **Q-F** Image publishing: which registry, and whether the institution needs to approve.


---

## 14. Decisions of the second round (2026-10-21) - these supersede conflicting statements above

| Topic | Decision |
|---|---|
| Screening schedule | The protocol has a parameter `screening_steps`: screen during the first N steps, then switch to plain random proposals. **Default: no limit, screening is used until the end.** (Untested option, kept off by default.) |
| Calibration | On by default with `calibration_mutants = M = 8` and `calibration_steps = S = 5`; `S <= 0` switches it off. During the S steps, **all M scored candidates of each step are also folded by bagel, including the ones that are not proposed**, and the rank correlation between the pseudo-energy change and the bagel energy change is computed over the S x M pairs. If it is **below 0.25** the protocol writes a clear message (also to `Mosaic_pseudo_energy.txt`) and switches to pure random sampling. |
| Selection among candidates | A parameter `selection_temperature` T_sel (in energy units) sets the softmax used to pick the proposed candidate: probability proportional to exp(-pseudo_energy_change / T_sel). **Default T_sel = 0, i.e. the candidate with the lowest pseudo-energy is always proposed (this is what all experiments so far did).** No other value has been tested. |
| Boltz2 | Mosaic side uses mosaic's cheap defaults (1 recycling step, 25 diffusion steps, empty binder MSA); the calibration is the check on the mismatch with bagel's setting (3 recycling steps, 200 steps, MSA for every chain). |
| Embeddings | **No embedding oracles in v1** (no ESM-C, no ESM2); the embedding-based and SAE terms (rows 22-24 of section 3.1) stay unmapped. |
| Cluster | HX1 is accepted as suitable by the user; the undocumented points (Apptainer, internet, drivers) remain to be confirmed. |
| Image | **GitHub Container Registry (`ghcr.io`) and CI, see section 16** (supersedes the Docker Hub discussion of 14.3/14.4); the licence steps of section 13.5 still come first. |

### 14.1 Calibration: how reliable is a correlation measured on 40 pairs?
With S = 5 and M = 8 there are 40 pairs. The uncertainty of a rank correlation measured on n pairs is roughly 1/sqrt(n-3) on the Fisher-z scale (about 0.16 for n = 40; the pairs of one step share the same current sequence and the bagel oracle is noisy, so the real uncertainty is somewhat larger). Probability that the *measured* correlation falls below the threshold 0.25, by true correlation:

| true correlation | 0.0 | 0.1 | 0.25 | 0.4 | 0.5 | 0.6 |
|---|---|---|---|---|---|---|
| n = 40 | 94% | 83% | 50% | 15% | 4% | 0% |
| n = 60 | 97% | 88% | 50% | 10% | 1% | 0% |
| n = 100 | 99% | 94% | 50% | 5% | 0% | 0% |

Reading: a useful predictor (true correlation 0.5 or more; the in-run values in the benchmarks were 0.59-0.65) is almost never switched off wrongly, but one in the zone 0.3-0.4 is switched off in roughly 15% of runs at n = 40, and a useless one (true 0.1) is wrongly kept in about 17% of runs. Rule options: (A) switch off when the measured value is below 0.25 (your rule; default); (B) switch off only if the upper end of the 95% interval is below 0.25 (needs a measured value below about -0.07 at n = 40, so it almost never switches); (C) increase S x M. The message will always print the measured correlation and its 95% interval. Single folds per mutant also lower the measured correlation (oracle noise); a `calibration_folds` parameter (default 1) can average several folds.

### 14.2 What boileroom does with images today (read from its code and workflows)
- Each model family has a Dockerfile in the repository (a shared base image plus one per model); the images contain code and dependencies, **not weights**.
- A GitHub Actions workflow builds and pushes them to Docker Hub under a personal namespace (`docker.io/jakublala`) on each push to `main` (prerelease tags such as `0.4.5-alpha.12` and `sha-<commit>`) and on published releases; tags are CUDA-qualified (`cuda12.6-<tag>`) with an unqualified alias for the default CUDA; a second workflow cleans up old tags and an ARM smoke-test exists. The namespace is configurable (`BOILEROOM_DOCKER_REPOSITORY`) as is the tag (`BOILEROOM_IMAGE_TAG`).
- **Modal** pulls them with `Image.from_registry(<repository>/<image>:<tag>)`; **Apptainer** runs `apptainer pull --force <file>.sif docker://<repository>/<image>:<tag>` and caches the `.sif` under the model/cache directory; weights are downloaded at run time into a Modal volume (`model-weights`) or into that cache directory.
- The repository commits a `uv.lock`. The optimisation "kit" images are not published by anyone: they are built on Modal from their Dockerfile, or pulled from a user-provided registry.
- A practical consequence seen in this session: the pinned boileroom (0.4.5) asks for an image tag that the workflow had not published (only prerelease tags exist), which is why a tag had to be set by hand.
Proposal for the mosaic scorer: the same pattern: a Dockerfile and a committed lock file in the bagel repository, a CI workflow that publishes `<namespace>/bagel-mosaic-scorer:<mosaic-commit>-<lock-hash>`, Modal using `Image.from_registry` (today the service is built from a local checkout, which is less reproducible), Apptainer pulling the same image (on HX1 the `.sif` may need to be built elsewhere and copied, because internet access is not documented), weights downloaded at run time or pre-staged.


### 14.3 Docker Hub: setting up a lab space (from Docker's documentation, fetched 2026-10-21)
- I could not find a Docker Hub organisation or user named `softnanolab` (the Hub API returned "not found"); `jakublala` is a personal account (the boileroom images live there). The lab may use another name, so please check whether one exists.
- **Create an organisation:** sign in at https://app.docker.com/, choose "Create new organization", select a subscription, billing cycle and number of seats, give the organisation a **namespace** (for example `softnanolab`) and a name, then enter billing and payment details. The documentation page I read does **not** say whether a free subscription is available for a new organisation; it says to consult Docker's pricing page, so the cost has to be checked there before creating it.
- **Alternatives if a paid plan is not wanted:** a free personal namespace of one lab member (as boileroom does), or GitHub's container registry (`ghcr.io`) under the lab's GitHub organisation (I did not verify its current terms in this session).
- **Create the repository** (for example `<namespace>/bagel-mosaic-scorer`, public) in the namespace.
- **Credentials for CI:** Docker recommends **organisation access tokens** for automation because they are not tied to a person; their scopes are `Image Pull`, `Image Push` and `Image Delete`, and an organisation token can be restricted to chosen repositories. A personal access token needs the level "Repo Read & Write" to push and **cannot** be restricted to one repository. Store the token as a secret of the GitHub repository (boileroom's workflow uses `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`). The exact login user name to use with an organisation token is not stated on the pages I read; check the token documentation.
- **Pull limits:** per 6 hours, 100 pulls for unauthenticated users per IPv4 address or IPv6 /64 subnet, 200 for a personal account, unlimited ("fair use") for paid plans. Many users behind one address (a cluster login node) share the anonymous limit, so on a cluster the image should be pulled once and the `.sif` file kept (as boileroom's Apptainer backend does).

### 14.4 CI publishing versus building by hand
| | CI (GitHub Actions) | By hand |
|---|---|---|
| Build machine | GitHub's Linux x86_64 runners: the same every time | your own machine; on an Apple-silicon laptop an amd64 CUDA image needs cross-building or emulation (slow, not tested here) |
| Reproducibility and traceability | every image tagged with the mosaic commit and lock hash, with the build log kept | depends on the person; easy to forget or to publish from a dirty tree |
| Checks before publishing | automatic (import smoke test, licence scan, offline start test) | manual, if done at all |
| Secrets | a push token stored in GitHub | the token stays on your machine |
| Limits and effort | a workflow to write and maintain; runner disk and time limits matter for a large CUDA/JAX/PyTorch image (size of ours not known; boileroom's workflow uses long timeouts and one build worker) | nothing to set up; works for the first image |
Suggested path (my judgement): build once by hand on a Linux machine to validate the Dockerfile, then move to CI triggered by a version tag or by a manual dispatch, as boileroom does.

---

## 15. Parameter reference (to be reproduced in the docstring of the protocol class)

The class docstring must describe every parameter with type, default, units, the effect, and when to change it. Planned parameters (names provisional):

| Parameter | Default | Meaning |
|---|---|---|
| `backend` | Modal | where the mosaic scorer runs: `Modal`, a local GPU process/container, or a PBS/SLURM job; carries the connection settings (app name, address, queue options) |
| `models` | by class | mapping from each bagel folding oracle used by a State to a mosaic model spec: model (ESMFold2 variant or Boltz2), recycling steps, diffusion sampling steps. Defaults are mosaic's cheap settings (ESMFold2: 3 loops and 14 steps as in the experiments; Boltz2: 1 recycling step, 25 steps, empty binder MSA). Oracles without a mosaic model are unmapped. |
| `n_candidates` (B) | 8 | number of random single substitutions scored per step; the batch sent to mosaic always has B + 1 sequences (current one first). Larger B finds better proposals but costs more time per step. |
| `selection_temperature` (T_sel) | 0 | softmax temperature, in energy units, for choosing among the B candidates (probability proportional to exp(-change/T_sel)); 0 means always the lowest pseudo-energy. Only 0 has been tested. |
| `screening_steps` | none | if set to N, screening is used for the first N steps and plain random proposals afterwards; default: screening until the end. |
| `calibration_mutants` (M) | 8 | candidates folded by bagel per calibration step (including the ones not proposed). |
| `calibration_steps` (S) | 5 | number of initial steps used for calibration; `S <= 0` switches the calibration off. |
| `calibration_threshold` | 0.25 | minimum rank correlation between pseudo-energy change and bagel energy change; below it the protocol warns and switches to pure random sampling. |
| `calibration_rule` | `'point'` | `'point'`: switch if the measured correlation is below the threshold; `'upper95'`: switch only if the upper end of the 95% confidence interval is below the threshold (stricter about switching, almost never does). The measured value and its interval are always reported. |
| `calibration_folds` | 1 | folds averaged per calibration candidate (the bagel oracle is noisy; more folds raise the measured correlation). |
| `mutation_bias` | no cysteine | amino acids with zero weight are never proposed; the weights are not used as probabilities. |
| `report_name` | `Mosaic_pseudo_energy.txt` | name of the report written into the run's log folder (needs the minimizer to hand its log path to the protocol; a small change in bagel). |
Not parameters, but fixed rules: substitutions only (chain-length changes are refused), one designed chain per State, no terms without a mosaic counterpart in the pseudo-energy.


---

## 16. Registry decision: GitHub Container Registry, published by CI (2026-10-21)

Decision: the scorer image is published to **`ghcr.io`** by **GitHub Actions**; there is no hand-built publishing step. What GitHub's documentation (read 2026-10-21) says that matters here:
- **Name:** `ghcr.io/<namespace>/<image>:<tag>`, the namespace being the personal account or organisation that owns the repository (for the bagel repository this would presumably be `softnanolab`; the organisation's package settings, which can restrict who may publish packages or make them public, were not checked).
- **Push from CI without a stored secret:** the workflow's built-in `GITHUB_TOKEN` can publish packages of its own repository when the job has the permission `packages: write`; the documented example also uses `contents: read`, `attestations: write` and `id-token: write`, the actions `docker/login-action`, `docker/metadata-action`, `docker/build-push-action` and an attestation step. Pushing from a command line needs a *classic* personal access token with `write:packages`.
- **Do not push by hand first:** the documentation warns that a package that was pushed to the namespace without being connected to the repository may refuse `GITHUB_TOKEN` pushes. Publishing from the workflow links the package to the repository automatically; a label `org.opencontainers.image.source=<repository URL>` in the Dockerfile does the same.
- **Visibility:** a newly published package is **private by default**. It has to be switched to **public once** in the package settings so that Modal, Apptainer and the cluster can pull it anonymously (the page I read does not describe the steps). Public images can be accessed anonymously.
- **Cost:** GitHub Packages is free for public packages; for the container registry the page states that storage and bandwidth are "currently free" and that any change would be announced at least one month in advance.
- **Pull limits:** the page does not mention any; not known.
Consequences for the design: the image must contain no secrets and no weights (consistent with section 13.5); Modal pulls it with `Image.from_registry("ghcr.io/...")` and Apptainer with `apptainer pull docker://ghcr.io/...` **[both not tested for ghcr.io in this session]**; tags: `<mosaic-commit>-<lock-hash>` (what the scorer reports at start, guardrail G12) plus `sha-<bagel commit>`; triggers: a version tag and a manual dispatch (boileroom also builds on pushes to `main` and on releases); the workflow runs the checks of section 12 (import smoke test, licence scan) before it pushes. Open points: whether the organisation allows public packages, and the size of the image against the runners' disk limit **[not verified]**.
