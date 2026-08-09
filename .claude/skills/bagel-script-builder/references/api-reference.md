# BAGEL API reference

Exact signatures and conventions for building scripts. Everything is reached through
`import bagel as bg`. When in doubt about a current signature, read the source file named in
each section under `src/bagel/` — this framework changes and the source is authoritative.

## Table of contents
1. Data objects (Residue, Chain, State, System) — `src/bagel/chain.py`, `state.py`, `system.py`
2. Oracles — `src/bagel/oracles/`
3. Energy terms — `src/bagel/energies.py`
4. Minimizers — `src/bagel/minimizer.py`
5. Mutation protocols — `src/bagel/mutation.py`
6. Callbacks — `src/bagel/callbacks.py`
7. Constants & utils — `src/bagel/constants.py`, `src/bagel/utils.py`

---

## 1. Data objects (build the system bottom-up)

```python
bg.Residue(name: str, chain_ID: str, index: int, mutable: bool = True)
```
- `name`: 1-letter amino acid code. `chain_ID`: string under 5 chars. `index`: 0-based.
- `mutable`: whether the minimizer may change this residue.

```python
bg.Chain(residues: list[bg.Residue])
```
- One monomeric chain; all residues must share the same `chain_ID`.
- Attributes/methods: `.sequence`, `.mutable_residues`, `bg.Chain.from_pdb(file_path, chain_id)`.

```python
bg.State(name: str, chains: list[bg.Chain], energy_terms: list[EnergyTerm])
```
- A complex (one or more chains) plus the energy terms scored on it. Computes `.energy`.

```python
bg.System(states: list[bg.State], name: str | None = None)
```
- Top-level object passed to the minimizer. Total energy = sum over states.
- Multiple states enable multi-objective / negative / selective design. A mutable chain
  can be shared across states (same residue objects) so one design is scored in several
  contexts at once.
- `System.dump_config(path)` serializes the system; the minimizer calls this for you.

**Idiom for building residues + mutability in parallel:**
```python
residues = [
    bg.Residue(name=aa, chain_ID='CD20', index=i, mutable=mut)
    for i, (aa, mut) in enumerate(zip(sequence, mutability))
]
```
Slice residue **groups** (hotspots, epitopes, catalytic sites) from this list to hand to
energy terms, e.g. `hotspot = [residues[i] for i in range(10, 20)]`.

---

## 2. Oracles (`bg.oracles.*`)

Construct with `backend='modal'` (remote via Modal, default) or `'apptainer'` /
`'apptainer:<image-tag>'` (local GPU), plus an optional `config` dict. One instance can back
many energy terms.

### Folding oracles — predict structure, expose pLDDT / PAE / pTM
- `bg.oracles.ESMFold(backend=..., config={...})` — default single-sequence folder, used in
  most scripts. For multi-chain states set
  `config={'glycine_linker': 'G'*25, 'position_ids_skip': 1024}` (skip is 512 or 1024).
- `bg.oracles.ESMFold2(...)` — newer ESMFold.
- `bg.oracles.Chai1(...)`, `bg.oracles.Boltz2(...)` — higher-accuracy complex folders.
- Method: `.fold(...)`; generic entry `.predict(...)`.

### Embedding oracles — per-residue embeddings / tracks
- `bg.oracles.ESM2(backend=..., config={'model_name': 'esm2_t33_650M_UR50D'})`.
- `bg.oracles.ESMC(...)` — ESM Cambrian.
- `bg.oracles.ESM3(...)` — also decodes tracks (SASA, SS8, function/annotation logits).
- Method: `.embed(chains=[chain])` returns a result with `.embeddings` (index it with a
  boolean mask to grab conserved-residue reference embeddings). Generic entry `.predict(...)`.

---

## 3. Energy terms (`bg.energies.*`)

All subclass `EnergyTerm(name, oracle, inheritable, weight=1.0)`. `weight` is relative;
**negative weight discourages** the quantity (used for negative design). `name` is optional
but recommended — it labels the term in per-term logs.

**Residue-group convention (the #1 source of bugs):**
- Single-group terms take a flat `residues: list[bg.Residue]`.
- Interface terms take a **list of two groups**: `residues=[group_a, group_b]`.

### Confidence / fold quality (folding oracle)
- `PTMEnergy(oracle, weight=1.0, name=None)` — global pTM.
- `OverallPLDDTEnergy(oracle, weight=1.0, name=None)` — whole-structure pLDDT.
- `PLDDTEnergy(oracle, residues, inheritable=True, weight=1.0, name=None)` — pLDDT of a group.

### Interface / binding (folding oracle) — take two groups
- `PAEEnergy(oracle, residues=[group_a, group_b], cross_term_only=True, weight=1.0, name=None)`
  — inter-group predicted aligned error (interface confidence).
- `SeparationEnergy(oracle, residues=(group_a, group_b), function=None, weight=1.0, name=None)`
  — distance between the two groups' centroids.
- `LISEnergy(oracle, residues, pae_cutoff=12.0, intensive=True, ...)` — local interaction score.
- `FlexEvoBindEnergy(oracle, residues, plddt_weighted, symmetrized, ...)`.

### Developability (folding oracle)
- `HydrophobicEnergy(oracle, residues=None, mode='all'|'surface'|'core', weight=1.0, name=None)`.
- `HydropathyEnergy(...)`.
- `SurfaceAreaEnergy(oracle, residues, probe_radius, max_sasa, ...)`.

### Shape / topology (folding oracle)
- `SecondaryStructureEnergy(oracle, groups, structure_type='beta-sheet', weight=1.0, name=None)`
  — note `groups=` (list of residue groups) and `structure_type=`. Confirm allowed
  `structure_type` values in `src/bagel/energies.py`.
- `RingSymmetryEnergy(oracle, groups, direct_neighbours_only=False, weight=1.0, name=None)`
  — note `groups=` (list of the symmetric residue groups).
- `TemplateMatchEnergy(oracle, template_atoms, residues, backbone_only, distogram_separation, ...)`
  — RMSD/distogram match to a PDB motif (`template_atoms` is a biotite `AtomArray`).
- `GlobularEnergy(...)`.

### Similarity / length (embedding oracle)
- `EmbeddingsSimilarityEnergy(oracle, residues, reference_embeddings, weight=1.0, name=None)`
  — keep the group's embeddings near a captured reference (conservation / mimicry).
- `ChemicalPotentialEnergy(oracle, power, target_size, chemical_potential, weight=1.0, name=None)`
  — controls sequence length; pair with `GrandCanonical`.

> Signatures above reflect the current source but arguments drift — open
> `src/bagel/energies.py` and grep for the class to confirm before relying on an argument.

---

## 4. Minimizers (`bg.minimizer.*`)

Common constructor args: `mutator`, `acceptance_criterion='metropolis'`, `experiment_name`,
`preserve_best_system_every_n_steps`, `log_path`, `callbacks`. Entry point:
`.minimize_system(system=...) -> System` (returns the best system found).

```python
bg.minimizer.MonteCarloMinimizer(
    mutator, temperature, n_steps,
    acceptance_criterion='metropolis', experiment_name=None,
    preserve_best_system_every_n_steps=None, log_path=None, callbacks=None,
)
```
- `temperature`: a `float` (constant) **or** a `list`/`np.ndarray` of length exactly
  `n_steps` — this is how you express an arbitrary custom schedule.

```python
bg.minimizer.SimulatedAnnealing(
    mutator, initial_temperature, final_temperature, n_steps, ...)
```
- Linear temperature ramp from initial to final over `n_steps` (internally `np.linspace`).

```python
bg.minimizer.SimulatedTempering(
    mutator, high_temperature, low_temperature, n_steps_high, n_steps_low, n_cycles, ...)
```
- Each cycle = `n_steps_low` at low T then `n_steps_high` at high T; repeated `n_cycles`
  times. Total steps = `(n_steps_low + n_steps_high) * n_cycles`. Workhorse for binders.

**Chaining:** because `minimize_system` returns a `System` and accepts one, run phases in
sequence, feeding `best_system` forward. Use distinct `experiment_name`s to keep logs apart.

**`preserve_best_system_every_n_steps`:** periodically resets the current system back to the
best-so-far (restart-from-best). Binder scripts often set it to one full tempering cycle
(`n_steps_low + n_steps_high`).

---

## 5. Mutation protocols (`bg.mutation.*`)

```python
bg.mutation.Canonical(n_mutations=1, mutation_bias=mutation_bias_no_cystein,
                      exclude_self=True)
```
- Substitutions only; **fixed length**. `n_mutations` mutations per step.
- `exclude_self=False` allows a residue to "mutate" to itself (some legacy scripts use this).

```python
bg.mutation.GrandCanonical(n_mutations=1, mutation_bias=mutation_bias_no_cystein,
                          move_probabilities={'substitution': 0.5, 'addition': 0.25,
                                              'removal': 0.25},
                          exclude_self=True)
```
- Insertions/deletions/substitutions; **variable length**. Pair with `ChemicalPotentialEnergy`
  to keep length controlled.

Both default to `mutation_bias_no_cystein`, so **cysteine is excluded** from proposals.

---

## 6. Callbacks (`bg.callbacks.*`)

Prefer these over the deprecated `log_frequency` argument to control logging cadence.
- `bg.callbacks.DefaultLogger(log_interval)` — CSV/step logging.
- `bg.callbacks.FoldingLogger(folding_oracle, log_interval)` — dumps predicted structures.
- `bg.callbacks.EarlyStopping(monitor, patience, min_delta=0.0, mode='min')`.
- `bg.callbacks.WandBLogger(project, config=None)`.

Typical: `callbacks=[bg.callbacks.DefaultLogger(log_interval=1),
bg.callbacks.FoldingLogger(folding_oracle=esmfold, log_interval=50)]`.

---

## 7. Constants & utilities

- `bg.constants.aa_dict` — 1-letter → 3-letter map of the 20 amino acids.
- `bg.constants.aminoacids_letters`, `bg.constants.mutation_bias`,
  `bg.constants.mutation_bias_no_cystein`.
- `bagel.utils`: `get_sequence_from_pdb_id`, `get_reconciled_sequence`,
  `get_atomarray_in_residue_range`, `sequence_from_atomarray`.
- `bg.oracles.folding.utils.pdb_file_to_atomarray` — load a fetched PDB into an `AtomArray`
  (used with `biotite.database.rcsb.fetch` for templates / catalytic-site atoms).

---

## 8. Gotchas that break scripts

These come from the real API and cost debugging time if missed:

- **Multi-chain folding needs linker config.** When an `ESMFold` state has more than one chain,
  pass `config={'glycine_linker': 'G'*25, 'position_ids_skip': 1024}` (512 or 1024 are the
  in-repo values). Without it the chains fold as one fused sequence.
- **Interface terms take a list of two groups**: `PAEEnergy(residues=[group_a, group_b])`, same
  for `SeparationEnergy` / `LISEnergy`. Single-group terms (`PLDDTEnergy`, `HydrophobicEnergy`)
  take a flat `residues=[...]`. Mixing these up is the most common error.
- **Negative design = negative weights** on the off-target state's interface terms
  (`weight=-5.0`), with the mutable binder chain shared across both states.
- **`RingSymmetryEnergy` / `SecondaryStructureEnergy` use `groups=`** (a list of residue groups),
  not `residues=`; `SecondaryStructureEnergy` uses `structure_type='beta-sheet'` (confirm allowed
  values in `src/bagel/energies.py`).
- **Residue basics:** `index` is 0-based; `chain_ID` must be under 5 characters; `name` is a
  1-letter amino acid code.
- **Residue-numbering conventions are a landmine.** Catalytic/active-site labels such as
  "His57 / Asp102 / Ser195" are usually a *domain convention* (e.g. chymotrypsin numbering), NOT
  the raw 1-based index into the given sequence — the same triad can sit at completely different
  linear positions. When residues are named by number, confirm the amino-acid identity and have
  the script **assert the identity at runtime** (`assert residue.name == 'H'`) so a wrong index
  fails loudly instead of silently freezing the wrong residues.
- **`GrandCanonical` without a `ChemicalPotentialEnergy`** lets length drift unchecked — add the
  term (small weight, e.g. `1e-3`) to keep size sane.
- **Backend:** `backend='modal'` runs oracles remotely (needs `modal token new`); `'apptainer'`
  (or `'apptainer:<tag>'`) runs locally on GPU. Expose it as a `main()` arg.
- **Parallel Modal runs collide on a fixed app name.** The Modal backend (`boileroom`) hard-codes
  `modal.App("boileroom")` — one app name shared by every `backend='modal'` process, with no
  override. Running several concurrently conflicts (and races to create the shared `model-weights`
  Volume on first cold-start). For parallel sweeps, give each run its own Modal environment via
  `MODAL_ENVIRONMENT` and warm up one run before fanning out — see `execution-harness.md`. A single
  run is unaffected; so is serial/background execution.
- **`log_frequency` is deprecated** and warns — control cadence via callbacks instead:
  `callbacks=[bg.callbacks.DefaultLogger(log_interval=1), bg.callbacks.FoldingLogger(folding_oracle=esmfold, log_interval=50)]`.
- **Python 3.12 only.** Run scripts with `uv run python scripts/....py` in this repo.

## 9. Smoke test

A smoke test runs the whole pipeline for a trivial number of steps to catch import/residue/oracle
errors before a long run. Use minimal counts and a throwaway `log_path`:
- Tempering: `--n_cycles=1 --n_steps_low=1 --n_steps_high=1`.
- Plain MC / annealing: `--n_steps=1`.
Check Modal auth first (`~/.modal.toml` present, or `MODAL_TOKEN_ID` set). Running it uses a
little Modal credit; report the exact error and likely cause on failure.
