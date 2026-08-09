# Phase-1 clarification checklist

The goal of Phase 1 is that the user has **explicitly defined every piece of the System** — no
silent defaults on anything design-critical. Walk the draft top to bottom and, for each item
below, either confirm it's already specified in the request or ask. Batch related questions;
don't interrogate one field at a time. Present your current assumption alongside each question so
the user can just confirm ("I'm assuming X — right?").

## System / States
- How many **states**, and what does each represent physically? (bound complex, apo, off-target…)
- Is any state one the design should be pushed *away from*? → that state's interface terms get
  **negative weights** (negative/selective design).
- Which **chains** belong to each state? (a mutable design chain is often shared across states.)

## Chains
- For every chain: its **sequence** (or, if de novo, its **length** and how it's seeded —
  random, from a PDB, from a given sequence).
- Its **`chain_ID`** (must be under 5 characters; pick something readable like `TGT`, `BIND`).
- Which chains are **targets** (usually fully immutable) vs the **design** chain(s).

## Mutability / protected residues  (never guess these)
- Exactly which residues are **mutable** vs **protected/immutable**.
- Active sites, **catalytic residues**, structural **cysteines/disulfides**, epitope residues on
  a fixed target, and any experimentally required positions → immutable.
- **Numbering check:** when residues are named by number, confirm the numbering convention and
  the actual amino-acid identity. Catalytic labels like "His57/Asp102/Ser195" are frequently a
  domain convention (e.g. chymotrypsin numbering) that is **not** the raw 1-based index into the
  supplied sequence. Resolve this now, and have the generated script assert the residue
  identities at runtime so a wrong index fails loudly.

## EnergyTerms — the heart of the design
For **each state**, pin down the list of terms. For **each term**:
- **Which term** and why (what property it rewards/penalizes). See `api-reference.md` for the menu.
- **Weight** (relative; negative to discourage). Confirm the intended sign, especially in
  negative-design states.
- **Residue group(s)** it applies to. Single-group terms (`PLDDTEnergy`, `HydrophobicEnergy`)
  take a flat `residues=[...]`; **interface terms** (`PAEEnergy`, `SeparationEnergy`, `LISEnergy`)
  take a list of **two** groups `residues=[group_a, group_b]`. Confirm exactly which residues form
  each group (hotspot/epitope/catalytic set/whole chain).
- **Oracle** that evaluates it — folding (`ESMFold`, …) for structure/confidence/geometry terms;
  embedding (`ESM2`, `ESMC`, `ESM3`) for `EmbeddingsSimilarityEnergy`/`ChemicalPotentialEnergy`.
  One oracle instance can back many terms; confirm the backend (`modal`/`apptainer`).
- **Inheritance** — only if the user wants non-default behavior. `inheritable` controls whether a
  term's cached evaluation carries across mutation steps; leave at the class default unless the
  user has a specific reason, and if they do, capture which terms and why.
- For terms with extra inputs, confirm them: `HydrophobicEnergy(mode=...)`,
  `SecondaryStructureEnergy(structure_type=..., groups=...)`, `RingSymmetryEnergy(groups=...)`,
  `TemplateMatchEnergy(template_atoms=..., backbone_only=...)`, `ChemicalPotentialEnergy`
  length-control parameters, `EmbeddingsSimilarityEnergy(reference_embeddings=...)`.

## Oracle / backend
- Multi-chain folding needs `config={'glycine_linker': ..., 'position_ids_skip': ...}` — confirm
  the state is multi-chain and set it.
- Backend: `modal` (remote, needs `modal token new`) vs `apptainer` (local GPU). Which do they have?

## Optimizer
- **Minimizer**: `MonteCarloMinimizer` (fixed T or custom per-step schedule), `SimulatedAnnealing`
  (linear ramp), or `SimulatedTempering` (explore/refine cycles). Which fits the goal?
- **Temperatures and step counts** — production scale. If a **custom schedule** is wanted, capture
  its shape (an array). If **chained phases** are wanted (e.g. broad MC → annealing refinement),
  capture the sequence and each phase's parameters.
- **Mutation protocol**: `Canonical` (substitutions, fixed length) vs `GrandCanonical`
  (insert/delete/substitute, variable length — pair with `ChemicalPotentialEnergy`). How many
  mutations per step?
- Any **callbacks** beyond default logging (structure dumps via `FoldingLogger`, `EarlyStopping`,
  `WandBLogger`)?

## Run scope (feeds Phase 4–5)
- Single run, or a **sweep** over some difference?
- If a sweep: the **axis/axes** to vary, the **list of values**, a **run-naming scheme**, and the
  **execution mode** (SLURM / PBS / Modal-parallel / serial-background).

When every item above is either specified by the user or an explicitly-confirmed default, Phase 1
is complete.
