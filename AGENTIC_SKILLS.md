# Agentic skills

BAGEL ships a set of **agent skills** under [`.claude/skills/`](.claude/skills/). Each
skill is a self-contained, plain-Markdown playbook (plus reference files, scripts, and
ready-to-use templates) that teaches a coding agent how to perform a specific BAGEL
workflow end to end.

A coding agent running in this repo — [Claude Code](https://claude.com/claude-code)
discovers them automatically — reads a skill's `SKILL.md`, consults its `references/` as
needed, and follows the workflow. The skills are model-agnostic: they are just Markdown and
helper files, so they work with any capable coding agent (Cursor, Aider, a Claude API
harness, or your own), not only Claude. To make a skill available in every project, copy it
into your personal skills directory:

```bash
cp -r .claude/skills/<skill-name> ~/.claude/skills/
```

This file documents every skill in `.claude/skills/`. It is intentionally kept separate
from the main [`README.md`](README.md) (which covers the library itself); some overlap with
the README's script-builder section is expected.

---

## `bagel-script-builder` — turn a design goal into runnable BAGEL scripts

**Location:** [`.claude/skills/bagel-script-builder/`](.claude/skills/bagel-script-builder/)

Turns a plain-language protein-design goal — *"design a 30-residue binder to CD20 and sweep
8 seeds on SLURM"* — into runnable, reviewable BAGEL scripts (the `biobagel` /
`import bagel as bg` framework). Use it whenever someone describes a protein-design or
sequence-optimization problem in words and wants BAGEL code: binders, de novo scaffolds,
sequence optimization, enzyme miniaturization, negative/selective design, or any custom
energy-minimization over a System of States.

It is **not** a one-shot code generator. It runs a guided, question-driven workflow:

0. **Understand & draft** — read the goal and write a first-pass draft script.
1. **Interview to close gaps** — walk through the draft and ask about everything the
   description left unspecified, until every State, Chain, and EnergyTerm (and each term's
   inputs) is explicitly pinned down. Silently guessing the unspecified parts of a design is
   the main failure mode this skill exists to prevent.
2. **Write the design script** — verbose, heavily commented, saved to disk for review.
3. **Smoke test** — generate a minimal test script and run it on Modal when credentials are
   available.
4. **Execution harness** — optionally generate the harness to run a **SWEEP** that repeats a
   minimization while varying some "difference" (protected residues, step counts, seeds,
   weights, …) in parallel on a SLURM/PBS cluster or on Modal, or serially in the
   background.

It knows the BAGEL building blocks — States/Chains/EnergyTerms, oracles (ESMFold,
ESM2/ESM-C/ESM3, Chai1, Boltz2), and minimizers (Monte Carlo, Simulated Annealing,
Simulated Tempering, custom temperature schedules, chained optimizers). Trigger it even when
the user doesn't say "script" — e.g. *"design a binder to CD20"*, *"optimize this enzyme
conserving the catalytic triad"*, *"sweep over 10 seeds on SLURM"*.

**How to invoke** — just describe what you want; Claude Code runs the interview and writes
the scripts into `bagel_designs/<name>/`. With any other agent, point it at the skill:

> "Read `.claude/skills/bagel-script-builder/SKILL.md` and its `references/`, then follow
> that workflow to build a BAGEL script for `<goal>`."

**Structure**

- `SKILL.md` — the self-contained workflow playbook.
- `references/` — `api-reference.md`, `patterns.md`, `clarification-checklist.md`,
  `execution-harness.md`.
- `assets/` — launcher templates: `slurm_job.sh`, `pbs_job.sh`, `submit_cluster.py`,
  `sweep_config.py`, `sweep_runner.py`.

Every generated file begins with a comment noting it was produced with AI assistance, so it
is clear the code should be reviewed before use.

---

## `sae-feature-annotations` — turn ESM-C SAE feature indices into biology

**Location:** [`.claude/skills/sae-feature-annotations/`](.claude/skills/sae-feature-annotations/)

Looks up what a Biohub ESM-C **sparse-autoencoder (SAE) feature** *means* — its label,
description, top-activating proteins, decoder neighbours, and activation statistics — by
querying the Biohub feature-annotation API live. Use it whenever someone has SAE feature
indices (e.g. from `boileroom`'s `SAE` model / `pooled_features`, or from BAGEL's `SAEnergy`
/ `ResidueSAEnergy` terms) and wants to interpret them: *"what is SAE feature 12345"*,
*"which features fired on my protein and what do they correspond to"*, *"annotate these
feature indices"*, *"what proteins most activate this feature"*, *"find the zinc-binding SAE
feature"*, *"decoder nearest neighbours of feature N"*.

**Important caveat the skill leads with:** these annotations are **only valid for the SAE
`ESMC-6B-sae-layer60-k64-codebook16384`** — ESM-C 6B, transformer layer 60, TopK k=64,
codebook of `2**14 = 16384` features. That is the default Forge SAE that `boileroom`'s `SAE`
model uses (`feature_source="forge"`). A `feature_index` (0…16383) means something
completely different under any other model, layer, k, or codebook, so if the features came
from a local 300M/600M SAE or a different layer, the labels do not apply and the skill says
so and stops. (This is the same identity that BAGEL's SAE energy terms require.)

**How it works** — there is no bundled offline table; it queries two **public read**
endpoints of the Biohub API (base URL `https://biohub.ai`):

1. `GET /esm/protein/api/v1alpha1/features` — the whole catalogue (`feature_index`, `label`,
   short `description` for all 16384 features) in one call; backs `list` and `search`.
2. The per-feature detail endpoint — rich metadata for a single feature: longform
   description, activation statistics, top-activating proteins, and decoder nearest
   neighbours.

An API key is optional (`ESM_API_KEY` / `FORGE_TOKEN`, the same credential `boileroom`'s
Forge backend uses); the annotation reads work with an anonymous token, and a key is only
needed if a deployment enforces auth. Producing the SAE features themselves is a separate
tool — that is `boileroom`'s `SAE` model, not this skill.

**How to invoke** — ask any of the questions above; Claude Code runs the lookup. With
another agent:

> "Read `.claude/skills/sae-feature-annotations/SKILL.md` and its `references/api.md`, then
> annotate these ESM-C SAE feature indices: `<indices>`."

**Structure**

- `SKILL.md` — the workflow and the model-identity guardrail.
- `references/api.md` — the Biohub annotation API reference.
- `scripts/sae_features.py` — helper script for querying the endpoints.
- `evals/evals.json` — evaluation cases for the skill.
