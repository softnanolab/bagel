---
name: bagel-script-builder
description: >-
  Interactively turn a natural-language protein-design goal into runnable, reviewable BAGEL
  scripts (the `biobagel` / `import bagel as bg` framework). Use this whenever the user
  describes a protein-design or sequence-optimization problem in plain language and wants
  BAGEL code — binders, de novo scaffolds, sequence optimization, enzyme miniaturization,
  negative/selective design, or any custom energy-minimization over a System of States. Also
  use it when the user wants to "build a BAGEL system", define States/Chains/EnergyTerms, pick
  oracles (ESMFold, ESM2/ESM-C/ESM3, Chai1, Boltz2) or a minimizer (Monte Carlo, Simulated
  Annealing, Simulated Tempering, custom temperature schedules, chained optimizers), smoke-test
  a script on Modal, or run a SWEEP that repeats a minimization while varying some "difference"
  (protected residues, step counts, seeds, weights, …) in parallel on a SLURM/PBS cluster or
  Modal, or serially in the background. This skill is a guided, question-driven workflow: it
  drafts, then interviews the user to pin down every undefined piece before finalizing, writes
  verbose well-explained scripts to disk for review, and can generate the execution harness.
  Trigger it even when the user doesn't say "script" — e.g. "design a binder to CD20",
  "optimize this enzyme conserving the catalytic triad", "sweep over 10 seeds on SLURM".
---

# Building BAGEL protein-design scripts, interactively

BAGEL (`pip` name `biobagel`, imported as `import bagel as bg`) frames protein design as
**minimizing an energy landscape**: you describe what a good design looks like as a sum of
weighted energy terms, then a Monte-Carlo minimizer mutates the mutable residues to lower that
energy. The scripts *are* the documentation — there is no config/YAML layer.

This skill is **not** a one-shot code generator. It runs an interactive workflow so the user
ends up with correct, fully-specified, well-explained, reviewable scripts. Work through the
phases below in order. Do not skip the interview: silently guessing the parts of a design the
user left unspecified is the main failure mode this skill exists to prevent.

## The workflow at a glance

0. **Understand & draft** — read the natural-language goal, write a first-pass draft script.
1. **Interview to close gaps** — walk the user through the draft and ask about everything the
   description left unclear, until every State, Chain, and EnergyTerm (and each term's inputs)
   is explicitly pinned down.
2. **Write the design script** — verbose, heavily commented, saved to disk for review.
3. **Smoke test** — generate a minimal test script; run it on Modal if credentials are
   available, and report exactly what worked or failed.
4. **Single run or sweep?** — ask whether this is one run or a repeat over some "difference".
5. **Execution harness** — if a sweep, generate the launcher (SLURM / PBS / Modal-parallel /
   serial-background), one output folder per run, runnable via a single command.

Always **save every script to disk** (never only paste into chat) so the user can review, edit,
and re-run them. Default to a `bagel_designs/<short-project-name>/` directory in the working
directory unless the user names a location; put the design script, smoke test, and any sweep
harness there together.

**Every file this skill generates must begin with an agent-disclaimer comment**, so it is always
clear the file was written with AI assistance and should be reviewed before use. Use this block
verbatim at the very top of every `.py` and `.sh` file you create (adjust the comment character
for shell scripts; fill in today's date if you know it):

```
# -----------------------------------------------------------------------------
# Generated with the assistance of an AI agent (Claude, via the
# `bagel-script-builder` skill). Review before running — you are responsible for
# its correctness. Generated: <YYYY-MM-DD>.
# -----------------------------------------------------------------------------
```

## Guiding principle — correct but simple

The scripts this skill produces are meant to be read, trusted, and edited by a scientist, not
admired for their engineering. Optimize for a human reading the file top to bottom: **correct
first, then as simple and readable as possible.**

- Write the design as a straight-line `main()` that reads like the protocol it encodes. Don't
  split it into a web of micro-functions unless a helper genuinely earns its name by removing
  real duplication or clarifying a fiddly step.
- Build only what was asked. No speculative features, no abstractions for single-use code, no
  configurability nobody requested. Expose as CLI flags only the quantities this task actually
  varies (the backend, the swept axis, the few step counts/weights the user will tune) — not
  every conceivable knob.
- Don't guard against impossible states. Validate the inputs that a user could realistically get
  wrong (e.g. a protected-residue identity), and let genuinely-impossible cases just not happen.
- If it could be half the length, rewrite it. A 50-line script that a reviewer understands in one
  pass beats a 200-line one bristling with options.

"Verbose" here means **verbose explanation, not verbose machinery**: rich comments and docstrings
explaining *why* each energy term and optimizer choice serves the goal, wrapped around simple,
direct code. The two go together — plain code with a clear narrative is exactly the target.

---

## Phase 0 — Understand and draft

Read the user's description and map it onto BAGEL's building blocks (see
`references/api-reference.md` for exact signatures, `references/patterns.md` for worked scripts
of each common goal — read these before drafting so names and arguments are exact). Identify the
closest pattern and write a **draft** script. The draft is a starting point for discussion, not
the final artifact — it's fine for it to contain clearly-marked assumptions and `TODO`s where
the description was silent. Prefer to imitate the repo's own scripts in `scripts/` (especially
`scripts/car/lnk_cd20_monomer.py`); when unsure whether a class/argument still exists, read the
relevant file under `src/bagel/` rather than guessing.

## Phase 1 — Interview to close gaps (do not skip)

Walk the user through the draft section by section, and for anything the description did not
make explicit, **ask** rather than assume. The user should end up having explicitly defined
every piece of the System. Use `references/clarification-checklist.md` as the full checklist;
the essentials that must be pinned down before finalizing:

- **States** — how many, and what each represents. Multiple states mean multi-objective or
  negative/selective design; confirm whether any state should be *discouraged* (negative weights).
- **Chains** — every chain's sequence (or length, if de novo), its `chain_ID`, and which chains
  appear in which states.
- **Mutability** — exactly which residues are mutable vs **protected/immutable**. Never guess
  this; protected residues (active sites, catalytic triads, structural cysteines, epitopes on a
  fixed target) are design-critical.
- **EnergyTerms** — for each state, which terms, their **weights**, and for each term its inputs:
  the **residue group(s)** it applies to (and the two-group `[a, b]` convention for interface
  terms), the **oracle** that evaluates it, and any non-default **inheritance** (`inheritable`).
- **Optimizer** — minimizer type, temperatures/schedule, step counts, and mutation protocol
  (`Canonical` fixed-length vs `GrandCanonical` variable-length). If the user wants a custom
  temperature schedule or chained phases (e.g. MC then annealing), capture the shape.

Surface every assumption you are carrying explicitly ("I assumed the target is fully immutable
and the epitope is residues 30–45 — correct?"). Ask focused questions; batch related ones.
Residue-numbering is a classic trap: **catalytic/active-site labels are often a domain
convention (e.g. chymotrypsin numbering) that does not equal the raw sequence index** — when the
user names residues by number, confirm the identity (the actual amino acid) and have the script
verify it at runtime rather than trusting the index. Only proceed to Phase 2 once the pieces are
defined or the user explicitly says to use your proposed defaults.

## Phase 2 — Write the design script (verbose, saved)

Write the final design script to disk. It must be **runnable**, follow the repo idioms, and be
**verbose**: every State, Chain, and EnergyTerm gets a comment explaining what it represents and
*why* it serves the design goal, not just what the line does. Concretely:

- A `main(...)` function wrapped in `fire.Fire(main)`, exposing as CLI flags only the quantities
  this task will actually vary (backend, the swept axis, the step counts/weights the user tunes) —
  not every possible knob (see the correct-but-simple principle above).
- **Named** energy terms (`name='epitope_binder'`) so per-term energies are logged separately.
- Target sequences as clearly-labeled constants near the top.
- Logging via callbacks (`DefaultLogger`, `FoldingLogger`), not the deprecated `log_frequency`.
- A module/function docstring restating the design goal and the role of each energy term.
- The agent-disclaimer comment block (see above) as the very first lines of the file.

Tell the user the saved path and give a one-line summary of each energy term so they can review.

## Phase 3 — Smoke test

A smoke test runs the *whole pipeline* for a trivial number of steps to catch import errors,
bad residue groups, oracle/backend problems, and shape mismatches — before committing to a long
run. Generate a small `smoke_test.py` next to the design script that calls the design's `main`
with minimal counts and a throwaway `log_path` (for tempering: `n_cycles=1, n_steps_low=1,
n_steps_high=1`; for plain MC: `n_steps=1`), printing a clear `SMOKE TEST PASSED` / `FAILED`.

Running the oracle needs a Modal backend. Check whether Modal is authenticated:
```bash
python -c "import modal, pathlib, os; print('AUTH' if (pathlib.Path.home()/'.modal.toml').exists() or os.getenv('MODAL_TOKEN_ID') else 'NO_AUTH')"
```
- If authenticated (or the user provides credentials), **run the smoke test** — the user has
  asked for this, so it's authorized; give a one-line heads-up that it will use a little Modal
  credit, then run it and report precisely: passed, or the exact error and the likely cause
  (bad residue index, missing linker config for a multi-chain state, auth/credit, etc.).
- If not authenticated, don't run it. Show the user the command and how to authenticate
  (`modal token new`, or set `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`), and offer to run it once
  they have.

Never enter or store the user's Modal credentials yourself — direct them to `modal token new`.

## Phase 4 — Single run or sweep?

Ask explicitly: **is this a single run, or should it repeat the minimization while varying some
"difference" between runs?** The "difference" is any axis the user wants to sweep — a different
set of protected residues, a different `n_steps`, different weights, or the *same* script with a
different random seed. Have the user specify:
- the **axis/axes** to vary and the **list of values** (e.g. seeds `[0,1,2,3,4]`; or three
  protected-residue sets; or `n_steps ∈ {1000, 5000, 20000}`), and
- a short **run-naming scheme** so each run is identifiable.

If it's a single run, you're done after Phase 3 (offer to kick off the full run). If it's a
sweep, go to Phase 5.

## Phase 5 — Execution harness for a sweep

Every run must write to its **own folder** so runs never overwrite each other — pass a distinct
`log_path` (and/or `experiment_name`) per run; `runs/<run_name>/` is the default convention. The
user must be able to launch everything with **one command** (e.g. `python send_all_scripts.py`),
CLI-style if parameters are needed.

Ask how the runs should execute, then generate the matching harness (full templates and guidance
in `references/execution-harness.md`; ready-to-fill templates in `assets/`):

- **Parallel on a cluster** — ask **SLURM or PBS**. Generate a submitter that writes one job
  script per run (correct `#SBATCH`/`#PBS` headers, resource requests, module/env activation)
  into its run folder and submits them (`sbatch`/`qsub`). Template: `assets/submit_cluster.py`
  with `assets/slurm_job.sh` / `assets/pbs_job.sh`.
- **Parallel on Modal** — launch N design processes concurrently, each offloading folding to
  Modal. **Critical:** the Modal backend (`boileroom`) hard-codes a single app named `"boileroom"`
  with no override, so concurrent runs share one app name and **collide** (the conflict the user
  may have already hit). Namespace each parallel run into its **own Modal environment** by setting
  a distinct `MODAL_ENVIRONMENT` per run, and **warm up once** (run one job or the smoke test to
  completion first) so the shared `model-weights` volume and app are initialized before fanning
  out. `assets/sweep_runner.py` + `sweep_config.py` support a per-run `modal_environment`; see
  `references/execution-harness.md` for the full pattern (creating environments, staggering, and
  when to prefer serial instead).
- **Serial in the background** — a launcher that runs one design at a time, submitting the next
  only when the previous finishes, so a single background process walks the whole sweep.
  Template: `assets/sweep_runner.py` (`--mode serial`; `--mode background` detaches it).

Adapt the chosen template to the actual design script's CLI flags and the sweep axis, save it
next to the design script, and tell the user the exact one-line command to launch it (and where
each run's outputs and logs will land).

---

## Reference files

Read these as needed — don't hold every detail in context:

- **`references/api-reference.md`** — the full verified catalog: every energy term with its real
  signature and residue-group convention, oracles and their config, minimizers, mutation
  protocols, callbacks, constants, utilities, and the gotchas that break scripts. Read before
  writing any design script.
- **`references/patterns.md`** — copy-and-adapt worked scripts: binder, selective/negative
  multi-state design, de novo symmetric scaffold, embedding-based enzyme miniaturization, and
  custom temperature schedules + chained optimizers.
- **`references/clarification-checklist.md`** — the Phase-1 interview checklist: every question
  to ask so the System is fully specified.
- **`references/execution-harness.md`** — Phase-5 detail: folder conventions, per-run isolation,
  and how to build the SLURM/PBS/Modal-parallel/serial-background launchers from the `assets/`
  templates.

## Output style (recap)

Match the repo: `main()` + `fire.Fire(main)`, only the tunables the task needs as flags, **named**
energy terms, callbacks over `log_frequency`, sequences as labeled constants, and comments that
explain the *reasoning* behind each energy term and optimizer choice. Keep the code correct but
simple (see the guiding principle) — plain, straight-line, no speculative machinery. Start every
generated file with the agent-disclaimer block. Save everything to disk. Favor asking a quick
question over shipping a silent assumption.
