# Phase 3 detail — the two documentation files

Every design script ships with two companion Markdown files in the same folder:

| File | Reader | Purpose |
|---|---|---|
| `Instructions_for_script.md` | Claude (or another agent) | A self-contained prompt that regenerates the same script from scratch. |
| `Extended_description_and_goals.md` | A human scientist | What the run is trying to achieve and exactly how, in plain English. |

Both are written **after** the design script is final, and both are **rewritten whenever the
script changes**, so a reader never finds documentation that disagrees with the code.

---

## How to write (applies to both files)

These files exist to make a run reproducible and interpretable months later by someone who was
not in this conversation. That goal drives every style rule below.

- **Plain English, scientific register.** Short declarative sentences. Say what is true. A
  colleague reading it should be able to re-implement the run without opening the script.
- **No filler.** Cut "comprehensive", "robust", "powerful", "seamlessly", "leverages",
  "cutting-edge", "it's worth noting", "in today's landscape". Cut sentences that would survive
  unchanged in a document about a completely different design — they carry no information.
- **No unexplained jargon.** BAGEL class names are fine, but gloss them the first time
  ("`PAEEnergy`, the predicted aligned error between two groups of residues — a measure of how
  confident the structure predictor is about their relative placement").
- **Numbers, not adjectives.** "20,000 steps at temperature 0.01" beats "a long low-temperature
  run". Every weight, step count, temperature, cutoff and residue index appears explicitly.
- **Be explicit about residues.** Never write "the epitope" alone. Write the chain ID, the index
  range, the numbering convention (0-based Python index vs. the author's 1-based numbering), and
  the one-letter identities: "epitope = chain `CD20` residues 166–182 (0-based Python indices,
  `YNCEPANPSEKNSPSTQ`)". For terms that act on residue *pairs* or two groups, spell both groups
  out in full.
- **State what is not known.** If something was a default nobody consciously chose, say so:
  "`inheritable` left at the class default; not deliberately tuned." That is more useful than
  silence, and it flags what to revisit.
- **No invention.** Only describe what is in the script and what the user actually said. If the
  scientific rationale for a term was never discussed, ask the user for it in Phase 3 rather
  than inventing a plausible-sounding reason.

---

## File 1 — `Instructions_for_script.md`

A prompt that could be pasted into a fresh Claude session, with this skill available and no other
context, and yield the same script. It must therefore carry **every** fact the script depends on:
sequences in full, every index, every weight, every step count, backends, file paths, CLI flags.
No references to "the sequence we discussed" or "as before".

Write it addressed to the agent, imperative, and end with the reproduction constraint.

```markdown
# Prompt to regenerate `<script_name>.py`

Write a BAGEL (`biobagel`, `import bagel as bg`) design script that does the following.
Reproduce it exactly as specified — do not substitute your own defaults for anything below.

## Goal
<one paragraph: what is being designed and what counts as success>

## Chains
- `<chain_ID>` — <role: target / binder / de novo>, length <N>.
  Sequence: `<full sequence, no ellipsis>`
  Mutable: <explicit indices or ranges, with numbering convention; or "none">
  Immutable: <explicit indices or ranges, and why each is protected>
<repeat for every chain>

## States
- `<state_name>` — contains chains <...>; represents <physical meaning>;
  <positive or negative design>.
<repeat for every state>

## Oracles
- `<name> = bg.oracles.<Class>(backend='<modal|apptainer[:tag]>', config=<dict or None>)`
  <model/image/weights identifier, and which terms use it>

## Energy terms
For state `<state_name>`:
- `<term name>` — `bg.energies.<Class>(oracle=<oracle>, residues=<groups spelled out>,
  weight=<w>, <other args>)`. <what it rewards or penalizes>
<repeat for every term in every state>

## Mutation protocol
`bg.mutation.<Canonical|GrandCanonical>(n_mutations=<n>, mutation_bias=<...>, <other args>)`

## Optimizer
`bg.minimizer.<Class>(<all arguments, including the full temperature schedule>)`
<if phases are chained: each phase in order, and what is fed forward>
Callbacks: <...>. Log path / experiment name: <...>.

## Script structure
`main()` wrapped in `fire.Fire(main)`. Expose these CLI flags only: <list>.
Save to `<path>`. Start the file with the standard agent-disclaimer comment block.

## Steps outside BAGEL
<any external tool, pre/post-processing, or interleaved optimization — or "none">

## Reproduction constraint
Random seed(s): <...>. Package versions / environment: <...>.
The regenerated script must produce the same System, energy function and optimizer settings as
described above; flag any point where this specification is ambiguous instead of guessing.
```

---

## File 2 — `Extended_description_and_goals.md`

The human-facing record. Use these section headings in this order; they map one-to-one onto what
a reviewer or a future analyst needs. Keep every section, writing "None" where a section does not
apply, so a missing section always means a mistake rather than an absence.

```markdown
# <Project name> — what this run does and why

## 1. Goal in one paragraph
<The design objective and the hypothesis being tested, in plain English. What result would
count as success, and what would falsify the hypothesis.>

## 2. The system
<The States and, within each, the Chains — in prose and/or a table, not Python. For each state:
what it physically represents (bound complex, free binder, off-target complex), which chains it
holds, and whether it is being pushed towards or away from.>

## 3. Chains and sequences
<For each fixed chain: chain_ID, role, length, full sequence in a fenced block, and the mutable
and immutable positions written out explicitly with the numbering convention stated. For each
runtime-generated chain, give the generation method, length, seed, and source inputs instead of
inventing a sequence. Give the reason each immutable region is protected (catalytic residue,
disulfide, epitope on a fixed target...).>

## 4. Energy terms
<One subsection per state. For each term: its name in the script, the BAGEL class, its weight and
sign, the oracle that evaluates it, every non-default parameter, and the residue group(s) it acts
on — both groups spelled out explicitly for interface terms, including chain and indices. State
what the term measures physically.>

## 5. What the energy function is trying to do
<Plain-English paragraph, no jargon: the combined intent of the terms above. For example:
"Together these terms push the binder to fold confidently on its own, to sit against residues
166–182 of CD20 rather than anywhere else on the target, and to avoid a large exposed
hydrophobic patch." Say explicitly how the positive and negative terms trade off, and which
term is expected to dominate.>

## 6. Mutation protocol
<Canonical or GrandCanonical and what that means here (fixed vs. variable length), mutations per
step, mutation bias (e.g. cysteines excluded), move probabilities, and which residues the moves
can actually touch.>

## 7. Optimization protocol and temperature schedule
<Minimizer, total number of steps (record step counts, not derived oracle-call counts: a step can evaluate several oracles and caching can skip evaluations; measure calls if they matter), the temperature schedule
written out (constant value, ramp endpoints, or cycle structure with step counts), restart /
preserve-best behaviour, chained phases in order, callbacks and what they log, and where outputs
are written.>

## 8. Oracles
<For each oracle: the class, the exact model or checkpoint, the backend and image (Modal app or
`apptainer:<image-tag>`), any config (glycine linker, position_ids_skip, model_name, recycles),
any optimization such as caching, batching or reduced precision, and which energy terms it
serves.>

## 9. Steps outside BAGEL
<Anything in the protocol that the design script does not do itself: an external folding or
docking run every N steps, a greedy or manual refinement between BAGEL phases, filtering or
clustering of outputs, a hand-edited starting sequence. For each: the tool and version, when it
runs, what it consumes and produces, and how its output re-enters the BAGEL run. Write "None —
the entire protocol is contained in the script" if that is the case.>

## 10. How to read the results
<Which log files and per-term energy traces to look at, and what pattern would support or
undermine the hypothesis in section 1.>
```

Sections 1, 5, 9 and 10 are the ones that make the file worth having — they are the parts that
cannot be reconstructed from the code. Give them real thought rather than padding them.
