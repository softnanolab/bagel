# Phase-5 execution harness

When the user wants a **sweep** — repeat one design while varying some "difference" — generate a
launcher so the whole sweep runs from a single command, with every run isolated in its own
folder. Ready-to-adapt templates live in `assets/`. Copy them next to the design script, fill in
the sweep and the cluster/env specifics, then tell the user the exact launch command.

## Core conventions (apply to every mode)

- **One folder per run.** Pass a distinct `--log_path=runs/<run_name>` to each invocation.
  BAGEL's minimizer creates `log_path/experiment_name/` and writes its logs/structures there, so
  distinct `log_path`s guarantee runs never overwrite each other.
- **Single source of truth for the sweep.** `assets/sweep_config.py` defines `DESIGN_SCRIPT` and
  `SWEEP` (a list of `{name, args}`). Both launchers import it, so the sweep is edited in one
  place. Adapt `cli_args_for()` so the flag names (`--seed`, `--n_steps`, `--protected`, …) match
  the design script's `main()` signature.
- **The "difference" is just the varied args.** Same script + different seed → vary `seed`.
  Different protected residues → vary a `--protected` flag the design parses. Different step
  counts → vary `--n_steps`. Anything the design exposes as a flag can be swept.
- **Resumable.** A run is marked done by a `DONE` file in its folder; re-launching skips
  finished runs. Good for recovering from partial failures.
- **One-command launch, CLI-style.** Every launcher takes flags (`--mode`, `--scheduler`,
  `--only`, `--dry-run`) so the user runs e.g. `python sweep_runner.py` or
  `python submit_cluster.py --scheduler slurm`.

Always start the user with `--dry-run` so they can inspect the exact commands/job scripts before
anything executes or is submitted.

## Choosing the mode (ask the user)

### Serial, in the background — `assets/sweep_runner.py --mode background`
A single background process walks the sweep, starting the next run only when the previous
finishes. Best when GPU/credit throughput is the bottleneck and you just want the list worked
through unattended. `--mode serial` does the same in the foreground.
```
python sweep_runner.py --mode background      # detaches; progress in runs/sweep_runner.out
```

### Parallel on Modal — `assets/sweep_runner.py --mode parallel`
Runs up to `--max-parallel` design processes at once; with `backend='modal'` each offloads its
folding to Modal GPU instances.

**The fixed-app-name conflict (must handle for parallel Modal).** The Modal backend
(`boileroom`) hard-codes a single app: `app = modal.App("boileroom")`, with no setting to rename
it. Every BAGEL process using `backend='modal'` therefore attaches to the *same* app name. Run
them concurrently and they collide — and on a cold machine they also race to create the shared
`model-weights` Modal Volume. Two mitigations, use both:

1. **One Modal environment per run.** Modal namespaces apps by *environment*, and honors the
   `MODAL_ENVIRONMENT` env var. Give each run a distinct environment so the shared `"boileroom"`
   app name lives in separate namespaces and can't collide. Environments must exist first
   (`modal environment create <name>`; creating an existing one is a harmless error). In
   `sweep_config.py`, set a per-run `modal_environment`; `sweep_runner.py` exports it for that
   run's subprocess and pre-creates it:
   ```python
   # sweep_config.py
   SWEEP = [_run(name=f"seed_{s}", seed=s, backend="modal",
                 modal_environment=f"bagel-seed-{s}") for s in range(5)]
   ```
2. **Warm up once, then fan out.** Run a single job (or the smoke test) to completion first so the
   app and the `model-weights` volume are initialized, then launch the rest in parallel. This
   avoids a cold-start stampede even within one environment.

If this is more orchestration than the user wants, prefer **serial/background on Modal** (no
concurrency, no conflict). **Cluster parallelism does not fix this by itself:** separate SLURM/PBS
jobs are separate OS processes, but they still share the one `"boileroom"` Modal app unless each
sets its own `MODAL_ENVIRONMENT`. Set a per-run `modal_environment` for concurrent Modal-backed
cluster jobs too — `submit_cluster.py` then exports it into each job script. Reserve high
`--max-parallel` on Modal for when per-run environments are set.

*Advanced (true per-instance Modal):* to run each entire minimization as a separate remote Modal
function, wrap the design call in your own `modal.App` and `.map()` over `SWEEP`. This needs a
Modal image with `biobagel` installed and is more setup; only build it if the user wants the whole
job remote. Note it does not remove the shared-app-name issue unless each map task also uses a
distinct environment.

### Parallel on a cluster — `assets/submit_cluster.py --scheduler slurm|pbs`
One job per run, submitted with `sbatch`/`qsub`. `submit_cluster.py` renders `slurm_job.sh` /
`pbs_job.sh` into each run folder with the per-run command filled in. **Edit the template first**
for the cluster's partition/queue, GPU request, walltime, memory, and environment activation
(`module load`, `conda activate`, or `cd` into the repo for `uv`).
```
python submit_cluster.py --scheduler slurm --dry-run   # inspect rendered job.slurm.sh files
python submit_cluster.py --scheduler slurm             # sbatch them
```
- For `backend=modal`, the cluster job needs Modal auth in its environment (`modal token new`
  beforehand, or export `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`) and does **not** need a cluster
  GPU — drop the GPU request. For `backend=apptainer`, keep the GPU request and load the image.

## Adapting the templates — checklist
- Point `DESIGN_SCRIPT` at the real design file; set `SWEEP` to the user's axis and values.
- Make `cli_args_for()` emit the design's actual flags; ensure the design's `main()` accepts
  `log_path` (and whatever axis is swept).
- For clusters: set resources + env activation in `slurm_job.sh` / `pbs_job.sh`; confirm the
  backend (GPU for apptainer vs Modal auth for modal).
- Hand the user the one-line launch command and point at `runs/<name>/run.log` (local),
  `runs/<name>/slurm.*.out` / `pbs.out` (cluster), and each run's `log_path` for BAGEL's own
  outputs.
