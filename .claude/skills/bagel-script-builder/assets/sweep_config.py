# -----------------------------------------------------------------------------
# Generated with the assistance of an AI agent (Claude, via the
# `bagel-script-builder` skill). Review before running — you are responsible for
# its correctness.
# -----------------------------------------------------------------------------
"""Single source of truth for a BAGEL sweep.

The launchers (`sweep_runner.py` for local serial/background/parallel, and
`submit_cluster.py` for SLURM/PBS) both import DESIGN_SCRIPT and SWEEP from here, so you
edit the sweep in exactly one place.

A sweep repeats one design many times while varying some "difference" between runs. Each
entry in SWEEP is one run: a unique `name` (its output folder) and `args`, the CLI flags
passed to the design script for that run. Every run writes to its own folder, so runs never
overwrite each other.

PARALLEL MODAL RUNS: the Modal backend (boileroom) hard-codes a single app named "boileroom"
with no override, so running several backend='modal' designs concurrently collides. To fan
out on Modal, give each run its own Modal environment via `modal_environment=...` below
(sweep_runner.py exports it as MODAL_ENVIRONMENT and pre-creates it), and warm up one run
first. A single run, or serial/background execution, needs none of this.

>>> FILL THIS IN for your design. The example below sweeps a random seed; the commented
    alternatives show sweeping protected residues or step counts instead.
"""
from __future__ import annotations

import pathlib as pl

# Path to the verbose design script this sweep runs (the one built in Phase 2).
DESIGN_SCRIPT = pl.Path(__file__).resolve().parent / "design.py"

# Where all runs live. Each run gets RUNS_DIR/<name>/.
RUNS_DIR = pl.Path(__file__).resolve().parent / "runs"


def _run(name: str, *, modal_environment: str | None = None, **args: object) -> dict:
    """One sweep entry: a folder name, optional Modal environment, and CLI overrides.

    `modal_environment` (optional) isolates this run's Modal app namespace so parallel
    Modal runs don't collide on the fixed "boileroom" app name. Leave it None for serial
    runs, single runs, or cluster/local-CPU work.
    """
    return {"name": name, "modal_environment": modal_environment, "args": args}


# --- The sweep: EDIT THIS LIST -------------------------------------------------------
# Example A — same script, different random seed (the simplest "difference").
SWEEP: list[dict] = [_run(name=f"seed_{s}", seed=s) for s in range(5)]

# Example A' — the same, but SAFE FOR PARALLEL MODAL (one environment per run):
# SWEEP = [_run(name=f"seed_{s}", seed=s, backend="modal",
#               modal_environment=f"bagel-seed-{s}") for s in range(5)]

# Example B — vary the number of minimization steps:
# SWEEP = [_run(name=f"steps_{n}", n_steps=n) for n in (1000, 5000, 20000)]

# Example C — vary which residues are protected (pass as a flag your design parses):
# SWEEP = [
#     _run(name="protect_triad",        protected="57,102,195"),
#     _run(name="protect_triad_buffer", protected="55,56,57,58,59,100,101,102,103,104"),
#     _run(name="protect_none",         protected=""),
# ]
# -------------------------------------------------------------------------------------


def cli_args_for(run: dict, log_path: pl.Path) -> list[str]:
    """Build the design-script CLI argument list for one run.

    Always passes --log_path so the run writes into its own folder. Adjust the flag names
    (--seed, --n_steps, --protected, --backend, …) to match your design script's `main()`.
    """
    args = [f"--{k}={v}" for k, v in run["args"].items()]
    args.append(f"--log_path={log_path}")
    return args


def env_for(run: dict) -> dict:
    """Extra environment variables for this run's subprocess.

    Sets MODAL_ENVIRONMENT when the run specifies `modal_environment`, so parallel Modal
    runs are namespaced apart. Returns an empty dict otherwise.
    """
    env: dict[str, str] = {}
    if run.get("modal_environment"):
        env["MODAL_ENVIRONMENT"] = str(run["modal_environment"])
    return env
