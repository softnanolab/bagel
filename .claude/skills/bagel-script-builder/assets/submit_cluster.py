#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Generated with the assistance of an AI agent (Claude, via the
# `bagel-script-builder` skill). Review before running — you are responsible for
# its correctness.
# -----------------------------------------------------------------------------
"""Submit a BAGEL sweep to a SLURM or PBS cluster — one job per run.

Reads the sweep from `sweep_config.py`. For each run it renders a job script from a
template (`slurm_job.sh` / `pbs_job.sh`), writes it into that run's own folder, and submits
it with `sbatch` / `qsub`. Every run writes to its own --log_path, so jobs never collide.

    python submit_cluster.py --scheduler slurm
    python submit_cluster.py --scheduler pbs --only seed_3
    python submit_cluster.py --scheduler slurm --dry-run   # render scripts, don't submit

Before using: open the matching template (slurm_job.sh / pbs_job.sh) and set the resource
requests (partition/queue, GPU, time, memory) and the environment activation for your
cluster (module load / conda activate / uv). The {{PLACEHOLDER}} tokens are filled in here.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import pathlib as pl

import sweep_config as cfg

HERE = pl.Path(__file__).resolve().parent
TEMPLATES = {"slurm": HERE / "slurm_job.sh", "pbs": HERE / "pbs_job.sh"}
SUBMIT_CMD = {"slurm": "sbatch", "pbs": "qsub"}


def render(scheduler: str, run: dict) -> tuple[pl.Path, str]:
    run_dir = cfg.RUNS_DIR / run["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    design_cmd = " ".join([
        "python", str(cfg.DESIGN_SCRIPT), *cfg.cli_args_for(run, run_dir),
    ])
    # Isolate parallel Modal runs by environment (avoids the fixed "boileroom" app-name
    # clash). No-op for runs without a modal_environment.
    env = cfg.env_for(run)
    if env.get("MODAL_ENVIRONMENT"):
        modal_env_export = (
            f'export MODAL_ENVIRONMENT="{env["MODAL_ENVIRONMENT"]}"\n'
            f'modal environment create "{env["MODAL_ENVIRONMENT"]}" 2>/dev/null || true'
        )
    else:
        modal_env_export = "# (no per-run Modal environment set)"
    template = TEMPLATES[scheduler].read_text()
    filled = (
        template
        .replace("{{JOB_NAME}}", f"bagel_{run['name']}")
        .replace("{{RUN_DIR}}", str(run_dir))
        .replace("{{MODAL_ENV_EXPORT}}", modal_env_export)
        .replace("{{DESIGN_CMD}}", design_cmd)
    )
    job_path = run_dir / f"job.{scheduler}.sh"
    job_path.write_text(filled)
    return job_path, design_cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit a BAGEL sweep to SLURM or PBS.")
    ap.add_argument("--scheduler", choices=["slurm", "pbs"], required=True)
    ap.add_argument("--only", default=None, help="submit only this sweep entry")
    ap.add_argument("--dry-run", action="store_true", help="render job scripts but do not submit")
    args = ap.parse_args()

    runs = cfg.SWEEP
    if args.only:
        runs = [r for r in runs if r["name"] == args.only]
        if not runs:
            sys.exit(f"No sweep entry named {args.only!r}")

    # Cluster jobs run concurrently, so warn if their Modal environments would collide.
    warning = cfg.parallel_modal_warning(runs)
    if warning:
        print(f"WARNING: {warning}")

    submit = SUBMIT_CMD[args.scheduler]
    print(f"{'Rendering' if args.dry_run else 'Submitting'} {len(runs)} job(s) to {args.scheduler}")
    for run in runs:
        # Skip runs already completed by a previous launch (matches sweep_runner.py).
        if (cfg.RUNS_DIR / run["name"] / "DONE").exists():
            print(f"  [skip] {run['name']} (already DONE)")
            continue
        job_path, _ = render(args.scheduler, run)
        if args.dry_run:
            print(f"  rendered {job_path}")
            continue
        result = subprocess.run([submit, str(job_path)], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  [submitted] {run['name']}: {result.stdout.strip()}")
        else:
            print(f"  [FAIL] {run['name']}: {result.stderr.strip()}")
    if args.dry_run:
        print("Dry run — inspect the rendered job.*.sh scripts, then submit without --dry-run.")


if __name__ == "__main__":
    main()
