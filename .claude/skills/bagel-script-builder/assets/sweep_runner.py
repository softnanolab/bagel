#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Generated with the assistance of an AI agent (Claude, via the
# `bagel-script-builder` skill). Review before running — you are responsible for
# its correctness.
# -----------------------------------------------------------------------------
"""Run a BAGEL sweep locally: serially, in the background, or in parallel.

Reads the sweep from `sweep_config.py` (DESIGN_SCRIPT + SWEEP). Each run executes the
design script as its own subprocess with a per-run --log_path, so runs are isolated and
never overwrite each other. A run is marked complete by a `DONE` file in its folder, so
re-running the launcher resumes and skips finished runs.

Launch with one command:

    python sweep_runner.py                     # serial: one run after another
    python sweep_runner.py --mode parallel --max-parallel 4
    python sweep_runner.py --mode background   # detach; the whole sweep runs in the background
    python sweep_runner.py --only seed_3       # just one run
    python sweep_runner.py --dry-run           # print the commands, run nothing

Modes:
  serial     — run each design to completion, then the next. Simple and cheap on local CPU.
  parallel   — run up to --max-parallel designs at once. With backend='modal' each design
               offloads folding to its own Modal GPU instances, so this gives real parallel
               Modal usage (this is the "parallel on Modal" option).
  background — re-launch this script detached (nohup-style) in serial mode and return
               immediately; a single background process walks the whole sweep, submitting the
               next run only when the previous finishes. Progress is in sweep_runner.out.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import pathlib as pl

import sweep_config as cfg


def _ensure_modal_env(name: str) -> None:
    """Create the Modal environment if it doesn't exist (harmless if it already does)."""
    subprocess.run(["modal", "environment", "create", name], capture_output=True, text=True)


def _prepare(run: dict, *, initialize: bool = True) -> tuple[pl.Path, list[str], dict]:
    """Build the command + environment for a run. With initialize=False (used by --dry-run)
    it has no side effects — no directory, no command.txt, no Modal CLI call."""
    run_dir = cfg.RUNS_DIR / run["name"]
    cmd = [sys.executable, str(cfg.DESIGN_SCRIPT), *cfg.cli_args_for(run, run_dir)]
    # Per-run environment (e.g. MODAL_ENVIRONMENT to isolate parallel Modal runs).
    extra_env = cfg.env_for(run)
    env = {**os.environ, **extra_env}
    if initialize:
        run_dir.mkdir(parents=True, exist_ok=True)
        if extra_env.get("MODAL_ENVIRONMENT"):
            _ensure_modal_env(extra_env["MODAL_ENVIRONMENT"])
        (run_dir / "command.txt").write_text(
            " ".join(f"{k}={v}" for k, v in extra_env.items()) + (" " if extra_env else "")
            + " ".join(cmd) + "\n"
        )
    return run_dir, cmd, env


def _done(run_dir: pl.Path) -> bool:
    return (run_dir / "DONE").exists()


def _start(run: dict):
    """Start one design subprocess; returns (run_name, popen, log_file_handle, run_dir)."""
    run_dir, cmd, env = _prepare(run)
    log = open(run_dir / "run.log", "w")
    print(f"  -> {run['name']}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=cfg.DESIGN_SCRIPT.parent, env=env)
    return run["name"], proc, log, run_dir


def _finish(name: str, proc: subprocess.Popen, log, run_dir: pl.Path) -> bool:
    rc = proc.wait()
    log.close()
    if rc == 0:
        (run_dir / "DONE").write_text("ok\n")
        print(f"  [OK]   {name}")
        return True
    print(f"  [FAIL] {name} (exit {rc}) — see {run_dir / 'run.log'}")
    return False


def run_serial(runs: list[dict]) -> int:
    failures = 0
    for run in runs:
        run_dir = cfg.RUNS_DIR / run["name"]
        if _done(run_dir):
            print(f"  [skip] {run['name']} (already DONE)")
            continue
        name, proc, log, run_dir = _start(run)
        if not _finish(name, proc, log, run_dir):
            failures += 1
    return failures


def run_parallel(runs: list[dict], max_parallel: int) -> int:
    pending = [r for r in runs if not _done(cfg.RUNS_DIR / r["name"])]
    for r in runs:
        if _done(cfg.RUNS_DIR / r["name"]):
            print(f"  [skip] {r['name']} (already DONE)")
    running: list[tuple] = []
    failures = 0
    while pending or running:
        while pending and len(running) < max_parallel:
            running.append(_start(pending.pop(0)))
        time.sleep(1.0)
        still = []
        for name, proc, log, run_dir in running:
            if proc.poll() is None:
                still.append((name, proc, log, run_dir))
            else:
                if not _finish(name, proc, log, run_dir):
                    failures += 1
        running = still
    return failures


def relaunch_background(only: str | None = None) -> None:
    out = cfg.RUNS_DIR / "sweep_runner.out"
    cfg.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, os.path.abspath(__file__), "--mode", "serial"]
    if only:  # preserve the user's --only selection in the detached run
        cmd.extend(["--only", only])
    with open(out, "w") as fh:
        proc = subprocess.Popen(
            cmd, stdout=fh, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            start_new_session=True,  # detach from this terminal
        )
    print(f"Sweep running in the background (PID {proc.pid}). Progress: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a BAGEL sweep locally.")
    ap.add_argument("--mode", choices=["serial", "parallel", "background"], default="serial")
    ap.add_argument("--max-parallel", type=int, default=4, help="concurrency for --mode parallel")
    ap.add_argument("--only", default=None, help="run only the sweep entry with this name")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = ap.parse_args()
    if args.mode == "parallel" and args.max_parallel < 1:
        ap.error("--max-parallel must be at least 1")

    runs = cfg.SWEEP
    if args.only:
        runs = [r for r in runs if r["name"] == args.only]
        if not runs:
            sys.exit(f"No sweep entry named {args.only!r}")

    if args.dry_run:
        for r in runs:
            _, cmd, _ = _prepare(r, initialize=False)
            print(" ".join(cmd))
        return

    if args.mode == "background":
        relaunch_background(args.only)
        return

    if args.mode == "parallel":
        warning = cfg.parallel_modal_warning(runs)
        if warning:
            print(f"WARNING: {warning}")

    print(f"Sweep: {len(runs)} run(s), mode={args.mode}")
    failures = run_serial(runs) if args.mode == "serial" else run_parallel(runs, args.max_parallel)
    print(f"Done. {len(runs) - failures} ok, {failures} failed. Outputs in {cfg.RUNS_DIR}/")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
