#!/bin/bash
# -----------------------------------------------------------------------------
# Generated with the assistance of an AI agent (Claude, via the
# `bagel-script-builder` skill). Review before running — you are responsible for
# its correctness.
# -----------------------------------------------------------------------------
# PBS/Torque job for one BAGEL sweep run. submit_cluster.py fills the {{...}} tokens per run.
# EDIT the resource requests and environment activation for your cluster before submitting.
#PBS -N {{JOB_NAME}}
#PBS -o {{RUN_DIR}}/pbs.out
#PBS -e {{RUN_DIR}}/pbs.err
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:ngpus=1:mem=32gb    # <-- adjust; omit ngpus for backend=modal
#PBS -q gpu                                    # <-- your GPU queue

set -euo pipefail
cd "$PBS_O_WORKDIR"

# --- Environment (EDIT for your cluster) --------------------------------------------
# module load cuda/12.6
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate bagel

# If using backend=modal, ensure Modal auth is available (modal token new, or
# export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... in this environment).
# Per-run Modal environment (isolates parallel Modal runs; no-op if unset):
{{MODAL_ENV_EXPORT}}

echo "Run dir: {{RUN_DIR}}"
echo "Started: $(date)"

# --- The design run -----------------------------------------------------------------
{{DESIGN_CMD}}

echo "Finished: $(date)"
touch {{RUN_DIR}}/DONE
