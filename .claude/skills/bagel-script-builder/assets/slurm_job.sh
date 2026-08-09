#!/bin/bash
# -----------------------------------------------------------------------------
# Generated with the assistance of an AI agent (Claude, via the
# `bagel-script-builder` skill). Review before running — you are responsible for
# its correctness.
# -----------------------------------------------------------------------------
# SLURM job for one BAGEL sweep run. submit_cluster.py fills the {{...}} tokens per run.
# EDIT the resource requests and environment activation for your cluster before submitting.
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output={{RUN_DIR}}/slurm.%j.out
#SBATCH --error={{RUN_DIR}}/slurm.%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu          # <-- your GPU partition
#SBATCH --gres=gpu:1             # <-- request 1 GPU (needed for backend=apptainer; omit for backend=modal)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# --- Environment (EDIT for your cluster) --------------------------------------------
# module load cuda/12.6
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate bagel
# or, if using uv in the repo:
# cd /path/to/bagel

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
