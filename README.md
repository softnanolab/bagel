# BAGEL: Protein Engineering via Exploration of an Energy Landscape

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/biobagel.svg)](https://pypi.org/project/biobagel/)
[![GitHub last commit](https://img.shields.io/github/last-commit/softnanolab/bagel.svg)](https://github.com/softnanolab/bagel/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/softnanolab/bagel.svg)](https://github.com/softnanolab/bagel/issues)
[![DOI](https://zenodo.org/badge/968747892.svg)](https://doi.org/10.5281/zenodo.15808838)

BAGEL is a model-agnostic, modular, fully customizable Python framework for programmable protein design.

The package formalizes the protein design task as an optimization (sampling) over an energy landscape.

<p align="center">
  <img src="https://raw.githubusercontent.com/softnanolab/bagel/main/docs/demo.gif" alt="BAGEL demo" width="600"/>
</p>

The BAGEL package is made up of several components that need to be specified to form a protein engineering task:

| **Component**      | **Description**                                                                                      | **Examples**                                         |
|--------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `EnergyTerms`      | Define specific design constraints as terms in the energy function.                                  | `TemplateMatchEnergy`, `PLDDTEnergy`, `HydrophobicEnergy` |
| `Oracles`          | Provide information (often via ML models) to compute optimization/sampling metrics.<br>Oracles are typically wrappers around models from [boileroom](https://github.com/softnanolab/boileroom). | `ESMFold`, `ESMFold2`, `Boltz2`, `Chai1`, `ESM3`, `ESM-C`, `ESM-2` |
| `Minimizers`       | Algorithms that sample or optimize sequences to find optima or diverse variants.                     | Monte Carlo, `SimulatedTempering`, `SimulatedAnnealing` |
| `MutationProtocols`| Methods for perturbing sequences to generate new candidates.                                         | `Canonical`, `GrandCanonical`                            |

For more details, consult the [published paper](https://doi.org/10.1371/journal.pcbi.1013774).

## Installation

### From PyPI (Recommended)

The easiest way to install BAGEL is through PyPI:

```bash
pip install biobagel
```

**Optional Extras:**

- For development (testing, linting, documentation):
```bash
pip install biobagel[dev]
```

Local model execution uses BoilerRoom's Apptainer backend and requires Apptainer
plus a suitable GPU on the host; there is no separate Python `local` extra.

### From Source

If you want to install from source or contribute to development:

1. Clone the repository:

```bash
git clone https://github.com/softnanolab/bagel
```

2. Install `uv` (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Navigate to the repository:

```bash
cd bagel
```

4. Install the environment:

```bash
uv sync
```

**Optional Extras:**

- For development (testing, linting, documentation):
```bash
uv sync --extra dev
```

- For Weights & Biases logging:
```bash
uv sync --extra wandb
```

- For all extras:
```bash
uv sync --all-extras
```


## Usage

Run any of the provided [example scripts](scripts/) to get started. For instance, to design a simple binder:

```bash
# With PyPI installation
python scripts/binders/simple_binder.py

# With source installation
uv run python scripts/binders/simple_binder.py
```

To execute templates reproducibly from the [published paper](https://doi.org/10.1371/journal.pcbi.1013774) (within statistical noise due to the nature of Monte Carlo sampling), follow release v0.1.0, also stored on Zenodo [![DOI](https://zenodo.org/badge/968747892.svg)](https://doi.org/10.5281/zenodo.15812348). Otherwise, use the most recent `biobagel` distribution.

## Oracles
Oracles are powered by [boileroom](https://github.com/softnanolab/boileroom) 0.4.1 and selected via a `backend` keyword on every oracle constructor. BAGEL exposes `ESMFold`, `ESMFold2`, `Chai1`, `Boltz2`, `ESM2`, `ESM-C`, and `ESM3` through the same interface.

- `backend="modal"` (default): Run on [Modal](https://www.modal.com). No local GPU required, but a Modal account with credits is needed. Authenticate via `modal token new`. Pin a specific image release via the `BOILEROOM_IMAGE_TAG` environment variable set **before** importing bagel/boileroom.
- `backend="apptainer"`: Run locally via an [Apptainer](https://apptainer.org) image pulled by boileroom. Requires `apptainer` on the host machine and a GPU with enough memory for the chosen model. Optionally pin the image tag inline: `backend="apptainer:<image-tag>"`.

All current oracles expose `modal` and `apptainer` backends. The Modal paths are
covered by the integration suite; Apptainer support for Boltz2 and Chai1 is
available through BoilerRoom but has not yet been validated in BAGEL's CI.

### Upgrading to BAGEL 0.2

BAGEL 0.2 requires Python 3.12. Oracle constructors now use
`backend="modal" | "apptainer"` and an optional `device`; the previous
`use_modal` and `modal_app_context` arguments and the local-Python backend were
removed. Replace `use_modal=True` with `backend="modal"` and
`use_modal=False` with `backend="apptainer"`.

### Google Colab
A prototyping, but unscalable alternative is to run BAGEL in Google Colab, having access to a T4 processing unit for free. See this [notebook](https://colab.research.google.com/drive/1dtX8j6t5VhSed4iiqSrjM35DyPSFE1yF?usp=sharing), which includes the installation, and the template script for [simple binder](scripts/binders/simple_binder.py).

### Examples
[Templates](scripts/) and [example applications from the paper](scripts/technical-report/) are included as ready-to-run Python scripts. For a case study on enzyme miniaturization using PLM embeddings, see the [mini-enzymes scripts](scripts/mini-enzymes/).

## Building BAGEL scripts with an AI agent

BAGEL ships an **agent skill**, [`bagel-script-builder`](.claude/skills/bagel-script-builder/), that turns a plain-language design goal — *"design a 30-residue binder to CD20 and sweep 8 seeds on SLURM"* — into runnable, reviewable BAGEL scripts. It is a guided, question-driven workflow: it drafts a script, then **interviews you** to pin down every undefined piece (States, Chains, EnergyTerms, protected residues, optimizer), writes a **verbose, well-commented** script to disk, offers a Modal **smoke test**, and can generate an **execution harness** for parameter sweeps.

The skill is plain Markdown plus reference files and ready-to-use launcher templates, so it is model-agnostic — it works with any capable coding agent, not only Claude.

### With Claude Code

The skill lives in this repo under [`.claude/skills/bagel-script-builder/`](.claude/skills/bagel-script-builder/), so [Claude Code](https://claude.com/claude-code) running in the repo discovers it automatically. Just describe what you want:

> "Use BAGEL to design a binder against `<target>`, focus on residues 40–60, and sweep 5 seeds serially in the background."

Claude runs the interview and writes the scripts into `bagel_designs/<name>/`. To make the skill available in every project, copy it to your personal skills directory:

```bash
cp -r .claude/skills/bagel-script-builder ~/.claude/skills/
```

### With any other agent

[`SKILL.md`](.claude/skills/bagel-script-builder/SKILL.md) is a self-contained playbook. With any coding agent (e.g. Cursor, Aider, a Claude API harness, or your own), point it at the skill directory and ask it to follow the workflow:

> "Read `.claude/skills/bagel-script-builder/SKILL.md` and its `references/`, then follow that workflow to build a BAGEL script for `<goal>`."

The agent reads `SKILL.md`, consults the reference files (`api-reference.md`, `patterns.md`, `clarification-checklist.md`, `execution-harness.md`) as needed, and adapts the launcher templates in `assets/`. No Claude-specific features are required.

### What it produces

- A **verbose, commented** design script (a single `main()` exposed via a CLI), saved for review.
- An optional **smoke test** that runs the whole pipeline for one step on Modal to catch errors early.
- For sweeps, a **one-command launcher** (`sweep_runner.py` / `submit_cluster.py`) that runs each configuration in its own folder — serially, in the background, on a SLURM/PBS cluster, or in parallel on Modal.

Every generated file begins with a comment noting it was produced with AI assistance, so it is clear the code should be reviewed before use. When running sweeps in parallel on Modal, the skill sets a distinct `MODAL_ENVIRONMENT` per run to avoid the shared-app-name conflict inherent to the default backend.

## Contributing

For development setup, testing, and contribution guidelines, see [Development Guide](docs/development.md).

## Citation
```bibtex
@article{Lala_2025,
  title={BAGEL: Protein engineering via exploration of an energy landscape},
  volume={21},
  ISSN={1553-7358},
  url={http://dx.doi.org/10.1371/journal.pcbi.1013774},
  DOI={10.1371/journal.pcbi.1013774},
  number={12},
  journal={PLOS Computational Biology},
  publisher={Public Library of Science (PLoS)},
  author={Lála, Jakub and Al-Saffar, Ayham and Angioletti-Uberti, Stefano},
  editor={Singh, Amar},
  year={2025},
  month=dec,
  pages={e1013774}
}
```

## Acknowledgments
BAGEL's development was led by Jakub Lála, Ayham Al-Saffar, and Dr Stefano Angioletti-Uberti at Imperial College London.
We thank Shanil Panara, Dr Daniele Visco, Arnav Cheruku, and Harsh Agrawal for helpful discussions.
We also thank [Hie et al. 2022](https://doi.org/10.1101/2022.12.21.521526), whose work inspired the creation of this package.
