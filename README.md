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
| `Oracles`          | Provide information (often via ML models) to compute optimization/sampling metrics.<br>Oracles are typically wrappers around models from [boileroom](https://github.com/softnanolab/boileroom). | `ESMFold`, `ESM-2`                                   |
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

- For local protein model execution (requires GPU):
```bash
pip install biobagel[local]
```

- For development (testing, linting, documentation):
```bash
pip install biobagel[dev]
```

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

- For local protein model execution (requires GPU):
```bash
uv sync --extra local
```

- For development (testing, linting, documentation):
```bash
uv sync --extra dev
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
Oracles are powered by [boileroom](https://pypi.org/project/boileroom) and selected via a `backend` keyword on each oracle constructor (`ESMFold`, `ESM2`, `Chai1`, `Boltz2`).

- `backend="modal"` (default): Run on [Modal](https://www.modal.com). No local GPU required, but a Modal account with credits is needed. Authenticate via `modal token new`. Pin a specific image release via the `BOILEROOM_IMAGE_TAG` environment variable set **before** importing bagel/boileroom.
- `backend="apptainer"`: Run locally via an [Apptainer](https://apptainer.org) image pulled by boileroom. Requires `apptainer` on the host machine and a GPU with enough memory for the chosen model. Optionally pin the image tag inline: `backend="apptainer:<image-tag>"`.

Boltz-2 and Chai-1 only support `modal` and `apptainer`; ESMFold and ESM-2 likewise (the previous `use_modal=False`/local-Python backend was removed in boileroom 0.3).

### Google Colab
A prototyping, but unscalable alternative is to run BAGEL in Google Colab, having access to a T4 processing unit for free. See this [notebook](https://colab.research.google.com/drive/1dtX8j6t5VhSed4iiqSrjM35DyPSFE1yF?usp=sharing), which includes the installation, and the template script for [simple binder](scripts/binders/simple_binder.py).

### Examples
[Templates](scripts/) and [example applications from the paper](scripts/technical-report/) are included as ready-to-run Python scripts. For a case study on enzyme miniaturization using PLM embeddings, see the [mini-enzymes scripts](scripts/mini-enzymes/).

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
