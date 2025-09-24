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

For more details, consult the available [pre-print](https://www.biorxiv.org/content/10.1101/2025.07.05.663138v1).

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

### With PyPI Installation

```bash
python scripts/script.py
```

### With Source Installation

```bash
uv run python scripts/script.py
```

To execute templates reproducibly from the [technical report manuscript](https://www.biorxiv.org/content/10.1101/2025.07.05.663138v1) (within statistical noise due to the nature of Monte Carlo sampling), follow release v0.1.0, also stored on Zenodo [![DOI](https://zenodo.org/badge/968747892.svg)](https://doi.org/10.5281/zenodo.15812348). Otherwise, use the most recent `biobagel` distribution.

## Oracles
One can either run Oracles locally, or remotely.

- `use_modal=True`: Run Oracles on [Modal](www.modal.com). Using the [boileroom](https://pypi.org/project/boileroom) package, running remotely is made seamless and does not require installing any dependencies. However, you need to have credits to use Modal.
- `use_modal=False`: Run Oracles locally through [boileroom](https://pypi.org/project/boileroom). You need a GPU with suitable memory requirements.

To use Modal, one needs to create an account and authenticate through:

        modal token new

You also need to set `MODEL_DIR` to an accessible folder, where deep learning models will be stored (i.e. cached).

Note on cache location and persistence:
- By default, examples may resolve `MODEL_DIR` to an XDG-compliant cache directory such as `~/.cache/bagel/models` (or the path in `$XDG_CACHE_HOME`). This directory is user-writable and persists across runs.
- The cache is not automatically cleaned by the application. If you wish to reclaim disk space, remove models manually (e.g., `rm -rf ~/.cache/bagel/models`) or configure your own housekeeping policy. Advanced users on Linux can use `systemd-tmpfiles` rules per their environment.

### Google Colab
A prototyping, but unscalable alterantive is to run BAGEL in Google Colab, having an access to a T4 processing unit for free. See this [notebook](https://colab.research.google.com/drive/1dtX8j6t5VhSed4iiqSrjM35DyPSFE1yF?usp=sharing), which includes the installation, and the template script for [simple binder](scripts/binders/simple_binder.py).

### Examples
[Templates](scripts/) and [example applications from the manuscript](scripts/technical-report/) are included as ready-to-run Python scripts.

## Contributing

For development setup, testing, and contribution guidelines, see [Development Guide](docs/development.md).

## Citation
```
@article{lala2025bagel,
        author = {L{\'a}la, Jakub and Al-Saffar, Ayham and Angioletti-Uberti, Stefano},
        title = {BAGEL: Protein Engineering via Exploration of an Energy Landscape},
        journal = {bioRxiv},
        year = {2025},
        doi = {10.1101/2025.07.05.663138},
        url = {https://www.biorxiv.org/content/early/2025/07/08/2025.07.05.663138},
        note = {Preprint}
}
```

## Acknowledgments
BAGEL's development was lead by Jakub Lála, Ayham Al-Saffar, and Dr Stefano Angioletti-Uberti at Imperial College London.
We thank Shanil Panara, Dr Daniele Visco, Arnav Cheruku, and Harsh Agrawal for helpful discussions.
We also thank [Hie et. al 2022](https://doi.org/10.1101/2022.12.21.521526), whose work inspired the creation of this package.
