# BAGEL: Protein Engineering via Exploration of an Energy Landscape

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/biobagel.svg)](https://pypi.org/project/biobagel/)
[![GitHub last commit](https://img.shields.io/github/last-commit/softnanolab/bagel.svg)](https://github.com/softnanolab/bagel/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/softnanolab/bagel.svg)](https://github.com/softnanolab/bagel/issues)

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

To execute templates reliably from the technical report manuscript [citation to be added], follow release v0.1.0, also stored on Zenodo.
[![DOI](https://zenodo.org/badge/968747892.svg)](https://doi.org/10.5281/zenodo.15808838)

## Oracles
One can either run Oracles locally, or remotely.

- `use_modal=True`: Run Oracles on [Modal](www.modal.com). Using the [boileroom](https://pypi.org/project/boileroom) package, running remotely is made seamless and does not require installing any dependencies. However, you need to have credits to use Modal.
- `use_modal=False`: Run Oracles locally through [boileroom](https://pypi.org/project/boileroom). You need a GPU with suitable memory requirements.

To use Modal, one needs to create an account and authenticate through:

        modal token new

You also need to set `HF_MODEL_DIR` to an accessible folder, where HuggingFace models will be stored.

## Examples
[Templates](scripts/) and [example applications from the manuscript](scripts/technical-report/) are included as ready-to-run Python scripts.

## Contributing

For development setup, testing, and contribution guidelines, see [Development Guide](docs/development.md).
