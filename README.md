# Bagel: Protein Engineering through Optimization

![Tests](https://img.shields.io/github/actions/workflow/status/softnanolab/bagel-wip/python-modal-tests.yaml?branch=main)


> 🚨 **Warning**
> This is the development repository for Bagel. A public repo with the most current release is going to be published in a separate repository.
> Instructions on how to properly develop this repo, and then push to the public repo are not yet finalized.


### Installation

1. Clone the repo.

        git clone https://github.com/softnanolab/bagel-wip

2. Install `uv`

        curl -LsSf https://astral.sh/uv/install.sh | sh

3. Set directory to repo.

        cd bagel

4. Download the environment.

        uv sync

6. (Optional) If you want to run all protein models locally.

        uv sync --extra local

### Use

Run code within the environment.

        uv run python scripts/script.py

### Oracles
One can either run Oracles locally, or remotely.

- `use_modal=True`: Run Oracles on [Modal](www.modal.com). Using the [boileroom](https://pypi.org/project/boileroom) package, running remotely is made seamless and does not require installing any dependencies. However, you need to have credits to use Modal.
- `use_modal=False`: Run Oracles locally through [boileroom](https://pypi.org/project/boileroom). You need a GPU with suitable memory requirements.

To use Modal, one needs to create an account and authenticate through:

        modal token new

You also need to set `HF_MODEL_DIR` to an accessible folder, where HuggingFace models will be stored.

### Testing

To run the tests, you must specify how to handle Oracles, i.e. whether to run remotely or not.

        uv run pytest --oracles skip

Alternative options for --oracles is "modal" and "local".

## Development Notes

First, you need to install the development dependencies

        uv sync --extra dev



## Docs

        uv run pydoclint src/bagel/* --style=sphinx

### Commit Checking

On commit, MyPy, Ruff, and PyTest checks are all run to ensure code quality.

If any of these tests fail, this will be displayed on the corresponding GitHub pull request.

We are not currently enforcing all these checks pass, but this behaviour can be enabled.

        uv run pre-commit install

This behaviour can subsequently be disabled aswell.

        uv run pre-commit uninstall

These checks can also be run on demand.

        uv run pre-commit run --all-files
