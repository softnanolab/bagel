# Bagel: Protein Engineering through Optimization

### Installation

1. Clone the repo.

        git clone https://github.com/softnanolab/bagel-wip

2. Install `uv`

        curl -LsSf https://astral.sh/uv/install.sh | sh

3. Set directory to repo.

        cd bagel

4. Download the environment.

        uv sync

5. (Optional) If you also want to develop this package.

        uv sync --extra dev

### Use

Run code within the environment.

        uv run python scripts/script.py


To add libraries to the enviroment:

        uv add numpy

If the library is only required for development and not production:

        uv add --dev pytest

### Folding

We are optionally using `modalfold` for protein folding. This is a standalone package implementing protein structure prediction tools with Modal, abstracting all the problematic dependencies.

To use this, one needs a Modal account with credits. After creating an account, one needs to authenticate through:

        modal token new

Otherwise protein folding can be run locally. The temporary solution is requires:

        uv pip install git+https://github.com/jakublala/my_transformers.git

And then make sure to set `ESMFold(use_modal=False)`.

### Testing

To run the tests, you must specify how to handle any tests that require folding.

        uv run pytest --folding skip

Alternative options for --folding is "modal" and "local".

By default, --numprocesses is set to auto. When running locally, this may have to be overided to avoid running out of CUDA memory.

### Commit Checking

On commit, MyPy, Ruff, and PyTest checks are all run to ensure code quality.

If any of these tests fail, this will be displayed on the corresponding GitHub pull request.

We are not currently enforcing all these checks pass, but this behaviour can be enabled.

        uv run pre-commit install

This behaviour can subsequently be disabled aswell.

        uv run pre-commit uninstall

These checks can also be run on demand.

        uv run pre-commit run --all-files
