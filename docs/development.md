# Development Guide

For development, we suggest one uses `uv`, as detailed in the main README.md

## Installation

First, you need to install the development dependencies:

```bash
uv sync --extra dev
```

If you are working on the optimization logic, using Modal should be sufficient; however, if you also need to debug individual Oracles locally, we suggest using a GPU and installing the local dependencies:

```bash
uv sync --extra local
# or install all extras
uv sync --all-extras
```

## Documentation [Work In Progress]

Generate documentation:

```bash
uv run pydoclint src/bagel/* --style=sphinx
```

## Testing

To run the tests, you must specify how to handle Oracles, i.e. whether to run remotely or not.

```bash
# Run tests while skipping Oracle execution
uv run pytest --oracles skip

# Alternative options:
# --oracles modal   # Use remote execution via Modal
# --oracles local   # Use local GPU-based execution
```

## Commit Checking

On commit, MyPy, Ruff, and PyTest checks are all run to ensure code quality.

If any of these tests fail, this will be displayed on the corresponding GitHub pull request.

To automatically run code quality checks before each commit (recommended for contributors):

```bash
uv run pre-commit install
```

To disable the hooks:

```bash
uv run pre-commit uninstall
```

You can also run all checks manually at any time:

```bash
uv run pre-commit run --all-files
```