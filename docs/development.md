# Development Guide

Thank you for your interest in contributing to `bagel`!

To contribute, please follow these general steps:

1. **Fork** the repository and clone your fork locally.
2. Create a **feature branch** for your changes.
3. Use `uv` for dependency and environment management (see below).
4. Follow the coding style and linting rules (enforced via pre-commit).
5. Submit a **pull request (PR)** to the main repository for review.

We welcome contributions of all kinds — new features, bug fixes, documentation improvements, and test coverage.

For more detailed guidelines consult these [Notion Docs](https://jakublala.notion.site/Development-Guide-24f22c74126780898c2bec333c7963ec?source=copy_link).

## Installation

First, you need to install the development dependencies:

```bash
uv sync --extra dev
```

BAGEL currently targets boileroom 0.4.1. Until that version is published to PyPI,
`uv` resolves the tagged GitHub release through the temporary source override in
`pyproject.toml`; the lock file records the exact commit for reproducibility.

BAGEL 0.2 requires Python 3.12. Its oracle API replaces `use_modal` and
`modal_app_context` with explicit `backend` and optional `device` parameters.

Host-side model dependencies are supplied by BoilerRoom's images. To run an
oracle locally, install Apptainer on the host and use a suitable GPU; no separate
Python `local` extra is required.

## Documentation [Work In Progress]

Generate documentation:

```bash
uv run pydoclint src/bagel/* --style=sphinx
```

## Testing

To run the tests, you must specify how to handle Oracles, i.e. whether to run remotely or locally.

```bash
# Run tests while skipping Oracle execution
uv run pytest --oracles skip

# Alternative options:
# --oracles modal      # Run oracles remotely via Modal (requires `modal token new`)
# --oracles apptainer  # Run oracles locally via Apptainer (requires `apptainer` on the host + a GPU)
```

Boltz-2 and Chai-1 fixtures currently skip under `--oracles apptainer` until the apptainer path is validated in CI; use `--oracles modal` to exercise them.

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
