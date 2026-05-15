# BAGEL-CAR Recipes

These scripts are the manuscript-facing BAGEL-CAR recipe translations for the
two BAGEL design stages:

- `bgl_sol_cd20_dimer.py`: the CD20-dimer objective used for `BGL` and as the
  source backbone objective for `SOL`.
- `lnk_cd20_monomer.py`: the CD20-monomer seed objective used before the
  `LNK` RFdiffusion linker-fusion stage.

The labels follow the manuscript (`BGL`, `SOL`, `LNK-AA`, `LNK-AB`, `LNK-BB`,
`CD20A`, `CD20B`, and `binder`). `LNK` is used only as the umbrella stage label.
The dimer script uses BAGEL-compatible chain IDs `C20A`, `C20B`, and `BIND`
because current BAGEL restricts AtomArray chain IDs to fewer than five
characters.

## Usage

Both scripts default to running ESMFold on Modal:

```bash
uv run python scripts/car/bgl_sol_cd20_dimer.py --help
uv run python scripts/car/lnk_cd20_monomer.py --help
```

For local execution, set `--use_modal=False` and point `MODEL_DIR` at a
location with the ESMFold weights (see the project README for details):

```bash
MODEL_DIR=/path/to/models uv run python scripts/car/lnk_cd20_monomer.py \
  --use_modal=False
```

For a minimal smoke run, override the default production-scale tempering
schedule:

```bash
uv run python scripts/car/bgl_sol_cd20_dimer.py --use_modal=True \
  --n_cycles=1 --n_steps_low=1 --n_steps_high=1 --log_frequency=1
uv run python scripts/car/lnk_cd20_monomer.py --use_modal=True \
  --n_cycles=1 --n_steps_low=1 --n_steps_high=1 --log_frequency=1
```

Omit those counter overrides for full recipe runs.
