# SAE-feature energy minimisation

`minimize_sae.py` runs a **canonical Monte Carlo** design loop at **low
temperature**, scoring mutations with a
[`SAEnergy`](../../src/bagel/energies.py) term. SAEnergy is a linear function of a
user-chosen set of ESM-C sparse-autoencoder (SAE) features of the design, with
default coefficients all equal to `1`, so the walk drives the sequence toward
higher values of that feature combination.

Because `SAEnergy` negates the combination by default (`maximize=True`) and BAGEL
minimizes energy:

- a **positive** coefficient (default) designs *toward* a feature's concept;
- a **negative** coefficient designs *away* from it.

Pass `maximize=False` to flip this.

## Run

```bash
# Modal backend (default)
python scripts/sae_features/minimize_sae.py

# Local containerised execution
BAGEL_BACKEND=apptainer python scripts/sae_features/minimize_sae.py
```

Edit `DEFAULT_SEQUENCE` and `DEFAULT_FEATURE_INDICES` at the top of the script, or
call `run(sequence=..., feature_indices=..., coefficients=..., temperature=...,
n_steps=...)` from your own code. The SAE oracle and MC loop are factored into
`build_sae_oracle`, `build_system`, `build_minimizer`, and `run` so the wiring can
be exercised with a fake oracle (see
`tests/unit_tests/test_sae_minimize_script.py`).

Feature indices correspond to the SAE codebook; use the
[ESM Atlas feature API](https://biohub.ai/esm/protein/atlas/api-docs/examples/feature_browse.html)
to look up what concept each feature encodes.
