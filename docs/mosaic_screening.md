# Mosaic screening (`bagel.mosaic_screening`)

`MosaicScreening` is a bagel `MutationProtocol` that decides *which* substitution to propose by running a fast forward pass of
[mosaic](https://github.com/escalante-bio/mosaic) (JAX) on a batch of random candidate mutations and keeping the one with the lowest
**pseudo-energy**. The proposal is then folded and evaluated by bagel's own oracles with the full energy, and accepted or rejected
by the minimizer as for any other protocol. Mosaic never replaces the energy that is optimised: if it ranks candidates badly the
cost is efficiency, not correctness.

It is meant for optimisation. The proposal is not symmetric and no Hastings correction is applied.

All parameters are documented in the class docstring (`help(bagel.mosaic_screening.MosaicScreening)`).

## Use

```python
import bagel as bg
from bagel.mosaic_screening import MosaicScreening, ModalMosaicBackend

mutator = MosaicScreening(backend=ModalMosaicBackend())          # defaults: 8 candidates, calibration on
minimizer = bg.minimizer.SimulatedAnnealing(mutator=mutator, ...)
```

Where mosaic runs is a *backend*:

| Backend | For | Needs |
|---|---|---|
| `ModalMosaicBackend()` (default) | Modal | the service deployed once: `modal deploy src/bagel/mosaic_screening/modal_service.py` (secret `huggingface-secret`) |
| `SubprocessMosaicBackend.local(<mosaic env>/bin/python)` | a machine with a GPU | an environment with mosaic (separate from bagel's: torch versions conflict) |
| `SubprocessMosaicBackend.apptainer("bagel-mosaic-screening.sif", binds=["/path/cache:/cache"])` | SLURM / PBS jobs | the published image as a `.sif`, and the model weights in the bound cache directory |

On a cluster whose compute nodes have no internet access, pull the image and download the weights (ESMFold2 is gated on Hugging
Face and needs a token) on a login node first.

## What is screened with, and what is not

The pseudo-energy of a State is the weighted sum of the energy terms that can be computed from a mosaic prediction, **using bagel's
own term code** (residue groups included): `PTMEnergy`, `PLDDTEnergy`, `OverallPLDDTEnergy`, `PAEEnergy`, `LISEnergy`,
`ipSAEEnergy`, on an `ESMFold2` or `Boltz2` oracle. These terms read only pLDDT, PAE and pTM. Every other term (embeddings, other States, and,
unless `use_structure_terms=True`, anything that reads atom coordinates) is left out of the pseudo-energy. At the start of a run the protocol logs which terms are used
and writes `Mosaic_pseudo_energy.txt` into the log folder of the run, with the coverage by number of terms and by weight
(sum of |weight| used / total). Low coverage means the pseudo-energy ignores much of what the real energy rewards; the calibration
check below is the safeguard.

With several States, the score of a candidate is the sum of the pseudo-energy changes of the States that contain the mutated chain.

### Structure-based terms (opt-in)

With `use_structure_terms=True`, mosaic also returns its all-atom coordinates, bagel rebuilds a structure with the same atoms, names and
order as its oracles' structures (a round-trip test shows the term values are identical on the rebuilt and the original structure), and
`SurfaceAreaEnergy`, `HydrophobicEnergy`, `HydropathyEnergy`, `GlobularEnergy`, `BuriedHistidineEnergy` and
`HistidineCarboxylateContactEnergy` join the pseudo-energy. It is **off by default** because the agreement with bagel's own ESMFold2
structures was weak in the one check made (`scripts/mosaic_screening/structure_terms_check.py`, 25 mutants of a 40-residue binder against
ubiquitin): Spearman correlation -0.12 for the whole-complex surface area, 0.35 for the binder's, -0.28 for the globular term, -0.04 for buried
histidine (and `his_carboxylate` was constant); the hydrophobic and hydropathy terms without a surface/core mode depend on the sequence only and
agree exactly. With 25 points a correlation is uncertain by about ±0.2, so this shows no demonstrated benefit, not proven uselessness. Mosaic's
structures were also systematically more compact (mean surface-area term 0.28 against 0.34). Other structure terms (salt bridges, shape
complementarity, secondary structure, template match, ring symmetry, separation) are not supported yet.

## Calibration

In the first `calibration_steps` (default 5) screening steps, `calibration_mutants` (default 8) candidates are also folded with
bagel, including the ones that are not proposed, and the Spearman correlation between pseudo-energy change and real energy change is
computed. Below `calibration_threshold` (default 0.25) the protocol warns, records the failure in the report and switches to random
sampling for the rest of the run. 40 pairs estimate a rank correlation to about ±0.16 (one standard error); `calibration_rule='upper95'`
fails only when even the upper end of the 95% interval is below the threshold. `calibration_steps <= 0` switches the check off.

## Guardrails (raised before any call to mosaic)

* more than one chain with mutable residues in a State (mosaic injects the designed sequence into the first chain only),
* `GrandCanonical` as fallback, or any change of a chain length during the run,
* residues whose `index` differs from their position in the chain; letters outside the 20 amino acids,
* no energy term of a designed State that can be evaluated on a mosaic prediction.

## The mosaic environment and the image

Mosaic needs torch 2.11 and JAX; bagel's boileroom pins torch < 2.7. They cannot share an environment, so mosaic runs from
`docker/mosaic-screening/Dockerfile`: mosaic at a pinned commit, a pinned dependency lock (`uv.lock`: upstream ships none and has
unpinned git dependencies), and `worker.py`. Weights are not in the image.

CI (`.github/workflows/mosaic-image.yaml`) builds it on changes to those files and pushes
`ghcr.io/softnanolab/bagel-mosaic-screening` with the automatic `GITHUB_TOKEN` (no stored credentials).

### Making the package public (free)

Per GitHub's documentation, a container package pushed by a workflow starts private. Public packages are free (storage and transfer),
and for the container registry GitHub states that storage and bandwidth are currently free, with at least a month's notice of any
change. To make it public, once, after the first CI push:

1. organisation page → **Packages** → open `bagel-mosaic-screening`;
2. **Package settings** → at the bottom, **Danger Zone** → **Change visibility** → **Public**.

This cannot be undone. It needs admin permission on the package, and an organisation owner can forbid members from creating public
packages (organisation **Settings → Packages → Package creation**); if the option is greyed out, that is the reason. The image links
itself to this repository through the `org.opencontainers.image.source` label, so workflows of the repository can pull and push it
without extra secrets.

## Limits of the evidence

The benchmark evidence behind this protocol (one family of terms, three small targets, ESMFold2 only) is in
`docs/mosaic_proposal.md` and `docs/mosaic_mutation_protocol_proposal.md`. Not measured: Boltz2 as the screening model, mismatched
screening/evaluation models, multi-State systems, and complexes larger than a few hundred residues.
