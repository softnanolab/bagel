# Mini-Enzymes PLM Optimization Scripts

Monte Carlo optimization scripts for generating miniaturized enzyme variants using ESM-2 embeddings. These scripts accompany the paper:

> **An Energy Landscape Approach to Miniaturizing Enzymes using Protein Language Model Embeddings**
> Jakub Lála, Harsh Agrawal, Fanfei Dong, Jude Wells, Stefano Angioletti-Uberti (2026).

Each script preserves the catalytic and functional residues (plus a buffer zone) while allowing the rest of the sequence to be mutated via grand canonical MC sampling with `EmbeddingsSimilarityEnergy` and `ChemicalPotentialEnergy`.

## Enzymes

| Script | Enzyme | PDB | Catalytic Residues |
|--------|--------|-----|-------------------|
| `petase.py` | PETase (PET-degrading enzyme) | 5XJH | S160, D206, H237 |
| `vioA.py` | VioA (tryptophan 2-monooxygenase) | 6FW9 | R64, K269, Y309 |
| `protease.py` | Subtilisin Carlsberg (serine protease) | 1SBC | D32, H64, N155, S221 |
| `taq.py` | Taq DNA polymerase | 1TAQ | D610, F667, D785, E786 |

## Usage

```bash
# Run with Modal (recommended)
python petase.py --use_modal=True --n_steps=100000

# Run locally
python petase.py --use_modal=False --n_steps=100000
```

## Data

All generated sequences and data from the paper are available on Zenodo: https://doi.org/10.5281/zenodo.18854113
