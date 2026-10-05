# TNF-alpha Multivalent Binder Design

This directory contains 4 BAGEL scripts for designing binders to TNF-alpha (human and mouse) with pH-sensitive properties.

## Overview

The goal is to design binders that:
1. **Bind to TNF-alpha trimers** (both human and mouse forms) via min ipSAE (minimized PAE)
2. **Pack buried histidines** in the core for pH-dependent unfolding
3. **Avoid His-Asp/Glu pairs** that would invert the pH switch
4. Work in **two target modes**: non-interface region of a monomer, or the full trimer

## Scripts

### Epitope Identification
- **`identify_epitope.py`** — Pre-processing step. Folds the TNF-alpha homo-trimer and identifies interface residues (within 9 Å of other chains). Outputs epitope files for use in monomer-epitope scripts.

### Design Scripts (4 total)

| Script | Binder Type | Target | Description |
|--------|-------------|--------|-------------|
| `binder_generic_monomer_epitope.py` | Generic (60-100 aa) | TNF monomer epitope | Binds to non-interface region only |
| `binder_generic_trimer.py` | Generic (60-100 aa) | Full TNF trimer | Binds to all residues |
| `binder_darpin_monomer_epitope.py` | DARPin (124 aa, 12 mutable) | TNF monomer epitope | Scaffold-based, minimal mutable positions |
| `binder_darpin_trimer.py` | DARPin (124 aa, 12 mutable) | Full TNF trimer | Scaffold-based for trimer binding |

## Workflow

### 1. Identify Epitope (if using monomer scripts)
```bash
python identify_epitope.py --backend modal
```
Creates:
- `tnf_human_epitope.txt` — epitope residues for human TNF
- `tnf_mouse_epitope.txt` — epitope residues for mouse TNF

### 2. Run Design Scripts
Each design script has the same interface:
```bash
python <script_name> --backend modal --seed <N> --n_steps <steps>
```

Examples:
```bash
# Generic binder vs monomer epitope, seed 0
python binder_generic_monomer_epitope.py --backend modal --seed 0 --n_steps 2000

# DARPin vs trimer, seed 1
python binder_darpin_trimer.py --backend modal --seed 1 --n_steps 2000
```

### 3. Smoke Test (before full runs)
```bash
python smoke_test.py
```
Runs all 4 scripts with minimal steps to verify they work.

## Energy Terms

All scripts use:
- **PTMEnergy** (weight=2.0) — confidence in fold
- **OverallPLDDTEnergy** (weight=2.0) — per-residue confidence
- **GlobularEnergy** (weight=0.5) — compactness
- **BuriedHistidineEnergy** (weight=4.0, target=6 for generic, 4 for DARPin) — bury histidines in the core, saturating to prevent over-packing
- **HistidineCarboxylateContactEnergy** (weight=3.0) — prevent His-Asp/Glu pairs that would raise pKa
- **ipSAEEnergy** (weight=-4.0) — minimize PAE (maximize binding confidence) to the target

## Optimizer

**SimulatedTempering** with:
- High temperature: 0.2
- Low temperature: 0.02
- Cycles: 50 (default, for full runs)
- Steps per cycle: 20 low, 5 high

## Multi-Target Design

Each design script optimizes against **both human and mouse TNF-alpha simultaneously** using a multi-state system. This drives the binder to be panspecific (binding both species).

## Utilities

- **`utils.py`** — Custom ipSAE energy term, DARPin scaffold builder, epitope loader
- **`ph_switch_histidine.py`** — Imported from `/scripts/ph_switch/`: BuriedHistidineEnergy, HistidineCarboxylateContactEnergy

## Expected Output

Each design script outputs:
- Log files in `runs/<experiment_name>/` with per-term energies and fold metrics
- Final best binder sequence printed to console
- FASTA output (if saving is enabled in callbacks)

## Notes

- **DARPin scaffolds** have only 12 mutable positions (2 variable regions of 6 each); the rest is fixed framework
- **Generic binders** have all residues mutable (random length 60-100)
- **Monomer-epitope scripts** require running `identify_epitope.py` first
- Both human and mouse TNF are optimized against in every run — no separate per-species scripts needed
