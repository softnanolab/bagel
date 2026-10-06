# TNF-Alpha Multivalent Binder Design Scripts — Specification for Rebuild

**Date**: 2026-10-06  
**Status**: Initial implementation complete, but requires validation and correction of BAGEL API usage

## Executive Summary

This document specifies the design of 4 BAGEL scripts for de novo binders to TNF-alpha homo-trimers (human and mouse) that combine pH-dependent unfolding (via buried histidines) with specific binding affinity. The initial implementation (in this conversation) completed the workflow but may have incorrect or suboptimal use of BAGEL's energy terms and APIs. This document provides the specification for a correct rebuild.

---

## Design Goals

### Primary Objectives
1. **Multi-valent binding** to TNF-alpha homo-trimers (homotrimer of 3 identical monomers)
   - Target both human and mouse forms simultaneously in a single run (multi-state design)
   - Use **min ipSAE** (minimum PAE, bidirectional) to measure binding affinity
2. **pH-dependent unfolding** via buried histidines
   - Pack 4–6 histidines in the binder core (depending on scaffold)
   - No His-Asp/Glu pairs (these raise pKa and invert the switch)
3. **Two target modes**:
   - **Monomer epitope**: bind only to non-interface region of a TNF monomer (surface residues NOT within 9Å of other chains in the trimer)
   - **Full trimer**: bind to all residues of the homo-trimer

### Binder Architectures
1. **Generic binder**: 60–100 residues, all mutable
2. **DARPin scaffold**: 124 residues fixed, exactly 12 mutable positions (2 variable regions: ~6 positions each, rest framework)

### Matrix of Scripts
```
                Generic Binder          DARPin Scaffold
Epitope         generic_monomer         darpin_monomer
Trimer          generic_trimer          darpin_trimer
```

---

## Target Sequences

### Human TNF-α Monomer (157 residues)
```
VRSSSRTPSDKPVAHVVANPQAEGQLQWLNRRANALLANGVELRDNQLVVPSEGLYLIYSQVLFKGQGCPSTHVLLTHTISRIAVSYQTKVNLLSAIKSPCQRETPEGAEAKPWYEPIYLGGVFQLEKGDRLSAEINRPDYLDFAESGQVYFGIIAL
```

### Mouse TNF-α Monomer (156 residues)
```
LRSSSQNSSDKPVAHVVANHQVEEQLEWLSQRANALLANGMDLKDNQLVVPADGLYLVYSQVLFKGQGCPDYVLLTHTVSRFAISYQEKVNLLSAVKSPCPKDTPEGAELKPWYEPIYLGGVFQLEKGDQLSAEVNLPKYLDFAESGQVYFGVIAL
```

**Note**: Sequences are slightly different length; this is normal and must be handled.

---

## Energy Terms and Rationale

### Required Custom Energy Terms

#### 1. **BuriedHistidineEnergy**
- **Source**: From pH-switch script (reusable)
- **Function**: Rewards histidines with low relative SASA (buried in the core)
- **Implementation**:
  - Compute relative SASA for each residue: `rel_SASA = residue_SASA / max_theoretical_SASA`
  - For each His: burial score = `1 - min(rel_SASA / sasa_cutoff, 1.0)` where `sasa_cutoff = 0.15`
  - Sum burial scores, but **saturate at a target count** (e.g., 6 for generic, 4 for DARPin)
  - Energy = `-min(sum_burial_scores, target_count) / target_count` ∈ [-1, 0]
- **Parameters**:
  - `target_count`: 6 (generic) or 4 (DARPin) — saturation prevents over-packing
  - `sasa_cutoff`: 0.15 — conventional "buried" threshold
  - `weight`: 4.0
  - Apply to: binder residues only

#### 2. **HistidineCarboxylateContactEnergy**
- **Source**: From pH-switch script (reusable)
- **Function**: Penalizes imidazole nitrogens (ND1, NE2) within ~4.5Å of carboxylate oxygens (OD1, OD2, OE1, OE2) on Asp/Glu
- **Rationale**: His-Asp/Glu dyads raise histidine pKa and stabilize the protonated form, inverting the switch
- **Implementation**:
  - For each His residue: identify ND1/NE2 atoms
  - For each Asp/Glu residue: identify OD1/OD2/OE1/OE2 atoms
  - Count pairs within `distance_cutoff = 4.5Å`
  - Energy = `min(n_contacts, max_contacts) / max_contacts` where `max_contacts = 4`
- **Parameters**:
  - `distance_cutoff`: 4.5Å
  - `max_contacts`: 4 — cap the penalty so one contact doesn't outweigh the burial reward
  - `weight`: 3.0
  - Apply to: all residues (binder + target, but binder will mutate to avoid contacts)

#### 3. **ipSAE Energy Term (Binding Affinity)**
- **Function**: Minimize PAE (Predicted Aligned Error) between binder and target residues
- **Definition**: ipSAE = inverse PAE = (1 - normalized PAE) or simply minimize PAE directly
- **Implementation**:
  - Compute PAE matrix from folding oracle (ESMFold/ESMFold2/etc.)
  - For each pair (binder_residue, target_residue), extract PAE[binder_idx, target_idx]
  - Compute forward PAE: mean PAE from binder → target residues
  - Compute reverse PAE: mean PAE from target → binder residues
  - Use **min(forward_PAE, reverse_PAE)** — the stricter direction
  - Energy = min_PAE (lower is better, so minimize it)
- **Parameters**:
  - `weight`: -4.0 (negative because lower PAE is good; with negative weight, lower value = lower total energy)
  - For **epitope scripts**: restrict target residues to the non-interface set only
  - For **trimer scripts**: use all 3×157=471 (human) or 3×156=468 (mouse) residues

### Standard Stock Energy Terms

#### 4. **PTMEnergy** (Predicted TM-score)
- Measures confidence in overall structure
- `weight`: 2.0
- Applied to: whole system

#### 5. **OverallPLDDTEnergy** (Mean pLDDT)
- Per-residue confidence, averaged
- `weight`: 2.0
- Applied to: whole system

#### 6. **GlobularEnergy** (Compactness via moment of inertia)
- Drives structure toward a sphere
- `weight`: 0.5
- Applied to: backbone atoms (CA, N, C)
- Prevents chain collapse while still allowing disorder

---

## Epitope Identification

### Workflow
1. **Pre-processing**: Fold the TNF-α homo-trimer (human and mouse separately)
2. **Interface detection**: For each monomer, identify residues within 9Å of atoms from *other* chains
3. **Epitope**: Non-interface residues = all residues minus interface residues
4. **Output**: Save epitope indices (0-indexed) to `tnf_{species}_epitope.txt`

### Rationale
When designing against the "monomer epitope," we restrict the binding interface to regions that are NOT at the trimer interface. This allows us to study binding to the exposed surface without needing to break apart the trimer.

---

## Script Architecture

### General Structure for Each Design Script

```python
# Design a binder against TNF-alpha (human + mouse, multi-state)

def main(backend='modal', seed=0, n_steps=2000):
    """
    Parameters
    ----------
    backend : str
        'modal' (default, requires auth) or 'apptainer' (local GPU)
    seed : int
        Random seed for reproducibility
    n_steps : int
        For SimulatedTempering: 
        - n_steps is split into cycles: n_steps_low + n_steps_high per cycle
        - Typical: n_steps_low=20, n_steps_high=5, n_cycles=50 → ~1250 total steps
    """
    
    # 1. Create binder chain
    #    - Generic: random length 60-100, all mutable
    #    - DARPin: fixed 124 residues, exactly 12 mutable (2 var regions)
    
    # 2. Create states for BOTH human and mouse TNF
    #    - Each state: [binder, TNF_target_chains]
    #    - For epitope scripts: restrict ipSAE energy to epitope residues
    #    - For trimer scripts: use all TNF residues
    
    # 3. Energy terms (same for both states, but applied per-state)
    #    - PTMEnergy (2.0)
    #    - OverallPLDDTEnergy (2.0)
    #    - GlobularEnergy (0.5)
    #    - BuriedHistidineEnergy (4.0, target=6 or 4)
    #    - HistidineCarboxylateContactEnergy (3.0)
    #    - ipSAE/PAEEnergy (-4.0)
    
    # 4. Create multi-state system with both states
    
    # 5. SimulatedTempering optimizer
    #    - high_temp=0.2, low_temp=0.02
    #    - n_cycles=50 (default for full runs)
    #    - n_steps_low=20, n_steps_high=5 per cycle
    #    - Canonical mutator (1 mutation per step)
    #    - Callbacks: DefaultLogger + FoldingLogger
    
    # 6. Run and return best sequence
```

### Script Variants

**Generic vs. Monomer Epitope**
```python
# Load epitope residues from file (pre-computed by identify_epitope.py)
epitope_residues = load_epitope_residues(f'tnf_{species}_epitope.txt', tnf_chain)

# ipSAE energy: only measure PAE between binder and epitope
ipSAE_energy = ipSAEEnergy(
    oracle=esmfold,
    residues=[binder_residues, epitope_residues],  # ← epitope subset
    weight=-4.0
)
```

**Generic vs. Trimer**
```python
# Build homo-trimer: 3 copies of the monomer
tnf_chains = []
for i in range(3):
    residues = [bg.Residue(..., chain_ID=f'T{i}', ...) for aa in tnf_seq]
    tnf_chains.append(bg.Chain(residues=residues))

# ipSAE energy: measure PAE against all TNF residues
all_tnf_residues = [r for chain in tnf_chains for r in chain.residues]
ipSAE_energy = ipSAEEnergy(
    oracle=esmfold,
    residues=[binder_residues, all_tnf_residues],  # ← all 471 or 468 residues
    weight=-4.0
)
```

**DARPin Scaffold**
```python
# Fixed sequence + positions: 12 mutable only
# Structure: N-cap (immutable) | N-variable (6 mutable) | core (immutable) 
#            | C-variable (6 mutable) | C-cap (immutable)
# Total: 124 residues

# Example positions (adjust as needed):
#   0-20: N-cap (immutable)
#   21-26: N-variable (mutable)
#   27-97: core (immutable)
#   98-103: C-variable (mutable)
#   104-123: C-cap (immutable)

darpin_residues = make_darpin_scaffold(length=124, n_mutable=12)
# Or construct manually with alternating mutable/immutable flags
```

---

## Multi-State Design Pattern

Each script optimizes against **both** human and mouse TNF simultaneously:

```python
states = []
for species, tnf_seq in [('human', TNF_HUMAN), ('mouse', TNF_MOUSE)]:
    # Create TNF chain(s)
    # Create energy terms (same terms, applied to this state)
    state = bg.State(name=f'binder_vs_tnf_{species}_...', chains=[binder_chain, ...], energy_terms=[...])
    states.append(state)

system = bg.System(states)
# Minimizer optimizes total energy across all states
# → binder must satisfy both human and mouse constraints simultaneously
```

**Rationale**: This ensures the binder is **panspecific** (works for both species) without requiring separate runs.

---

## Known Issues and Limitations in Initial Implementation

### Potential Bugs/Limitations
1. **ipSAE energy term**: The initial `ipSAEEnergy` class may not correctly handle multi-chain systems or may have issues with residue group indexing. **Recommendation**: Verify against BAGEL's `PAEEnergy` term and ensure bidirectional PAE (forward + reverse, taking min) is correctly implemented.

2. **DARPin scaffold**: The specific residue positions (21-26 for N-variable, 98-103 for C-variable) are arbitrary placeholders. **Recommendation**: Validate against an actual DARPin structure or consensus sequence; confirm that 12 mutable positions is sufficient for meaningful variation.

3. **Epitope masking**: The `identify_epitope.py` script uses `CellList` and distance queries, which may be slow or incorrectly mask residues. **Recommendation**: Validate that the output epitope residues are biologically sensible (surface-facing, away from trimer interface).

4. **Energy term weights**: The weights (2.0, 4.0, 3.0, -4.0) are copied from the pH-switch script but may not be optimal for binding design. **Recommendation**: Start with these, but be prepared to sweep over weight ratios (especially burial vs. hydropathy trade-off).

5. **Optimizer parameters**: `n_cycles=50, n_steps_low=20, n_steps_high=5` total ~1250 steps, which may be too short or too long. **Recommendation**: Smoke-test with minimal counts (n_cycles=1), then scale up based on convergence.

---

## Correct BAGEL API Usage (Checklist for Rebuild)

### Energy Terms
- [ ] Verify `PTMEnergy`, `OverallPLDDTEnergy`, `GlobularEnergy` exist in current BAGEL and have correct signatures
- [ ] Check if `PAEEnergy` already supports bidirectional PAE or if a custom term is needed
- [ ] Ensure residue-group convention is `[group_a, group_b]` for interface terms
- [ ] Verify `inheritable` flag is correctly set (False for interface/binding terms; True for others)

### Custom Terms
- [ ] `BuriedHistidineEnergy`: Check that relative SASA computation matches BAGEL's `sasa()` function
- [ ] `HistidineCarboxylateContactEnergy`: Verify atom name and residue type queries work as expected
- [ ] ipSAE/PAE: Ensure PAE matrix is indexed correctly and bidirectional min is computed

### Chains and States
- [ ] Verify `chain_ID` strings are handled correctly for multi-chain systems (trimers)
- [ ] Confirm residue indices (0-based in BAGEL) align with returned structures
- [ ] Test multi-state system with >2 states to ensure energy summation is correct

### Optimizer
- [ ] `SimulatedTempering` signature: `high_temperature`, `low_temperature`, `n_cycles`, `n_steps_low`, `n_steps_high`
- [ ] `Canonical` mutation protocol: `n_mutations=1` per step, with optional `mutation_bias`
- [ ] Callbacks: `DefaultLogger(log_interval=5)`, `FoldingLogger(log_interval=20)`

### Oracles
- [ ] ESMFold (or ESMFold2) must return PAE in result
- [ ] Modal backend requires authentication; apptainer runs locally on GPU

---

## Next Steps for Rebuild

1. **Validate the ipSAE energy term** against BAGEL's existing PAEEnergy; may just wrap or adjust it
2. **Test epitope masking** on a single human TNF trimer fold; visualize masked vs. full regions
3. **Smoke-test each of the 4 scripts** with minimal parameters (n_cycles=1) to catch import/setup errors
4. **Tune weights** if convergence is poor (esp. burial vs. hydropathy trade-off for generic binders)
5. **Run full production** with 2000+ steps per script, multiple seeds, and compare results

---

## Files Provided

- `identify_epitope.py` — epitope identification (may need refinement)
- `binder_generic_monomer_epitope.py` — generic vs. epitope (verify PAE energy term)
- `binder_generic_trimer.py` — generic vs. trimer (verify multi-chain setup)
- `binder_darpin_monomer_epitope.py` — DARPin vs. epitope (verify scaffold)
- `binder_darpin_trimer.py` — DARPin vs. trimer (verify multi-chain + scaffold)
- `utils.py` — ipSAE energy, DARPin scaffold builder, epitope loader
- `ph_switch_histidine.py` — imported pH-switch utilities (BuriedHistidineEnergy, HistidineCarboxylateContactEnergy)
- `verify_setup.py` — offline verification (imports + structure checks)
- `README.md` — user-facing workflow guide

---

## Questions for Rebuild

- Should DARPin variable regions be longer/shorter? (Currently 6 positions each, 12 total)
- Are there existing BAGEL examples of multi-chain design (trimers, etc.) to reference?
- What is the correct way to handle PAE for multi-chain systems in BAGEL?
- Should epitope be identified per-monomer or globally across the trimer?
- Any known issues with SimulatedTempering on multi-state systems?

