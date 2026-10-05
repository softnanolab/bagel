# -----------------------------------------------------------------------------
# Generated with the assistance of an AI agent (Claude, via the
# `bagel-script-builder` skill). Review before running — you are responsible for
# its correctness. Generated: 2025-01-XX.
# -----------------------------------------------------------------------------
"""
Shared utilities for TNF-alpha binder design: custom ipSAE energy term, 
DARPin scaffold, and epitope loading.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Any

import bagel as bg
from bagel.energies import EnergyTerm
from bagel.oracles import OraclesResultDict
from bagel.oracles.folding import FoldingOracle


class ipSAEEnergy(EnergyTerm):
    """
    Inverse PAE (ipSAE) energy for measuring binding affinity.
    
    ipSAE = 1 - mean(PAE) for residues in the two groups. Higher ipSAE
    means better binding (lower predicted error). This term is minimized
    when ipSAE is high, driving down PAE and improving binding.
    
    Uses the minimum of forward (group_a→group_b) and reverse (group_b→group_a)
    directions, as specified in the user's binding definition.
    """
    
    def __init__(
        self,
        oracle: FoldingOracle,
        residues: list[list[bg.Residue]],
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Initialize ipSAE energy.
        
        Parameters
        ----------
        oracle : FoldingOracle
            Folding oracle (e.g., ESMFold) that returns PAE.
        residues : list[list[Residue]]
            Two groups: [binder_residues, target_residues].
        weight : float
            Weight of this energy term.
        name : str or None
            Optional name suffix.
        """
        name = 'ipSAE' if name is None else f'ipSAE_{name}'
        super().__init__(name=name, oracle=oracle, inheritable=False, weight=weight)
        
        # Store the two residue groups
        if len(residues) != 2:
            raise ValueError('ipSAE requires exactly 2 residue groups [binder, target]')
        
        from bagel.energies import residue_list_to_group
        self.residue_groups = [residue_list_to_group(residues[0]), 
                               residue_list_to_group(residues[1])]
        
        assert isinstance(self.oracle, FoldingOracle)
        assert 'pae' in self.oracle.result_class.model_fields
    
    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        """Compute ipSAE binding energy."""
        folding_result = oracles_result[self.oracle]
        pae = folding_result.pae[0]  # [n_res, n_res]
        
        group_a_ids, group_a_indices = self.residue_groups[0]
        group_b_ids, group_b_indices = self.residue_groups[1]
        
        if len(group_a_indices) == 0 or len(group_b_indices) == 0:
            return 0.0, 0.0
        
        # PAE[i, j] is error at residue i aligned to j
        # Forward direction: binder→target
        forward_pae = pae[np.ix_(group_a_indices, group_b_indices)]
        forward_mean_pae = np.mean(forward_pae) if forward_pae.size > 0 else 0.0
        
        # Reverse direction: target→binder
        reverse_pae = pae[np.ix_(group_b_indices, group_a_indices)]
        reverse_mean_pae = np.mean(reverse_pae) if reverse_pae.size > 0 else 0.0
        
        # Min ipSAE (better binding = lower PAE)
        min_mean_pae = min(forward_mean_pae, reverse_mean_pae)
        
        # ipSAE = 1 - PAE, but we want to minimize energy when ipSAE is high
        # So energy = PAE (minimizing PAE is good)
        value = float(min_mean_pae)
        return value, value * self.weight


def load_epitope_residues(filename: str, chain: bg.Chain) -> list[bg.Residue]:
    """
    Load epitope residue indices from a file (space-separated integers).
    Return the Residue objects for those indices.
    """
    with open(filename) as f:
        indices = list(map(int, f.read().split()))
    
    epitope = [chain.residues[i] for i in indices]
    return epitope


def make_darpin_scaffold(length: int = 124, n_variable: int = 12) -> list[bg.Residue]:
    """
    Create a DARPin scaffold with 2 variable regions and fixed framework.
    
    Total length: 124 residues
    Variable regions: 12 mutable positions total (distributed between 2 regions)
    
    Structure (approximate):
      - N-cap (immutable): residues 0-20
      - N-variable (6 mutable): residues 21-26
      - Core (immutable): residues 27-97
      - C-variable (6 mutable): residues 98-103
      - C-cap (immutable): residues 104-123
    """
    # Start with a random sequence
    sequence = np.random.choice(list(bg.constants.aa_dict.keys()), size=length)
    
    # Define mutable positions: 6 in N-variable, 6 in C-variable
    n_variable_start, n_variable_end = 21, 27  # 6 positions
    c_variable_start, c_variable_end = 98, 104  # 6 positions
    
    residues = []
    for i, aa in enumerate(sequence):
        mutable = (n_variable_start <= i < n_variable_end) or (c_variable_start <= i < c_variable_end)
        residues.append(bg.Residue(name=aa, chain_ID='D', index=i, mutable=mutable))
    
    return residues
