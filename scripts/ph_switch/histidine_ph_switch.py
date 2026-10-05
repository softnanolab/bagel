"""
De novo design of an acid-labile protein: folded near pH 7.4, unfolded at pH ~6.

PHYSICAL IDEA
-------------
Histidine is the only canonical side chain whose pKa (~6.0-6.5 free in solution)
sits between pH 6 and pH 7.4. The switch is built out of one thermodynamic fact:

  - At pH 7.4 the imidazole is neutral. A neutral imidazole is a perfectly
    acceptable core residue: it is roughly as apolar as tyrosine and packs well.
  - At pH 6 it protonates to the imidazolium cation. Burying a *charged* group in
    a low-dielectric core with no counter-ion and no hydrogen-bond acceptor costs
    a large desolvation penalty. That penalty is paid only by the folded state,
    because in the unfolded state the same cation is fully solvated.
  - Each buried histidine therefore subtracts from the folding free energy as the
    pH drops. Several of them, mutually repelling, subtract a lot.

So the design objective is: a well-folded, compact monomer whose core contains
several histidines that are (a) deeply buried and (b) NOT hydrogen bonded to a
carboxylate. Point (b) matters more than it looks: a His-Asp or His-Glu dyad
raises the histidine pKa and stabilises the protonated form, which would make the
protein *more* stable at low pH. That is the opposite switch.

IMPORTANT LIMITATION, STATED UP FRONT
-------------------------------------
None of BAGEL's folding oracles (ESMFold, ESMFold2, Chai1, Boltz2) take pH as an
input. There is no way to ask them to "fold this at pH 6". The pH dependence here
is therefore *encoded by construction* (buried, unpaired histidines) rather than
predicted. The `ProtonationMimicEnergy` term below is a deliberately crude
surrogate for the low-pH state: it folds a copy of the sequence with every
histidine replaced by arginine and penalises that copy for folding well. Arginine
is not imidazolium -- wrong size, wrong geometry, permanently charged -- so treat
this term as a soft "a positive charge at these positions should break the fold"
prior, not as a prediction of the acid state. Everything downstream still needs
wet-lab confirmation by pH-dependent circular dichroism or tryptophan
fluorescence.

Run with:
    python scripts/ph_switch/histidine_ph_switch.py
Set BAGEL_BACKEND=apptainer to run the oracle locally instead of on Modal.
"""

from __future__ import annotations

import os
import copy
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

import bagel as bg
from bagel.energies import EnergyTerm, residue_list_to_group
from bagel.chain import Residue
from bagel.oracles import OraclesResultDict
from bagel.oracles.folding import FoldingOracle
from bagel.constants import (
    probe_radius_water,
    max_theoretical_sasa_for_residues,
    max_residue_sasa,
)
from biotite.structure import sasa


# ----------------------------------------------------------------------------
# Helper: per-residue relative SASA, shared by the custom terms below.
# ----------------------------------------------------------------------------
def _relative_residue_sasa(structure: Any) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.int_], npt.NDArray[np.str_], npt.NDArray[np.float64]]:
    """
    Return (chain_ids, res_ids, res_names, relative_sasa) with one entry per residue.

    `relative_sasa` is the residue's total SASA divided by its maximum theoretical
    SASA (Tien et al. values already shipped in bagel.constants), clipped to [0, 1].
    0 means fully buried, 1 means fully exposed.
    """
    atom_sasa = sasa(structure, probe_radius=probe_radius_water)

    # Unique (chain_id, res_id) pairs in the order they appear in the structure.
    residue_ids = np.empty(
        len(structure),
        dtype=[('chain_id', structure.chain_id.dtype), ('res_id', structure.res_id.dtype)],
    )
    residue_ids['chain_id'] = structure.chain_id
    residue_ids['res_id'] = structure.res_id
    first_index = np.sort(np.unique(residue_ids, return_index=True)[1])

    chain_ids = residue_ids['chain_id'][first_index]
    res_ids = residue_ids['res_id'][first_index]
    res_names = structure.res_name[first_index]

    relative = np.zeros(len(first_index), dtype=float)
    for i, (chain_id, res_id, res_name) in enumerate(zip(chain_ids, res_ids, res_names)):
        atom_mask = (structure.chain_id == chain_id) & (structure.res_id == res_id)
        total = float(np.sum(atom_sasa[atom_mask]))
        max_sasa = max_theoretical_sasa_for_residues.get(res_name, max_residue_sasa)
        relative[i] = np.clip(total / max_sasa, 0.0, 1.0) if max_sasa > 0 else 0.0

    return chain_ids, res_ids, res_names, relative


# ----------------------------------------------------------------------------
# Energy term 1 -- the actual switch: bury histidines.
# ----------------------------------------------------------------------------
class BuriedHistidineEnergy(EnergyTerm):
    """
    Rewards histidines that are buried, saturating at `target_count` of them.

    Each histidine contributes a burial score in [0, 1]:

        burial_i = 1 - min(relative_sasa_i / sasa_cutoff, 1)

    i.e. 1 when the side chain is completely occluded, falling linearly to 0 once
    its relative SASA reaches `sasa_cutoff`. The returned energy is

        E = -min(sum_i burial_i, target_count) / target_count       in [-1, 0]

    The saturation matters. Without it the optimiser keeps stuffing in histidines
    forever and you end up with a polyhistidine blob that does not fold at any pH.
    Capping the reward at `target_count` says "six buried histidines is the goal,
    a seventh buys you nothing", which leaves the remaining core positions free to
    be filled by ordinary hydrophobics that pay for the fold at neutral pH.

    Parameters
    ----------
    oracle : FoldingOracle
        Oracle supplying the predicted structure.
    target_count : int, default=6
        Number of buried histidines that saturates the reward. For a ~100-residue
        single domain, 4-8 is a sensible window: below 4 the pH-6 destabilisation
        is likely too small to unfold the protein, above ~8 you are unlikely to
        retain a folded state at pH 7.4 at all.
    sasa_cutoff : float, default=0.15
        Relative SASA at and above which a histidine counts as fully exposed.
        0.15 is a conventional "buried" threshold.
    residues : list[Residue] or None
        Restrict the count to these positions (e.g. only the designed core).
        Default considers every residue in the state.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        target_count: int = 6,
        sasa_cutoff: float = 0.15,
        residues: list[Residue] | None = None,
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        name = 'buried_histidine' if name is None else f'buried_histidine_{name}'
        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = [residue_list_to_group(residues)] if residues is not None else []
        self.target_count = int(target_count)
        self.sasa_cutoff = float(sasa_cutoff)
        assert self.target_count > 0, 'target_count must be positive'
        assert 0.0 < self.sasa_cutoff <= 1.0, 'sasa_cutoff must lie in (0, 1]'
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be a FoldingOracle'

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        if len(structure) == 0:
            return 0.0, 0.0

        chain_ids, res_ids, res_names, relative = _relative_residue_sasa(structure)

        selected = np.full(len(res_ids), True)
        if len(self.residue_groups) > 0:
            group_chains, group_indices = self.residue_groups[0]
            selected = np.zeros(len(res_ids), dtype=bool)
            for cid in np.unique(group_chains):
                wanted = group_indices[group_chains == cid]
                selected |= (chain_ids == cid) & np.isin(res_ids, wanted)

        his_mask = (res_names == 'HIS') & selected
        if not np.any(his_mask):
            return 0.0, 0.0

        burial = 1.0 - np.minimum(relative[his_mask] / self.sasa_cutoff, 1.0)
        value = -float(min(burial.sum(), self.target_count)) / self.target_count
        return value, value * self.weight


# ----------------------------------------------------------------------------
# Energy term 2 -- keep the switch from being short-circuited.
# ----------------------------------------------------------------------------
class HistidineCarboxylateContactEnergy(EnergyTerm):
    """
    Penalises imidazole nitrogens sitting close to Asp/Glu carboxylate oxygens.

    A buried His-Asp or His-Glu pair is the classic pKa-raising motif: the
    carboxylate stabilises the protonated imidazolium, so the folded state
    becomes *more* stable as the pH falls. That inverts the switch you want, and
    an optimiser rewarded only for burying histidines will happily build these
    pairs because they are excellent buried hydrogen bonds. This term prices them
    out.

    Energy = min(n_contacts, max_contacts) / max_contacts, in [0, 1]; a contact is
    any ND1/NE2 atom within `distance_cutoff` of any OD1/OD2/OE1/OE2 atom.

    Set `weight` high enough that one contact outweighs the burial reward for one
    histidine, otherwise the trade is still worth making.
    """

    IMIDAZOLE_NITROGENS = ('ND1', 'NE2')
    CARBOXYLATE_OXYGENS = ('OD1', 'OD2', 'OE1', 'OE2')

    def __init__(
        self,
        oracle: FoldingOracle,
        distance_cutoff: float = 4.5,
        max_contacts: int = 4,
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        name = 'his_carboxylate' if name is None else f'his_carboxylate_{name}'
        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = []
        self.distance_cutoff = float(distance_cutoff)
        self.max_contacts = int(max_contacts)
        assert self.max_contacts > 0, 'max_contacts must be positive'
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be a FoldingOracle'

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        structure = oracles_result.get_structure(self.oracle)
        if len(structure) == 0:
            return 0.0, 0.0

        his_mask = (structure.res_name == 'HIS') & np.isin(structure.atom_name, self.IMIDAZOLE_NITROGENS)
        acid_mask = np.isin(structure.res_name, ('ASP', 'GLU')) & np.isin(
            structure.atom_name, self.CARBOXYLATE_OXYGENS
        )
        if not np.any(his_mask) or not np.any(acid_mask):
            return 0.0, 0.0

        his_coords = structure.coord[his_mask]
        acid_coords = structure.coord[acid_mask]
        distances = np.linalg.norm(his_coords[:, None, :] - acid_coords[None, :, :], axis=-1)
        n_contacts = int(np.count_nonzero(distances < self.distance_cutoff))

        value = float(min(n_contacts, self.max_contacts)) / self.max_contacts
        return value, value * self.weight


# ----------------------------------------------------------------------------
# Energy term 3 -- surrogate negative design of the protonated state.
# ----------------------------------------------------------------------------
class ProtonationMimicEnergy(EnergyTerm):
    """
    Folds a charge-mimic copy of the sequence and penalises it for folding well.

    Every histidine is substituted by `mimic_residue` (arginine by default) and
    the mutated sequence is sent through the same oracle. The returned energy is
    the mimic's pTM, so minimising the total energy drives the mimic's confidence
    *down* while the real sequence's own PTMEnergy drives its confidence up.

    Read the caveat in the module docstring before you weight this heavily.
    Arginine is a poor stand-in for imidazolium: it is larger, its charge is
    delocalised differently, and it is charged unconditionally. The term captures
    only the coarse statement "a permanent positive charge at these buried
    positions should be incompatible with this fold". It cannot tell you the
    midpoint of the transition, and a sequence that satisfies it may still have a
    pH midpoint of 4.5 or 7.0 rather than the 6 you asked for.

    Cost note: this doubles the number of oracle calls per Monte Carlo step. The
    per-sequence cache below means the extra fold is skipped whenever the
    histidine-masked sequence is unchanged, which happens often since most
    accepted mutations touch non-histidine positions.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        mimic_residue: str = 'R',
        metric: Literal['ptm', 'plddt'] = 'ptm',
        inheritable: bool = True,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        name = 'protonation_mimic' if name is None else f'protonation_mimic_{name}'
        super().__init__(name=name, oracle=oracle, inheritable=inheritable, weight=weight)
        self.residue_groups = []
        self.mimic_residue = mimic_residue
        self.metric = metric
        self._cache: dict[tuple[str, ...], float] = {}
        assert isinstance(self.oracle, FoldingOracle), 'Oracle must be a FoldingOracle'

    def compute(self, oracles_result: OraclesResultDict) -> tuple[float, float]:
        input_chains = oracles_result.get_input_chains(self.oracle)

        mimic_chains = copy.deepcopy(input_chains)
        n_substituted = 0
        for chain in mimic_chains:
            for residue in chain.residues:
                if residue.name == 'H':
                    residue.name = self.mimic_residue
                    n_substituted += 1

        # No histidines yet: nothing to say about the protonated state.
        if n_substituted == 0:
            return 0.0, 0.0

        key = tuple(chain.sequence for chain in mimic_chains)
        if key not in self._cache:
            mimic_result = self.oracle.predict(chains=mimic_chains)
            if self.metric == 'ptm':
                score = float(np.asarray(mimic_result.ptm).reshape(-1)[0])
            else:
                score = float(np.mean(mimic_result.local_plddt[0]))
            # Keep the cache bounded; this runs for thousands of MC steps.
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = score

        # Positive value: minimising the total energy pushes the mimic's
        # confidence down, i.e. makes the charged variant fold badly.
        value = self._cache[key]
        return value, value * self.weight


# ----------------------------------------------------------------------------
# The design run.
# ----------------------------------------------------------------------------
def build_system(length: int = 100, seed: int | None = None) -> tuple[bg.System, Any]:
    """Assemble the single-state system for the acid-labile design."""
    rng = np.random.default_rng(seed)
    backend = os.getenv('BAGEL_BACKEND', 'modal')
    print(f'Backend: {backend}')

    # Start from a random sequence. Cysteine is excluded by the default mutation
    # bias; a disulfide would clamp the fold shut and blunt the switch.
    alphabet = [aa for aa in bg.constants.aa_dict.keys() if aa != 'C']
    sequence = rng.choice(alphabet, size=length)
    residues = [bg.Residue(name=aa, chain_ID='A', index=i, mutable=True) for i, aa in enumerate(sequence)]
    chain = bg.Chain(residues)

    esmfold = bg.oracles.ESMFold(backend=backend)

    energy_terms = [
        # --- Fold quality: without these the switch terms design a disordered
        # peptide that is "unfolded at pH 6" only because it is unfolded always.
        bg.energies.PTMEnergy(oracle=esmfold, weight=2.0),
        bg.energies.OverallPLDDTEnergy(oracle=esmfold, weight=2.0),
        bg.energies.GlobularEnergy(oracle=esmfold, weight=0.5),

        # --- Keep a real hydrophobic core and a polar surface. The histidines
        # are a perturbation on an otherwise normal globular protein, not a
        # replacement for one.
        bg.energies.HydrophobicEnergy(oracle=esmfold, mode='surface', weight=2.0),
        # NOTE: this term and BuriedHistidineEnergy below pull against each other.
        # Histidine's Kyte-Doolittle index is -3.2, so every buried histidine makes
        # the burial-weighted hydropathy worse. That tension is deliberate -- it is
        # what stops the core becoming all histidine -- but it means the ratio of
        # these two weights is the single most important knob in this script. Sweep
        # it (0.1/8.0 through 1.5/2.0) before trusting any one run.
        bg.energies.HydropathyEnergy(oracle=esmfold, mode='core', weight=-0.5),

        # --- The switch itself.
        BuriedHistidineEnergy(oracle=esmfold, target_count=6, sasa_cutoff=0.15, weight=4.0),

        # --- Stop the optimiser building pKa-raising His/Asp dyads, which would
        # invert the switch. Weighted above the per-histidine burial reward.
        HistidineCarboxylateContactEnergy(oracle=esmfold, distance_cutoff=4.5, max_contacts=4, weight=3.0),

        # --- Surrogate negative design of the protonated state. Start low; this
        # term is the least trustworthy one here and it doubles oracle cost.
        ProtonationMimicEnergy(oracle=esmfold, mimic_residue='R', metric='ptm', weight=1.0),
    ]

    state = bg.State(name='folded_neutral_pH', chains=[chain], energy_terms=energy_terms)
    return bg.System([state]), esmfold


def main() -> Any:
    system, esmfold = build_system(length=100, seed=0)

    # Enrich histidine in the proposal distribution. Uniform sampling proposes H
    # about 5% of the time, so reaching six *buried* histidines by chance is slow.
    # This biases proposals only -- acceptance is still decided by the energy, so
    # it changes mixing speed, not the target distribution's optimum.
    bias = dict(bg.constants.mutation_bias_no_cystein)
    bias['H'] = 4.0 * bias['H']
    total = sum(bias.values())
    bias = {aa: p / total for aa, p in bias.items()}

    minimizer = bg.minimizer.SimulatedTempering(
        mutator=bg.mutation.Canonical(n_mutations=1, mutation_bias=bias),
        high_temperature=1.0,
        low_temperature=0.1,
        n_cycles=100,
        n_steps_low=200,
        n_steps_high=50,
        experiment_name='ph6_histidine_switch',
        callbacks=[
            bg.callbacks.DefaultLogger(log_interval=1),
            bg.callbacks.FoldingLogger(folding_oracle=esmfold, log_interval=50),
        ],
    )

    best_system = minimizer.minimize_system(system=system)
    print('Best sequence:', best_system.states[0].chains[0].sequence)
    return best_system


if __name__ == '__main__':
    main()
