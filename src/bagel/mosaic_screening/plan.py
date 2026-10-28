"""
Static analysis of a bagel :class:`~bagel.system.System` for mosaic screening.

For every State that contains a designed chain, :func:`build_plans` works out

* which chain is designed and in which order the chains are handed to mosaic (designed chain first, then the others in
  State order: every official mosaic model assumes this),
* which energy terms can be evaluated on a mosaic prediction (and, for the rest, why not),
* the guardrails (see :class:`~bagel.mosaic_screening.errors.MosaicIncompatibleError`).

It is pure Python: it needs neither mosaic nor JAX nor a GPU.

Terms are re-fetched from the *current* State by index at every step (:meth:`StatePlan.terms_of`), because bagel's
minimizer hands protocols deep copies of the system, so term objects from the first step are not the ones of later steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..chain import Chain
from ..constants import aa_dict
from ..energies import (
    BuriedHistidineEnergy,
    EnergyTerm,
    GlobularEnergy,
    HistidineCarboxylateContactEnergy,
    HydrophobicEnergy,
    HydropathyEnergy,
    LISEnergy,
    OverallPLDDTEnergy,
    PAEEnergy,
    PLDDTEnergy,
    PTMEnergy,
    SurfaceAreaEnergy,
    ipSAEEnergy,
)
from ..oracles.base import Oracle
from ..state import State
from ..system import System
from .errors import MosaicIncompatibleError
from .models import MosaicModelSpec, default_spec_for

#: Energy terms that bagel can compute from a mosaic prediction using only ``local_plddt``, ``pae`` and ``ptm`` (the structure
#: is used only to turn their residue groups into masks). Exact types, not subclasses, because a subclass may override ``compute``.
CONFIDENCE_TERMS: tuple[type[EnergyTerm], ...] = (
    PTMEnergy,
    PLDDTEnergy,
    OverallPLDDTEnergy,
    PAEEnergy,
    LISEnergy,
    ipSAEEnergy,
)

#: Terms that read atom coordinates. They are evaluated on a structure rebuilt from mosaic's all-atom output (see
#: :func:`bagel.mosaic_screening.adapter.structure_from_atom37`), which has the same heavy atoms, names and order as the
#: structures bagel gets from its oracles.
STRUCTURE_TERMS: tuple[type[EnergyTerm], ...] = (
    SurfaceAreaEnergy,
    HydrophobicEnergy,
    HydropathyEnergy,
    GlobularEnergy,
    BuriedHistidineEnergy,
    HistidineCarboxylateContactEnergy,
)

SUPPORTED_TERMS: tuple[type[EnergyTerm], ...] = CONFIDENCE_TERMS + STRUCTURE_TERMS

_REASONS = {
    'BinderRMSDEnergy': 'compares structures across States and needs all-atom coordinates',
    'TemplateMatchEnergy': 'needs all-atom coordinates',
    'EmbeddingsSimilarityEnergy': 'needs language-model embeddings (no embedding oracle in this version)',
    'ChemicalPotentialEnergy': 'depends on chain length changes (GrandCanonical), which this protocol does not support',
}


@dataclass
class TermStatus:
    """Whether one energy term of one State enters the pseudo-energy."""

    state_name: str
    name: str
    weight: float
    used: bool
    reason: str = ''
    spec_label: str = ''


@dataclass
class ScoringGroup:
    """Terms of one State that are evaluated on the same mosaic model (hence one batched mosaic call)."""

    spec: MosaicModelSpec
    term_indices: list[int]  # positions in ``state.energy_terms``
    term_names: list[str]
    needs_structure: bool = False  # some term reads atom coordinates, so mosaic must return them


@dataclass
class StatePlan:
    """Everything the protocol needs to know about one State, computed once."""

    state_index: int
    state_name: str
    design_chain_id: str
    chain_order: list[str]  # design chain first, then the other chains in State order
    chain_lengths: dict[str, int]
    mutable_positions: list[int]
    fixed_sequences: dict[str, str]  # sequences of the non-designed chains (they must not change)
    groups: list[ScoringGroup]
    statuses: list[TermStatus] = field(default_factory=list)

    @property
    def target_sequences(self) -> list[str]:
        return [self.fixed_sequences[c] for c in self.chain_order[1:]]

    def ordered_chains(self, state: State) -> list[Chain]:
        by_id = {c.chain_ID: c for c in state.chains}
        return [by_id[c] for c in self.chain_order]

    def terms_of(self, state: State, group: ScoringGroup) -> list[EnergyTerm]:
        terms = []
        for index, name in zip(group.term_indices, group.term_names):
            term = state.energy_terms[index]
            if term.name != name:
                raise MosaicIncompatibleError(
                    f"Energy terms of state '{state.name}' changed after the mosaic plan was built "
                    f"(expected '{name}' at position {index}, found '{term.name}')."
                )
            terms.append(term)
        return terms

    def check_current(self, state: State) -> None:
        """Raise if the State no longer matches the plan (most commonly because a chain length changed)."""
        by_id = {c.chain_ID: c for c in state.chains}
        for chain_id in self.chain_order:
            if chain_id not in by_id:
                raise MosaicIncompatibleError(f"Chain '{chain_id}' disappeared from state '{state.name}'.")
            expected = self.chain_lengths[chain_id]
            if by_id[chain_id].length != expected:
                raise MosaicIncompatibleError(
                    f"Chain '{chain_id}' of state '{state.name}' has length {by_id[chain_id].length} but the mosaic "
                    f'plan was built for {expected}. Mosaic losses are compiled for fixed shapes, so moves that '
                    'change chain lengths (GrandCanonical) cannot be combined with MosaicScreening.'
                )
        for chain_id, sequence in self.fixed_sequences.items():
            if by_id[chain_id].sequence != sequence:
                raise MosaicIncompatibleError(
                    f"Chain '{chain_id}' of state '{state.name}' is not designed but its sequence changed; the mosaic "
                    'features of a State are built for fixed partner sequences.'
                )


def _validate_chain(chain: Chain, state_name: str) -> None:
    for position, residue in enumerate(chain.residues):
        if residue.index != position:
            raise MosaicIncompatibleError(
                f"Chain '{chain.chain_ID}' of state '{state_name}': residue at position {position} has index "
                f'{residue.index}. Residue masks would address the wrong tokens (residues were added or removed, '
                'as GrandCanonical does).'
            )
        if residue.name not in aa_dict:
            raise MosaicIncompatibleError(
                f"Chain '{chain.chain_ID}' of state '{state_name}': '{residue.name}' is not one of the 20 standard "
                'amino acids supported by the mosaic models.'
            )


def _classify_term(
    term: EnergyTerm, models: Mapping[Oracle, MosaicModelSpec], structure_terms: bool
) -> tuple[MosaicModelSpec | None, str]:
    """Return (spec, '') if the term can be evaluated on a mosaic prediction, else (None, reason)."""
    if type(term) in STRUCTURE_TERMS and not structure_terms:
        return None, (
            f'{type(term).__name__}: evaluated on mosaic coordinates only if use_structure_terms=True '
            '(agreement with the oracle structure is weak for most structural terms)'
        )
    if type(term) not in SUPPORTED_TERMS:
        reason = _REASONS.get(
            type(term).__name__, 'needs the all-atom structure, which is not reconstructed from mosaic'
        )
        return None, f'{type(term).__name__}: {reason}'
    spec = models.get(term.oracle) or default_spec_for(term.oracle)
    if spec is None:
        return None, f'oracle {type(term.oracle).__name__} has no mosaic counterpart (give one in `models`)'
    return spec, ''


def build_plans(
    system: System, models: Mapping[Oracle, MosaicModelSpec] | None = None, structure_terms: bool = False
) -> list[StatePlan | None]:
    """
    Build one :class:`StatePlan` per State of ``system`` (``None`` for States without a designed chain, which no move
    can change and which therefore never need mosaic).

    Raises
    ------
    MosaicIncompatibleError
        If a State has more than one designed chain, if none of the terms of a State with a designed chain can be
        evaluated on a mosaic prediction, or if a chain is incompatible with mosaic (see :func:`_validate_chain`).
    """
    models = models or {}
    plans: list[StatePlan | None] = []
    for state_index, state in enumerate(system.states):
        for chain in state.chains:
            _validate_chain(chain, state.name)
        designed = [c for c in state.chains if len(c.mutable_residues) > 0]
        if len(designed) == 0:
            plans.append(None)
            continue
        if len(designed) > 1:
            raise MosaicIncompatibleError(
                f"State '{state.name}' has {len(designed)} chains with mutable residues "
                f'({[c.chain_ID for c in designed]}); MosaicScreening supports one designed chain per State, because '
                'mosaic injects the designed sequence into the first chain only and the features of the other chains '
                'depend on their sequences.'
            )
        if len({c.chain_ID for c in state.chains}) != len(state.chains):
            raise MosaicIncompatibleError(f"Chain IDs of state '{state.name}' must be unique.")
        for term in state.energy_terms:
            # mosaic models one prediction per term, so it cannot stand in for an average over structures
            multiple = [o for o in term.oracles if getattr(o, 'n_samples', 1) > 1]
            if len(term.oracles) > 1 or multiple:
                raise MosaicIncompatibleError(
                    f"Energy term '{term.name}' of state '{state.name}' averages over several structures "
                    f'({len(term.oracles)} oracles, n_samples>1 for {[type(o).__name__ for o in multiple]}); '
                    'MosaicScreening models a single prediction per term. Use oracles with n_samples=1 and one oracle '
                    'per term, or another mutation protocol.'
                )
        design = designed[0]
        others = [c for c in state.chains if c is not design]
        statuses: list[TermStatus] = []
        by_spec: dict[MosaicModelSpec, ScoringGroup] = {}
        for index, term in enumerate(state.energy_terms):
            spec, reason = _classify_term(term, models, structure_terms)
            statuses.append(
                TermStatus(
                    state_name=state.name,
                    name=term.name,
                    weight=float(term.weight),
                    used=spec is not None,
                    reason=reason,
                    spec_label=spec.label if spec is not None else '',
                )
            )
            if spec is not None:
                group = by_spec.setdefault(spec, ScoringGroup(spec=spec, term_indices=[], term_names=[]))
                group.term_indices.append(index)
                group.term_names.append(term.name)
                group.needs_structure = group.needs_structure or type(term) in STRUCTURE_TERMS
        if not by_spec:
            raise MosaicIncompatibleError(
                f"No energy term of state '{state.name}' has a mosaic counterpart, so there is nothing to screen "
                f'with. Unused terms: {[(s.name, s.reason) for s in statuses]}'
            )
        plans.append(
            StatePlan(
                state_index=state_index,
                state_name=state.name,
                design_chain_id=design.chain_ID,
                chain_order=[design.chain_ID, *[c.chain_ID for c in others]],
                chain_lengths={c.chain_ID: c.length for c in state.chains},
                mutable_positions=[int(i) for i in design.mutable_residue_indexes],
                fixed_sequences={c.chain_ID: c.sequence for c in others},
                groups=list(by_spec.values()),
                statuses=statuses,
            )
        )
    if all(p is None for p in plans):
        raise MosaicIncompatibleError('No State of the system has a chain with mutable residues.')
    return plans


@dataclass
class Coverage:
    n_used: int
    n_total: int
    weight_used: float
    weight_total: float

    @property
    def fraction_terms(self) -> float:
        return self.n_used / self.n_total if self.n_total else 1.0

    @property
    def fraction_weight(self) -> float:
        return self.weight_used / self.weight_total if self.weight_total else 1.0


def coverage(statuses: list[TermStatus]) -> Coverage:
    """Terms used / total and sum of |weight| used / total."""
    used = [s for s in statuses if s.used]
    return Coverage(
        n_used=len(used),
        n_total=len(statuses),
        weight_used=sum(abs(s.weight) for s in used),
        weight_total=sum(abs(s.weight) for s in statuses),
    )


def format_report(plans: list[StatePlan | None], extra: dict[str, Any] | None = None) -> str:
    """Text of ``Mosaic_pseudo_energy.txt``: which terms make up the pseudo-energy and which do not."""
    statuses = [s for p in plans if p is not None for s in p.statuses]
    cov = coverage(statuses)
    lines = [
        'Mosaic pseudo-energy: terms used to rank candidate mutations (the accept/reject step always uses the full bagel energy)',
        '',
        f'Coverage by number of terms : {cov.n_used}/{cov.n_total} ({100 * cov.fraction_terms:.0f}%)',
        f'Coverage by weight          : {cov.weight_used:.4g}/{cov.weight_total:.4g} (sum of |weight|, {100 * cov.fraction_weight:.0f}%)',
        '',
    ]
    for plan in plans:
        if plan is None:
            continue
        sc = coverage(plan.statuses)
        lines.append(
            f"State '{plan.state_name}': designed chain {plan.design_chain_id}, chain order for mosaic {plan.chain_order}, "
            f'{sc.n_used}/{sc.n_total} terms, {100 * sc.fraction_weight:.0f}% of the weight'
        )
        for status in plan.statuses:
            if status.used:
                lines.append(f'  USED    {status.name:<40s} weight {status.weight:<8.4g} model {status.spec_label}')
        for status in plan.statuses:
            if not status.used:
                lines.append(f'  UNUSED  {status.name:<40s} weight {status.weight:<8.4g} {status.reason}')
        lines.append('')
    for key, value in (extra or {}).items():
        lines.append(f'{key}: {value}')
    return '\n'.join(lines) + '\n'
