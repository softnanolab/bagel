"""
:class:`MosaicScreening`: a :class:`~bagel.mutation.MutationProtocol` that uses mosaic forward passes to choose which
substitution to propose.
"""

from __future__ import annotations

from collections import OrderedDict
import logging
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..constants import mutation_bias_no_cystein
from ..mutation import Canonical, GrandCanonical, Mutation, MutationProtocol, MutationRecord
from ..oracles.base import Oracle
from ..system import System
from .adapter import pseudo_energy
from .backends import MosaicBackend, ModalMosaicBackend
from .errors import MosaicIncompatibleError
from .models import MosaicModelSpec
from .plan import StatePlan, build_plans, coverage, format_report

logger = logging.getLogger(__name__)


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (average ranks for ties); nan if either side is constant."""
    a, b = pd.Series(x).rank(), pd.Series(y).rank()
    if a.nunique() < 2 or b.nunique() < 2:
        return float('nan')
    return float(a.corr(b))


def correlation_upper95(rho: float, n: int) -> float:
    """Approximate upper end of the 95% interval of a correlation from ``n`` pairs (Fisher z, SE = 1/sqrt(n-3))."""
    if n <= 3 or not math.isfinite(rho):
        return float('nan')
    rho = min(max(rho, -0.999999), 0.999999)
    return math.tanh(math.atanh(rho) + 1.96 / math.sqrt(n - 3))


class MosaicScreening(MutationProtocol):
    """
    Substitution protocol that screens random candidate mutations with a fast mosaic forward pass and proposes the one
    with the lowest *pseudo-energy*.

    Every step samples ``n_candidates`` single substitutions at random, scores the mutated sequences (and the current
    one) with mosaic in one batched call per State, and proposes the best-scoring candidate. The proposal is then folded
    and evaluated with bagel's own oracles and the full energy, and accepted or rejected by the minimizer as for any
    other protocol: **mosaic only chooses what is proposed; the energy that is optimised is never replaced.** If the
    pseudo-energy ranks candidates badly the cost is efficiency (proposals no better than random, plus the screening
    time), not correctness.

    The pseudo-energy of a State is the weighted sum of those of its energy terms that can be evaluated on a mosaic
    prediction, computed by bagel's own term code on the mosaic output (residue groups included). The supported terms
    are ``PTMEnergy``, ``PLDDTEnergy``, ``OverallPLDDTEnergy``, ``PAEEnergy``, ``LISEnergy`` and ``ipSAEEnergy``, on an
    ``ESMFold2`` or ``Boltz2`` oracle; with ``use_structure_terms=True`` also ``SurfaceAreaEnergy``, ``HydrophobicEnergy``,
    ``HydropathyEnergy``, ``GlobularEnergy``, ``BuriedHistidineEnergy`` and ``HistidineCarboxylateContactEnergy``. Other terms are simply left out of the pseudo-energy; which terms were used, which
    not and why, and the coverage (terms used/total and sum of |weight| used/total) are logged and written to
    ``Mosaic_pseudo_energy.txt`` in the log folder of the run. The score of a candidate is the sum, over the States that
    contain the mutated chain, of the pseudo-energy change of that State (the real total energy is a sum over States too).

    Constraints (``MosaicIncompatibleError`` is raised before any call to mosaic):

    * a State has at most one chain with mutable residues; mosaic writes the designed sequence into the first chain only,
    * no move that changes a chain length: do not use :class:`~bagel.mutation.GrandCanonical` with this protocol,
    * 20 standard amino acids only, residue ``index`` equal to the position in the chain,
    * at least one energy term per designed State must be supported.

    The proposal distribution is not symmetric and no Hastings correction is applied: this protocol is for optimisation,
    not for exact sampling.

    Parameters
    ----------
    backend : MosaicBackend, optional
        Where mosaic runs. :class:`~bagel.mosaic_screening.backends.ModalMosaicBackend` (the default; needs the Modal app
        of ``modal_service.py`` to be deployed) or :class:`~bagel.mosaic_screening.backends.SubprocessMosaicBackend`
        (local GPU, or a SLURM/PBS job through Apptainer).
    models : Mapping[Oracle, MosaicModelSpec], optional
        Mosaic model and settings used for the terms of each bagel oracle. Oracles that are not in the mapping get
        :func:`~bagel.mosaic_screening.models.default_spec_for` (ESMFold2: 3 recycling and 14 sampling steps; Boltz2: 1
        recycling and 25 sampling steps, empty MSA). The bagel oracle and the mosaic model are independent, so a term
        evaluated with one model in bagel can be screened with another in mosaic.
    n_candidates : int
        Candidate substitutions scored per step (default 8, the only value tested). Candidates are sampled uniformly over
        the mutable positions and the amino acids with a non-zero ``mutation_bias``, without replacement and excluding the
        moves already proposed from the current sequence. The mosaic batch always has ``n_candidates + 1`` sequences
        (the current one plus the candidates) because every new batch size recompiles the mosaic loss.
    selection_temperature : float
        0 (default, the only tested value): propose the candidate with the lowest pseudo-energy change. If positive, the
        candidate is drawn from ``softmax(-delta / T)``, with ``delta`` the pseudo-energy change in energy units (the
        same units as the weighted energy of bagel) and ``T`` this value; large ``T`` approaches random choice among the
        candidates. Use it to trade exploitation of the pseudo-energy against diversity.
    screening_steps : int, optional
        Screen only the first ``screening_steps`` calls to :meth:`one_step`; afterwards the ``fallback`` protocol is used
        and mosaic is no longer called. ``None`` (default): screen until the end of the run.
    calibration_steps : int
        Number of screening steps used to check that the pseudo-energy agrees with the real energy (default 5). In each of
        them ``calibration_mutants`` candidates are folded with bagel's oracles, including candidates that are not
        proposed, and the pseudo-energy change is compared with the real change in total energy. This costs
        ``calibration_mutants - 1`` extra bagel folds per calibration step (the proposed one is folded anyway). Zero or
        negative switches the check off.
    calibration_mutants : int
        Candidates folded per calibration step (default 8); must not exceed ``n_candidates``.
    calibration_threshold : float
        Smallest acceptable Spearman rank correlation between pseudo-energy changes and real energy changes (default
        0.25). If the check fails a warning is logged, the finding is written to the report, and the protocol permanently
        switches to the ``fallback`` (random) sampling. Mind that 40 pairs (5 steps x 8 candidates) estimate a rank
        correlation to about +-0.16 (one standard error), which is why ``calibration_rule`` exists.
    calibration_rule : {'point', 'upper95'}
        ``'point'`` (default): fail if the estimated correlation is below the threshold. ``'upper95'``: fail only if even
        the upper end of an approximate 95% interval of the correlation (Fisher z, assuming independent pairs) is below
        the threshold, i.e. switch to random only when there is good evidence that screening does not work. Despite the
        name, ``'upper95'`` is the *less* strict rule: it tolerates a noisy estimate that happens to be low.
    calibration_folds : int
        Bagel folds averaged per calibrated candidate (default 1). Increase it when the bagel oracle is stochastic, to
        average its noise out of the comparison; every extra fold multiplies the calibration cost.
    use_structure_terms : bool
        Default False. If True, the terms that read atom coordinates (surface area, hydrophobic, hydropathy, globular,
        buried histidine, histidine-carboxylate contacts) are also part of the pseudo-energy, evaluated on a structure
        rebuilt from mosaic's all-atom output (this makes mosaic return the coordinates, a larger transfer). Off by default
        because the structural terms agreed poorly with the oracle's structures in the one check made (25 mutants of one
        binder: Spearman correlation between -0.3 and 0.35 for surface area, globular and buried histidine; the exceptions
        are the hydrophobic and hydropathy terms without a residue-group mode, which depend on the sequence only and agree
        exactly). If you enable it, keep the calibration on.
    mutation_bias : dict[str, float]
        Amino acids with zero (or missing) bias are never proposed. The non-zero values are not used as weights; the
        candidates are sampled uniformly over the allowed amino acids.
    fallback : MutationProtocol, optional
        Protocol used when screening is off (after ``screening_steps`` or after a failed calibration). Defaults to
        :class:`~bagel.mutation.Canonical` with the same ``mutation_bias``. It must not be GrandCanonical.
    report_name : str
        Name of the report file written to the log folder of the run (default ``'Mosaic_pseudo_energy.txt'``).
    report_dir : pathlib.Path or str, optional
        Where to write the report. By default the log folder of the minimizer that runs this protocol.

    Attributes
    ----------
    last_info : dict
        Diagnostics of the last step: ``screened`` (bool), ``predicted_delta`` (pseudo-energy change of the proposed move),
        ``candidate_deltas``, ``mosaic_calls`` and ``mosaic_seconds``.
    stats : dict
        Counters over the whole run: ``screened_steps``, ``fallback_steps``, ``mosaic_calls``, ``mosaic_seconds``,
        ``calibration_folds``.
    calibration : dict
        Result of the calibration check (empty until it has run): ``rho``, ``n``, ``upper95``, ``passed``, ``rule``.
    """

    def __init__(
        self,
        backend: MosaicBackend | None = None,
        models: Mapping[Oracle, MosaicModelSpec] | None = None,
        n_candidates: int = 8,
        selection_temperature: float = 0.0,
        screening_steps: int | None = None,
        calibration_steps: int = 5,
        calibration_mutants: int = 8,
        calibration_threshold: float = 0.25,
        calibration_rule: str = 'point',
        calibration_folds: int = 1,
        use_structure_terms: bool = False,
        mutation_bias: dict[str, float] = mutation_bias_no_cystein,
        fallback: MutationProtocol | None = None,
        report_name: str = 'Mosaic_pseudo_energy.txt',
        report_dir: Path | str | None = None,
    ) -> None:
        if n_candidates < 1:
            raise ValueError('n_candidates must be at least 1.')
        if selection_temperature < 0.0:
            raise ValueError('selection_temperature must be >= 0.')
        if screening_steps is not None and screening_steps < 0:
            raise ValueError('screening_steps must be None or >= 0.')
        if calibration_rule not in ('point', 'upper95'):
            raise ValueError("calibration_rule must be 'point' or 'upper95'.")
        if calibration_folds < 1:
            raise ValueError('calibration_folds must be at least 1.')
        if calibration_steps > 0 and not 1 <= calibration_mutants <= n_candidates:
            raise ValueError('calibration_mutants must be between 1 and n_candidates.')
        if isinstance(fallback, GrandCanonical):
            raise MosaicIncompatibleError(
                'MosaicScreening cannot be combined with GrandCanonical sampling: mosaic losses are compiled for fixed '
                'chain lengths, so no call to mosaic is made when chain lengths can change. Use substitution-only moves.'
            )
        self.backend: MosaicBackend = backend if backend is not None else ModalMosaicBackend()
        self.models = dict(models or {})
        self.n_candidates = n_candidates
        self.selection_temperature = selection_temperature
        self.screening_steps = screening_steps
        self.calibration_steps = calibration_steps
        self.calibration_mutants = calibration_mutants
        self.calibration_threshold = calibration_threshold
        self.calibration_rule = calibration_rule
        self.calibration_folds = calibration_folds
        self.use_structure_terms = use_structure_terms
        self.mutation_bias = mutation_bias
        self.n_mutations = 1
        self.exclude_self = True
        self.fallback: MutationProtocol = fallback if fallback is not None else Canonical(mutation_bias=mutation_bias)
        self.report_name = report_name
        self.report_dir: Path | None = Path(report_dir) if report_dir is not None else None

        self.plans: list[StatePlan | None] | None = None
        self._step = 0
        self._screening_on = True
        self._pairs: list[tuple[float, float]] = []
        self._calibration_steps_done = 0
        self._score_cache: OrderedDict[tuple[int, MosaicModelSpec, str], float] = OrderedDict()
        self._tabu: OrderedDict[tuple[str, ...], set[tuple[int, str]]] = OrderedDict()
        self.last_info: dict[str, Any] = {}
        self.stats: dict[str, float] = {
            'screened_steps': 0,
            'fallback_steps': 0,
            'mosaic_calls': 0,
            'mosaic_seconds': 0.0,
            'calibration_folds': 0,
        }
        self.calibration: dict[str, Any] = {}

    # ------------------------------------------------------------------------------------------------------------
    def set_log_path(self, path: Path) -> None:
        """Called by the minimizer: the report goes to the log folder of the run unless ``report_dir`` was given."""
        if self.report_dir is None:
            self.report_dir = Path(path)

    def _write_report(self) -> None:
        assert self.plans is not None
        extra: dict[str, Any] = {
            'screening': 'on' if self._screening_on else 'off (random sampling)',
            'n_candidates': self.n_candidates,
            'selection_temperature': self.selection_temperature,
        }
        if self.calibration:
            extra['calibration'] = self.calibration
        text = format_report(self.plans, extra)
        if self.report_dir is not None:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            (self.report_dir / self.report_name).write_text(text)

    def _initialise(self, system: System) -> None:
        self.plans = build_plans(system, self.models, self.use_structure_terms)
        statuses = [s for p in self.plans if p is not None for s in p.statuses]
        cov = coverage(statuses)
        unused = [f'{s.name} ({s.reason})' for s in statuses if not s.used]
        logger.info(
            f'Mosaic pseudo-energy uses {cov.n_used}/{cov.n_total} energy terms, '
            f'{100 * cov.fraction_weight:.0f}% of the total |weight|.'
        )
        if unused:
            logger.warning(f'Energy terms NOT used by the mosaic pseudo-energy: {"; ".join(unused)}')
        self._write_report()

    # ------------------------------------------------------------------------------------------------------------
    def _screening_active(self) -> bool:
        if not self._screening_on:
            return False
        return self.screening_steps is None or self._step <= self.screening_steps

    def _pseudo_energies(self, system: System, plan: StatePlan, design_sequences: list[str]) -> list[float]:
        """Pseudo-energy of the State for each designed-chain sequence (cached; one batched mosaic call for the rest)."""
        state = system.states[plan.state_index]
        chains = plan.ordered_chains(state)
        per_group: list[list[float]] = []
        for group in plan.groups:
            keys = [(plan.state_index, group.spec, s) for s in design_sequences]
            fresh = [s for s, k in zip(design_sequences, keys) if k not in self._score_cache]
            fresh = list(dict.fromkeys(fresh))
            if fresh:
                # constant batch size: the mosaic loss is recompiled for every new batch size
                batch = fresh + [fresh[-1]] * max(0, self.n_candidates + 1 - len(fresh))
                t0 = time.time()
                predictions = self.backend.predict(
                    group.spec, plan.target_sequences, batch, structure=group.needs_structure
                )
                self.stats['mosaic_calls'] += 1
                self.stats['mosaic_seconds'] += time.time() - t0
                terms = plan.terms_of(state, group)
                for sequence, prediction in zip(fresh, predictions):
                    value, _ = pseudo_energy(terms, chains, prediction)
                    self._score_cache[(plan.state_index, group.spec, sequence)] = value
                while len(self._score_cache) > 50000:
                    self._score_cache.popitem(last=False)
            per_group.append([self._score_cache[k] for k in keys])
        return [float(sum(g[i] for g in per_group)) for i in range(len(design_sequences))]

    def _real_energy(self, system: System) -> float:
        """Total bagel energy, averaged over ``calibration_folds`` folds (the first one uses the State caches)."""
        energies = [system.get_total_energy()]
        for _ in range(self.calibration_folds - 1):
            for state in system.states:
                state._cache_key = None  # forces a new fold on the next energy access
            energies.append(system.get_total_energy())
        self.stats['calibration_folds'] += self.calibration_folds
        return float(np.mean(energies))

    def _calibrate(
        self,
        system: System,
        mutated: System,
        chain_id: str,
        candidates: list[tuple[int, str]],
        deltas: np.ndarray,
        chosen: int,
    ) -> None:
        """Fold ``calibration_mutants`` candidates with bagel and store (pseudo change, real change) pairs."""
        base = system.get_total_energy()
        order = [chosen] + [i for i in range(len(candidates)) if i != chosen]
        for i in order[: self.calibration_mutants]:
            if i == chosen:
                target = mutated  # already mutated; its caches are reused by the minimizer
            else:
                target = system.__copy__()
                tchain = next(c for st in target.states for c in st.chains if c.chain_ID == chain_id)
                tchain.mutate_residue(index=candidates[i][0], amino_acid=candidates[i][1])
            self._pairs.append((float(deltas[i]), self._real_energy(target) - base))
        self._calibration_steps_done += 1
        if self._calibration_steps_done >= self.calibration_steps:
            self._finish_calibration()

    def _finish_calibration(self) -> None:
        pseudo, real = zip(*self._pairs)
        rho = spearman(list(pseudo), list(real))
        n = len(self._pairs)
        upper = correlation_upper95(rho, n)
        statistic = rho if self.calibration_rule == 'point' else upper
        passed = bool(math.isfinite(statistic) and statistic >= self.calibration_threshold)
        self.calibration = {
            'rho': round(rho, 4) if math.isfinite(rho) else rho,
            'n': n,
            'upper95': round(upper, 4) if math.isfinite(upper) else upper,
            'threshold': self.calibration_threshold,
            'rule': self.calibration_rule,
            'passed': passed,
        }
        if passed:
            logger.info(f'Mosaic calibration passed: {self.calibration}')
        else:
            logger.warning(
                f'Mosaic calibration FAILED ({self.calibration}): the pseudo-energy does not rank mutations like the '
                'real energy for this system. Switching to random sampling for the rest of the run.'
            )
            self._screening_on = False
        self._write_report()

    # ------------------------------------------------------------------------------------------------------------
    def one_step(self, system: System, temperature: float | None = None) -> tuple[System, MutationRecord]:
        # temperature (the Monte Carlo temperature, passed by newer minimizers) is accepted for interface compatibility only
        if self.plans is None:
            self._initialise(system)
        assert self.plans is not None
        for plan in self.plans:  # gate first: nothing below may reach mosaic if the System no longer fits
            if plan is not None:
                plan.check_current(system.states[plan.state_index])
        self._step += 1

        if not self._screening_active():
            self.stats['fallback_steps'] += 1
            self.last_info = {'screened': False}
            return self.fallback.one_step(system)

        mutated = system.__copy__()
        chain = self.choose_chain(mutated)
        affected = [
            p for p in self.plans if p is not None and any(c is chain for c in mutated.states[p.state_index].chains)
        ]
        if not affected:  # cannot happen for a mutable chain, but never guess
            raise MosaicIncompatibleError(f"Chain '{chain.chain_ID}' belongs to no State with a mosaic plan.")

        current = chain.sequence
        key = tuple(c.sequence for st in mutated.states for c in st.chains)
        tabu = self._tabu.setdefault(key, set())
        while len(self._tabu) > 16:
            self._tabu.popitem(last=False)
        allowed = [aa for aa, w in self.mutation_bias.items() if w > 0.0]
        positions = chain.mutable_residue_indexes
        moves = [(i, aa) for i in positions for aa in allowed if aa != current[i] and (i, aa) not in tabu]
        if not moves:  # every move from this sequence has been proposed already: start over
            tabu.clear()
            moves = [(i, aa) for i in positions for aa in allowed if aa != current[i]]
        if not moves:
            raise ValueError('No valid substitution: check mutable residues and mutation_bias.')
        pick = np.random.choice(len(moves), size=min(self.n_candidates, len(moves)), replace=False)
        candidates = [moves[k] for k in pick]
        sequences = [current] + [current[:i] + aa + current[i + 1 :] for i, aa in candidates]

        calls_before, seconds_before = self.stats['mosaic_calls'], self.stats['mosaic_seconds']
        deltas = np.zeros(len(candidates))
        for plan in affected:
            values = self._pseudo_energies(mutated, plan, sequences)
            deltas += np.asarray(values[1:]) - values[0]

        if self.selection_temperature > 0.0:
            logits = -(deltas - deltas.min()) / self.selection_temperature
            p = np.exp(logits)
            chosen = int(np.random.choice(len(candidates), p=p / p.sum()))
        else:
            chosen = int(np.argmin(deltas))
        position, amino_acid = candidates[chosen]
        tabu.add((position, amino_acid))
        old = chain.residues[position].name
        chain.mutate_residue(index=position, amino_acid=amino_acid)
        mutation = Mutation(
            chain_id=chain.chain_ID,
            move_type='substitution',
            residue_index=position,
            old_amino_acid=old,
            new_amino_acid=amino_acid,
        )

        if self.calibration_steps > 0 and self._calibration_steps_done < self.calibration_steps:
            self._calibrate(system, mutated, chain.chain_ID, candidates, deltas, chosen)

        self.stats['screened_steps'] += 1
        self.last_info = {
            'screened': True,
            'predicted_delta': float(deltas[chosen]),
            'candidate_deltas': [float(d) for d in deltas],
            'mosaic_calls': self.stats['mosaic_calls'] - calls_before,
            'mosaic_seconds': self.stats['mosaic_seconds'] - seconds_before,
        }
        return mutated, MutationRecord(mutations=[mutation])
