"""
Objects defining the different protocols for mutating the Chains.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

import numpy as np

# from .folding import FoldingAlgorithm
from .chain import Chain
from .state import State
from .system import System, validate_unique_chain_ids
from .constants import aminoacids_letters, mutation_bias_no_cystein
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Dict, Optional
from abc import ABC, abstractmethod
import logging
import pathlib as pl

if TYPE_CHECKING:
    from .oracles.folding.base import FoldingOracle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Mutation:
    """
    Single mutation operation.

    For GrandCanonical, the Mutation can be skipped if the move is impossible. In that case, move_type is None,
    but we keep track of the chain_id.

    """

    chain_id: str
    move_type: str | None  # 'substitution', 'addition', 'removal', or None if skipped
    residue_index: int | None  # None for skipped moves (e.g. impossible removals)
    old_amino_acid: str | None  # None for additions
    new_amino_acid: str | None  # None for removals
    parent_residue_index_by_state: Dict[str, Optional[int]] | None = (
        None  # For additions: state_name -> parent_residue_index
    )


@dataclass(frozen=True)
class MutationRecord:
    """Record of all mutations performed in a single one_step() call."""

    mutations: list[Mutation]  # Ordered list of mutations in this step


@dataclass
class MutationProtocol(ABC):
    mutation_bias: Dict[str, float] = field(default_factory=lambda: mutation_bias_no_cystein)
    n_mutations: int = 1
    exclude_self: bool = True

    @abstractmethod
    def one_step(
        self,
        system: System,
        temperature: float | None = None,
    ) -> tuple[System, MutationRecord]:
        """
        Abstract method for performing a single mutation step.

        Parameters
        ----------
        system : System
            The system to be mutated
        temperature : float, optional
            Temperature of the current Monte Carlo step, i.e. the one the minimizer uses to accept or reject the
            proposed mutation. Protocols that propose mutations at random ignore it; protocols that sample from a
            model may use it to temper their proposal distribution (see ``InverseFoldingSampling``).

        Returns
        -------
        System
            The mutated system (new copy)
        MutationRecord
            Record of all mutations performed in this step
        """
        pass

    def set_log_path(self, path: pl.Path) -> None:
        """Called by the minimizer with the log folder of the run. Does nothing unless a protocol writes its own files."""
        return

    def choose_chain(self, system: System, states: list[State] | None = None) -> Chain:
        """
        Choose one of the chains in the whole System that needs to be mutated. This is done by selecting a chain
        proportionally to the number of mutable aminoacid it has compared to the total number of mutable aminoacids
        available within the whole system. Because the number of mutable aminoacids can change during the simulation,
        probability must be recalculated at each step (i.e. after each mutation).

        If ``states`` is given, only the chains of those States are candidates (the proportionality is then over the
        mutable residues of those chains).
        """
        validate_unique_chain_ids(system.states)  # chains may have been added to States after the System was built
        unique_chain_list: list[Chain] = []
        for state in system.states if states is None else states:
            for chain in state.chains:
                # each chain (by identity: shared Chain objects are the same object) is only counted once
                if not any(chain is other for other in unique_chain_list):
                    unique_chain_list.append(chain)

        n_total_mutables = sum([len(chain.mutable_residues) for chain in unique_chain_list])
        if n_total_mutables == 0:
            where = 'the system' if states is None else f'State(s) {[state.name for state in states]}'
            raise ValueError(f'There is no mutable residue in {where}, so no chain can be chosen for mutation.')
        probability = np.zeros(len(unique_chain_list))
        for i, chain in enumerate(unique_chain_list):
            n_mutables = len(chain.mutable_residues)
            probability[i] = n_mutables / n_total_mutables
        # Step 2:
        # the chain is mutated according to the protocol chosen. Side note: a chain can be part of multiple states, and
        # mutations need to be made so that they are consistent across all states. This is taken care of by the fact
        # that the same object is used.
        return np.random.choice(unique_chain_list, p=probability)  # type: ignore

    def mutate_random_residue(self, chain: Chain) -> Mutation:
        """
        Mutate a random residue on a chain.

        Parameters
        ----------
        chain : Chain
            The chain to mutate.

        Returns
        -------
        Mutation
            The mutation performed. This includes the chain_id, the move_type, the residue_index, the old_amino_acid, and the new_amino_acid.
        """
        # Choose a residue to mutate
        index = np.random.choice(chain.mutable_residue_indexes)
        # Choose a new aminoacid
        current_aa = chain.residues[index].name
        aa_keys = list(self.mutation_bias.keys())
        probs = np.array([self.mutation_bias[a] for a in aa_keys], dtype=float)
        if self.exclude_self:  # exclude the current amino acid from the probability distribution
            mask = np.array([a != current_aa for a in aa_keys], dtype=bool)
            probs = probs * mask
            total = probs.sum()
            if total <= 0:
                raise ValueError(
                    f'No valid mutation targets after excluding current AA={current_aa}. '
                    'Check mutation_bias provides non-zero probability to at least one alternative.'
                )
            probs = probs / total
        amino_acid = np.random.choice(aa_keys, p=probs)
        chain.mutate_residue(index=index, amino_acid=amino_acid)
        return Mutation(
            chain_id=chain.chain_ID,
            move_type='substitution',
            residue_index=index,
            old_amino_acid=current_aa,
            new_amino_acid=amino_acid,
        )

    def replay(self, system: System, mutation_record: MutationRecord) -> System:
        """
        Replay a mutation record on a system, reusing existing mutation logic.

        Parameters
        ----------
        system : System
            The system to apply mutations to
        mutation_record : MutationRecord
            Record of mutations to replay

        Returns
        -------
        System
            System with mutations applied
        """
        replayed_system = system.__copy__()

        # Precompute chain mapping for O(1) lookups
        chain_map: Dict[str, Chain] = {}
        for state in replayed_system.states:
            for state_chain in state.chains:
                if state_chain.chain_ID not in chain_map:
                    chain_map[state_chain.chain_ID] = state_chain

        for mutation in mutation_record.mutations:
            # Find the chain by chain_id using precomputed mapping
            chain: Chain | None = chain_map.get(mutation.chain_id)

            if chain is None:
                raise ValueError(f'Chain with ID {mutation.chain_id} not found in system')

            if mutation.move_type is None:
                # Skip this mutation (e.g., removal was skipped because chain.length == 1)
                continue
            elif mutation.move_type == 'substitution':
                if mutation.residue_index is None:
                    raise ValueError('Residue index is required for substitution')
                if mutation.new_amino_acid is None:
                    raise ValueError('New amino acid is required for substitution')
                if mutation.old_amino_acid is None:
                    raise ValueError('Old amino acid is required for substitution')
                # Validation: ensure current residue matches expected old_amino_acid
                assert chain.residues[mutation.residue_index].name == mutation.old_amino_acid, (
                    f'Substitution mismatch: expected {mutation.old_amino_acid} at index {mutation.residue_index}, '
                    f'but found {chain.residues[mutation.residue_index].name}'
                )
                chain.mutate_residue(mutation.residue_index, mutation.new_amino_acid)
            elif mutation.move_type == 'addition':
                if mutation.new_amino_acid is None:
                    raise ValueError('New amino acid is required for addition')
                if mutation.residue_index is None:
                    raise ValueError('Residue index is required for addition')
                chain.add_residue(mutation.new_amino_acid, mutation.residue_index)
                # Update energy terms for all states using stored parent choices
                for state in replayed_system.states:
                    if mutation.parent_residue_index_by_state is None:
                        state.add_residue_to_all_energy_terms(mutation.chain_id, mutation.residue_index)
                    else:
                        if state.name not in mutation.parent_residue_index_by_state:
                            raise ValueError(
                                f'State {state.name} not found in parent_residue_index_by_state. '
                                f'Available states: {list(mutation.parent_residue_index_by_state.keys())}'
                            )
                        parent_residue_index = mutation.parent_residue_index_by_state[state.name]
                        state.add_residue_to_all_energy_terms(
                            mutation.chain_id, mutation.residue_index, parent_residue_index=parent_residue_index
                        )
            elif mutation.move_type == 'removal':
                if mutation.residue_index is None:
                    raise ValueError('Residue index is required for removal')
                if mutation.old_amino_acid is None:
                    raise ValueError('Old amino acid is required for removal')
                # Validation: ensure current residue matches expected old_amino_acid
                assert chain.residues[mutation.residue_index].name == mutation.old_amino_acid, (
                    f'Removal mismatch: expected {mutation.old_amino_acid} at index {mutation.residue_index}, '
                    f'but found {chain.residues[mutation.residue_index].name}'
                )
                chain.remove_residue(mutation.residue_index)
                # Update energy terms for all states
                for state in replayed_system.states:
                    state.remove_residue_from_all_energy_terms(mutation.chain_id, mutation.residue_index)
            else:
                raise ValueError(f'Unknown move_type: {mutation.move_type}')

        return replayed_system


class Canonical(MutationProtocol):
    """
    Canonical mutation protocol, making a substitution at random n residues.
    It cannot add or remove residues.

    Parameters
    ----------
    n_mutations : int, optional
        Number of mutations to perform in each step.
    mutation_bias : Dict[str, float], optional
        Bias for the substitution. The keys are the amino acids, the values are the probabilities.
    """

    def __init__(
        self,
        n_mutations: int = 1,
        mutation_bias: Dict[str, float] = mutation_bias_no_cystein,
        exclude_self: bool = True,
    ):
        self.n_mutations = n_mutations
        self.mutation_bias = mutation_bias
        self.exclude_self = exclude_self

    def one_step(
        self,
        system: System,
        temperature: float | None = None,
    ) -> tuple[System, MutationRecord]:
        # temperature is accepted for interface compatibility only: proposals here are random, not tempered
        mutated_system = system.__copy__()
        mutations: list[Mutation] = []

        for _ in range(self.n_mutations):
            chain = self.choose_chain(mutated_system)
            mutations.append(self.mutate_random_residue(chain))
        mutation_record = MutationRecord(mutations=mutations)
        return mutated_system, mutation_record


class GrandCanonical(MutationProtocol):
    """
    Grand canonical mutation protocol, making a substitution, addition or removal at random n residues.

    Parameters
    ----------
    n_mutations : int, optional
        Number of mutations to perform in each step.
    mutation_bias : Dict[str, float], optional
        Bias for the substitution. The keys are the amino acids, the values are the probabilities.
    move_probabilities : dict[str, float], optional
        Probabilities for the different moves. The keys are 'substitution', 'addition', and 'removal', and the values
        are the probabilities of performing the corresponding move. These should be normalized to sum to 1.
    """

    def __init__(
        self,
        n_mutations: int = 1,
        mutation_bias: Dict[str, float] = mutation_bias_no_cystein,
        move_probabilities: dict[str, float] = {
            'substitution': 0.5,
            'addition': 0.25,
            'removal': 0.25,
        },
        exclude_self: bool = True,
    ):
        self.n_mutations = n_mutations
        self.mutation_bias = mutation_bias
        self.move_probabilities = move_probabilities
        self.exclude_self = exclude_self
        # Check that no probabilities are negative
        if any([prob < 0 for prob in self.move_probabilities.values()]):
            raise ValueError('Probabilities must be positive')

        # Check that the sum of the probabilities is 1 otherwise normalize
        if sum(self.move_probabilities.values()) != 1.0:
            self.move_probabilities = {
                move: prob / sum(self.move_probabilities.values()) for move, prob in self.move_probabilities.items()
            }
            logger.warning('Recalculated move probabilties to ensure they sum to 1')
            logger.info(self.move_probabilities)

    def remove_random_residue(self, chain: Chain, system: System) -> Mutation:
        # First of all, only try this if it does not bring chains to 0 length
        chain_ID = chain.chain_ID
        if chain.length > 1:
            # Choose a residue to remove
            residue_index = np.random.choice(chain.mutable_residue_indexes)
            old_amino_acid = chain.residues[residue_index].name
            # Sanity check
            assert chain_ID == chain.residues[residue_index].chain_ID
            # Remove the residue from the chain it is part of
            chain.remove_residue(index=residue_index)
            # Remove the residue from energy terms of all the states in the system
            for state in system.states:
                state.remove_residue_from_all_energy_terms(chain_ID=chain_ID, residue_index=residue_index)
            move_type = 'removal'
            new_amino_acid = None
        else:
            residue_index = None
            move_type = None
            old_amino_acid = None
            new_amino_acid = None

        return Mutation(
            chain_id=chain_ID,
            move_type=move_type,
            residue_index=residue_index,
            old_amino_acid=old_amino_acid,
            new_amino_acid=new_amino_acid,
            parent_residue_index_by_state=None,
        )

    def add_random_residue(self, chain: Chain, system: System) -> Mutation:
        # Choose where to add the residue
        residue_index = np.random.choice(range(chain.length + 1))
        chain_ID = chain.residues[0].chain_ID
        # Choose a new aminoacid
        amino_acid = np.random.choice(list(self.mutation_bias.keys()), p=list(self.mutation_bias.values()))
        chain.add_residue(index=residue_index, amino_acid=amino_acid)
        # Now you need to decide which energy terms you want to associate to this residue. You do it based on its
        # neighbours. You look within the same chain and the same state and you add the residue to the same energy terms
        # the neighbours are part of. You actually look left and right, and randomly decide between the two. If the
        # residue is at the beginning or at the end of the chain, you just look at one of them.
        # This method records parent choices per state by storing the int|None returned
        # by add_residue_to_all_energy_terms so deterministic replays can reuse them.
        parent_residue_index_by_state: Dict[str, Optional[int]] = {}
        for state in system.states:
            parent_residue_index = state.add_residue_to_all_energy_terms(chain_ID=chain_ID, residue_index=residue_index)
            parent_residue_index_by_state[state.name] = parent_residue_index
        return Mutation(
            chain_id=chain_ID,
            move_type='addition',
            residue_index=residue_index,
            old_amino_acid=None,
            new_amino_acid=amino_acid,
            parent_residue_index_by_state=parent_residue_index_by_state,
        )

    def one_step(
        self,
        system: System,
        temperature: float | None = None,
    ) -> tuple[System, MutationRecord]:
        # temperature is accepted for interface compatibility only: proposals here are random, not tempered
        mutated_system = system.__copy__()
        mutations: list[Mutation] = []

        for _ in range(self.n_mutations):
            chain = self.choose_chain(mutated_system)

            # Now pick a move to make among removal, addition, or mutation
            assert self.move_probabilities.keys() == {'substitution', 'addition', 'removal'}, (
                'Move probabilities must be mutation, addition and removal'
            )
            move = np.random.choice(
                list(self.move_probabilities.keys()),
                p=list(self.move_probabilities.values()),
            )

            if move == 'substitution':
                mutations.append(self.mutate_random_residue(chain))
            elif move == 'addition':
                mutations.append(self.add_random_residue(chain, mutated_system))
            elif move == 'removal':
                mutations.append(self.remove_random_residue(chain, mutated_system))

        mutation_record = MutationRecord(mutations=mutations)
        return mutated_system, mutation_record


class InverseFoldingSampling(MutationProtocol):
    """
    Substitution protocol that proposes the new residue(s) by inverse folding.

    ``n_mutations`` is the number of residues the step mutates, and it also selects how the model is asked.

    **One residue (``n_mutations = 1``, the default).** (1) a random mutable residue is masked while all others
    stay fixed, (2) the inverse folding model gives the probability of every amino acid at that position,
    conditioned on the rest of the sequence and the currently folded structure, (3) the probability of the
    current amino acid is set to 0, the distribution renormalised, and the new residue sampled from it.

    **Several residues (``n_mutations > 1``).** That many random mutable residues of one chain are masked
    *together* and then decoded one residue per round, so each is sampled in the context of the ones already
    decoded rather than against a mask. This is the multi-residue redesign ESM3 does natively. Three
    consequences: every residue comes from the same chain (the model is asked about one chain at
    a time), the current amino acid is *not* excluded (the model may re-propose it, and only the residues that
    actually changed are recorded), and the excluded amino acids are enforced by forbidding their tokens in the
    model rather than by renormalising afterwards. The count is clamped to the number of mutable residues on the
    chosen chain.

    Either way the whole proposal is one Monte Carlo move, accepted or rejected as a unit, and chains cannot be
    added or removed.

    The model is sampled at ``t_sampling_proposals``, which is the protocol's own temperature and is independent
    of the Monte Carlo temperature the minimizer uses to accept or reject the move. With one residue it tempers
    the inverse folding probabilities: once the current amino acid and the excluded ones are removed, the
    remaining probabilities are raised to ``1 / T`` and renormalised (equivalent to ``softmax(logits / T)`` over
    the allowed alternatives). With several it is the sampling temperature of the decoding itself. It defaults to
    0.1 because ESM3 recommends not sampling sequences above that: beyond it the distribution is so flat that the
    proposals are essentially random, with little to no difference from the ``Canonical`` random-substitution
    protocol. Higher values are allowed and not capped; it is your call.

    The structure is taken from the (cached, else freshly computed) result of a folding oracle used by a State
    containing the chosen chain. By default, if several States contain it, one is picked at random. With
    ``conditioning_state`` the structure always comes from that State, and only the chains of that State are mutated
    (chains that are not part of it are never proposed by this protocol, and the chain is chosen proportionally to
    the mutable residues of that State's chains only, not of the whole system). State names must then be unique.
    A State's structure is always that of its current sequence: a cached result is reused only while the sequences
    are unchanged, otherwise it is recomputed. A State with several folding oracles needs ``folding_oracle`` to say
    which one supplies it; with several samples per prediction, it is the oracle's best sample. Amino acids with zero
    (or missing) ``mutation_bias`` are excluded: with one residue their probability is set to 0 before renormalising,
    and with several they are passed to the model as amino acids it may not sample. Non-zero bias values are not
    used as weights; the inverse folding probabilities are.

    Parameters
    ----------
    inverse_folding_method : str, optional
        Inverse folding model to use. Only ``'ESM3'`` is currently supported; anything else raises ValueError.
    oracle : ESM3, optional
        ESM3 oracle to use. If None, ``bagel.oracles.embedding.ESM3()`` is created on first use.
    n_mutations : int, optional
        Number of residues mutated in each step. ``1`` (the default) keeps the single-residue behaviour; above
        that the residues are masked and redesigned jointly, all on the same chain.
    mutation_bias : Dict[str, float], optional
        Amino acids whose value is zero (or that are missing) can never be proposed.
    t_sampling_proposals : float | Sequence[float] | Callable[[int], float], keyword-only, optional
        Temperature the inverse folding model is sampled at, defaulting to ``0.1``. A float keeps it constant; a
        sequence is read per step (``t_sampling_proposals[step]``, with ``step`` counted from 0 over the calls to
        ``one_step``, like the minimizer's own temperature schedule); a callable is passed the step and returns
        the temperature, which is how a scheduler is plugged in. It must resolve to a positive number.
        Keyword-only, so it cannot shift the positional arguments that came before it.
    conditioning_state : State | str, keyword-only, optional
        State (or its name) whose structure conditions every proposal. Only the chains of this State are then
        mutated. If None (the default), the chain is chosen among all mutable chains of the system and the State among
        those that contain it, at random.
    folding_oracle : FoldingOracle, keyword-only, optional
        The folding oracle (an instance used by the State's energy terms) that supplies the structure. It can be left
        out (None) only if the State has a single folding oracle; with several, not naming one is an error, since which
        structure conditions the proposals would otherwise be a silent choice. With several samples per prediction it
        is the oracle's best sample that is used.
    """

    SUPPORTED_METHODS = ('ESM3',)
    # ESM3 recommends against sampling above 0.1: beyond that proposals are basically random.
    DEFAULT_T_SAMPLING_PROPOSALS = 0.1

    def __init__(
        self,
        inverse_folding_method: str = 'ESM3',
        oracle: Any = None,
        n_mutations: int = 1,
        mutation_bias: Dict[str, float] = mutation_bias_no_cystein,
        *,
        t_sampling_proposals: float | Sequence[float] | Callable[[int], float] = DEFAULT_T_SAMPLING_PROPOSALS,
        conditioning_state: State | str | None = None,
        folding_oracle: 'FoldingOracle | None' = None,
    ):
        if inverse_folding_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f'Unsupported inverse folding method {inverse_folding_method!r}. '
                f'Supported methods: {list(self.SUPPORTED_METHODS)}'
            )
        if n_mutations < 1:
            raise ValueError(f'n_mutations must be at least 1, got {n_mutations}.')
        self.inverse_folding_method = inverse_folding_method
        self.oracle = oracle
        self.n_mutations = n_mutations
        self.t_sampling_proposals = t_sampling_proposals
        self.mutation_bias = mutation_bias
        # States are copied with the system, so they are identified by name
        self.conditioning_state_name: str | None = (
            conditioning_state
            if conditioning_state is None or isinstance(conditioning_state, str)
            else conditioning_state.name
        )
        self.folding_oracle = folding_oracle
        # Joint decoding samples from the model directly, so it cannot forbid the current residue.
        self.exclude_self = n_mutations == 1
        self._step = 0  # counts calls to one_step, so a schedule can be read per step

    def _get_oracle(self) -> Any:
        if self.oracle is None:
            from .oracles.embedding.esm3 import ESM3

            self.oracle = ESM3()
        return self.oracle

    @staticmethod
    def _get_structure(state: Any, oracle: Any = None) -> Any:
        """Folded structure of the state's current sequence (cached if valid, else computed).

        With several samples (or several oracles) this is the best structure of the chosen oracle. ``oracle`` selects the
        folding oracle; it can be left out only if the state has a single folding oracle.

        Raises
        ------
        ValueError
            If the state has no folding oracle, if it has several and ``oracle`` is not given, or if ``oracle`` is not
            used by the state's energy terms.
        """
        from .oracles.folding.base import FoldingOracle

        folding_oracles = [o for o in state.oracles_list if isinstance(o, FoldingOracle)]
        if not folding_oracles:
            raise ValueError(f"State '{state.name}' has no folding oracle, so no structure is available.")
        if oracle is None:
            if len(folding_oracles) > 1:
                raise ValueError(
                    f"State '{state.name}' has {len(folding_oracles)} folding oracles "
                    f'({[type(o).__name__ for o in folding_oracles]}), so it is ambiguous which one provides the structure '
                    f'that conditions the proposals. Pass the one to use as folding_oracle=.'
                )
            oracle = folding_oracles[0]
        elif oracle not in folding_oracles:
            raise ValueError(
                f"The requested folding oracle is not used by any energy term of State '{state.name}', "
                f'so it cannot provide its structure.'
            )
        state._invalidate_cache_if_needed()
        if oracle not in state._oracles_result:
            state._predict(oracle)
        return state._oracles_result.get_structure(oracle)

    def proposal_temperature(self, step: int) -> float:
        """
        Resolve ``t_sampling_proposals`` for a given step into the temperature the model is sampled at.

        This is the protocol's own temperature, not the Monte Carlo one the minimizer uses to accept or reject
        the move. ESM3 recommends staying at or below 0.1, but a higher value is allowed and not capped.

        Parameters
        ----------
        step : int
            Index of the current step, counted from 0 over the calls to ``one_step``. Only used when
            ``t_sampling_proposals`` is a sequence or a callable.

        Returns
        -------
        float
            The temperature to sample at.

        Raises
        ------
        ValueError
            If a sequence is too short for ``step``, or the resolved temperature is not finite and
            positive (NaN and infinity are rejected: both survive a bare positivity check, and infinity
            would silently flatten the proposal distribution to uniform).
        """
        schedule = self.t_sampling_proposals
        if callable(schedule):
            temperature = float(schedule(step))
        elif isinstance(schedule, (int, float)):
            temperature = float(schedule)
        else:
            values = list(schedule)
            if step >= len(values):
                raise ValueError(
                    f't_sampling_proposals has {len(values)} entries, too few for step {step}. Give a float, a '
                    f'callable, or a sequence at least as long as the number of steps.'
                )
            temperature = float(values[step])
        # NaN and inf both slip past `<= 0.0` (every NaN comparison is False). NaN then reaches numpy as a
        # NaN probability vector, and inf silently flattens the distribution to uniform, which turns inverse
        # folding into random substitution without raising. Reject both here.
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f't_sampling_proposals must resolve to a finite positive temperature, got {temperature}.')
        return temperature

    def excluded_amino_acids(self) -> str:
        """One-letter codes that ``mutation_bias`` forbids, as a string for the inverse folding model.

        The joint (``n_mutations > 1``) path cannot renormalise a distribution after the fact, so the excluded
        amino acids are forbidden inside the model instead. Amino acids missing from ``mutation_bias``, or
        with a value of zero or less, are excluded, matching :meth:`propose_amino_acid`.

        Returns
        -------
        str
            The excluded one-letter codes, sorted.
        """
        return ''.join(sorted(aa for aa in aminoacids_letters if self.mutation_bias.get(aa, 0.0) <= 0.0))

    def propose_amino_acid(self, probabilities: Dict[str, float], current_aa: str, temperature: float = 1.0) -> str:
        """
        Sample the new amino acid from the tempered inverse folding distribution.

        ``current_aa`` and the amino acids excluded by ``mutation_bias`` (zero or missing) are removed, the remaining
        probabilities are raised to ``1 / temperature`` and renormalised, and the new amino acid is sampled. The
        tempering is done in log space and only over the allowed alternatives, so a very likely current amino acid
        cannot make every alternative underflow to zero at low temperatures.
        """
        if not np.isfinite(temperature) or temperature <= 0.0:
            # See `proposal_temperature`: NaN and inf pass a bare `<= 0.0` check and corrupt the sampling.
            raise ValueError(f'temperature must be finite and positive, got {temperature}')
        aa_keys = list(probabilities.keys())
        allowed = np.array(
            [a != current_aa and self.mutation_bias.get(a, 0.0) > 0.0 and probabilities[a] > 0.0 for a in aa_keys]
        )
        if not allowed.any():
            raise ValueError(f'No valid mutation targets after excluding current AA={current_aa}.')
        with np.errstate(divide='ignore'):  # log(0) = -inf for the excluded amino acids is intended
            log_probs = np.log(np.array([probabilities[a] for a in aa_keys], dtype=float))
        # shift by the largest allowed log-probability *before* dividing, so that the best alternative always has
        # weight exp(0) = 1 and tiny temperatures cannot turn every weight into -inf (and the ratios into NaN)
        log_probs = np.where(allowed, (log_probs - log_probs[allowed].max()) / temperature, -np.inf)
        probs = np.exp(log_probs)
        return str(np.random.choice(aa_keys, p=probs / probs.sum()))

    def one_step(
        self,
        system: System,
        temperature: float | None = None,
    ) -> tuple[System, MutationRecord]:
        """
        Propose ``n_mutations`` inverse-folding substitutions as a single Monte Carlo move.

        Parameters
        ----------
        system : System
            The system to be mutated.
        temperature : float, optional
            Monte Carlo temperature of the current step (the one BAGEL uses to accept or reject the mutation).
            Accepted so the minimizer can pass it, but not used: the model is sampled at ``t_sampling_proposals``,
            which is set on the protocol and resolved per step by ``proposal_temperature``.
        """
        sampling_temperature = self.proposal_temperature(self._step)
        self._step += 1
        mutated_system = system.__copy__()
        mutations: list[Mutation] = []
        structures: dict[str, Any] = {}  # the structure the proposal was conditioned on, for this step

        if self.conditioning_state_name is None:
            chain = self.choose_chain(mutated_system)
            candidate_indices = [i for i, s in enumerate(mutated_system.states) if any(c is chain for c in s.chains)]
            state_index = candidate_indices[np.random.randint(len(candidate_indices))]
        else:
            names = [s.name for s in mutated_system.states]
            if self.conditioning_state_name not in names:
                raise ValueError(
                    f"conditioning_state '{self.conditioning_state_name}' is not a State of the system; "
                    f'available States: {names}'
                )
            if names.count(self.conditioning_state_name) != 1:
                raise ValueError(
                    f"More than one State is named '{self.conditioning_state_name}', so conditioning_state is "
                    f'ambiguous. Give every State of the system a unique name.'
                )
            state_index = names.index(self.conditioning_state_name)
            chain = self.choose_chain(mutated_system, states=[mutated_system.states[state_index]])
        state = mutated_system.states[state_index]
        # Take the structure from the corresponding State of the input system, not of the mutated copy: the
        # cached prediction then stays on the system that is kept and is not lost with the throwaway copy.
        structures[state.name] = self._get_structure(system.states[state_index], self.folding_oracle)
        structure = structures[state.name]

        if self.n_mutations == 1:
            proposal = self._propose_one(state, structure, chain, sampling_temperature)
        else:
            proposal = self._propose_several(state, structure, chain, sampling_temperature)

        for index, amino_acid in proposal.items():
            current_aa = chain.residues[index].name
            if amino_acid == current_aa:
                # Joint decoding can re-propose the residue that is already there; nothing to record.
                continue
            chain.mutate_residue(index=index, amino_acid=amino_acid)
            mutations.append(
                Mutation(
                    chain_id=chain.chain_ID,
                    move_type='substitution',
                    residue_index=index,
                    old_amino_acid=current_aa,
                    new_amino_acid=amino_acid,
                )
            )
        return mutated_system, MutationRecord(mutations=mutations)

    def _propose_one(
        self,
        state: Any,
        structure: Any,
        chain: Chain,
        sampling_temperature: float,
    ) -> dict[int, str]:
        """Propose a new residue for one random mutable position, from the tempered inverse folding distribution.

        Parameters
        ----------
        state : State
            State whose chains and structure condition the model.
        structure : AtomArray
            Folded structure of ``state``'s chains.
        chain : Chain
            Chain to mutate.
        sampling_temperature : float
            Temperature the proposal distribution is tempered at.

        Returns
        -------
        dict[int, str]
            The single ``{residue index: new amino acid}`` entry proposed.
        """
        index = int(np.random.choice(chain.mutable_residue_indexes))
        probabilities = self._get_oracle().inverse_fold(
            chains=state.chains, structure=structure, chain_id=chain.chain_ID, residue_index=index
        )
        amino_acid = self.propose_amino_acid(
            probabilities, chain.residues[index].name, temperature=sampling_temperature
        )
        return {index: amino_acid}

    def _propose_several(
        self,
        state: Any,
        structure: Any,
        chain: Chain,
        sampling_temperature: float,
    ) -> dict[int, str]:
        """Jointly redesign ``n_mutations`` random mutable positions of ``chain`` by iterative decoding.

        The positions are masked together, so each residue is sampled in the context of the ones already
        decoded. The current residues are not excluded, so some positions may come back unchanged.

        Parameters
        ----------
        state : State
            State whose chains and structure condition the model.
        structure : AtomArray
            Folded structure of ``state``'s chains.
        chain : Chain
            Chain to mutate.
        sampling_temperature : float
            Temperature the model samples at.

        Returns
        -------
        dict[int, str]
            The proposed ``{residue index: amino acid}`` for every masked position, changed or not.
        """
        mutable = np.asarray(chain.mutable_residue_indexes)
        n_chosen = min(self.n_mutations, int(mutable.size))
        indices = [int(index) for index in np.random.choice(mutable, size=n_chosen, replace=False)]
        proposed: dict[int, str] = self._get_oracle().inverse_generate(
            chains=state.chains,
            structure=structure,
            chain_id=chain.chain_ID,
            residue_indices=indices,
            temperature=sampling_temperature,
            excluded_amino_acids=self.excluded_amino_acids(),
        )
        return proposed
