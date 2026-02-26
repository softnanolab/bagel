"""Custom plugin classes/functions for config compiler tests."""

from __future__ import annotations

import pathlib as pl
from typing import Any

import bagel as bg
from bagel.chain import Chain
from bagel.oracles.base import OracleResult


class DummyOracleResult(OracleResult):
    input_chains: list[Chain]

    def save_attributes(self, filepath: pl.Path) -> None:
        filepath.with_suffix('.dummy').write_text('ok\n')


class DummyOracle(bg.oracles.Oracle):
    result_class = DummyOracleResult

    def predict(self, chains: list[Chain]) -> DummyOracleResult:
        return DummyOracleResult(input_chains=chains)


class DummyMutator(bg.mutation.MutationProtocol):
    def __init__(self, n_mutations: int = 1, custom_rate: float = 0.0):
        self.n_mutations = n_mutations
        self.custom_rate = custom_rate

    def one_step(self, system: bg.System) -> tuple[bg.System, bg.mutation.MutationRecord]:
        return system.__copy__(), bg.mutation.MutationRecord(mutations=[])


class DummyEnergy(bg.energies.EnergyTerm):
    def __init__(self, oracle: bg.oracles.Oracle, bias: float = 0.0, weight: float = 1.0):
        super().__init__(name='dummy_energy', oracle=oracle, inheritable=False, weight=weight)
        self.bias = bias

    def compute(self, oracles_result: bg.oracles.OraclesResultDict) -> tuple[float, float]:
        chains = oracles_result.get_input_chains(self.oracle)
        total_res = float(sum(chain.length for chain in chains))
        value = total_res + self.bias
        return value, value * self.weight


class DummyCallback(bg.callbacks.Callback):
    def __init__(self, tag: str = 'dummy', extra_option: int = 0):
        self.tag = tag
        self.extra_option = extra_option
        self.steps: list[int] = []

    def on_step_end(self, context: bg.callbacks.CallbackContext) -> None:
        self.steps.append(context.step)


class DummyMinimizer(bg.minimizer.Minimizer):
    def __init__(
        self,
        mutator: bg.mutation.MutationProtocol,
        experiment_name: str = 'dummy_min',
        log_frequency: int = 1,
        log_path: str | pl.Path | None = None,
        callbacks: list[bg.callbacks.Callback] | None = None,
        custom_alpha: float = 0.1,
        **kwargs: Any,
    ) -> None:
        self.custom_alpha = custom_alpha
        super().__init__(
            mutator=mutator,
            experiment_name=experiment_name,
            log_frequency=log_frequency,
            log_path=log_path,
            callbacks=callbacks,
            **kwargs,
        )

    def minimize_system(self, system: bg.System) -> bg.System:
        system.get_total_energy()
        return system


def square_distance(x: float) -> float:
    return x * x
