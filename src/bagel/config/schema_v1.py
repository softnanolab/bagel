"""Pydantic schema for BAGEL YAML config version 1."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WandBSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    config_mode: Literal['raw', 'resolved', 'both'] = 'resolved'


class RunSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    experiment_name: str | None = None
    log_path: str | None = None
    seed: int | None = None
    wandb: WandBSpec = Field(default_factory=WandBSpec)


class ChainSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    sequence: str
    mutable: list[bool]
    chain_id: str

    @model_validator(mode='after')
    def _validate_lengths(self) -> 'ChainSpec':
        if len(self.sequence) != len(self.mutable):
            raise ValueError('chain sequence and mutable lists must be the same length')
        return self


class ComponentSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class EnergySpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: str
    oracle: str
    residues: Any | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class StateSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str
    chains: list[str]
    energies: list[EnergySpec]


class MinimizerSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfigV1(BaseModel):
    model_config = ConfigDict(extra='forbid')

    schema_version: Literal[1]
    run: RunSpec = Field(default_factory=RunSpec)
    chains: dict[str, ChainSpec]
    oracles: dict[str, ComponentSpec]
    mutators: dict[str, ComponentSpec]
    states: list[StateSpec]
    callbacks: dict[str, ComponentSpec] = Field(default_factory=dict)
    minimizer: MinimizerSpec

    @model_validator(mode='after')
    def _validate_non_empty(self) -> 'RunConfigV1':
        if not self.chains:
            raise ValueError('config must define at least one chain')
        if not self.states:
            raise ValueError('config must define at least one state')
        return self
