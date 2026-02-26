# BAGEL Config Schema v1

This document describes the `schema_version: 1` YAML format used by `bagel run`.

## Versioning

- `schema_version` refers to the config contract version.
- It is not the same as the BAGEL package version.
- Current supported versions: `1`.

## CLI

- `bagel run <config.yaml>`: compile and run optimization.
- `bagel validate <config.yaml>`: validate schema/imports/refs without running optimization.
- `bagel schema --version 1`: print JSON schema for v1.

## Type Field

`type` supports:

1. Built-in alias, e.g. `simulated_tempering`, `canonical`, `esmfold`, `pTM`.
2. Explicit class path:
   - `my_pkg.my_module:MyClass`
   - `./plugins/my_class.py:MyClass` (relative to YAML file directory)

## Callable Marker

Nested callable values can be declared as:

```yaml
__callable__: "my_pkg.transforms:soft_cap"
```

or

```yaml
__callable__: "./plugins/fns.py:my_fn"
```

## Baseline Example

```yaml
schema_version: 1

run:
  experiment_name: my_run
  log_path: ./outputs
  seed: 42
  wandb:
    enabled: true
    project: bagel-project
    entity: my-team
    config_mode: resolved

chains:
  binder:
    sequence: ACDEFGHIK
    mutable: [true, true, true, true, true, true, true, true, true]
    chain_id: bind
  target:
    sequence: MKT...
    mutable: [false, false, false]
    chain_id: targ

oracles:
  fold_main:
    type: esmfold
    params:
      use_modal: true
      config:
        glycine_linker: GGGG
        position_ids_skip: 1024

mutators:
  mut_main:
    type: canonical
    params:
      n_mutations: 1

states:
  - name: state_A
    chains: [binder, target]
    energies:
      - type: pTM
        oracle: "@oracles.fold_main"
        params: {weight: 1.0}
      - type: PAE
        oracle: "@oracles.fold_main"
        residues:
          - - {chain: target, start: 10, end: 20}
          - - {chain: binder, start: 0, end: 8}
        params: {weight: 5.0}

callbacks:
  default_logger:
    type: default_logger
    params: {log_interval: 1}
  folding_logger:
    type: folding_logger
    params:
      folding_oracle: "@oracles.fold_main"
      log_interval: 50

minimizer:
  type: simulated_tempering
  params:
    mutator: "@mutators.mut_main"
    high_temperature: 2.0
    low_temperature: 0.1
    n_cycles: 10
    n_steps_low: 100
    n_steps_high: 20
    callbacks:
      - "@callbacks.default_logger"
      - "@callbacks.folding_logger"
```
