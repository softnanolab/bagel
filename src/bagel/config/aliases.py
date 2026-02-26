"""Alias registry for built-in BAGEL components in YAML configs."""

from __future__ import annotations

import re
from collections.abc import Mapping

SectionName = str


def normalize_alias(alias: str) -> str:
    """Normalize user-provided alias names for case-insensitive matching."""
    return re.sub(r'[^a-z0-9_]', '', alias.strip().lower())


_BUILTIN_ALIASES_RAW: dict[SectionName, dict[str, str]] = {
    'minimizers': {
        'monte_carlo': 'bagel.minimizer:MonteCarloMinimizer',
        'montecarlo': 'bagel.minimizer:MonteCarloMinimizer',
        'simulated_annealing': 'bagel.minimizer:SimulatedAnnealing',
        'annealing': 'bagel.minimizer:SimulatedAnnealing',
        'simulated_tempering': 'bagel.minimizer:SimulatedTempering',
        'tempering': 'bagel.minimizer:SimulatedTempering',
    },
    'mutators': {
        'canonical': 'bagel.mutation:Canonical',
        'grand_canonical': 'bagel.mutation:GrandCanonical',
        'grandcanonical': 'bagel.mutation:GrandCanonical',
    },
    'oracles': {
        'esmfold': 'bagel.oracles.folding.esmfold:ESMFold',
        'esm2': 'bagel.oracles.embedding.esm2:ESM2',
    },
    'callbacks': {
        'default_logger': 'bagel.callbacks:DefaultLogger',
        'defaultlogger': 'bagel.callbacks:DefaultLogger',
        'folding_logger': 'bagel.callbacks:FoldingLogger',
        'foldinglogger': 'bagel.callbacks:FoldingLogger',
        'early_stopping': 'bagel.callbacks:EarlyStopping',
        'earlystopping': 'bagel.callbacks:EarlyStopping',
        'wandb_logger': 'bagel.callbacks:WandBLogger',
        'wandblogger': 'bagel.callbacks:WandBLogger',
    },
    'energies': {
        'ptm': 'bagel.energies:PTMEnergy',
        'plddt': 'bagel.energies:PLDDTEnergy',
        'overall_plddt': 'bagel.energies:OverallPLDDTEnergy',
        'global_plddt': 'bagel.energies:OverallPLDDTEnergy',
        'surface_area': 'bagel.energies:SurfaceAreaEnergy',
        'hydrophobic': 'bagel.energies:HydrophobicEnergy',
        'pae': 'bagel.energies:PAEEnergy',
        'lis': 'bagel.energies:LISEnergy',
        'ring_symmetry': 'bagel.energies:RingSymmetryEnergy',
        'separation': 'bagel.energies:SeparationEnergy',
        'flex_evobind': 'bagel.energies:FlexEvoBindEnergy',
        'globular': 'bagel.energies:GlobularEnergy',
        'template_match': 'bagel.energies:TemplateMatchEnergy',
        'secondary_structure': 'bagel.energies:SecondaryStructureEnergy',
        'embeddings_similarity': 'bagel.energies:EmbeddingsSimilarityEnergy',
        'chemical_potential': 'bagel.energies:ChemicalPotentialEnergy',
    },
}

BUILTIN_ALIASES: dict[SectionName, dict[str, str]] = {
    section: {normalize_alias(alias): class_path for alias, class_path in aliases.items()}
    for section, aliases in _BUILTIN_ALIASES_RAW.items()
}

BUILTIN_CLASS_PATHS: dict[SectionName, set[str]] = {
    section: set(aliases.values()) for section, aliases in BUILTIN_ALIASES.items()
}


def resolve_alias(section: SectionName, type_name: str) -> str | None:
    """Resolve a built-in alias to a class path if known."""
    aliases = BUILTIN_ALIASES.get(section, {})
    return aliases.get(normalize_alias(type_name))


def is_builtin_class_path(section: SectionName, class_path: str) -> bool:
    """Return True if class_path matches a known built-in for the section."""
    return class_path in BUILTIN_CLASS_PATHS.get(section, set())


def aliases_for_section(section: SectionName) -> Mapping[str, str]:
    """Expose aliases for docs/validation tooling."""
    return BUILTIN_ALIASES.get(section, {})
