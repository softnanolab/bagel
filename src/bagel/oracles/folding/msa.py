"""User-supplied multiple sequence alignments (MSAs) for folding oracles.

An MSA is supplied once, when the oracle is built, as ``{chain_ID: A3M text or A3M file path}``. Files are read a single
time at construction; every later fold only forwards the stored text. Folding oracles that accept an MSA do not search
for one by default, so the search cost is paid once (by the user, offline) and not again during a minimization.

An MSA describes a fixed sequence: its first row must equal the chain it is attached to. It is therefore meant for chains
that are never mutated (e.g. a target); a designed chain simply has no entry and is folded from its sequence alone.
"""

import logging
from collections.abc import Mapping
from pathlib import Path

from ...chain import Chain

logger = logging.getLogger(__name__)

MSAInputs = Mapping[str, str | Path]


def _query_row(a3m: str) -> str:
    """Return the aligned (match-state) columns of the first A3M row, i.e. the query sequence."""
    sequence = ''
    seen_header = False
    for raw in a3m.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('>'):
            if seen_header:
                break
            seen_header = True
        elif seen_header:
            sequence += line
    return ''.join(char for char in sequence if not char.islower() and char != '.')


def load_msas(msa: MSAInputs | None) -> dict[str, str]:
    """
    Read user MSAs into A3M text keyed by chain_ID, so that files are only touched once.

    A value is A3M text if it starts with ``>``, otherwise it is treated as the path of an A3M file.
    """
    if msa is None:
        return {}
    texts: dict[str, str] = {}
    for chain_id, value in msa.items():
        if isinstance(value, Path) or not str(value).lstrip().startswith('>'):
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f'MSA file for chain {chain_id!r} does not exist: {path}')
            text = path.read_text(encoding='utf-8')
        else:
            text = str(value)
        if not text.lstrip().startswith('>') or not _query_row(text):
            raise ValueError(f'MSA for chain {chain_id!r} is not valid A3M text')
        texts[chain_id] = text
    return texts


def msas_for_chains(
    msas: Mapping[str, str],
    chains: list[Chain],
    keep_empty: bool = False,
    identical_chains_share: bool = False,
) -> list[str | None] | None:
    """
    Return one A3M text (or None) per chain, in chain order, or None when no chain has an MSA.

    With ``keep_empty`` the all-None list is returned instead of None when the oracle holds MSAs but none of them
    matches these chains; models that search an MSA whenever none is passed need it to know that none is wanted.

    With ``identical_chains_share`` (models that give chains of equal sequence one common MSA) chains with the same
    sequence must have the same MSA entry, since an entry on only one of them would silently apply to all of them.

    Raises
    ------
    ValueError
        If the first row of a chain's MSA is not that chain's current sequence, which means the chain was mutated, or
        if ``identical_chains_share`` is set and chains of equal sequence have different MSA entries.
    """
    if not msas:
        return None
    entries: list[str | None] = []
    for chain in chains:
        text = msas.get(chain.chain_ID)
        if text is not None and _query_row(text) != chain.sequence:
            raise ValueError(
                f'The MSA of chain {chain.chain_ID!r} does not match its sequence. An MSA describes a fixed sequence: '
                f'only give MSAs for chains that are not mutated.'
            )
        entries.append(text)
    if identical_chains_share:
        by_sequence: dict[str, str | None] = {}
        for chain, entry in zip(chains, entries):
            if chain.sequence in by_sequence and by_sequence[chain.sequence] != entry:
                raise ValueError(
                    f'Chain {chain.chain_ID!r} has the same sequence as another chain but a different MSA entry. '
                    f'This model uses one MSA for all chains of equal sequence: give every such chain the same MSA.'
                )
            by_sequence[chain.sequence] = entry
    if all(entry is None for entry in entries):
        logger.debug('No MSA entry matches the chains being folded; folding without MSA.')
        return entries if keep_empty else None
    return entries
