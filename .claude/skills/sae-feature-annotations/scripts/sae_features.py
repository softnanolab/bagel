#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Part of the `sae-feature-annotations` Claude skill.
# Queries Biohub for annotations of ESM-C sparse-autoencoder (SAE) features.
# Standard library only (urllib) so it runs anywhere without installing anything.
#
# IMPORTANT: the annotations returned here are only meaningful for the SAE
# ESMC-6B-sae-layer60-k64-codebook16384 (ESM-C 6B, layer 60, k=64, codebook
# 16384). A feature_index means something completely different under any other
# layer / k / codebook, so never mix these annotations with a different SAE.
# -----------------------------------------------------------------------------
"""Look up SAE feature annotations from Biohub.

The Biohub feature-annotation endpoints are public reads, so this tool always
queries them live — there is no bundled offline table. Two endpoints, two levels
of detail:

  * The **map** (feature_index -> label + short description) for *every* feature,
    from ``GET {base}/esm/protein/api/v1alpha1/features``. Backs ``list`` /
    ``search``.
  * The **detail** for a single feature (longform description, activation pattern,
    category, top-activating UniRef90 & SwissProt proteins, decoder
    nearest-neighbour feature indices, activation statistics), from
    ``GET {base}/esm/protein/api/v1alpha1/features/{index}``. Backs ``get``.

Auth: if a real API key is available (``--token`` / ``ESM_API_KEY`` /
``FORGE_TOKEN``) it is sent as ``Authorization: Bearer <token>``. If none is set,
the request is still made with a placeholder token, since the annotation
endpoints are public. If a deployment enforces auth and the anonymous call is
rejected, set ``ESM_API_KEY`` (the same credential ``boileroom``'s Forge backend
uses).

Subcommands
-----------
  list                    List features (map endpoint).
  search <text>           Search labels/descriptions for <text> (map endpoint).
  get <feature_index>     Full detail for one feature (detail endpoint).

Configuration (flags override environment override defaults)
------------------------------------------------------------
  --base-url   BIOHUB_BASE_URL      default https://biohub.ai
  --token      ESM_API_KEY / FORGE_TOKEN

Examples
--------
  python sae_features.py search "zinc"           # find features by keyword
  python sae_features.py get 42 --json           # full detail for feature 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# The one SAE these annotations describe. Everything here is meaningless for any
# other layer / k / codebook, so we lead with it in the banner.
DEFAULT_SAE_MODEL = 'esmc-6b-2024-12-sae-layer60-k64-codebook16384'
DEFAULT_BASE_URL = 'https://biohub.ai'
API_PREFIX = '/esm/protein/api/v1alpha1'

# The annotation endpoints are public; when the user has no key of their own we
# still authenticate the request with this placeholder so the header is present.
ANONYMOUS_TOKEN = 'anonymous'

VALIDITY_BANNER = (
    'These SAE feature annotations are only valid for '
    'ESMC-6B-sae-layer60-k64-codebook16384 (ESM-C 6B, layer 60, k=64, '
    'codebook 16384). They do not transfer to any other SAE.'
)


def _eprint(*args: object) -> None:
    """Print to stderr so machine-readable stdout (``--json``) stays clean."""
    print(*args, file=sys.stderr)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _resolve_token(explicit: str | None) -> tuple[str, bool]:
    """Return ``(token_to_send, is_real)``.

    Prefers an explicit ``--token``, then ``ESM_API_KEY``, then ``FORGE_TOKEN``.
    If none is set, falls back to the anonymous placeholder so the public
    endpoints can still be queried.
    """
    real = explicit or os.environ.get('ESM_API_KEY') or os.environ.get('FORGE_TOKEN')
    if real:
        return real, True
    return ANONYMOUS_TOKEN, False


def _http_get_json(url: str, token: str, timeout: int = 60) -> object:
    """GET a URL and parse JSON. Raises on HTTP/network/JSON errors.

    Auth is a bearer token (the same credential ``ESMCForgeInferenceClient`` uses,
    or the anonymous placeholder). If Biohub ever expects a different header,
    adjust it here — it is the single place auth is applied.
    """
    request = urllib.request.Request(url, headers={'Accept': 'application/json'})
    request.add_header('Authorization', f'Bearer {token}')
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (trusted host)
        return json.loads(response.read().decode('utf-8'))


def _features_url(base_url: str) -> str:
    return f'{base_url.rstrip("/")}{API_PREFIX}/features'


def _feature_detail_url(base_url: str, feature_index: int) -> str:
    return f'{base_url.rstrip("/")}{API_PREFIX}/features/{feature_index}'


def _note_auth(is_real: bool) -> None:
    if not is_real:
        _eprint(
            '[info] no API key set (ESM_API_KEY / FORGE_TOKEN / --token); querying '
            'Biohub anonymously. Set a key for authenticated access if required.'
        )


def _live_feature_map(base_url: str, token: str) -> list[dict]:
    """Fetch the full feature map, normalized to a list of dicts with at least
    ``feature_index``, ``label``, ``description``."""
    payload = _http_get_json(_features_url(base_url), token)
    # Be liberal about the envelope: accept a bare list, {"data": [...]},
    # or {"features": [...]}. Biohub currently returns {"data": [...]}.
    if isinstance(payload, dict):
        payload = payload.get('data', payload.get('features', []))
    if not isinstance(payload, list):
        raise ValueError(f'Unexpected features payload shape: {type(payload).__name__}')
    return payload


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def _match(feature: dict, text: str) -> bool:
    hay = f'{feature.get("label", "")} {feature.get("description", "")}'.lower()
    return text.lower() in hay


def cmd_list(args: argparse.Namespace) -> int:
    token, is_real = _resolve_token(args.token)
    _note_auth(is_real)
    try:
        features = _live_feature_map(args.base_url, token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as exc:
        _eprint(f'[error] could not fetch feature map from Biohub: {exc}')
        return 1
    if getattr(args, 'query', None):
        features = [f for f in features if _match(f, args.query)]
    features = sorted(features, key=lambda f: f.get('feature_index', 0))
    if args.limit:
        features = features[: args.limit]
    if args.json:
        print(json.dumps({'count': len(features), 'features': features}, indent=2))
    else:
        _eprint(VALIDITY_BANNER)
        _eprint(f'{len(features)} feature(s):')
        for f in features:
            print(f'  {f.get("feature_index"):>6}  {f.get("label", "")}')
            if f.get('description'):
                print(f'          {f["description"]}')
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    token, is_real = _resolve_token(args.token)
    _note_auth(is_real)
    url = _feature_detail_url(args.base_url, args.feature_index)
    _eprint(f'[info] GET {url}')
    try:
        detail = _http_get_json(url, token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as exc:
        _eprint(f'[error] could not fetch feature {args.feature_index}: {exc}')
        return 1
    if not args.json:
        _eprint(VALIDITY_BANNER)
    print(json.dumps(detail, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument('--base-url', default=os.environ.get('BIOHUB_BASE_URL', DEFAULT_BASE_URL))
        sub.add_argument('--token', default=None, help='API key; else ESM_API_KEY / FORGE_TOKEN, else anonymous.')

    subparsers = parser.add_subparsers(dest='command', required=True)

    p_list = subparsers.add_parser('list', help='List features from the Biohub map endpoint.')
    add_common(p_list)
    p_list.add_argument('--limit', type=int, default=0, help='Max rows (0 = all).')
    p_list.add_argument('--json', action='store_true')
    p_list.set_defaults(func=cmd_list, query=None)

    p_search = subparsers.add_parser('search', help='Search labels/descriptions on the map endpoint.')
    add_common(p_search)
    p_search.add_argument('query', help='Text to match in label or description.')
    p_search.add_argument('--limit', type=int, default=0)
    p_search.add_argument('--json', action='store_true')
    p_search.set_defaults(func=cmd_list)

    p_get = subparsers.add_parser('get', help='Full detail for one feature from the detail endpoint.')
    add_common(p_get)
    p_get.add_argument('feature_index', type=int)
    p_get.add_argument('--json', action='store_true')
    p_get.set_defaults(func=cmd_get)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
