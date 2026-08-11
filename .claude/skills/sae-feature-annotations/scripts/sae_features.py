#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Part of the `sae-feature-annotations` Claude skill.
# Queries Biohub for annotations of ESM-C sparse-autoencoder (SAE) features and
# maintains a local fallback lookup table. Standard library only (urllib) so it
# runs anywhere without installing dependencies.
#
# IMPORTANT: the annotations returned here are only meaningful for the SAE
# ESMC-6B-sae-layer60-k64-codebook16384 (ESM-C 6B, layer 60, k=64, codebook
# 16384). A feature_index means something completely different under any other
# layer / k / codebook, so never mix these annotations with a different SAE.
# -----------------------------------------------------------------------------
"""Look up SAE feature annotations from Biohub, with an offline fallback table.

Two data sources, mirroring the two Biohub endpoints:

  * The **map** (feature_index -> label + short description) for *every* feature,
    from ``GET {base}/esm/protein/api/v1alpha1/features``. This is small enough to
    cache locally, so ``refresh`` snapshots it to a JSON file that ``list`` /
    ``search`` / ``get`` can fall back to when no API key or network is available.
  * The **detail** for a single feature (longform description, top-activating
    UniRef90 & SwissProt proteins, decoder nearest neighbours, activation
    statistics), from ``GET {base}/esm/protein/api/v1alpha1/features/{index}``.
    This is only available live — there is no offline substitute.

Policy: always try Biohub first (it is authoritative), and only fall back to the
cached table when the live call cannot be made — printing a clear reminder that
the cache may have drifted from Biohub since it was last refreshed.

Subcommands
-----------
  refresh                 Fetch the full feature map from Biohub and write the cache.
  list                    List features (live if possible, else cache).
  search <text>           Search labels/descriptions for <text> (live or cache).
  get <feature_index>     Full detail for one feature (live only).

Configuration (flags override environment override defaults)
------------------------------------------------------------
  --base-url   BIOHUB_BASE_URL      default https://biohub.ai
  --token      ESM_API_KEY / FORGE_TOKEN
  --cache      SAE_FEATURE_CACHE    default <skill>/assets/feature_annotations.json
  --sae-model  SAE_MODEL            default esmc-6b-2024-12-sae-layer60-k64-codebook16384

Examples
--------
  python sae_features.py refresh                 # seed/refresh the offline table (needs key)
  python sae_features.py search "zinc binding"   # find features by keyword
  python sae_features.py get 12345 --json        # full detail for feature 12345
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# The one SAE these annotations describe. Everything here is meaningless for any
# other layer / k / codebook, so we carry it around and stamp it into the cache.
DEFAULT_SAE_MODEL = "esmc-6b-2024-12-sae-layer60-k64-codebook16384"
DEFAULT_BASE_URL = "https://biohub.ai"
API_PREFIX = "/esm/protein/api/v1alpha1"

# assets/feature_annotations.json sits next to this scripts/ directory.
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "assets" / "feature_annotations.json"

VALIDITY_BANNER = (
    "These SAE feature annotations are only valid for "
    "ESMC-6B-sae-layer60-k64-codebook16384 (ESM-C 6B, layer 60, k=64, "
    "codebook 16384). They do not transfer to any other SAE."
)


def _eprint(*args: object) -> None:
    """Print to stderr so machine-readable stdout (``--json``) stays clean."""
    print(*args, file=sys.stderr)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _resolve_token(explicit: str | None) -> str | None:
    return explicit or os.environ.get("ESM_API_KEY") or os.environ.get("FORGE_TOKEN")


def _http_get_json(url: str, token: str | None, timeout: int = 60) -> object:
    """GET a URL and parse JSON. Raises on HTTP/network/JSON errors.

    Auth defaults to a bearer token (the same credential ``ESMCForgeInferenceClient``
    uses). If Biohub expects a different header, adjust it here and in the skill's
    references/api.md.
    """
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (trusted host)
        return json.loads(response.read().decode("utf-8"))


def _features_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}{API_PREFIX}/features"


def _feature_detail_url(base_url: str, feature_index: int) -> str:
    return f"{base_url.rstrip('/')}{API_PREFIX}/features/{feature_index}"


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _load_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _eprint(f"[warn] could not read cache {cache_path}: {exc}")
        return None
    # A never-seeded placeholder has an empty feature list; treat as no cache.
    if not data.get("features"):
        return None
    return data


def _cache_features(cache: dict) -> list[dict]:
    return cache.get("features", [])


def _live_feature_map(base_url: str, token: str | None) -> list[dict]:
    """Fetch the full feature map. Normalizes to a list of dicts with at least
    ``feature_index``, ``label``, ``description``."""
    payload = _http_get_json(_features_url(base_url), token)
    # Be liberal about the envelope: accept a bare list or {"features": [...]}.
    if isinstance(payload, dict):
        payload = payload.get("features", payload.get("data", []))
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected features payload shape: {type(payload).__name__}")
    return payload


def _get_map(args: argparse.Namespace) -> tuple[list[dict], str]:
    """Return (features, source) where source is 'live' or 'cache'.

    Live first (authoritative); fall back to the cache with a loud divergence
    reminder when the live call cannot be made.
    """
    token = _resolve_token(args.token)
    if token:
        try:
            return _live_feature_map(args.base_url, token), "live"
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as exc:
            _eprint(f"[warn] Biohub live query failed ({exc}); falling back to local cache.")
    else:
        _eprint("[warn] no API key (ESM_API_KEY / FORGE_TOKEN / --token); using local cache.")

    cache = _load_cache(args.cache)
    if cache is None:
        _eprint(
            f"[error] no usable cache at {args.cache}. Seed it with a key:\n"
            f"        python {Path(__file__).name} refresh"
        )
        raise SystemExit(2)
    fetched = cache.get("fetched_at", "unknown time")
    _eprint(
        f"[cache] using local table snapshotted at {fetched} for SAE "
        f"{cache.get('sae_model', '?')}. Biohub may have changed since; refresh "
        f"with an API key to be sure the labels are current."
    )
    return _cache_features(cache), "cache"


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_refresh(args: argparse.Namespace) -> int:
    token = _resolve_token(args.token)
    if not token:
        _eprint(
            "[error] refresh needs an API key. Set ESM_API_KEY (or FORGE_TOKEN), "
            "or pass --token. Get one from your Biohub/Forge account."
        )
        return 2
    _eprint(f"[info] fetching feature map from {_features_url(args.base_url)} ...")
    try:
        features = _live_feature_map(args.base_url, token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as exc:
        _eprint(f"[error] refresh failed: {exc}")
        return 1
    cache = {
        "sae_model": args.sae_model,
        "base_url": args.base_url,
        "endpoint": _features_url(args.base_url),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": VALIDITY_BANNER,
        "features": features,
    }
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(cache, indent=2, sort_keys=False))
    _eprint(f"[ok] wrote {len(features)} features to {args.cache}")
    return 0


def _match(feature: dict, text: str) -> bool:
    hay = f"{feature.get('label', '')} {feature.get('description', '')}".lower()
    return text.lower() in hay


def cmd_list(args: argparse.Namespace) -> int:
    features, source = _get_map(args)
    if getattr(args, "query", None):
        features = [f for f in features if _match(f, args.query)]
    features = sorted(features, key=lambda f: f.get("feature_index", 0))
    if args.limit:
        features = features[: args.limit]
    if args.json:
        print(json.dumps({"source": source, "count": len(features), "features": features}, indent=2))
    else:
        _eprint(VALIDITY_BANNER)
        _eprint(f"[{source}] {len(features)} feature(s):")
        for f in features:
            print(f"  {f.get('feature_index'):>6}  {f.get('label', '')}")
            if f.get("description"):
                print(f"          {f['description']}")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    token = _resolve_token(args.token)
    if not token:
        _eprint(
            "[error] Extended feature detail (longform description, top-activating "
            "UniRef90/SwissProt proteins, decoder nearest neighbours, activation "
            "statistics) is only available live from Biohub and needs an API key.\n"
            "        Set ESM_API_KEY / FORGE_TOKEN or pass --token. For just the "
            "label + short description you can use the offline table:\n"
            f"        python {Path(__file__).name} search <text>"
        )
        return 2
    url = _feature_detail_url(args.base_url, args.feature_index)
    _eprint(f"[info] GET {url}")
    try:
        detail = _http_get_json(url, token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError) as exc:
        _eprint(f"[error] could not fetch feature {args.feature_index}: {exc}")
        return 1
    if args.json:
        print(json.dumps(detail, indent=2))
    else:
        _eprint(VALIDITY_BANNER)
        print(json.dumps(detail, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--base-url", default=os.environ.get("BIOHUB_BASE_URL", DEFAULT_BASE_URL))
        sub.add_argument("--token", default=None, help="API key; else ESM_API_KEY / FORGE_TOKEN.")
        sub.add_argument(
            "--cache",
            type=Path,
            default=Path(os.environ.get("SAE_FEATURE_CACHE", DEFAULT_CACHE)),
        )
        sub.add_argument("--sae-model", default=os.environ.get("SAE_MODEL", DEFAULT_SAE_MODEL))

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_refresh = subparsers.add_parser("refresh", help="Fetch full feature map and write the cache (needs key).")
    add_common(p_refresh)
    p_refresh.set_defaults(func=cmd_refresh)

    p_list = subparsers.add_parser("list", help="List features (live if possible, else cache).")
    add_common(p_list)
    p_list.add_argument("--limit", type=int, default=0, help="Max rows (0 = all).")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list, query=None)

    p_search = subparsers.add_parser("search", help="Search labels/descriptions (live if possible, else cache).")
    add_common(p_search)
    p_search.add_argument("query", help="Text to match in label or description.")
    p_search.add_argument("--limit", type=int, default=0)
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_list)

    p_get = subparsers.add_parser("get", help="Full detail for one feature (live only, needs key).")
    add_common(p_get)
    p_get.add_argument("feature_index", type=int)
    p_get.add_argument("--json", action="store_true")
    p_get.set_defaults(func=cmd_get)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
