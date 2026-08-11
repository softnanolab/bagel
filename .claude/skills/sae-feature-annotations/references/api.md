# Biohub SAE feature annotation API

Reference for the two endpoints the skill uses. These annotations come from the
sparse-autoencoder analysis in *Language Modeling Materializes a World Model of
Protein Biology* (Biohub, 2026) and describe exactly one SAE:

> **ESMC-6B-sae-layer60-k64-codebook16384** — ESM-C 6B, transformer layer 60,
> TopK with k=64, codebook of 2**14 = 16384 features.

A `feature_index` (0…16383) is only meaningful under this SAE. The same index
under a different layer / k / codebook is an unrelated direction, so never reuse
these labels for another SAE. This is the SAE the `boileroom` `SAE` model uses by
default via its Forge backend (`forge_sae_model =
esmc-6b-2024-12-sae-layer60-k64-codebook16384`, `forge_url = https://biohub.ai`).

## Base URL and auth

- **Base URL:** `https://biohub.ai` (same host as `boileroom`'s `forge_url`).
  Override with `--base-url` or `BIOHUB_BASE_URL`.
- **Auth:** an API key, the same credential `ESMCForgeInferenceClient` uses. The
  script reads `--token`, then `ESM_API_KEY`, then `FORGE_TOKEN`, and sends it as
  `Authorization: Bearer <token>`.
- If Biohub actually expects a different header name/scheme than bearer, adjust
  `_http_get_json` in `scripts/sae_features.py` — it is the single place auth is
  applied. (The exact header could not be verified from the build environment,
  which blocks `biohub.ai`; the bearer default matches the Forge SDK convention.)

## Endpoint 1 — the feature map (all features at once)

```
GET {base}/esm/protein/api/v1alpha1/features
```

Returns every SAE feature's `feature_index`, `label`, and short `description` in a
single call. This is the whole `feature -> annotation` map, small enough to cache.
`refresh` snapshots it to `assets/feature_annotations.json` so `list` / `search`
work offline as a fallback.

Expected item shape (the script is liberal about the envelope — a bare list or
`{"features": [...]}` / `{"data": [...]}` are all accepted):

```json
{ "feature_index": 12345, "label": "zinc-binding site", "description": "short description" }
```

## Endpoint 2 — one feature's full detail (live only)

```
GET {base}/esm/protein/api/v1alpha1/features/{feature_index}
```

Returns the extended record for a single feature:

- longform description,
- top-activating **UniRef90** and **SwissProt** proteins,
- decoder **nearest-neighbour** features,
- **activation statistics**.

There is no offline substitute for this — it is always a live call and needs an
API key. The skill only falls back to the cached table for the basic
label/description map (endpoint 1), never for this detail.

### How to read "decoder nearest neighbours"

Each SAE feature owns a **decoder vector** — a direction in ESM-C's
representation space that the feature writes back when it is active. The decoder
nearest neighbours are the other features whose decoder vectors are closest to
this one by **cosine similarity** (e.g. neighbour `512` at `0.91` points almost
the same way). Because nearby directions tend to encode nearby biology,
neighbours are usually semantically related features — helpful for interpreting a
vaguely-labelled feature, and for spotting **feature splitting**, where one real
concept is spread across several nearly-parallel features (a common SAE artifact).

This is a property of the learned dictionary's geometry, **not** co-activation:
two features can have near-parallel decoder directions yet rarely fire on the same
residues. For "which proteins light this feature up," use the top-activating
proteins and activation statistics instead. Like everything here, neighbour
relationships are specific to `ESMC-6B-sae-layer60-k64-codebook16384` and do not
carry over to another SAE.

## Divergence

The cached table is a point-in-time snapshot stamped with `fetched_at`. When the
script falls back to it (no key or Biohub unreachable) it prints a reminder that
Biohub is authoritative and may have changed since. Re-run `refresh` with a key to
bring the cache back in sync.
