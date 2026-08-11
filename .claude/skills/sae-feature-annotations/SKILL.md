---
name: sae-feature-annotations
description: >-
  Look up what a Biohub ESM-C sparse-autoencoder (SAE) feature *means* — its label,
  description, top-activating proteins, decoder neighbours, and activation statistics —
  by querying the Biohub feature-annotation API, with a local lookup table as an offline
  fallback. Use this whenever the user has SAE feature indices (e.g. from `boileroom`'s
  `SAE` model / `pooled_features`) and wants to interpret them: "what is SAE feature
  12345", "which features fired on my protein and what do they correspond to", "annotate
  these feature indices", "what proteins most activate this feature", "find the
  zinc-binding SAE feature", "decoder nearest neighbours of feature N", or any request to
  turn ESM-C SAE feature numbers into biology. Also trigger when the user mentions Biohub
  feature annotations, the feature viewer, or the ESMC-6B layer-60 SAE codebook. These
  annotations are only valid for the `ESMC-6B-sae-layer60-k64-codebook16384` SAE — the
  skill says so up front and refuses to pretend otherwise.
---

# Annotating ESM-C SAE features from Biohub

This skill turns **SAE feature indices into biology**. A sparse autoencoder trained on
ESM-C representations decomposes each residue into a sparse set of interpretable features
(directions in a codebook). Biohub publishes an annotation for each feature — a human
label, top-activating proteins, decoder neighbours, activation stats — and this skill
fetches and presents them, with a bundled offline table as a fallback.

## The one thing to say first, every time

**These annotations are only valid for the SAE `ESMC-6B-sae-layer60-k64-codebook16384`**
— ESM-C 6B, transformer layer 60, TopK k=64, codebook of 2**14 = 16384 features. This is
the SAE from *Language Modeling Materializes a World Model of Protein Biology* (Biohub,
2026), and the default Forge SAE that `boileroom`'s `SAE` model uses.

A `feature_index` means something completely different under any other layer, k, or
codebook. So before interpreting anything, **confirm the user produced their features with
this exact SAE** (in `boileroom` that is the default `feature_source="forge"` path, i.e.
`forge_sae_model = "esmc-6b-2024-12-sae-layer60-k64-codebook16384"`). If they used a local
300M/600M SAE or a different layer, tell them these labels do not apply and stop — do not
hand them annotations that describe a different feature basis.

Lead with this. Do not bury it under the results.

## Two sources, and which to trust

Biohub is authoritative. **Always query it first** when an API key is available; only fall
back to the local table when you can't reach Biohub, and say so.

| Source | What it gives | When |
| --- | --- | --- |
| **Biohub (live)** | Everything. The full map *and* per-feature detail. | Preferred, always try first. Needs an API key. |
| **Local table** (`assets/feature_annotations.json`) | Only the basic map: `feature_index -> label + short description`. | Fallback when there's no key or no network. Warn that it may have drifted. |

When you fall back to the local table, **remind the user it is a point-in-time snapshot and
may have diverged from Biohub** if the annotations were updated since it was last
refreshed — and that the extended detail below is simply not available offline. The script
prints this reminder automatically; reinforce it in your summary.

## Two endpoints = two levels of detail

1. **The whole map, in one call** — `GET /esm/protein/api/v1alpha1/features` returns every
   feature's `feature_index`, `label`, and short `description` at once. This is the
   `feature -> annotation` map. It is small, so it is the thing cached locally for offline
   use (`refresh` snapshots it).

2. **One feature, in full** — `GET /esm/protein/api/v1alpha1/features/{feature_index}`
   returns the longform description, **top-activating UniRef90 and SwissProt proteins**,
   **decoder nearest-neighbour features**, and **activation statistics**. This is only
   available **live** — there is no offline substitute. Any request that wants proteins,
   neighbours, or stats *requires* the live call and thus an API key.

See `references/api.md` for the exact request/response shapes, base URL, and auth details.

## The tool

Everything goes through one stdlib-only script (no dependencies to install):

```
scripts/sae_features.py
  refresh                 # fetch the full map from Biohub and (re)write the offline table
  list   [--limit N]      # list features (live if a key is set, else cache)
  search <text>           # find features whose label/description matches <text>
  get    <feature_index>  # full detail for one feature (LIVE ONLY, needs a key)
  # shared flags: --base-url, --token, --cache, --sae-model, --json
```

Auth resolves from `--token`, then `ESM_API_KEY`, then `FORGE_TOKEN` (the same key
`boileroom`'s Forge backend uses). Base URL defaults to `https://biohub.ai`
(`--base-url` / `BIOHUB_BASE_URL` to override).

## Workflow

1. **State the validity constraint** (the section above) and confirm the user's features
   came from `ESMC-6B-sae-layer60-k64-codebook16384`. If not, stop and explain why the
   labels don't apply.

2. **Figure out what they need**, because it decides whether a key is required:
   - Just labels / short descriptions for some indices, or a keyword search →
     `list` / `search`. Works offline against the cached table if needed.
   - Top-activating proteins, decoder neighbours, or activation stats for a feature →
     `get`. **Live only** — a key is mandatory.

3. **Get the API key.** Prefer the user's own key via `ESM_API_KEY` (or `--token`). Never
   ask them to paste a secret into the chat if an env var will do; never store it. If they
   have no key and only want basic labels, proceed against the cache.

4. **Query live first.** Run the relevant subcommand with the key. If Biohub is
   unreachable, the script falls back to the cache for `list`/`search` and prints a
   divergence warning — surface that to the user. `get` cannot fall back; if it fails,
   report the error rather than substituting anything.

5. **Seed the cache when appropriate.** If the offline table is empty (never seeded) or
   stale and the user has a key, run `refresh` so future offline lookups work. This writes
   `assets/feature_annotations.json` — mention that you updated it.

6. **Present results plainly.** Give the label and description for each index; for `get`,
   summarize the top proteins, nearest-neighbour features, and activation stats. Keep the
   feature index next to every annotation so the mapping is unambiguous.

## Notes

- The offline table may ship **unseeded** (empty `features`) if it was never refreshed with
  a key. In that case `list`/`search` will tell you to run `refresh`. That is expected —
  don't invent labels; get a key and refresh, or query live.
- If Biohub rejects the bearer token or the endpoint shape differs from
  `references/api.md`, the auth/parse logic lives in one place (`_http_get_json` /
  `_live_feature_map` in the script) — adjust there. The defaults follow the Forge SDK
  convention (`Authorization: Bearer <token>`, base `https://biohub.ai`).
- This skill only *reads* annotations. Producing the SAE features themselves (running
  ESM-C + the SAE to get `feature_index` values for a sequence) is `boileroom`'s `SAE`
  model, on the `features/sae` branch — a different tool.
