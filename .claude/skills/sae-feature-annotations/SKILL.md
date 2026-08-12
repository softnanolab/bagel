---
name: sae-feature-annotations
description: >-
  Look up what a Biohub ESM-C sparse-autoencoder (SAE) feature *means* — its label,
  description, top-activating proteins, decoder neighbours, and activation statistics —
  by querying the Biohub feature-annotation API. Use this whenever the user has SAE
  feature indices (e.g. from `boileroom`'s `SAE` model / `pooled_features`) and wants to
  interpret them: "what is SAE feature 12345", "which features fired on my protein and
  what do they correspond to", "annotate these feature indices", "what proteins most
  activate this feature", "find the zinc-binding SAE feature", "decoder nearest neighbours
  of feature N", or any request to turn ESM-C SAE feature numbers into biology. Also
  trigger when the user mentions Biohub feature annotations, the feature viewer, or the
  ESMC-6B layer-60 SAE codebook. These annotations are only valid for the
  `ESMC-6B-sae-layer60-k64-codebook16384` SAE — the skill says so up front and refuses to
  pretend otherwise.
---

# Annotating ESM-C SAE features from Biohub

This skill turns **SAE feature indices into biology**. A sparse autoencoder trained on
ESM-C representations decomposes each residue into a sparse set of interpretable features
(directions in a codebook). Biohub publishes an annotation for each feature — a human
label, top-activating proteins, decoder neighbours, activation stats — and this skill
fetches and presents them by querying the Biohub API live.

## The one thing to say first, every time

**These annotations are only valid for the SAE `ESMC-6B-sae-layer60-k64-codebook16384`**
— ESM-C 6B, transformer layer 60, TopK k=64, codebook of 2**14 = 16384 features. This is
the SAE from *Language Modeling Materializes a World Model of Protein Biology* (Biohub,
2026), and the default Forge SAE that `boileroom`'s `SAE` model uses.

A `feature_index` (0…16383) means something completely different under any other layer, k,
or codebook. So before interpreting anything, **confirm the user produced their features
with this exact SAE** (in `boileroom` that is the default `feature_source="forge"` path,
i.e. `forge_sae_model = "esmc-6b-2024-12-sae-layer60-k64-codebook16384"`). If they used a
local 300M/600M SAE or a different layer, tell them these labels do not apply and stop — do
not hand them annotations that describe a different feature basis.

Lead with this. Do not bury it under the results.

## One source: the live Biohub API

There is no bundled offline table. The two Biohub annotation endpoints are **public
reads**, so the skill always queries them live. An API key is *optional*: if the user has
one (`ESM_API_KEY` / `FORGE_TOKEN`, the same credential `boileroom`'s Forge backend uses)
it is sent; if not, the request is made with an anonymous placeholder token, which is
enough for the annotation endpoints. Base URL defaults to `https://biohub.ai`.

If a deployment ever enforces auth and the anonymous call is rejected, the fix is to set
`ESM_API_KEY`. That is the only case where a key matters for *reading annotations* —
producing the features themselves is a different tool (see the last note).

## Two endpoints = two levels of detail

1. **The whole map, in one call** — `GET /esm/protein/api/v1alpha1/features` returns every
   feature's `feature_index`, `label`, and short `description` at once (all 16384 in a
   `{"data": [...]}` envelope). This is the `feature -> annotation` map, behind `list` and
   `search`.

2. **One feature, in full** — `GET /esm/protein/api/v1alpha1/features/{feature_index}`
   returns the longform description, activation pattern, category, **top-activating
   UniRef90 and SwissProt proteins**, **decoder nearest-neighbour feature indices**, and
   **activation statistics**. This is behind `get`.

See `references/api.md` for the exact request/response shapes, base URL, and auth details.

## The tool

Everything goes through one stdlib-only script (no dependencies to install):

```
scripts/sae_features.py
  list   [--limit N]      # list features (map endpoint)
  search <text>           # find features whose label/description matches <text>
  get    <feature_index>  # full detail for one feature (detail endpoint)
  # shared flags: --base-url, --token, --json
```

Auth resolves from `--token`, then `ESM_API_KEY`, then `FORGE_TOKEN`, else the anonymous
placeholder. Base URL defaults to `https://biohub.ai` (`--base-url` / `BIOHUB_BASE_URL` to
override).

## Workflow

1. **State the validity constraint** (the section above) and confirm the user's features
   came from `ESMC-6B-sae-layer60-k64-codebook16384`. If not, stop and explain why the
   labels don't apply.

2. **Figure out what they need:**
   - Just labels / short descriptions for some indices, or a keyword search →
     `list` / `search`.
   - Top-activating proteins, decoder neighbours, or activation stats for a feature →
     `get`.

3. **Query.** Run the relevant subcommand. It works with or without an API key; prefer the
   user's own key via `ESM_API_KEY` (or `--token`) when they have one, but never ask them
   to paste a secret into the chat if an env var will do, and never store it.

4. **Present results plainly.** Give the label and description for each index; for `get`,
   summarize the top proteins, decoder nearest-neighbour feature indices, and activation
   statistics. Keep the feature index next to every annotation so the mapping is
   unambiguous.

## Notes

- Don't invent labels. If the live call fails, report the error rather than guessing —
  there is no offline substitute.
- If Biohub rejects the request or the endpoint shape differs from `references/api.md`, the
  auth/parse logic lives in one place (`_http_get_json` / `_live_feature_map` in the
  script) — adjust there. The defaults (`Authorization: Bearer <token>`, base
  `https://biohub.ai`, `{"data": [...]}` envelope) are verified against the live API.
- This skill only *reads* annotations. Producing the SAE features themselves (running
  ESM-C + the SAE to get `feature_index` values for a sequence) is `boileroom`'s `SAE`
  model, on the `features/sae` branch — a different tool.
