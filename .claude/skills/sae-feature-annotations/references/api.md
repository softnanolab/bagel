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

The shapes below were verified against the live API (`https://biohub.ai`).

## Base URL and auth

- **Base URL:** `https://biohub.ai` (same host as `boileroom`'s `forge_url`).
  Override with `--base-url` or `BIOHUB_BASE_URL`.
- **Auth:** the annotation endpoints are **public reads** — they return data for
  an anonymous request. The script sends `Authorization: Bearer <token>`, using
  `--token` / `ESM_API_KEY` / `FORGE_TOKEN` when set, otherwise an anonymous
  placeholder. A real key is only needed if a deployment enforces auth and rejects
  the anonymous call.
- Auth is applied in one place — `_http_get_json` in `scripts/sae_features.py`. If
  Biohub ever expects a different header name/scheme, change it there.

## Endpoint 1 — the feature map (all features at once)

```
GET {base}/esm/protein/api/v1alpha1/features
```

Returns every SAE feature's `feature_index`, `label`, and short `description` in a
single call (all 16384 features). This is the whole `feature -> annotation` map,
behind `list` and `search`.

The response is a `{"data": [...]}` envelope (the script is also liberal and
accepts a bare list or `{"features": [...]}`). Each item:

```json
{ "feature_index": 0, "label": "Nudix N-terminal substrate-binding loop", "description": "short description" }
```

## Endpoint 2 — one feature's full detail

```
GET {base}/esm/protein/api/v1alpha1/features/{feature_index}
```

Returns a single flat JSON object (no envelope) with the extended record. Verified
fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `feature_index` | int | The feature (0…16383). |
| `label` | str | Short human label. |
| `summary` | str | One-line summary. |
| `description` | str | Longform description (a `Summary: … / Activation pattern: … / Exemplars: …` block). |
| `activation_pattern` | str | Where along the sequence the feature fires. |
| `category` | str | Coarse category, e.g. `Compositional bias`. |
| `exemplar_protein_families` | str | Prose list of exemplar families. |
| `uniref90_frequency` | int | How many UniRef90 proteins activate it. |
| `uniref90_idf` | float | Inverse-document-frequency weight (rarer = higher). |
| `uniref90_max_activation` | float | Peak activation seen across UniRef90. |
| `top_100_uniref_ids` | list | Top-activating UniRef90 proteins: `[{"uniref_id": ..., "activation": ...}]`. |
| `top_swissprot_activations` | list | Top-activating SwissProt proteins: `[{"uniprot_id": ..., "activation": ...}]`. |
| `decoder_nearest_neighbors` | list[int] | Nearest-neighbour **feature indices** (see below). |
| `threshold` | float | Activation threshold for the feature. |

### How to read "decoder nearest neighbours"

Each SAE feature owns a **decoder vector** — a direction in ESM-C's representation
space that the feature writes back when it is active. The decoder nearest
neighbours are the other features whose decoder vectors point most nearly the same
way (cosine similarity in decoder space). The API returns them as a **plain list
of neighbour feature indices** (e.g. `[8686, 12927, 3985, …]`), highest-similarity
first; it does not return the similarity scores themselves. Look each neighbour up
with `get` (or `list`) to see what it means.

Because nearby directions tend to encode nearby biology, neighbours are usually
semantically related features — helpful for interpreting a vaguely-labelled
feature, and for spotting **feature splitting**, where one real concept is spread
across several nearly-parallel features (a common SAE artifact).

This is a property of the learned dictionary's geometry, **not** co-activation:
two features can have near-parallel decoder directions yet rarely fire on the same
residues. For "which proteins light this feature up," use `top_100_uniref_ids` /
`top_swissprot_activations` and the activation statistics instead. Like everything
here, neighbour relationships are specific to
`ESMC-6B-sae-layer60-k64-codebook16384` and do not carry over to another SAE.
