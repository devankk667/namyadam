# Package: Person 2, image features for the fusion model (image_model_v1)
| Path in this zip | What |
|---|---|
| `data/image_spectral_features_v1.parquet` | 39 spectral features per labelled event |
| `data/image_embeddings_v1.parquet` | CNN embeddings (512-d + PCA-32), frozen-ImageNet PCA-32, image-only probabilities |
| `data/spatial_folds_v1.csv` | the 5 spatial folds, **use these for evaluation** |
| `data/firms_detections_classified.parquet` | 85,819 labelled detections with `event_id`: the detection→event join map |
| `model_card.json`, `MODEL_REPORT.md` | how the features were made, image-only scores |
| `reports/model_comparison.csv`, `reports/metrics_v1.json` | image-only results to compare your fusion against |

Paths in the guide below that start with `outputs/...` refer to Person 1's repo; here the files are in `data/`.

---

# Handoff → Person 2 (ML / Multimodal Fusion)

From: Person 1 (Satellite / CV) · `image_model_v1` · background: `MODEL_REPORT.md`

## What you get
Image-derived features for all **18,929 labelled events**, keyed by `event_id` (the ids in
`firms_event_labels.parquet`). They come from Sentinel-2 imagery, so they are **independent of the OSM
features** the labels were generated from. That independence is what the fusion model needs to break the
circularity described in `TIER_1_2_CONTEXT.md`.

| File | Rows | Use |
|---|---|---|
| `image_spectral_features_v1.parquet` | 18,929 | 39 interpretable spectral features, good for RF/XGB |
| `image_embeddings_v1.parquet` | 18,929 | CNN features: compressed (PCA) and full embeddings, image-only class probabilities |
| `spatial_folds_v1.csv` | 18,929 | the 5 spatial folds all image numbers were produced with |
| `MODEL_REPORT.md` | – | how the features were made, and the image-only scores to compare against |

3 events have **no image** (`has_image = 0`; no cloud-free scene). Their image columns are NaN.

## Columns
**`image_spectral_features_v1.parquet`**: `event_id`, `has_image`, then:
- 7 indices × 3 statistics: `{ndvi, gndvi, mndwi, ndbi, nbr, nbr2, evi}_{mean, std, center}`. `mean`/`std`
  cover the whole 5.12 km patch, `center` covers the central 1 km around the event.
- `frac_vegetation`, `frac_builtup`, `frac_water`, `frac_low_nbr` (burnt/bare-looking): fractions of the patch
- `scl_{veg, bare, water, cloud}_frac`: Sentinel-2 scene-classification fractions
- `B02_center … B12_center`: reflectance per band in the central 1 km
- `b12_center_max`, `b12_hotspot_ratio`, `hot_px_center`, `hot_px_total`: SWIR active-fire / hot-surface
  signals (flares, coal fires, burning fields)

**`image_embeddings_v1.parquet`**: `event_id`, `label`, `label_confidence`, `trusted`, `fold`, `block_id`,
`has_image`, then:
- `p_agricultural_burning`, `p_industrial_fire`, `p_mining_activity`, `p_gas_flare`, `cnn_pred`: image-only
  prediction (out-of-fold, prior-corrected)
- `pca_00 … pca_31`: 32-d PCA of the fine-tuned CNN embedding (out-of-fold)
- `frozen_pca_00 … frozen_pca_31`: 32-d PCA of plain ImageNet ResNet18 features. **Never trained on our
  labels, so leakage-free for any split.** On trusted events these scored as well as the fine-tuned ones.
- `emb_000 … emb_511`: the full 512-d fine-tuned embedding (for a neural fusion model; too wide for trees)

## Joining to your data
Your models are per **detection** (observation); these features are per **event**. Every detection of an
event gets its event's features:
```python
import pandas as pd
obs = pd.read_parquet("firms_obs_event_map.parquet")     # obs_id -> event_id (or datasets_classified/firms_detections_classified.parquet)
spec = pd.read_parquet("image_spectral_features_v1.parquet")
emb = pd.read_parquet("image_embeddings_v1.parquet")
img_cols = [c for c in emb.columns if c.startswith(("pca_", "frozen_pca_", "p_"))]
X = (obs.merge(spec, on="event_id", how="left")
        .merge(emb[["event_id", *img_cols]], on="event_id", how="left"))
```
Suggested feature sets to compare:
| Set | Columns |
|---|---|
| A. tabular (current) | thermal + OSM + temporal |
| B. A + spectral | + 39 spectral features |
| C. A + spectral + frozen | + `frozen_pca_*` (safest, no leakage at all) |
| D. A + spectral + CNN | + `pca_*` + `p_*` |
| E. image only | spectral + `frozen_pca_*` + `pca_*` + `p_*` (no OSM): shows what the images alone carry |

## Leakage rules (important)
1. `pca_*`, `emb_*` and `p_*` are **out-of-fold**: each event was produced by a CNN that never saw its 0.1°
   block. They are clean **if you evaluate with the same folds** (`spatial_folds_v1.csv`). With a different
   split, a test event's image features may come from a CNN that trained on some of your test events.
   Prefer our folds, or use only the spectral + `frozen_pca_*` features, which carry no label information.
2. Do **not** create features with `models/cnn_final.pt` for evaluation. It has seen every labelled event.
3. Keep **whole blocks** together in any split. Also note that one large mining complex spans blocks
   223_825 / 223_826 / 223_827, so mining scores stay slightly optimistic.

## What to report (so the image contribution is measurable)
- Macro-F1 on **trusted** events (`trusted == 1`, n = 281) for sets A–E on the **same folds**. This is the
  number that shows whether circularity is reduced. All-events scores only show agreement with the rules.
- Also report macro-F1 **without gas_flare** (8 events at 2 sites cannot be validated).
- Image-only reference (out-of-fold, trusted): macro-F1 0.41 / 0.55 without flare. Spectral-only: 0.24.
- Feature importance: if the fusion model keeps relying only on OSM features, the image features are not
  being used. Check importance and ablations (A vs B vs D).

## Versions
Dataset: Sentinel-2 L2A via Earth Search, built 2026-09-28 (`dataset_info.json`). Labels: the
`firms_event_labels.parquet` version with 18,929 labelled events. Please confirm it matches yours (the team
doc mentions 18,946). Model: `image_model_v1` (`model_card.json`).
