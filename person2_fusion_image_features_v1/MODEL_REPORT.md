# Image Model v1: Sentinel-2 patch classifier and image features

**Team 1, Person 1 (Satellite / Computer Vision)** · version `image_model_v1` · 2026-09-29

## TL;DR
- For every labelled FIRMS thermal event we fetched a cloud-filtered Sentinel-2 image (5.12 km × 5.12 km,
  6 bands) and trained an **ImageNet-pretrained ResNet18** to classify the event from the image alone.
- On the **281 trusted events** (high/medium-confidence labels), evaluated with **5-fold spatial
  cross-validation**, the image-only model reaches **macro-F1 0.41 (0.55 without gas_flare)**. The majority
  baseline gets 0.12 and hand-made spectral indices get 0.24.
- Image-only, it does **not** reach the tabular OSM+FIRMS model (0.74–0.75). It isn't meant to. Its value is
  a **source of evidence independent of OSM**, which the tabular models lack (see *TIER_1_2_CONTEXT.md*,
  circular reasoning).
- Main outputs: **image features for the fusion model (Person 2)**, **527 image-vs-label disagreements for
  label review (Person 4)**, and a **model for scoring new events in the backend (Person 5)**.

---

## 1. Where it fits in the project
```
FIRMS detections ──► thermal events ──► rule labels (OSM + FRP + duration)
                          │
                          ▼  (Person 1)
             Sentinel-2 L2A patch per event   ← s2_pipeline/  (STAC search, cloud filter, 512 px patch)
                          │
                          ▼  (Person 1, this report)
      ┌───────────────────┼─────────────────────────┬──────────────────────────┐
      ▼                   ▼                         ▼                          ▼
 spectral features   CNN embeddings +          disagreement list          cnn_final.pt
 (39 per event)      class probabilities       (image vs rule label)      + infer.py
      └──────────┬────────┘                          │                          │
                 ▼                                   ▼                          ▼
     Person 2: multimodal fusion           Person 4: reviewed eval set    Person 5: backend API
     (tabular + temporal + image)
```
The tabular models learn from OSM proximity features, and the labels were created from those same
features, so the models mostly re-learn the labelling rule. Satellite imagery shows what is actually on the
ground (fields, factories, open pits, flares, burn scars), so it can confirm or contradict a rule label.

## 2. Data
| | |
|---|---|
| Events | 18,929 labelled events (`firms_event_labels.parquet`, label ≠ unknown) |
| With a usable image | **18,926** (3 had no cloud-free image within ±60 days) |
| Classes (with image) | agricultural_burning 18,133 · industrial_fire 685 · mining_activity 100 · gas_flare 8 |
| Trusted labels | **281** high/medium-confidence events (agri 90, industrial 83, mining 100, flare 8). The other 18,645 (98.5%) are low-confidence rule labels |
| Imagery | Sentinel-2 L2A (Earth Search `sentinel-2-c1-l2a`), scene nearest the event date (median 0 days off), patch cloud ≤ 20% via the SCL mask |
| Patch | 512 × 512 px at 10 m = 5.12 km, centred on the event centroid; bands B02 B03 B04 B08 B11 B12 (surface reflectance) |
| Sites | mining = 10 sites (72 events at one site); gas_flare = 2 sites; industrial trusted = 63 sites |

## 3. Model
- **Architecture:** torchvision ResNet18 with ImageNet weights. The first convolution is widened from 3 to
  6 input bands: the RGB filters are copied onto B04/B03/B02, NIR/SWIR start from the mean RGB filter, and
  everything is rescaled by 3/6. Head: dropout 0.2 → linear layer (4 classes). The 512-d pooled feature is the
  **embedding**.
- **Input:** 512 px patch averaged 2×2 → 256 px (20 m, full 5.12 km context), per-band z-score.
- **Training:** AdamW (lr 3e-4 backbone, 1.5e-3 head, weight decay 1e-4), cosine schedule with warm-up,
  15 epochs × 8,000 samples, batch 64, fp16 mixed precision, label smoothing 0.05. Augmentation: random 90°
  rotations and flips (an overhead view has no "up") and ±10% per-band gain.
- **Imbalance and label quality:** class-balanced sampling (class share ∝ count^0.25 → 54/24/15/8%). Within
  a class, trusted events are sampled 5× more often than rule-labelled ones.
- **Prior correction:** at prediction time, probabilities are divided by the training class share and
  renormalised. This removes the bias towards agricultural that the sampling leaves behind (+0.03 trusted
  macro-F1).
- **Hardware / time:** RTX 4060 Laptop, about 615 img/s, about 3.5 min per fold; the whole pipeline takes about 45 min.

## 4. How it was evaluated
- **Spatial 5-fold cross-validation** (`spatial_folds_v1.csv`): StratifiedGroupKFold on **0.1° blocks**,
  the team's block size. A site is never in train and test at the same time. Every event gets an
  **out-of-fold** prediction and embedding.
- **Headline metric = macro-F1 on trusted events.** Scores on all events mainly measure agreement with
  the OSM rules (the Tier 1 lesson: F1 = 1.00 there was meaningless). Plain accuracy is not reported as a
  headline, because "always agricultural" already scores 96% on all events.
- **Model selection:** two class-balancing settings were tried, with the selection rule fixed before seeing
  results (higher trusted macro-F1 without flare). They tied (0.548 vs 0.545), and the original was kept. Both
  are reported. Choosing on the same 281 events flatters the winner slightly.

## 5. Results
### 5.1 Model comparison (out-of-fold, same folds for every model)
| Model | Trusted macro-F1 | Trusted, no flare | Trusted balanced acc. | All-events macro-F1 |
|---|---|---|---|---|
| Majority class ("agricultural") | 0.12 | 0.16 | 0.25 | 0.25 |
| Spectral indices + gradient boosting | 0.24 | 0.32 | 0.30 | 0.42 |
| Frozen ImageNet ResNet18 + logistic regression | **0.41** | **0.55** | 0.42 | 0.42 |
| **Fine-tuned ResNet18, prior-corrected (selected)** | **0.41** | **0.55** | **0.44** | **0.51** |
| Fine-tuned ResNet18, raw probabilities | 0.38 | 0.50 | 0.42 | 0.50 |
| Fine-tuned, stronger balancing (α = 0.1), prior-corrected | 0.41 | 0.55 | 0.44 | 0.51 |

### 5.2 Selected model, per class
| Class | Trusted: precision / recall / F1 (n) | All events: precision / recall / F1 (n) |
|---|---|---|
| agricultural_burning | 0.43 / 0.94 / 0.59 (90) | 0.98 / 0.99 / 0.99 (18,133) |
| industrial_fire | 0.79 / 0.54 / 0.64 (83) | 0.68 / 0.68 / 0.68 (685) |
| mining_activity | 0.96 / 0.26 / 0.41 (100) | 0.67 / 0.26 / 0.37 (100) |
| gas_flare | 0 / 0 / 0 (8) | 0 / 0 / 0 (8) |

Trusted macro-F1 per fold: 0.51 · 0.31 · 0.29 · 0.52 · 0.86 (mean 0.50 ± 0.21). The large spread comes from
the tiny number of trusted events and sites in each fold.

Confusion matrices: `reports/cnn_confusion_trusted_events.png`, `reports/cnn_confusion_all_events.png`.

### 5.3 What the numbers say
1. **The imagery carries real signal.** Industrial sites are found with 79% precision, and mining predictions
   are right 96% of the time. The frozen ImageNet features (no training on our labels at all) already reach 0.41.
2. **Fine-tuning on rule labels reproduces the rule.** Fine-tuning improves agreement with the rule labels
   (all-events macro-F1 0.42 → 0.51) but not the trusted score (0.41 → 0.41). This is the circularity issue
   showing up in the image model: 98.5% of the training labels are rule outputs.
3. **Mining is limited by sites, not by the model.** 100 events come from only 10 locations, and 72 are at one
   mining complex. Recall on that complex is 21/72, and on the other 9 sites 5/28. The complex also spans
   neighbouring 0.1° blocks (223_825/826/827), so block-level splitting does not fully separate it.
4. **Gas flare cannot be learned from 2 sites.** All 8 flares were predicted *industrial_fire*, which is
   visually reasonable, since the flares sit inside refineries.
5. **Hand-made spectral indices are weak on their own** (0.24). The CNN learns far richer spatial context
   (layout of plants, pits and field patterns) than scene-average indices capture.

### 5.4 Comparison with the tabular baseline
The tabular RF/XGB reaches 0.74–0.75 macro-F1 on its trusted seed set. That number is **not directly
comparable**: it uses a different split and observation-level rows, and its inputs include the OSM features
the labels were derived from. The fair test of the image model's value is **Person 2's fusion model**:
tabular vs tabular + image features, on the same folds and trusted events.

## 6. How it contributes to the project
| Contribution | For | File |
|---|---|---|
| **OSM-independent image features** for fusion: 39 spectral features, 512-d embedding + 32-d PCA, class probabilities (out-of-fold), and leakage-free 32-d frozen-ImageNet PCA | Person 2 | `deliverables/image_spectral_features_v1.parquet`, `deliverables/image_embeddings_v1.parquet` |
| **Shared spatial folds**, so image, tabular and fusion models are compared on identical test events | Person 2, Person 4 | `deliverables/spatial_folds_v1.csv` |
| **Label-review candidates:** 527 events where the image model disagrees with the rule label (e.g. 50 "agricultural" events that look industrial, 85 "industrial" events that look agricultural), sorted by confidence, with scene ids and preview paths | Person 4 | `reports/cnn_label_disagreements.csv` |
| **Image-only evaluation** (the Person 1 deliverable "evaluate image-only classification") | Person 4, final report | `reports/model_comparison.csv`, `reports/metrics_v1.json` |
| **Deployable model** for new events: patch → class probabilities + embedding (about 50 ms per patch on GPU) | Person 5 | `deliverables/models/cnn_final.pt`, `image_model/infer.py`, `deliverables/model_card.json` |

## 7. Limitations and next steps
- **Labels:** 98.5% rule-generated. Real progress needs more reviewed labels. The disagreement list is the
  fastest way to find informative events to review (and `label_review.csv` from the team should be added).
- **Rare classes:** mining (10 sites) and gas flare (2 sites) need more independent locations. Candidates
  are known flare/mine locations from other sources, or events predicted as such among the 1.09M unknowns,
  after manual review.
- **One image per event:** no before/after comparison. A pre-fire scene would allow dNBR (burn severity) and
  change features, a strong signal for agricultural burning.
- **Sentinel-1 radar** (smoke/cloud-independent) is not included.
- **Split alignment:** once the team's official test blocks are shared, re-run the evaluation on them for a
  like-for-like number.
- **Resolution:** training used 256 px (20 m). A 512 px (10 m) run is possible (about 4× slower) if finer
  detail proves useful.

## 8. Reproduce
From the project folder with the `mlproj` environment (RTX 4060, total about 45 min):
```
python -m image_model.prepare             # 256 px cache + spectral features       (~5 min)
python -m image_model.splits              # spatial 5-fold split                   (seconds)
python -m image_model.baselines           # majority, spectral, frozen ImageNet    (~4 min)
python -m image_model.train --epochs 15 --epoch-size 8000            # 5-fold CV   (~18 min)
python -m image_model.train --final --epochs 15 --epoch-size 8000    # final model (~4 min)
python -m image_model.report --tag cv     # metrics, plots, deliverables           (~1 min)
```
Inputs: `datasets_classified/satellite_patches/` (built by `s2_pipeline`, see `s2_pipeline/README.md`).
Outputs: `outputs/image_model_v1/` (`deliverables/`, `reports/`, `models/`, `predictions/`, `cache/`).
