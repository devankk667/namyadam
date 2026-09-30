# Tier 1 & Tier 2 Semi-Supervised Training Pipeline

## Overview

This document provides complete context for the OSM+FIRMS baseline training work (Tier 1 & Tier 2) and serves as a handoff guide for the next phase: **Sentinel-1/2 imagery integration**.

**Current Status**: Feature branch `feature/osm-frism-baseline` contains all Tier 1 & 2 training results, trained models, and evaluation metrics.

---

## 🎯 The Problem We Solved

### Starting Point
- **300 high/medium-confidence labeled events** → 8,765 trainable observations
- **Macro F1 Score**: 0.74 (modest but reliable baseline)
- **Data bottleneck**: Too few labels to train robust models

### The Goal
Expand labeled data while maintaining quality, then scale with semi-supervised learning.

---

## 📊 Results Summary

### Tier 1: Low-Confidence Events Included
| Metric | Value |
|--------|-------|
| Labeled Events | 18,946 (+63x) |
| Trainable Observations | 86,425 (+10x) |
| RF Macro F1 | 1.00 ⚠️ |
| XGB Macro F1 | 1.00 ⚠️ |

**Status**: ❌ **Red Flag Detected** — Perfect F1 indicates circular reasoning

### Tier 2: Semi-Supervised Self-Training (3 Rounds)

| Round | Training Type | Total Obs | New Pseudo-Labels | Threshold | Macro F1 |
|-------|--------------|-----------|------------------|-----------|----------|
| 0 | Seed (high/med) | 5,922 | 0 | — | 0.7388 |
| 1 | Seed + pseudo | 66,559 | 60,637 | 0.95 | 0.75 |
| 2 | Cumulative | 126,841 | 60,282 | 0.93 | 0.75 ← **Plateau** |
| 3 | Cumulative | 177,325 | 50,484 | 0.91 | 0.75 ← **Plateau** |

**Status**: ⚠️ **Data Plateau** — 30x expansion, only +0.01 F1 improvement

---

## 🚨 Critical Finding: Circular Reasoning

### The Problem

The current models suffer from **circular reasoning**:

1. **Labels created from OSM rules**: Low-confidence events were labeled using OpenStreetMap proximity rules:
   - `nearest_osm_distance_km < threshold`
   - `industrial_count_1km > threshold`
   - `power_plant_count_5km > threshold`
   - etc.

2. **Features also use OSM metrics**: The model feature set includes:
   - `nearest_osm_distance_km`
   - `industrial_count_1km`, `industrial_count_2km`, `industrial_count_5km`
   - `power_plant_count_5km`
   - `flare_count_5km`
   - `petroleum_well_count_5km`
   - etc.

3. **Models learn the labeling rule, not fire patterns**: 
   - Model feature set ≈ Labeling function inputs
   - Models achieve "perfect" F1 (1.00) by reproducing the OSM proximity rule
   - Real-world fire detection relies on OSM proximity, not actual thermal/spectral signatures

4. **Self-training amplifies the circularity**:
   - Models trained on 5,922 seed observations
   - Predict on 1.89M unknowns using OSM features
   - High-confidence predictions (>0.95) become pseudo-labels
   - Retrain on seed + pseudo-labels
   - Result: Models just get better at the OSM rule, not real fire detection

### Why F1 Plateaus at 0.75

- **Round 0 (seed)**: F1 = 0.7388 (legitimate from true labels)
- **Round 1**: F1 = 0.75 (model + 60k pseudo-labels from OSM rule)
- **Rounds 2-3**: F1 = 0.75 (no improvement despite 2x more pseudo-labels)

The plateau proves the model has extracted all the signal that OSM proximity features can provide. Adding more of the same type of data doesn't help.

---

## 📁 Repository Structure

### Key Files

```
data/processed/
├── firms_osm_enriched.parquet           (1.1M observations with OSM features)
├── firms_temporal.parquet               (temporal persistence features)
├── firms_thermal_events.parquet         (event centroids: lat, lon, date)
├── firms_event_labels.parquet           (18,946 labeled events)
├── firms_obs_event_map.parquet          (obs_id → event_id mapping)
└── firms_1.1M_predictions.parquet       (full dataset predictions)

ml/scripts/
├── train_rf_baseline.py                 (RandomForest trainer, --min-conf flag)
├── train_xgb_temporal.py                (XGBoost with temporal features)
├── self_train.py                        (Tier 2 self-training pipeline)
└── predict_full_dataset.py              (Batch inference on 1.1M obs)

ml/models/
├── rf_baseline/
│   ├── model.joblib
│   └── metadata.json                    (features, F1, train/test splits)
├── xgb_temporal/
│   ├── model.joblib
│   └── metadata.json
└── self_trained_rf{0..3}/               (Tier 2 trained models)
    ├── model.joblib
    └── metadata.json

eda/outputs/
├── self_train_log.json                  (per-round Tier 2 metrics)
├── full_1.1M_classification_summary.json (predictions on full dataset)
├── label_audit.json                     (label confidence breakdown)
└── osm_enrichment_qa.json               (OSM feature coverage stats)
```

### Model Features

**OSM + Thermal Features (15 total)**:
```python
[
    "bright_ti4", "bright_ti5",          # Thermal bands (FIRMS)
    "temp_diff",                          # Temperature difference
    "frp",                                # Fire Radiative Power
    "conf_ord",                           # Confidence ordinal
    "is_night",                           # Day/night flag
    "nearest_osm_distance_km",            # Distance to nearest OSM feature
    "industrial_count_1km",               # Count of OSM features in 1km radius
    "industrial_count_2km",
    "industrial_count_5km",
    "power_plant_count_5km",
    "quarry_count_5km",
    "flare_count_5km",
    "petroleum_well_count_5km",
    "industrial_landuse_nearby",          # Boolean: is nearby OSM landuse industrial
]
```

**Temporal Features (11 additional for XGBoost)**:
```python
[
    "prev_7d", "prev_14d", "prev_30d",   # Fire counts in past N days
    "time_since_prev_days",               # Days since previous detection
    "days_since_first_seen",              # Days since first appearance
    "cell_prior_count",                   # Historical count for this cell
    "recurrence_per_day",                 # Detection rate
    "frp_prior_mean", "frp_prior_std",   # Historical FRP statistics
    "frp_trend_slope",                    # FRP trend
    "event_prev_count",                   # Historical event count
]
```

---

## 🔄 Training Pipelines

### Baseline: High/Medium Confidence Only
```bash
python ml/scripts/train_rf_baseline.py --min-conf medium
python ml/scripts/train_xgb_temporal.py --min-conf medium
```
- **Input**: 300 labeled events, 8,765 observations
- **Output**: `ml/models/rf_baseline/model.joblib`, `ml/models/xgb_temporal/model.joblib`
- **F1**: 0.74 (reliable baseline)

### Tier 1: Include Low-Confidence Events
```bash
python ml/scripts/train_rf_baseline.py --min-conf low
python ml/scripts/train_xgb_temporal.py --min-conf low
```
- **Input**: 18,946 labeled events (high/medium/low), 86,425 observations
- **Output**: Same paths (overwrites)
- **F1**: 1.00 ⚠️ (circular reasoning detected)

### Tier 2: Semi-Supervised Self-Training
```bash
python ml/scripts/self_train.py --rounds 3 --threshold 0.95 --decay 0.02
```
- **Input**: 5,922 seed observations (high/medium only)
- **Process**:
  - Round 0: Train on seed only
  - Rounds 1-3: Predict full 1.89M, accept >threshold probability as pseudo-labels, retrain
- **Output**: `ml/models/self_trained_rf{0..3}/model.joblib`, `eda/outputs/self_train_log.json`
- **F1 progression**: 0.7388 → 0.75 → 0.75 → 0.75 (plateau)

### Full Dataset Prediction
```bash
python ml/scripts/predict_full_dataset.py
```
- **Input**: 1.1M observations, trained models
- **Output**: `data/processed/firms_1.1M_predictions.parquet`, `eda/outputs/full_1.1M_classification_summary.json`

---

## 🎓 Self-Training Details (Tier 2)

### Algorithm
1. Train classifier on small seed set (5,922 high/medium observations)
2. Predict on all unlabeled observations (1.89M)
3. Accept predictions where `max_probability >= threshold` as pseudo-labels
4. **Per-class cap**: Limit pseudo-labels to 30,000 per class (prevent class imbalance)
5. Merge seed + pseudo-labeled observations
6. Retrain on combined set
7. Repeat: lower threshold, predict again, retrain (Rounds 1-3)

### Safeguards
- **Held-out test set**: Spatial GroupShuffleSplit creates test blocks that are never seen during training (prevents leakage)
- **Test set fixed across all rounds**: Same test data used to evaluate all 4 models (ensures fair comparison)
- **Per-class pseudo-label caps**: Prevents agricultural_burning from overwhelming other classes
- **Threshold decay**: 0.95 → 0.93 → 0.91 (gradually relax confidence requirement)

### Results
```json
{
  "rounds": [
    {"round": 0, "train_obs": 5922, "pseudo_obs": 0, "macro_f1": 0.7388},
    {"round": 1, "train_obs": 66559, "pseudo_obs": 60637, "macro_f1": 0.75},
    {"round": 2, "train_obs": 126841, "pseudo_obs": 60282, "macro_f1": 0.75},
    {"round": 3, "train_obs": 177325, "pseudo_obs": 50484, "macro_f1": 0.75}
  ]
}
```

---

## ⚡ Key Metrics

### Data Expansion
- Baseline: 8,765 obs
- Tier 1: 86,425 obs (+10x)
- Tier 2: 177,325 obs (+30x from seed, +20x from baseline)

### F1 Score Progression
- Baseline: 0.74 (high/medium only)
- Tier 1: 1.00 (low included, circular reasoning)
- Tier 2: 0.75 (plateau after Round 1)

### Why the Plateau?
The 30x data expansion only improved F1 by 0.01 (0.74 → 0.75). This is not a limitation of self-training — it's a symptom of **circular reasoning**. Models have learned all the signal that OSM proximity features can provide.

---

## 🛰️ Next Phase: Sentinel Imagery Integration

### The Opportunity

Sentinel-1/2 provide **independent ground truth** that breaks the OSM circularity:

| Data Source | Independence | Signal |
|-------------|--------------|--------|
| FIRMS + OSM | ❌ Circular | Location + OSM rules |
| Sentinel-2 | ✓ Independent | Burn scars, NDVI drops, spectral signatures |
| Sentinel-1 | ✓ Independent | Smoke/moisture patterns, SAR backscatter |

### Expected Improvements

- **Current F1**: 0.75 (OSM+FIRMS)
- **Projected F1**: 0.80–0.85 (OSM+FIRMS+Sentinel)
- **Why**: Sentinel features are independent of OSM labels, allowing models to learn real fire patterns

### Recommended Approach

1. **Extract Sentinel features**:
   - Sentinel-2: NDVI, NDBI, NBR, GNDVI, EVI, burn scars
   - Sentinel-1: VV/VH backscatter, coherence changes, σ0 trends
   - Resample to 30m, align with FIRMS observations

2. **Merge with existing features**:
   - Combine OSM + thermal + temporal + Sentinel
   - Create new feature set with ~30+ features

3. **Retrain models**:
   - Use same GroupShuffleSplit strategy (prevent leakage)
   - Compare against Tier 2 baseline (0.75 F1)
   - Document improvements in new metrics file

4. **Validate circularity is broken**:
   - If F1 improves to 0.80+, circularity is broken ✓
   - If F1 stays at 0.75, Sentinel features weren't useful ✗

---

## 📚 Files You'll Use

### Training Data (Required)
```
data/processed/
├── firms_thermal_events.parquet       (~26k events, use for Sentinel queries)
├── firms_event_labels.parquet         (18,946 labeled, ground truth)
├── firms_obs_event_map.parquet        (1.1M obs → event mapping)
├── firms_osm_enriched.parquet         (1.1M obs with OSM features, reference)
└── firms_1.1M_predictions.parquet     (current predictions, for comparison)
```

### Code to Reference
```
ml/scripts/
├── train_rf_baseline.py               (RF trainer pattern)
├── train_xgb_temporal.py              (XGBoost pattern)
├── self_train.py                      (Self-training logic)
└── predict_full_dataset.py            (Batch prediction pattern)
```

### Results to Beat
```
eda/outputs/
├── self_train_log.json                (Tier 2 baseline: 0.75 F1)
├── full_1.1M_classification_summary.json (current predictions)
└── label_audit.json                   (label confidence breakdown)
```

---

## 💡 Implementation Checklist for Sentinel Integration

### Before Starting
- [ ] Understand the circular reasoning issue (read this document + commit message)
- [ ] Review `self_train_log.json` (Tier 2 baseline metrics)
- [ ] Check the 4 classes: agricultural_burning, gas_flare, industrial_fire, mining_activity
- [ ] Understand GroupShuffleSplit on 0.1° spatial blocks (prevents leakage)

### Sentinel Data Preparation
- [ ] Set up Copernicus Data Space / Google Earth Engine access
- [ ] Create `ml/features/extract_sentinel_features.py`
- [ ] Query Sentinel-2 (10m, 5/10-day revisit) and Sentinel-1 SAR
- [ ] Generate `firms_sentinel_features.parquet` for all 1.1M observations
- [ ] Validate feature distributions, handle cloud cover, SAR gaps

### Model Retraining
- [ ] Combine features: OSM + temporal + Sentinel
- [ ] Modify `train_rf_baseline.py` to load Sentinel features
- [ ] Modify `train_xgb_temporal.py` similarly
- [ ] Retrain with same GroupShuffleSplit strategy
- [ ] Document results in `eda/outputs/sentinel_baseline_results.json`

### Validation
- [ ] Compare new F1 against Tier 2 baseline (0.75)
- [ ] If F1 > 0.80, circularity is broken ✓
- [ ] If F1 ≤ 0.75, Sentinel features weren't helpful (debug)
- [ ] Commit to new branch (e.g., `feature/sentinel-integration`)

---

## 📖 Git History

Latest commits on `feature/osm-frism-baseline`:

```
d76377d - feat: tier 1 & tier 2 semi-supervised training pipeline
  (Tier 1 results, Tier 2 3-round self-training, all models + logs)

ca9e99d - osm+frism +baseline model
a75da8e - docs: update README with application interface screenshots
```

---

## 🤔 FAQ

### Q: Why is the current F1 only 0.75 after 30x data expansion?
**A**: Circular reasoning. Models learned the OSM-proximity labeling rule rather than fire patterns. Self-training with more OSM-derived data doesn't help.

### Q: Should I run more self-training rounds?
**A**: No. The plateau after Round 1 proves the model extracted all available signal. Sentinel imagery is the next step to break the circularity.

### Q: What happens if Sentinel features don't improve F1?
**A**: Then Sentinel features aren't sufficiently independent from OSM, or the model isn't using them effectively. Debug by checking feature distributions, correlation with labels, and feature importance rankings.

### Q: Can I use Tier 3 (ground truth enrichment) instead of Sentinel?
**A**: Yes, but Sentinel is faster and more scalable. CPCB/IMD/news data would also work to break circularity, but requires manual effort. Combine both if possible.

### Q: What spatial resolution should I target?
**A**: 30m is ideal (aligns with Landsat legacy + Sentinel-2 SWIR). FIRMS detections are ~1km pixels, so anything finer than 100m will work.

---

## 📞 Contact & Notes

- **OSM Enrichment QA**: See `eda/outputs/osm_enrichment_qa.json` for feature coverage statistics
- **Label Audit**: See `eda/outputs/label_audit.json` for confidence breakdown by class
- **292 Flagged Candidates**: Exists in `label_review.csv` (not pushed to repo) — consider manual review as complementary ground truth

---

**Last Updated**: 2026-09-29  
**Branch**: `feature/osm-frism-baseline`  
**Status**: Ready for Sentinel integration
