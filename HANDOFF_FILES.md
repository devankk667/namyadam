# Data Files Handoff Checklist for Sentinel Integration

## 📋 Files to Forward to Next Team Member

Complete list of all files needed for the next person to integrate Sentinel-1/2 imagery.

---

## 🚨 CRITICAL - Must Forward

These files are **essential** for the next phase:

### Documentation
- [ ] **TIER_1_2_CONTEXT.md** — Primary handoff guide (397 lines, 15 KB)
  - Problem statement (circular reasoning)
  - Complete Tier 1 & 2 results
  - Training pipeline details
  - Sentinel integration roadmap
  - Implementation checklist
  - FAQ

### Training Data (All Required)
```
data/processed/
├── [ ] firms_thermal_events.parquet          (~26k events, 2 MB)
│       ├─ Use: Query Sentinel data from event centroids
│       ├─ Columns: event_id, centroid_lat, centroid_lon, acq_date
│       └─ Key join: event_id
│
├── [ ] firms_event_labels.parquet            (18,946 labeled events, 1 MB)
│       ├─ Use: Ground truth for training
│       ├─ Columns: event_id, label, label_confidence
│       ├─ Classes: agricultural_burning, gas_flare, industrial_fire, mining_activity
│       └─ Key join: event_id
│
├── [ ] firms_obs_event_map.parquet           (1.1M observations, 25 MB)
│       ├─ Use: Map observations to events
│       ├─ Columns: obs_id, event_id
│       ├─ Relationship: 1 event → many obs
│       └─ Key join: obs_id, event_id
│
├── [ ] firms_osm_enriched.parquet            (1.1M observations, 150 MB)
│       ├─ Use: Baseline OSM features to combine with Sentinel
│       ├─ Features: nearest_osm_distance_km, industrial_count_*, etc. (15 features)
│       └─ Key join: obs_id
│
└── [ ] firms_temporal.parquet                (1.1M observations, 30 MB)
        ├─ Use: Temporal features
        ├─ Features: prev_7d, prev_14d, prev_30d, days_since_first_seen, etc. (11 features)
        └─ Key join: obs_id
```

**Total size**: ~208 MB

### Trained Models (For Baseline Comparison)
```
ml/models/
├── [ ] rf_baseline/
│       ├─ model.joblib                       (2.3 MB)
│       └─ metadata.json                      (baseline: F1 0.74)
│
├── [ ] xgb_temporal/
│       ├─ model.joblib                       (2.9 MB)
│       └─ metadata.json                      (baseline: F1 0.74)
│
└── [ ] self_trained_rf3/
        ├─ model.joblib                       (4.0 MB) ← Use this for comparison
        └─ metadata.json                      (current best: F1 0.75)
```

**Total size**: ~9 MB

### Results & Metrics (For Baseline Reference)
```
eda/outputs/
├── [ ] self_train_log.json                   (1 KB)
│       └─ Shows why F1 plateaued at 0.75
│
├── [ ] full_1.1M_classification_summary.json (2 KB)
│       └─ Current predictions on full dataset
│
├── [ ] label_audit.json                      (2 KB)
│       └─ Label confidence breakdown by class
│
└── [ ] osm_enrichment_qa.json                (2 KB)
        └─ OSM feature coverage stats
```

**Total size**: ~7 KB

### Code Patterns (Training Scripts)
```
ml/scripts/
├── [ ] train_rf_baseline.py                  (100 lines)
│       └─ Template for modifying with Sentinel features
│
├── [ ] train_xgb_temporal.py                 (115 lines)
│       └─ Pattern for combining multiple feature groups
│
├── [ ] self_train.py                         (204 lines)
│       └─ Reference for self-training algorithm
│
└── [ ] predict_full_dataset.py               (88 lines)
        └─ Pattern for batch prediction
```

**Total size**: ~20 KB

---

## 📊 Optional but Helpful

These files provide context but aren't strictly required:

- [ ] `README.md` — Original project documentation (reference only)
- [ ] `eda/outputs/full_1.1M_classification_summary.json` — Current predictions (for comparison)
- [ ] `ml/models/self_trained_rf{0,1,2}/` — Other self-training rounds (reference only)

---

## 📦 Total Transfer Size

| Category | Size | Required |
|----------|------|----------|
| Training Data (5 files) | ~208 MB | ✅ YES |
| Trained Models (6 files) | ~9 MB | ✅ YES |
| Results/Metrics (4 files) | ~7 KB | ✅ YES |
| Code Scripts (4 files) | ~20 KB | ✅ YES |
| Documentation (1 file) | ~15 KB | ✅ YES |
| **TOTAL** | **~217 MB** | |

---

## 🚚 How to Transfer

### Method 1: Git Clone (Simplest)
All code and models are already in the repo. Only need to transfer large data files separately.

```bash
git clone https://github.com/devankk667/namyadam.git
cd namyadam
git checkout feature/osm-frism-baseline
```

Then transfer the `data/processed/` parquet files (~208 MB) via:
- Cloud storage (S3, GCS)
- File sharing service (Dropbox, WeTransfer)
- Direct download link

### Method 2: Compressed Archive
```bash
# Create a single archive with all required files
tar -czf nam_ya_dam_sentinel_handoff.tar.gz \
  data/processed/*.parquet \
  ml/models/ \
  ml/scripts/ \
  eda/outputs/ \
  TIER_1_2_CONTEXT.md

# Transfer the ~150 MB .tar.gz file
```

### Method 3: GitHub LFS (if available)
If your repo is set up with Git LFS for large files:
```bash
git lfs clone https://github.com/devankk667/namyadam.git
cd namyadam
git checkout feature/osm-frism-baseline
git lfs pull  # Downloads large parquet files
```

---

## ✅ Pre-Transfer Verification

Before handing off, verify all files are present:

```bash
# Check training data
ls -lh data/processed/*.parquet
# Should show 5 files: ~208 MB total

# Check models
ls -lh ml/models/*/model.joblib
# Should show 3 model files: ~9 MB total

# Check code scripts
ls -lh ml/scripts/*.py
# Should show 4 Python files

# Check documentation
cat TIER_1_2_CONTEXT.md | wc -l
# Should show 397 lines
```

---

## 🎯 What They'll Do With These Files

1. **Read TIER_1_2_CONTEXT.md** to understand the problem
2. **Load training data** (parquet files) into Python/pandas
3. **Reference trained models** for F1 baseline (0.75)
4. **Query Sentinel data** for all event centroids (firms_thermal_events)
5. **Extract Sentinel features** (NDVI, NBR, SAR, etc.)
6. **Combine features**: OSM (15) + Sentinel (~14) + Temporal (11) = ~40 total
7. **Retrain models** using train_rf_baseline.py and train_xgb_temporal.py as templates
8. **Validate improvement**: Goal is F1 > 0.80 (breaking circularity)
9. **Commit results** to `feature/sentinel-integration` branch

---

## 🔑 Key Points to Communicate

- **Current baseline**: F1 = 0.75 (Tier 2 Round 3, 177k observations)
- **Why it plateaued**: Circular reasoning (OSM features + OSM-derived labels)
- **Goal**: F1 > 0.80 with Sentinel features (independent signal)
- **Success metric**: If F1 improves, circularity is broken ✓
- **Timeline**: Estimate 2-3 weeks for Sentinel integration + retraining

---

## 📄 Summary Command

Forward this complete package:

```bash
# All files needed for Sentinel integration
FORWARD_TO_NEXT_PERSON:
  ✅ TIER_1_2_CONTEXT.md
  ✅ data/processed/*.parquet (~208 MB)
  ✅ ml/models/*/*.joblib (~9 MB)
  ✅ ml/scripts/*.py
  ✅ eda/outputs/*.json
  ✅ This checklist (handoff_files.txt)

Total: ~217 MB, fully documented, ready to integrate Sentinel imagery
```

---

**Last Updated**: 2026-09-29  
**Status**: Ready for handoff  
**Next Phase**: Sentinel-1/2 integration to break OSM circularity
