"""
Late fusion / stacking experiment.

Branches:
1. Reduced tabular + Sentinel spectral features -> XGBoost
2. Existing OOF CNN probabilities -> image branch

The two branches produce 8 probabilities total:
    4 tabular probabilities
    4 CNN probabilities

A Logistic Regression meta-model combines them.

Evaluation:
- Same 5 spatial folds
- Base tabular predictions are generated OOF
- Meta-model predictions are also generated OOF
- Primary metric: trusted Macro-F1
- Secondary: trusted Macro-F1 without gas flare
- All-event Macro-F1 is agreement with rule-derived labels
"""

import json
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

EVENT_PATH = "data/processed/event_training_table.parquet"

SPECTRAL_PATH = (
    r"C:\Users\Vedansh\Downloads\person2_fusion_image_features_v1"
    r"\person2_fusion_image_features_v1\data\image_spectral_features_v1.parquet"
)

EMBEDDING_PATH = (
    r"C:\Users\Vedansh\Downloads\person2_fusion_image_features_v1"
    r"\person2_fusion_image_features_v1\data\image_embeddings_v1.parquet"
)

FOLDS_PATH = (
    r"C:\Users\Vedansh\Downloads\person2_fusion_image_features_v1"
    r"\person2_fusion_image_features_v1\data\spatial_folds_v1.csv"
)

OUTPUT_DIR = "ml/models/late_fusion"


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

LABEL_ORDER = [
    "agricultural_burning",
    "gas_flare",
    "industrial_fire",
    "mining_activity",
]

LABEL_TO_INT = {
    label: i for i, label in enumerate(LABEL_ORDER)
}


# Same 26 OSM rule-proxy features removed in the
# reduced spatial fusion experiment.
REMOVED_OSM_FEATURES = [
    "industrial_count_1km_mean",
    "industrial_count_1km_max",
    "industrial_count_1km_std",
    "industrial_count_2km_mean",
    "industrial_count_2km_max",
    "industrial_count_2km_std",
    "industrial_count_5km_mean",
    "industrial_count_5km_max",
    "industrial_count_5km_std",
    "power_plant_count_5km_mean",
    "power_plant_count_5km_max",
    "power_plant_count_5km_std",
    "quarry_count_5km_mean",
    "quarry_count_5km_max",
    "quarry_count_5km_std",
    "flare_count_5km_mean",
    "flare_count_5km_max",
    "flare_count_5km_std",
    "petroleum_well_count_5km_mean",
    "petroleum_well_count_5km_max",
    "petroleum_well_count_5km_std",
    "industrial_landuse_nearby_mean",
    "industrial_landuse_nearby_max",
    "industrial_landuse_nearby_std",
    "nearest_osm_type_mode",
    "nearest_osm_kind_mode",
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading event table...")
events = pd.read_parquet(EVENT_PATH)

print("Event table:", events.shape)


print("Loading Sentinel spectral features...")
spectral = pd.read_parquet(SPECTRAL_PATH)

print("Spectral:", spectral.shape)


print("Loading existing CNN OOF predictions...")
embeddings = pd.read_parquet(EMBEDDING_PATH)

print("Embeddings:", embeddings.shape)


print("Loading spatial folds...")
folds = pd.read_csv(FOLDS_PATH)

print("Folds:", folds.shape)


# ============================================================
# MERGE
# ============================================================

# Spectral contains:
# event_id, has_image, 39 spectral features
spectral_features = [
    c for c in spectral.columns
    if c not in ["event_id", "has_image"]
]

events = events.merge(
    spectral[["event_id"] + spectral_features],
    on="event_id",
    how="left",
    validate="one_to_one",
)

# Embeddings already contains fold/trusted/block_id.
# We only need CNN probabilities here.
cnn_prob_cols = [
    "p_agricultural_burning",
    "p_industrial_fire",
    "p_mining_activity",
    "p_gas_flare",
]

events = events.merge(
    embeddings[["event_id"] + cnn_prob_cols],
    on="event_id",
    how="left",
    validate="one_to_one",
)

# Add fold/trusted metadata.
# Do NOT merge duplicate fold/trusted columns from embeddings.
events = events.merge(
    folds[["event_id", "fold", "trusted"]],
    on="event_id",
    how="left",
    validate="one_to_one",
)


print("Merged table:", events.shape)


# ============================================================
# BASIC VALIDATION
# ============================================================

assert events["event_id"].is_unique

assert events["fold"].notna().all()
assert events["trusted"].notna().all()

# 3 labelled events have no Sentinel image, so their CNN
# probabilities are missing. For these events, the CNN branch
# contributes no class preference.
#
# Use a uniform probability distribution rather than inventing
# an image-based prediction.
for col in cnn_prob_cols:
    events[col] = events[col].fillna(0.25)

print(
    "Events without CNN image:",
    int(events["p_agricultural_burning"].eq(0.25).sum())
)


# ============================================================
# BUILD REDUCED + SPECTRAL FEATURE SET
# ============================================================

excluded = {
    "event_id",
    "label",
    "label_confidence",
    "label_source",
    "fold",
    "trusted",
    *cnn_prob_cols,
}

feature_cols = [
    c
    for c in events.columns
    if c not in excluded
    and c not in REMOVED_OSM_FEATURES
]

print()
print("R + S feature count:", len(feature_cols))


X = events[feature_cols].copy()

y = events["label"].map(LABEL_TO_INT).to_numpy()

fold_values = events["fold"].astype(int).to_numpy()
trusted = events["trusted"].astype(int).to_numpy()


# ============================================================
# PREPROCESSOR
# ============================================================

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    c for c in X.columns
    if c not in numeric_features
]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# ============================================================
# XGBOOST FACTORY
# ============================================================

def make_xgb():
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# GENERATE TABULAR OOF PROBABILITIES
# ============================================================

print()
print("=" * 70)
print("GENERATING R+S OOF PREDICTIONS")
print("=" * 70)

tabular_oof = np.zeros(
    (len(events), len(LABEL_ORDER)),
    dtype=float,
)

unique_folds = sorted(np.unique(fold_values))

for test_fold in unique_folds:

    print()
    print(f"Tabular fold {test_fold}")

    train_idx = fold_values != test_fold
    test_idx = fold_values == test_fold

    print(
        f"  train={train_idx.sum()} "
        f"test={test_idx.sum()}"
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", make_xgb()),
        ]
    )

    model.fit(
        X.iloc[train_idx],
        y[train_idx],
    )

    probabilities = model.predict_proba(
        X.iloc[test_idx]
    )

    tabular_oof[test_idx] = probabilities

    print("  done")


# ============================================================
# CHECK OOF PROBABILITIES
# ============================================================

assert np.allclose(
    tabular_oof.sum(axis=1),
    1.0,
    atol=1e-5,
)

print()
print("Tabular OOF predictions generated.")


# ============================================================
# CNN PROBABILITY ORDER
# ============================================================

# The embedding file stores:
# agricultural, industrial, mining, gas
#
# Our desired order is:
# agricultural, gas, industrial, mining
#
# Therefore reorder them explicitly.

cnn_oof = events[
    [
        "p_agricultural_burning",
        "p_gas_flare",
        "p_industrial_fire",
        "p_mining_activity",
    ]
].to_numpy(dtype=float)

assert np.allclose(
    cnn_oof.sum(axis=1),
    1.0,
    atol=1e-4,
)


# ============================================================
# BUILD META FEATURES
# ============================================================

meta_X = np.concatenate(
    [
        tabular_oof,
        cnn_oof,
    ],
    axis=1,
)

print()
print("Meta feature shape:", meta_X.shape)
print("Expected: (18929, 8)")


# ============================================================
# META-MODEL OOF PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING META-MODEL OOF PREDICTIONS")
print("=" * 70)

meta_oof = np.zeros(
    (len(events), len(LABEL_ORDER)),
    dtype=float,
)

for test_fold in unique_folds:

    print()
    print(f"Meta fold {test_fold}")

    train_idx = fold_values != test_fold
    test_idx = fold_values == test_fold

    meta_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="auto",
        random_state=RANDOM_STATE,
    )

    meta_model.fit(
        meta_X[train_idx],
        y[train_idx],
    )

    meta_oof[test_idx] = meta_model.predict_proba(
        meta_X[test_idx]
    )

    print("  done")


# ============================================================
# FINAL PREDICTIONS
# ============================================================

pred = np.argmax(
    meta_oof,
    axis=1,
)


# ============================================================
# METRICS
# ============================================================

all_macro_f1 = f1_score(
    y,
    pred,
    average="macro",
)

accuracy = accuracy_score(
    y,
    pred,
)

weighted_f1 = f1_score(
    y,
    pred,
    average="weighted",
)


trusted_mask = trusted == 1

trusted_macro_f1 = f1_score(
    y[trusted_mask],
    pred[trusted_mask],
    labels=list(range(len(LABEL_ORDER))),
    average="macro",
    zero_division=0,
)

no_flare_mask = trusted_mask & (
    y != LABEL_TO_INT["gas_flare"]
)

trusted_no_flare_macro_f1 = f1_score(
    y[no_flare_mask],
    pred[no_flare_mask],
    labels=[
        LABEL_TO_INT["agricultural_burning"],
        LABEL_TO_INT["industrial_fire"],
        LABEL_TO_INT["mining_activity"],
    ],
    average="macro",
    zero_division=0,
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("LATE FUSION RESULTS")
print("=" * 70)

print(
    f"All-event Macro F1:              "
    f"{all_macro_f1:.6f}"
)

print(
    f"Trusted Macro F1:                "
    f"{trusted_macro_f1:.6f}"
)

print(
    f"Trusted no-flare Macro F1:       "
    f"{trusted_no_flare_macro_f1:.6f}"
)

print(
    f"Accuracy:                        "
    f"{accuracy:.6f}"
)

print(
    f"Weighted F1:                     "
    f"{weighted_f1:.6f}"
)


# ============================================================
# PER-CLASS TRUSTED RESULTS
# ============================================================

print()
print("Trusted class-wise F1:")

trusted_report = {}

for class_id, class_name in enumerate(LABEL_ORDER):

    mask = trusted_mask & (
        y == class_id
    )

    class_f1 = f1_score(
        y[mask],
        pred[mask],
        labels=[class_id],
        average="macro",
        zero_division=0,
    )

    trusted_report[class_name] = {
        "support": int(mask.sum()),
        "f1": float(class_f1),
    }

    print(
        f"  {class_name:25s} "
        f"support={mask.sum():3d} "
        f"F1={class_f1:.4f}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

prediction_output = events[
    [
        "event_id",
        "label",
        "fold",
        "trusted",
    ]
].copy()

prediction_output["prediction"] = [
    LABEL_ORDER[i]
    for i in pred
]

for i, class_name in enumerate(LABEL_ORDER):

    prediction_output[
        f"p_{class_name}"
    ] = meta_oof[:, i]


prediction_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "late_fusion_oof_predictions.csv",
    ),
    index=False,
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "experiment": "late_fusion",
    "feature_count_tabular": len(feature_cols),
    "meta_feature_count": 8,
    "n_events": int(len(events)),
    "n_trusted": int(trusted_mask.sum()),
    "all_event_macro_f1": float(all_macro_f1),
    "trusted_macro_f1": float(trusted_macro_f1),
    "trusted_no_flare_macro_f1": float(
        trusted_no_flare_macro_f1
    ),
    "accuracy": float(accuracy),
    "weighted_f1": float(weighted_f1),
    "trusted_classwise": trusted_report,
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "metrics.json",
    ),
    "w",
) as f:
    json.dump(
        metrics,
        f,
        indent=2,
    )


print()
print("Saved:")
print(
    f"  {OUTPUT_DIR}/late_fusion_oof_predictions.csv"
)
print(
    f"  {OUTPUT_DIR}/metrics.json"
)