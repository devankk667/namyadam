"""
Late-fusion branch ablation.

Compares:
1. R+S only
2. CNN only
3. R+S + CNN

All experiments use:
- Same 18,929 events
- Same 5 spatial folds
- Same R+S XGBoost OOF probabilities
- Same existing CNN OOF probabilities
- Same Logistic Regression meta-model
- Fold-safe meta-model evaluation

Primary metric:
    Trusted Macro-F1

Secondary:
    Trusted Macro-F1 without gas flare
"""


import json
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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

OUTPUT_DIR = "ml/models/late_fusion_ablation"

RANDOM_STATE = 42


# ============================================================
# LABELS
# ============================================================

LABEL_ORDER = [
    "agricultural_burning",
    "gas_flare",
    "industrial_fire",
    "mining_activity",
]

LABEL_TO_INT = {
    label: i
    for i, label in enumerate(LABEL_ORDER)
}


# ============================================================
# REMOVE SAME 26 OSM RULE-PROXY FEATURES
# ============================================================

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
# CNN PROBABILITY COLUMNS
# ============================================================

cnn_prob_cols = [
    "p_agricultural_burning",
    "p_industrial_fire",
    "p_mining_activity",
    "p_gas_flare",
]


# ============================================================
# LOAD
# ============================================================

print("Loading event table...")
events = pd.read_parquet(EVENT_PATH)

print("Loading Sentinel spectral features...")
spectral = pd.read_parquet(SPECTRAL_PATH)

print("Loading CNN OOF predictions...")
embeddings = pd.read_parquet(EMBEDDING_PATH)

print("Loading spatial folds...")
folds = pd.read_csv(FOLDS_PATH)


# ============================================================
# MERGE
# ============================================================

spectral_features = [
    c
    for c in spectral.columns
    if c not in ["event_id", "has_image"]
]

events = events.merge(
    spectral[["event_id"] + spectral_features],
    on="event_id",
    how="left",
    validate="one_to_one",
)

events = events.merge(
    embeddings[["event_id"] + cnn_prob_cols],
    on="event_id",
    how="left",
    validate="one_to_one",
)

events = events.merge(
    folds[["event_id", "fold", "trusted"]],
    on="event_id",
    how="left",
    validate="one_to_one",
)


# ============================================================
# HANDLE 3 EVENTS WITHOUT IMAGES
# ============================================================

for col in cnn_prob_cols:
    events[col] = events[col].fillna(0.25)


# ============================================================
# BUILD R+S FEATURES
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

X = events[feature_cols]

y = events["label"].map(LABEL_TO_INT).to_numpy()

fold_values = events["fold"].astype(int).to_numpy()

trusted = events["trusted"].astype(int).to_numpy()


print()
print("R+S feature count:", len(feature_cols))


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    c
    for c in X.columns
    if c not in numeric_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                    ),
                ]
            ),
            categorical_features,
        ),
    ]
)


# ============================================================
# XGBOOST
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
# GENERATE R+S OOF PROBABILITIES
# ============================================================

print()
print("=" * 70)
print("GENERATING R+S OOF PROBABILITIES")
print("=" * 70)

tabular_oof = np.zeros(
    (len(events), 4),
    dtype=float,
)

for test_fold in sorted(np.unique(fold_values)):

    train_idx = fold_values != test_fold
    test_idx = fold_values == test_fold

    print(
        f"Fold {test_fold}: "
        f"train={train_idx.sum()} "
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

    tabular_oof[test_idx] = model.predict_proba(
        X.iloc[test_idx]
    )


print("R+S OOF probabilities generated.")


# ============================================================
# CNN OOF PROBABILITIES
# ============================================================

# Existing CNN columns are ordered:
# agricultural, industrial, mining, gas
#
# Convert to:
# agricultural, gas, industrial, mining

cnn_oof = events[
    [
        "p_agricultural_burning",
        "p_gas_flare",
        "p_industrial_fire",
        "p_mining_activity",
    ]
].to_numpy(dtype=float)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_meta_features(
    meta_X,
    experiment_name,
):

    print()
    print("=" * 70)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 70)

    meta_oof = np.zeros(
        (len(events), 4),
        dtype=float,
    )

    for test_fold in sorted(np.unique(fold_values)):

        train_idx = fold_values != test_fold
        test_idx = fold_values == test_fold

        meta_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        meta_model.fit(
            meta_X[train_idx],
            y[train_idx],
        )

        meta_oof[test_idx] = (
            meta_model.predict_proba(
                meta_X[test_idx]
            )
        )

    predictions = np.argmax(
        meta_oof,
        axis=1,
    )

    all_macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
    )

    trusted_mask = trusted == 1

    trusted_macro_f1 = f1_score(
        y[trusted_mask],
        predictions[trusted_mask],
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0,
    )

    no_flare_mask = (
        trusted_mask
        & (y != LABEL_TO_INT["gas_flare"])
    )

    trusted_no_flare_macro_f1 = f1_score(
        y[no_flare_mask],
        predictions[no_flare_mask],
        labels=[0, 2, 3],
        average="macro",
        zero_division=0,
    )

    print(
        f"All-event Macro F1:        "
        f"{all_macro_f1:.6f}"
    )

    print(
        f"Trusted Macro F1:          "
        f"{trusted_macro_f1:.6f}"
    )

    print(
        f"Trusted no-flare Macro F1: "
        f"{trusted_no_flare_macro_f1:.6f}"
    )

    print(
        f"Accuracy:                  "
        f"{accuracy:.6f}"
    )

    print(
        f"Weighted F1:               "
        f"{weighted_f1:.6f}"
    )

    return {
        "experiment": experiment_name,
        "all_event_macro_f1": float(
            all_macro_f1
        ),
        "trusted_macro_f1": float(
            trusted_macro_f1
        ),
        "trusted_no_flare_macro_f1": float(
            trusted_no_flare_macro_f1
        ),
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
    }


# ============================================================
# RUN THREE ABLATIONS
# ============================================================

results = []


# ------------------------------------------------------------
# 1. R+S ONLY
# ------------------------------------------------------------

results.append(
    evaluate_meta_features(
        tabular_oof,
        "R+S only",
    )
)


# ------------------------------------------------------------
# 2. CNN ONLY
# ------------------------------------------------------------

results.append(
    evaluate_meta_features(
        cnn_oof,
        "CNN only",
    )
)


# ------------------------------------------------------------
# 3. R+S + CNN
# ------------------------------------------------------------

combined_meta_X = np.concatenate(
    [
        tabular_oof,
        cnn_oof,
    ],
    axis=1,
)

results.append(
    evaluate_meta_features(
        combined_meta_X,
        "R+S + CNN",
    )
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

results_df = pd.DataFrame(results)

print()
print("=" * 70)
print("ABLATION SUMMARY")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ablation_results.csv",
    ),
    index=False,
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "ablation_results.json",
    ),
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )

print()
print("Saved:")
print(
    f"  {OUTPUT_DIR}/ablation_results.csv"
)
print(
    f"  {OUTPUT_DIR}/ablation_results.json"
)