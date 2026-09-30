"""
Spatial fusion experiment using the REDUCED tabular feature set.

Experiments:
R  = Reduced tabular features
R+S = Reduced tabular + Sentinel spectral features
R+F = Reduced tabular + Sentinel spectral + frozen ImageNet PCA
R+C = Reduced tabular + Sentinel spectral + OOF fine-tuned CNN PCA + CNN probabilities

Evaluation:
- Same 5 spatial folds from spatial_folds_v1.csv
- Primary: trusted Macro-F1
- Secondary: trusted Macro-F1 without gas flare
- All-event Macro-F1 = agreement with rule-derived labels
"""

import json
import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


# ============================================================
# CONFIG
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

OUTPUT_DIR = "ml/models/reduced_spatial_fusion"

RANDOM_STATE = 42


# ============================================================
# 26 OSM RULE-PROXY FEATURES TO REMOVE
# ============================================================

OSM_RULE_PROXY_FEATURES = [
    # Industrial counts
    "industrial_count_1km_mean",
    "industrial_count_1km_max",
    "industrial_count_1km_std",

    "industrial_count_2km_mean",
    "industrial_count_2km_max",
    "industrial_count_2km_std",

    "industrial_count_5km_mean",
    "industrial_count_5km_max",
    "industrial_count_5km_std",

    # Power plants
    "power_plant_count_5km_mean",
    "power_plant_count_5km_max",
    "power_plant_count_5km_std",

    # Quarries
    "quarry_count_5km_mean",
    "quarry_count_5km_max",
    "quarry_count_5km_std",

    # Flares
    "flare_count_5km_mean",
    "flare_count_5km_max",
    "flare_count_5km_std",

    # Petroleum wells
    "petroleum_well_count_5km_mean",
    "petroleum_well_count_5km_max",
    "petroleum_well_count_5km_std",

    # Industrial land use
    "industrial_landuse_nearby_mean",
    "industrial_landuse_nearby_max",
    "industrial_landuse_nearby_std",

    # Nearest OSM categorical fields
    "nearest_osm_type_mode",
    "nearest_osm_kind_mode",
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading event table...")
events = pd.read_parquet(EVENT_PATH)

print("Loading Sentinel spectral features...")
spectral = pd.read_parquet(SPECTRAL_PATH)

print("Loading Sentinel embeddings...")
embeddings = pd.read_parquet(EMBEDDING_PATH)

print("Loading spatial folds...")
folds = pd.read_csv(FOLDS_PATH)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\nShapes:")
print("Events:", events.shape)
print("Spectral:", spectral.shape)
print("Embeddings:", embeddings.shape)
print("Folds:", folds.shape)


assert events["event_id"].is_unique
assert spectral["event_id"].is_unique
assert embeddings["event_id"].is_unique
assert folds["event_id"].is_unique

assert len(events) == 18929
assert len(spectral) == 18929
assert len(embeddings) == 18929
assert len(folds) == 18929


# ============================================================
# MERGE EVERYTHING ON event_id
# ============================================================

df = events.merge(
    spectral,
    on="event_id",
    how="inner",
    suffixes=("", "_spectral"),
)

df = df.merge(
    embeddings,
    on="event_id",
    how="inner",
    suffixes=("", "_embedding"),
)

df = df.merge(
    folds[
        [
            "event_id",
        ]
    ],
    on="event_id",
    how="inner",
)

print("\nMerged shape:", df.shape)

assert len(df) == 18929
assert df["event_id"].is_unique


# ============================================================
# CHECK LABEL CONSISTENCY
# ============================================================

if "label_embedding" in df.columns:
    assert (df["label"] == df["label_embedding"]).all()

print("\nLabel counts:")
print(df["label"].value_counts())


# ============================================================
# TARGET
# ============================================================

label_order = [
    "agricultural_burning",
    "gas_flare",
    "industrial_fire",
    "mining_activity",
]

label_to_id = {
    label: i
    for i, label in enumerate(label_order)
}

df["target"] = df["label"].map(label_to_id)

assert df["target"].notna().all()


# ============================================================
# BASE TABULAR FEATURES
# ============================================================

EXCLUDED_COLUMNS = {
    "event_id",
    "label",
    "label_confidence",
    "label_source",
    "target",
}


base_tabular_features = [
    c
    for c in events.columns
    if c not in EXCLUDED_COLUMNS
]


print("\nOriginal tabular feature count:")
print(len(base_tabular_features))


# Remove the 26 OSM rule-proxy features
reduced_tabular_features = [
    c
    for c in base_tabular_features
    if c not in OSM_RULE_PROXY_FEATURES
]


print("Reduced tabular feature count:")
print(len(reduced_tabular_features))

print("\nRemoved features:")
for c in OSM_RULE_PROXY_FEATURES:
    print("  ", c)

assert len(OSM_RULE_PROXY_FEATURES) == 26
assert len(reduced_tabular_features) == 48


# ============================================================
# SENTINEL FEATURE GROUPS
# ============================================================

spectral_features = [
    c
    for c in spectral.columns
    if c not in ["event_id", "has_image"]
]

frozen_features = [
    c
    for c in embeddings.columns
    if c.startswith("frozen_pca_")
]

cnn_pca_features = [
    c
    for c in embeddings.columns
    if c.startswith("pca_")
]

cnn_probability_features = [
    "p_agricultural_burning",
    "p_industrial_fire",
    "p_mining_activity",
    "p_gas_flare",
]


print("\nSentinel features:")
print("Spectral:", len(spectral_features))
print("Frozen PCA:", len(frozen_features))
print("CNN PCA:", len(cnn_pca_features))
print("CNN probabilities:", len(cnn_probability_features))


assert len(spectral_features) == 39
assert len(frozen_features) == 32
assert len(cnn_pca_features) == 32
assert len(cnn_probability_features) == 4


# ============================================================
# EXPERIMENT FEATURE SETS
# ============================================================

feature_sets = {
    "R_reduced": reduced_tabular_features,

    "R_S_spectral": (
        reduced_tabular_features
        + spectral_features
    ),

    "R_F_frozen": (
        reduced_tabular_features
        + spectral_features
        + frozen_features
    ),

    "R_C_cnn": (
        reduced_tabular_features
        + spectral_features
        + cnn_pca_features
        + cnn_probability_features
    ),
}


print("\nFeature-set sizes:")

for name, features in feature_sets.items():
    print(f"{name}: {len(features)}")


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(X):
    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
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
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


# ============================================================
# XGBOOST
# ============================================================

def build_model(n_classes):
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        num_class=n_classes,
    )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    trusted,
):
    results = {}

    # --------------------------------------------------------
    # All events
    # --------------------------------------------------------

    results["all_macro_f1"] = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    results["accuracy"] = accuracy_score(
        y_true,
        y_pred,
    )

    results["weighted_f1"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
    )

    # --------------------------------------------------------
    # Trusted events
    # --------------------------------------------------------

    trusted_mask = trusted == 1

    y_true_trusted = y_true[trusted_mask]
    y_pred_trusted = y_pred[trusted_mask]

    results["trusted_count"] = int(
        trusted_mask.sum()
    )

    results["trusted_macro_f1"] = f1_score(
        y_true_trusted,
        y_pred_trusted,
        average="macro",
    )

    # --------------------------------------------------------
    # Trusted events WITHOUT gas flare
    # --------------------------------------------------------

    gas_flare_id = label_to_id["gas_flare"]

    no_flare_mask = (
        trusted_mask
        & (y_true != gas_flare_id)
    )

    y_true_no_flare = y_true[no_flare_mask]
    y_pred_no_flare = y_pred[no_flare_mask]

    results["trusted_no_flare_count"] = int(
        no_flare_mask.sum()
    )

    results["trusted_no_flare_macro_f1"] = (
        f1_score(
            y_true_no_flare,
            y_pred_no_flare,
            average="macro",
        )
    )

    return results


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# SPATIAL CV
# ============================================================

all_results = {}

all_oof_predictions = {}


for experiment_name, feature_columns in feature_sets.items():

    print("\n")
    print("=" * 70)
    print(experiment_name)
    print("=" * 70)

    X = df[feature_columns].copy()
    y = df["target"].values
    trusted = df["trusted"].values
    fold_values = df["fold"].values

    print("Features:", X.shape[1])

    oof_predictions = np.full(
        len(df),
        -1,
        dtype=int,
    )

    oof_probabilities = np.full(
        (len(df), len(label_order)),
        np.nan,
    )

    fold_results = []

    # --------------------------------------------------------
    # 5 spatial folds
    # --------------------------------------------------------

    for fold_id in sorted(df["fold"].unique()):

        print(
            f"\nTraining fold {fold_id + 1}/5..."
        )

        train_mask = fold_values != fold_id
        test_mask = fold_values == fold_id

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]

        y_train = y[train_mask]
        y_test = y[test_mask]

        # ----------------------------------------------------
        # Preprocessing fitted ONLY on training fold
        # ----------------------------------------------------

        preprocessor = build_preprocessor(
            X_train
        )

        X_train_processed = (
            preprocessor.fit_transform(
                X_train
            )
        )

        X_test_processed = (
            preprocessor.transform(
                X_test
            )
        )

        print(
            "Processed train shape:",
            X_train_processed.shape,
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = build_model(
            len(label_order)
        )

        model.fit(
            X_train_processed,
            y_train,
        )

        # ----------------------------------------------------
        # OOF prediction
        # ----------------------------------------------------

        fold_pred = model.predict(
            X_test_processed
        )

        fold_prob = model.predict_proba(
            X_test_processed
        )

        oof_predictions[test_mask] = (
            fold_pred
        )

        oof_probabilities[test_mask] = (
            fold_prob
        )

        # ----------------------------------------------------
        # Fold metrics
        # ----------------------------------------------------

        fold_macro = f1_score(
            y_test,
            fold_pred,
            average="macro",
        )

        fold_trusted_mask = (
            trusted[test_mask] == 1
        )

        if fold_trusted_mask.sum() > 0:

            fold_trusted_macro = f1_score(
                y_test[fold_trusted_mask],
                fold_pred[fold_trusted_mask],
                average="macro",
            )

        else:
            fold_trusted_macro = np.nan

        print(
            f"Fold {fold_id}: "
            f"all Macro-F1={fold_macro:.4f}, "
            f"trusted Macro-F1={fold_trusted_macro:.4f}"
        )

        fold_results.append(
            {
                "fold": int(fold_id),
                "all_macro_f1": float(
                    fold_macro
                ),
                "trusted_macro_f1": float(
                    fold_trusted_macro
                ),
            }
        )

    # ========================================================
    # FINAL OOF METRICS
    # ========================================================

    assert (oof_predictions >= 0).all()

    metrics = calculate_metrics(
        y,
        oof_predictions,
        trusted,
    )

    metrics["experiment"] = experiment_name
    metrics["feature_count"] = len(
        feature_columns
    )

    metrics["fold_results"] = fold_results

    all_results[experiment_name] = metrics

    all_oof_predictions[
        experiment_name
    ] = {
        "predictions": oof_predictions,
        "probabilities": oof_probabilities,
    }

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nFINAL RESULTS")
    print("-" * 50)

    print(
        "All-event Macro F1:",
        f"{metrics['all_macro_f1']:.6f}",
    )

    print(
        "Trusted Macro F1:",
        f"{metrics['trusted_macro_f1']:.6f}",
    )

    print(
        "Trusted no-flare Macro F1:",
        f"{metrics['trusted_no_flare_macro_f1']:.6f}",
    )

    print(
        "Accuracy:",
        f"{metrics['accuracy']:.6f}",
    )

    print(
        "Weighted F1:",
        f"{metrics['weighted_f1']:.6f}",
    )

    # ========================================================
    # TRUSTED CLASSIFICATION REPORT
    # ========================================================

    trusted_mask = trusted == 1

    print("\nTrusted classification report:")

    print(
        classification_report(
            y[trusted_mask],
            oof_predictions[trusted_mask],
            labels=list(range(len(label_order))),
            target_names=label_order,
            zero_division=0,
        )
    )

    # ========================================================
    # SAVE OOF PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame(
        {
            "event_id": df["event_id"],
            "true_label": df["label"],
            "trusted": df["trusted"],
            "fold": df["fold"],
            "prediction": [
                label_order[i]
                for i in oof_predictions
            ],
        }
    )

    for class_id, class_name in enumerate(
        label_order
    ):
        prediction_df[
            f"p_{class_name}"
        ] = oof_probabilities[:, class_id]

    prediction_path = os.path.join(
        OUTPUT_DIR,
        f"{experiment_name}_oof_predictions.csv",
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print(
        "Saved:",
        prediction_path,
    )


# ============================================================
# SAVE COMPARISON TABLE
# ============================================================

comparison_rows = []

for experiment_name, metrics in all_results.items():

    comparison_rows.append(
        {
            "experiment": experiment_name,
            "feature_count": metrics[
                "feature_count"
            ],
            "all_macro_f1": metrics[
                "all_macro_f1"
            ],
            "trusted_macro_f1": metrics[
                "trusted_macro_f1"
            ],
            "trusted_no_flare_macro_f1": metrics[
                "trusted_no_flare_macro_f1"
            ],
            "accuracy": metrics[
                "accuracy"
            ],
            "weighted_f1": metrics[
                "weighted_f1"
            ],
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    "comparison.csv",
)

comparison_df.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics_json_path = os.path.join(
    OUTPUT_DIR,
    "metrics.json",
)

with open(
    metrics_json_path,
    "w",
) as f:
    json.dump(
        all_results,
        f,
        indent=2,
    )


# ============================================================
# PRINT FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(comparison_path)
print(metrics_json_path)