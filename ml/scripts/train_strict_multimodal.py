"""
Strict multimodal experiment.

Goal:
    Evaluate FIRMS + temporal + Sentinel features while excluding:
      - all OSM-derived features
      - label metadata
      - supervised/label-trained CNN features

Original datasets and existing model artifacts are read-only.

Experiments:
    ST-1: FIRMS
    ST-2: Temporal
    ST-3: FIRMS + Temporal
    ST-4: FIRMS + Temporal + Sentinel spectral
    ST-5: FIRMS + Temporal + frozen ImageNet representation
    ST-6: FIRMS + Temporal + Sentinel spectral + frozen ImageNet representation

Evaluation:
    5-fold spatial CV using spatial_folds_v1.csv
"""

import json
import os
import re
from pathlib import Path

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
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

EVENT_TABLE = ROOT / "data" / "processed" / "event_training_table.parquet"

IMAGE_ROOT = (
    Path.home()
    / "Downloads"
    / "person2_fusion_image_features_v1"
    / "person2_fusion_image_features_v1"
)

SPECTRAL_FILE = IMAGE_ROOT / "data" / "image_spectral_features_v1.parquet"
EMBEDDING_FILE = IMAGE_ROOT / "data" / "image_embeddings_v1.parquet"
FOLDS_FILE = IMAGE_ROOT / "data" / "spatial_folds_v1.csv"

OUTPUT_DIR = ROOT / "ml" / "models" / "strict_multimodal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------

FIRMS_EXACT = [
    "observation_count",
    "confidence_mode",
    "daynight_mode",
    "frp_mean",
    "frp_max",
    "frp_std",
    "frp_sum",
]

FIRMS_PREFIXES = [
    "bright_ti4_",
    "bright_ti5_",
    "scan_",
    "track_",
]

TEMPORAL_PREFIXES = [
    "prev_7d_",
    "prev_14d_",
    "prev_30d_",
    "cell_prior_count_",
    "recurrence_per_day_",
    "frp_prior_",
    "frp_trend_",
    "event_prev_count_",
    "time_gap_",
    "history_span_",
]

SPECTRAL_FEATURES = [
    "ndvi_mean",
    "ndvi_std",
    "ndvi_center",
    "gndvi_mean",
    "gndvi_std",
    "gndvi_center",
    "mndwi_mean",
    "mndwi_std",
    "mndwi_center",
    "ndbi_mean",
    "ndbi_std",
    "ndbi_center",
    "nbr_mean",
    "nbr_std",
    "nbr_center",
    "nbr2_mean",
    "nbr2_std",
    "nbr2_center",
    "evi_mean",
    "evi_std",
    "evi_center",
    "frac_vegetation",
    "frac_builtup",
    "frac_water",
    "frac_low_nbr",
    "scl_veg_frac",
    "scl_bare_frac",
    "scl_water_frac",
    "scl_cloud_frac",
    "B02_center",
    "B03_center",
    "B04_center",
    "B08_center",
    "B11_center",
    "B12_center",
    "b12_center_max",
    "b12_hotspot_ratio",
    "hot_px_center",
    "hot_px_total",
]

FROZEN_FEATURES = [
    f"frozen_pca_{i:02d}"
    for i in range(32)
]


# Anything in these categories must NEVER enter X.
FORBIDDEN_EXACT = {
    "event_id",
    "label",
    "label_confidence",
    "label_source",
    "label_reason",
    "trusted",
    "fold",
    "block_id",
    "has_image",
    "nearest_osm_type_mode",
    "nearest_osm_kind_mode",
}

FORBIDDEN_PREFIXES = [
    "nearest_osm_",
    "industrial_count_",
    "power_plant_count_",
    "quarry_count_",
    "flare_count_",
    "petroleum_well_count_",
    "industrial_landuse_",
    "emb_",
    "pca_",
    "p_",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def starts_with_any(column, prefixes):
    return any(column.startswith(prefix) for prefix in prefixes)


def build_feature_list(event_columns, use_firms, use_temporal):
    """Build only from explicitly approved FIRMS/temporal features."""

    features = []

    if use_firms:
        for col in event_columns:
            if col in FIRMS_EXACT:
                features.append(col)
            elif starts_with_any(col, FIRMS_PREFIXES):
                features.append(col)

    if use_temporal:
        for col in event_columns:
            if starts_with_any(col, TEMPORAL_PREFIXES):
                features.append(col)

    return list(dict.fromkeys(features))


def audit_features(features):
    """Fail loudly if an unsafe feature enters the experiment."""

    forbidden_found = []

    for col in features:
        if col in FORBIDDEN_EXACT:
            forbidden_found.append(col)
            continue

        if starts_with_any(col, FORBIDDEN_PREFIXES):
            forbidden_found.append(col)

    if forbidden_found:
        raise RuntimeError(
            "STRICT FEATURE AUDIT FAILED.\n"
            f"Forbidden features detected: {forbidden_found}"
        )


def add_image_features(df, spectral, embeddings, use_spectral, use_frozen):
    """Merge only explicitly approved Sentinel features."""

    result = df.copy()

    if use_spectral:
        spectral_keep = ["event_id"] + SPECTRAL_FEATURES

        missing = [
            col for col in SPECTRAL_FEATURES
            if col not in spectral.columns
        ]

        if missing:
            raise RuntimeError(
                f"Missing Sentinel spectral columns: {missing}"
            )

        result = result.merge(
            spectral[spectral_keep],
            on="event_id",
            how="left",
            validate="one_to_one",
        )

    if use_frozen:
        frozen_keep = ["event_id"] + FROZEN_FEATURES

        missing = [
            col for col in FROZEN_FEATURES
            if col not in embeddings.columns
        ]

        if missing:
            raise RuntimeError(
                f"Missing frozen Sentinel columns: {missing}"
            )

        result = result.merge(
            embeddings[frozen_keep],
            on="event_id",
            how="left",
            validate="one_to_one",
        )

    return result


def make_preprocessor(X):
    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical = [
        col for col in X.columns
        if col not in categorical
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def make_model():
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------
# One experiment
# ---------------------------------------------------------------------

def run_experiment(
    name,
    base_df,
    spectral,
    embeddings,
    folds,
    use_firms,
    use_temporal,
    use_spectral,
    use_frozen,
    label_to_int,
):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    df = add_image_features(
        base_df,
        spectral,
        embeddings,
        use_spectral,
        use_frozen,
    )

    features = build_feature_list(
        df.columns,
        use_firms,
        use_temporal,
    )

    if use_spectral:
        features.extend(SPECTRAL_FEATURES)

    if use_frozen:
        features.extend(FROZEN_FEATURES)

    features = list(dict.fromkeys(features))

    audit_features(features)

    missing_features = [
        col for col in features
        if col not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"{name}: missing features: {missing_features}"
        )

    print(f"Number of features: {len(features)}")
    print("Features:")
    print(features)

    X = df[features].copy()
    y = df["label"].map(label_to_int).values

    fold_values = folds.set_index("event_id").loc[
        df["event_id"],
        "fold",
    ].values

    predictions = np.full(len(df), -1, dtype=int)

    fold_metrics = []

    for fold in sorted(np.unique(fold_values)):
        print(f"\n{name} — spatial fold {fold}")

        train_idx = np.where(fold_values != fold)[0]
        test_idx = np.where(fold_values == fold)[0]

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        preprocessor = make_preprocessor(X_train)

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", make_model()),
            ]
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test).astype(int)

        predictions[test_idx] = pred

        macro_f1 = f1_score(
            y_test,
            pred,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_test,
            pred,
            average="weighted",
            zero_division=0,
        )

        accuracy = accuracy_score(y_test, pred)

        print(
            f"accuracy={accuracy:.6f} "
            f"macro_f1={macro_f1:.6f} "
            f"weighted_f1={weighted_f1:.6f}"
        )

        fold_metrics.append(
            {
                "fold": int(fold),
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "test_size": int(len(test_idx)),
            }
        )

    # -----------------------------------------------------------------
    # Overall OOF metrics
    # -----------------------------------------------------------------

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    accuracy = accuracy_score(y, predictions)

    print("\nOverall spatial OOF:")
    print(f"Accuracy:    {accuracy:.6f}")
    print(f"Macro F1:    {macro_f1:.6f}")
    print(f"Weighted F1: {weighted_f1:.6f}")

    report = classification_report(
        y,
        predictions,
        target_names=[
            name for name, _ in sorted(
                label_to_int.items(),
                key=lambda x: x[1],
            )
        ],
        output_dict=True,
        zero_division=0,
    )

    # -----------------------------------------------------------------
    # Save predictions
    # -----------------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "event_id": df["event_id"],
            "label": df["label"],
            "prediction": [
                next(
                    name
                    for name, value in label_to_int.items()
                    if value == pred
                )
                for pred in predictions
            ],
            "fold": fold_values,
        }
    )

    prediction_path = OUTPUT_DIR / f"{name}_oof_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False)

    metrics = {
        "experiment": name,
        "features": features,
        "num_features": len(features),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "fold_metrics": fold_metrics,
        "classification_report": report,
    }

    metrics_path = OUTPUT_DIR / f"{name}_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("Loading event table...")
    base_df = pd.read_parquet(EVENT_TABLE)

    print("Loading Sentinel spectral features...")
    spectral = pd.read_parquet(SPECTRAL_FILE)

    print("Loading Sentinel embeddings...")
    embeddings = pd.read_parquet(EMBEDDING_FILE)

    print("Loading spatial folds...")
    folds = pd.read_csv(FOLDS_FILE)

    # ---------------------------------------------------------------
    # Basic integrity checks
    # ---------------------------------------------------------------

    if base_df["event_id"].duplicated().any():
        raise RuntimeError(
            "event_training_table contains duplicate event_id values."
        )

    if spectral["event_id"].duplicated().any():
        raise RuntimeError(
            "Sentinel spectral file contains duplicate event_id values."
        )

    if embeddings["event_id"].duplicated().any():
        raise RuntimeError(
            "Sentinel embedding file contains duplicate event_id values."
        )

    if folds["event_id"].duplicated().any():
        raise RuntimeError(
            "Spatial fold file contains duplicate event_id values."
        )

    # Only classified events should be present.
    base_df = base_df[
        base_df["label"].notna()
        & (base_df["label"] != "unknown")
    ].copy()

    print(f"Classified events: {len(base_df):,}")

    # ---------------------------------------------------------------
    # Verify all event IDs have exactly one spatial fold.
    # ---------------------------------------------------------------

    fold_lookup = folds[
        ["event_id", "label", "fold"]
    ].copy()

    merged_check = base_df[
        ["event_id", "label"]
    ].merge(
        fold_lookup,
        on="event_id",
        how="left",
        validate="one_to_one",
        suffixes=("_event", "_fold"),
    )

    missing_folds = merged_check["fold"].isna().sum()

    if missing_folds:
        raise RuntimeError(
            f"{missing_folds} classified events have no spatial fold."
        )

    # ---------------------------------------------------------------
    # Verify labels agree between event table and spatial fold file.
    # ---------------------------------------------------------------

    label_mismatches = (
        merged_check["label_event"]
        != merged_check["label_fold"]
    )

    if label_mismatches.any():

        mismatch_count = int(label_mismatches.sum())

        examples = merged_check.loc[
            label_mismatches,
            [
                "event_id",
                "label_event",
                "label_fold",
            ],
        ].head(10)

        raise RuntimeError(
            "Label mismatch between event_training_table "
            "and spatial_folds_v1.csv.\n"
            f"Mismatched events: {mismatch_count}\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )

    print(
        "Spatial folds:",
        sorted(folds["fold"].dropna().unique().tolist()),
    )

    # ---------------------------------------------------------------
    # Class encoding
    # ---------------------------------------------------------------

    class_names = sorted(base_df["label"].unique())

    label_to_int = {
        label: idx
        for idx, label in enumerate(class_names)
    }

    print("\nClasses:")
    for label, idx in label_to_int.items():
        print(f"  {idx}: {label}")

    # ---------------------------------------------------------------
    # Verify Sentinel coverage
    # ---------------------------------------------------------------

    spectral_ids = set(spectral["event_id"])
    embedding_ids = set(embeddings["event_id"])
    event_ids = set(base_df["event_id"])

    print("\nSentinel coverage:")
    print(
        f"Spectral events available: "
        f"{len(event_ids & spectral_ids):,}/{len(event_ids):,}"
    )
    print(
        f"Embedding events available: "
        f"{len(event_ids & embedding_ids):,}/{len(event_ids):,}"
    )

    # ---------------------------------------------------------------
    # Experiment matrix
    # ---------------------------------------------------------------

    experiments = [
        {
            "name": "ST1_firms",
            "use_firms": True,
            "use_temporal": False,
            "use_spectral": False,
            "use_frozen": False,
        },
        {
            "name": "ST2_temporal",
            "use_firms": False,
            "use_temporal": True,
            "use_spectral": False,
            "use_frozen": False,
        },
        {
            "name": "ST3_firms_temporal",
            "use_firms": True,
            "use_temporal": True,
            "use_spectral": False,
            "use_frozen": False,
        },
        {
            "name": "ST4_firms_temporal_spectral",
            "use_firms": True,
            "use_temporal": True,
            "use_spectral": True,
            "use_frozen": False,
        },
        {
            "name": "ST5_firms_temporal_frozen",
            "use_firms": True,
            "use_temporal": True,
            "use_spectral": False,
            "use_frozen": True,
        },
        {
            "name": "ST6_firms_temporal_spectral_frozen",
            "use_firms": True,
            "use_temporal": True,
            "use_spectral": True,
            "use_frozen": True,
        },
    ]

    all_metrics = []

    for experiment in experiments:

        metrics = run_experiment(
            name=experiment["name"],
            base_df=base_df,
            spectral=spectral,
            embeddings=embeddings,
            folds=folds,
            use_firms=experiment["use_firms"],
            use_temporal=experiment["use_temporal"],
            use_spectral=experiment["use_spectral"],
            use_frozen=experiment["use_frozen"],
            label_to_int=label_to_int,
        )

        all_metrics.append(metrics)

    # ---------------------------------------------------------------
    # Compact comparison
    # ---------------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "experiment": m["experiment"],
                "num_features": m["num_features"],
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
            }
            for m in all_metrics
        ]
    )

    comparison_path = OUTPUT_DIR / "comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n" + "=" * 80)
    print("FINAL STRICT MULTIMODAL COMPARISON")
    print("=" * 80)
    print(comparison.to_string(index=False))

    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()