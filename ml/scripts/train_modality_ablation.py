from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH = Path("data/processed/event_training_table.parquet")
OUTPUT_DIR = Path("ml/models/modality_ablation")

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "label"


# =============================================================================
# FEATURE GROUPS
# =============================================================================

FIRMS_FEATURES = [
    "observation_count",

    "bright_ti4_mean",
    "bright_ti4_max",
    "bright_ti4_std",

    "scan_mean",
    "scan_max",
    "scan_std",

    "track_mean",
    "track_max",
    "track_std",

    "bright_ti5_mean",
    "bright_ti5_max",
    "bright_ti5_std",

    "frp_mean",
    "frp_max",
    "frp_std",
    "frp_sum",

    "confidence_mode",
    "daynight_mode",
]


TEMPORAL_FEATURES = [
    "prev_7d_mean",
    "prev_7d_max",

    "prev_14d_mean",
    "prev_14d_max",

    "prev_30d_mean",
    "prev_30d_max",

    "cell_prior_count_mean",
    "cell_prior_count_max",

    "recurrence_per_day_mean",
    "recurrence_per_day_max",
    "recurrence_per_day_last",

    "frp_prior_mean_mean",
    "frp_prior_mean_max",

    "frp_prior_std_mean",
    "frp_prior_std_max",

    "event_prev_count_mean",
    "event_prev_count_max",
    "event_prev_count_last",

    "time_gap_mean",
    "time_gap_median",
    "time_gap_max",

    "history_span_max",

    "frp_trend_mean",
    "frp_trend_max",
    "frp_trend_min",
    "frp_trend_last",
]


# =============================================================================
# MODEL
# =============================================================================

def create_model():

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
    )


# =============================================================================
# MAIN
# =============================================================================

print("=" * 80)
print("NAMYADAM — MODALITY ABLATION")
print("=" * 80)


# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------

print("\n[1/7] Loading event dataset...")

df = pd.read_parquet(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Unique events: {df['event_id'].nunique():,}")

print("\nClass distribution:")
print(df[TARGET].value_counts())


# -----------------------------------------------------------------------------
# 2. PREPARE TARGET
# -----------------------------------------------------------------------------

print("\n[2/7] Preparing target...")

class_names = sorted(df[TARGET].unique())

class_to_idx = {
    class_name: idx
    for idx, class_name in enumerate(class_names)
}

y = df[TARGET].map(class_to_idx)

print("\nClasses:")
for class_name, idx in class_to_idx.items():
    print(f"  {idx}: {class_name}")


# -----------------------------------------------------------------------------
# 3. CREATE COMMON EVENT-LEVEL SPLIT
# -----------------------------------------------------------------------------

print("\n[3/7] Creating common event-level split...")

# One split is created and reused for ALL three experiments.
# This is important because it makes the modality comparison fair.

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training events: {len(train_idx):,}")
print(f"Testing events:  {len(test_idx):,}")


# -----------------------------------------------------------------------------
# 4. DEFINE FEATURE SETS
# -----------------------------------------------------------------------------

print("\n[4/7] Preparing modality feature sets...")

feature_sets = {
    "firms_only": FIRMS_FEATURES,
    "temporal_only": TEMPORAL_FEATURES,
    "firms_temporal": FIRMS_FEATURES + TEMPORAL_FEATURES,
}

for name, features in feature_sets.items():

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing features: {missing}"
        )

    print(f"\n{name}:")
    print(f"  Features: {len(features)}")


# -----------------------------------------------------------------------------
# 5. TRAIN EACH MODALITY
# -----------------------------------------------------------------------------

print("\n[5/7] Training modality models...")
print("=" * 80)

results = {}
trained_models = {}

for modality_name, feature_columns in feature_sets.items():

    print(f"\nTraining: {modality_name}")
    print("-" * 80)

    X = df[feature_columns].copy()

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    transformers = []

    if numeric_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = create_model()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    results[modality_name] = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "num_features": len(feature_columns),
    }

    trained_models[modality_name] = pipeline

    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")


# -----------------------------------------------------------------------------
# 6. COMPARISON + REPORTS
# -----------------------------------------------------------------------------

print("\n")
print("=" * 80)
print("MODALITY COMPARISON")
print("=" * 80)

for modality_name, result in results.items():

    print(
        f"{modality_name:<20} | "
        f"Features: {result['num_features']:>2} | "
        f"Macro F1: {result['macro_f1']:.4f} | "
        f"Weighted F1: {result['weighted_f1']:.4f} | "
        f"Accuracy: {result['accuracy']:.4f}"
    )


best_modality = max(
    results,
    key=lambda name: results[name]["macro_f1"],
)

print("\n")
print("=" * 80)
print("BEST MODALITY")
print("=" * 80)

print(
    f"Selected by Macro F1: {best_modality}"
)


print("\n")
print("=" * 80)
print(f"CLASSIFICATION REPORT — {best_modality}")
print("=" * 80)

best_pipeline = trained_models[best_modality]

best_predictions = best_pipeline.predict(
    df[feature_sets[best_modality]].iloc[test_idx]
)

print(
    classification_report(
        y.iloc[test_idx],
        best_predictions,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
)


# -----------------------------------------------------------------------------
# 7. SAVE ARTIFACTS
# -----------------------------------------------------------------------------

print("\n[7/7] Saving artifacts...")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Save all three trained models
for modality_name, pipeline in trained_models.items():

    joblib.dump(
        pipeline,
        OUTPUT_DIR / f"{modality_name}.joblib",
    )


metadata = {
    "dataset": str(DATA_PATH),
    "target": TARGET,
    "class_names": class_names,
    "class_to_idx": class_to_idx,
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "split_type": "event-level stratified random split",
    "feature_sets": {
        name: features
        for name, features in feature_sets.items()
    },
    "results": results,
    "best_modality": best_modality,
    "note": (
        "Labels are rule-derived. Results measure agreement with "
        "the current labeling system and are not independent "
        "real-world ground-truth validation."
    ),
}

with open(
    OUTPUT_DIR / "metadata.json",
    "w",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


print("\nArtifacts saved:")

for modality_name in trained_models:

    print(
        f"  Model: "
        f"{OUTPUT_DIR / (modality_name + '.joblib')}"
    )

print(
    f"  Metadata: {OUTPUT_DIR / 'metadata.json'}"
)

print("\nDone.")