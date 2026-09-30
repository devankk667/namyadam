"""
Namyadam — Event-Level Tabular Baseline

Trains and evaluates:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost

Prediction unit:
    ONE EVENT

Input:
    data/processed/event_training_table.parquet

The current labels are rule-derived event labels, so this is
a supervised baseline against the current labeling system.

Split:
    Event-level stratified train/test split

Primary metric:
    Macro F1
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/processed/event_training_table.parquet"
)

OUTPUT_DIR = Path(
    "ml/models/tabular_reduced"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# MAIN
# ============================================================

def train_and_evaluate():

    print("=" * 80)
    print("NAMYADAM — EVENT-LEVEL TABULAR BASELINE")
    print("=" * 80)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n[1/8] Loading event dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Unique events: {df['event_id'].nunique():,}")

    print("\nClass distribution:")
    print(
        df["label"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # 2. DEFINE TARGET
    # ========================================================

    print("\n[2/8] Preparing target...")

    TARGET = "label"

    class_names = sorted(
        df[TARGET].unique()
    )

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(class_names)
    }

    y = df[TARGET].map(class_to_idx)

    print("\nClasses:")

    for class_name, idx in class_to_idx.items():
        print(
            f"  {idx}: {class_name}"
        )

    # ========================================================
    # 3. REMOVE NON-FEATURE COLUMNS
    # ========================================================

    print("\n[3/8] Preparing features...")

    DROP_COLUMNS = [
        "event_id",
        "label",
        "label_confidence",
        "label_source",
    ]

    X = df.drop(
        columns=DROP_COLUMNS
    )

    # --------------------------------------------------------
    # Remove OSM features that are closely related to the
    # rules used to generate the event labels.
    #
    # These are kept in the original baseline, but excluded
    # here to test how much predictive signal remains without
    # these direct rule-proxy features.
    # --------------------------------------------------------

    RULE_PROXY_FEATURES = [
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

    X = X.drop(
        columns=RULE_PROXY_FEATURES
    )

    print(
        f"Feature columns after removing rule-proxy features: {X.shape[1]}"
    )

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    print(
        f"Numeric features: {len(numeric_features)}"
    )

    print(
        f"Categorical features: {len(categorical_features)}"
    )

    print("\nCategorical features:")

    for col in categorical_features:
        print(f"  {col}")

    # ========================================================
    # 4. TRAIN / TEST SPLIT
    # ========================================================

    print("\n[4/8] Creating event-level split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Training events: {len(X_train):,}"
    )

    print(
        f"Testing events:  {len(X_test):,}"
    )

    print("\nTraining class distribution:")

    train_counts = y_train.value_counts().sort_index()

    for idx, count in train_counts.items():
        print(
            f"  {class_names[idx]:25s} {count:6,}"
        )

    print("\nTesting class distribution:")

    test_counts = y_test.value_counts().sort_index()

    for idx, count in test_counts.items():
        print(
            f"  {class_names[idx]:25s} {count:6,}"
        )

    # ========================================================
    # 5. PREPROCESSING
    # ========================================================

    print("\n[5/8] Building preprocessing pipelines...")

    # Numeric preprocessing for Logistic Regression.
    #
    # Median imputation:
    #   handles the 309 std-related NaNs and 203 gap-related NaNs.
    #
    # StandardScaler:
    #   important for Logistic Regression because features
    #   have very different numerical scales.

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    # Categorical preprocessing.

    categorical_pipeline = Pipeline(
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
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    # ========================================================
    # 6. DEFINE MODELS
    # ========================================================

    print("\n[6/8] Creating models...")

    models = {

        "logistic_regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

        "random_forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "xgboost":
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
    }

    trained_models = {}
    eval_results = {}
    predictions = {}

    # ========================================================
    # 7. TRAIN + EVALUATE
    # ========================================================

    print("\n[7/8] Training models...")
    print("=" * 80)

    for model_name, model in models.items():

        print(
            f"\nTraining: {model_name}"
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        y_pred = pipeline.predict(
            X_test
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        precision_macro = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        recall_macro = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        trained_models[model_name] = pipeline

        predictions[model_name] = y_pred

        eval_results[model_name] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "precision_macro": float(
                precision_macro
            ),
            "recall_macro": float(
                recall_macro
            ),
        }

        print(
            f"  Accuracy:    {accuracy:.4f}"
        )

        print(
            f"  Macro F1:    {macro_f1:.4f}"
        )

        print(
            f"  Weighted F1: {weighted_f1:.4f}"
        )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print("\n")
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    for model_name, metrics in eval_results.items():

        print(
            f"{model_name:<22}"
            f" | Macro F1: {metrics['macro_f1']:.4f}"
            f" | Weighted F1: {metrics['weighted_f1']:.4f}"
            f" | Accuracy: {metrics['accuracy']:.4f}"
        )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_model_name = max(
        eval_results,
        key=lambda name:
            eval_results[name]["macro_f1"]
    )

    best_model = trained_models[
        best_model_name
    ]

    best_predictions = predictions[
        best_model_name
    ]

    print("\n")
    print("=" * 80)
    print("BEST MODEL")
    print("=" * 80)

    print(
        f"Selected by Macro F1: {best_model_name}"
    )

    # ========================================================
    # DETAILED CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("=" * 80)
    print(
        f"CLASSIFICATION REPORT — {best_model_name}"
    )
    print("=" * 80)

    print(
        classification_report(
            y_test,
            best_predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        best_predictions,
        labels=list(range(len(class_names))),
    )

    print("\nConfusion matrix:")

    print(
        pd.DataFrame(
            cm,
            index=class_names,
            columns=class_names,
        )
    )

    # ========================================================
    # 8. SAVE ARTIFACTS
    # ========================================================

    print("\n[8/8] Saving artifacts...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save best pipeline.
    model_path = (
        OUTPUT_DIR /
        "best_model.joblib"
    )

    joblib.dump(
        best_model,
        model_path,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    feature_importances = {}

    underlying_model = best_model.named_steps[
        "model"
    ]

    fitted_preprocessor = best_model.named_steps[
        "preprocessor"
    ]

    try:

        transformed_feature_names = (
            fitted_preprocessor
            .get_feature_names_out()
        )

        if hasattr(
            underlying_model,
            "feature_importances_"
        ):

            importance_values = (
                underlying_model
                .feature_importances_
            )

            feature_importances = dict(
                zip(
                    transformed_feature_names,
                    importance_values.astype(float),
                )
            )

        elif hasattr(
            underlying_model,
            "coef_"
        ):

            coefficients = np.abs(
                underlying_model.coef_
            ).mean(axis=0)

            feature_importances = dict(
                zip(
                    transformed_feature_names,
                    coefficients.astype(float),
                )
            )

    except Exception as exc:

        print(
            f"Could not extract feature importance: {exc}"
        )

    # Sort feature importance.

    feature_importances = dict(
        sorted(
            feature_importances.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "model_name":
            best_model_name,

        "version":
            "1.0.0",

        "prediction_unit":
            "event",

        "dataset":
            str(DATA_PATH),

        "train_samples":
            int(len(X_train)),

        "test_samples":
            int(len(X_test)),

        "split_method":
            "event-level stratified train/test split",

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "classes":
            class_names,

        "class_to_index":
            class_to_idx,

        "feature_count_before_encoding":
            int(X.shape[1]),

        "numeric_features":
            numeric_features,

        "categorical_features":
            categorical_features,

        "metrics":
            eval_results[best_model_name],

        "all_model_benchmarks":
            eval_results,

        "confusion_matrix":
            cm.tolist(),

        "feature_importances":
            feature_importances,

        "label_warning":
            (
                "Labels are rule-derived. "
                "This model evaluates approximation of the "
                "current labeling system and is not independent "
                "validation against manually verified ground truth."
            ),
    }

    metadata_path = (
        OUTPUT_DIR /
        "metadata.json"
    )

    with open(
        metadata_path,
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save feature importance separately
    # --------------------------------------------------------

    importance_path = (
        OUTPUT_DIR /
        "feature_importance.csv"
    )

    if feature_importances:

        importance_df = pd.DataFrame(
            {
                "feature": list(
                    feature_importances.keys()
                ),
                "importance": list(
                    feature_importances.values()
                ),
            }
        )

        importance_df.to_csv(
            importance_path,
            index=False,
        )

    print("\nArtifacts saved:")

    print(
        f"  Model:       {model_path}"
    )

    print(
        f"  Metadata:    {metadata_path}"
    )

    if feature_importances:

        print(
            f"  Importance:  {importance_path}"
        )

    print("\nDone.")


if __name__ == "__main__":
    train_and_evaluate()