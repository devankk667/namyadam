"""Train and spatially evaluate event-level thermal/OSM/image fusion models.

The image handoff is keyed by event_id, so this script deliberately trains on
one row per event instead of duplicating an event for every FIRMS observation.
It uses the handoff's five spatial folds and reports trusted-label metrics as
primary; low-confidence rule labels are retained at reduced sample weight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "person2_fusion_image_features_v1"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
EVENTS_PATH = DATA_DIR / "firms_thermal_events.parquet"
LABELS_PATH = DATA_DIR / "firms_event_labels.parquet"
SPECTRAL_PATH = PACKAGE_DIR / "data" / "image_spectral_features_v1.parquet"
EMBEDDINGS_PATH = PACKAGE_DIR / "data" / "image_embeddings_v1.parquet"
FOLDS_PATH = PACKAGE_DIR / "data" / "spatial_folds_v1.csv"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "models" / "fusion_v1"
CLASSES = [
    "agricultural_burning",
    "industrial_fire",
    "mining_activity",
    "gas_flare",
]

THERMAL_FEATURES = [
    "max_frp",
    "median_frp",
    "mean_frp",
    "median_temp_diff",
    "n_detections",
    "n_days",
    "duration_days",
    "frac_night",
    "month_first",
    "month_last",
]
OSM_FEATURES = [
    "nearest_osm_distance_km",
    "industrial_count_1km",
    "industrial_count_2km",
    "industrial_count_5km",
    "power_plant_count_5km",
    "quarry_count_5km",
    "flare_count_5km",
    "petroleum_well_count_5km",
    "industrial_landuse_nearby",
]


def image_feature_groups(embedding_columns: list[str]) -> dict[str, list[str]]:
    """Return the named embedding groups without loading the 512-d embeddings."""
    return {
        "pca": sorted(c for c in embedding_columns if c.startswith("pca_")),
        "frozen_pca": sorted(c for c in embedding_columns if c.startswith("frozen_pca_")),
        "probabilities": sorted(c for c in embedding_columns if c.startswith("p_")),
    }


def build_feature_sets(
    base_features: list[str],
    spectral_features: list[str],
    groups: dict[str, list[str]],
    has_image_feature: str | None,
) -> dict[str, list[str]]:
    """Define handoff ablations A-E with a common base and shared rows/folds."""
    spectral = spectral_features + ([has_image_feature] if has_image_feature else [])
    fine_tuned = groups["pca"] + groups["probabilities"]
    frozen = groups["frozen_pca"]
    result = {
        "A_tabular": list(base_features),
        "B_tabular_spectral": list(base_features) + spectral,
        "C_tabular_spectral_frozen": list(base_features) + spectral + frozen,
        "D_tabular_spectral_cnn": list(base_features) + spectral + fine_tuned,
        "E_image_only": spectral + frozen + fine_tuned,
    }
    for name, columns in result.items():
        if not columns:
            raise ValueError(f"Feature set {name} is empty")
        if len(columns) != len(set(columns)):
            raise ValueError(f"Feature set {name} contains duplicate columns")
    return result


def trusted_sample_weights(trusted: np.ndarray, trusted_weight: float) -> np.ndarray:
    if trusted_weight < 1.0:
        raise ValueError("trusted_weight must be at least 1.0")
    return np.where(np.asarray(trusted, dtype=bool), trusted_weight, 1.0).astype(float)


def _parquet_columns(path: Path) -> list[str]:
    try:
        from pyarrow import parquet
    except ImportError as error:
        raise RuntimeError("Reading fusion Parquet files requires pyarrow; install backend/requirements.txt") from error
    return list(parquet.ParquetFile(path).schema.names)


def align_labels_to_spatial_folds(
    labels: pd.DataFrame,
    folds: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Use the handoff labels as canonical and audit local label differences."""
    label_event_ids = set(labels["event_id"])
    fold_event_ids = set(folds["event_id"])
    unassigned_labels = sorted(label_event_ids - fold_event_ids)
    labels_missing_locally = sorted(fold_event_ids - label_event_ids)

    comparison = folds[["event_id", "fold_label", "fold_label_confidence"]].merge(
        labels.rename(columns={
            "label": "local_label",
            "label_confidence": "local_label_confidence",
        }),
        on="event_id",
        how="left",
        validate="one_to_one",
    )
    has_local_label = comparison["local_label"].notna()
    label_mismatch = comparison["local_label"].astype(str) != comparison["fold_label"].astype(str)
    confidence_mismatch = (
        comparison["local_label_confidence"].astype(str)
        != comparison["fold_label_confidence"].astype(str)
    )
    label_mismatches = sorted(comparison.loc[has_local_label & label_mismatch, "event_id"].tolist())
    confidence_mismatches = sorted(
        comparison.loc[has_local_label & confidence_mismatch, "event_id"].tolist()
    )

    if unassigned_labels:
        print(
            f"[WARNING] {len(unassigned_labels)} local labeled event(s) have no supplied spatial fold "
            "and are outside this handoff evaluation cohort. "
            f"Examples: {unassigned_labels[:10]}",
            flush=True,
        )
    if labels_missing_locally:
        print(
            f"[WARNING] {len(labels_missing_locally)} handoff fold event(s) have no matching local label; "
            "using the versioned handoff labels for these events. "
            f"Examples: {labels_missing_locally[:10]}",
            flush=True,
        )
    if label_mismatches or confidence_mismatches:
        print(
            f"[WARNING] Local labels/confidence differ from the versioned handoff for "
            f"{len(label_mismatches)} label(s) and {len(confidence_mismatches)} confidence tier(s); "
            "using handoff values to keep labels aligned with spatial folds and OOF image features. "
            f"Label examples: {label_mismatches[:10]}; confidence examples: {confidence_mismatches[:10]}",
            flush=True,
        )

    canonical_labels = folds[["event_id", "fold_label", "fold_label_confidence"]].rename(columns={
        "fold_label": "label",
        "fold_label_confidence": "label_confidence",
    }).copy()
    return canonical_labels, {
        "local_labels_without_fold": unassigned_labels,
        "fold_events_without_local_label": labels_missing_locally,
        "local_label_mismatches": label_mismatches,
        "local_confidence_mismatches": confidence_mismatches,
    }


def load_event_dataset() -> tuple[pd.DataFrame, dict[str, list[str]], list[str], dict[str, list[str]]]:
    required = [EVENTS_PATH, LABELS_PATH, SPECTRAL_PATH, EMBEDDINGS_PATH, FOLDS_PATH]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing fusion input(s): " + ", ".join(missing))

    labels = pd.read_parquet(LABELS_PATH)
    labels = labels.loc[labels["label"].isin(CLASSES), ["event_id", "label", "label_confidence"]].copy()
    if labels["event_id"].isna().any():
        raise ValueError("Event labels contain a missing event_id")
    labels["event_id"] = labels["event_id"].astype(str)
    if labels["event_id"].duplicated().any():
        raise ValueError("Event labels must contain at most one row per event_id")

    folds = pd.read_csv(FOLDS_PATH)
    fold_columns = ["event_id", "fold", "trusted", "label", "label_confidence", "has_image"]
    if not set(fold_columns).issubset(folds.columns):
        raise ValueError(f"Spatial fold file must contain {fold_columns}")
    if folds["event_id"].duplicated().any():
        raise ValueError("Spatial folds must contain at most one row per event_id")
    if folds[fold_columns].isna().any().any():
        raise ValueError("Spatial fold metadata contains missing values")
    fold_values = pd.to_numeric(folds["fold"], errors="coerce")
    if fold_values.isna().any() or not set(fold_values.unique()).issubset(set(range(5))):
        raise ValueError("Spatial fold values must be integers in the range 0-4")
    folds["fold"] = fold_values.astype(int)
    folds["event_id"] = folds["event_id"].astype(str)
    folds = folds[fold_columns].rename(columns={
        "label": "fold_label",
        "label_confidence": "fold_label_confidence",
        "has_image": "fold_has_image",
    }).copy()
    labels, cohort_alignment = align_labels_to_spatial_folds(labels, folds)

    event_columns = _parquet_columns(EVENTS_PATH)
    if "event_id" not in event_columns:
        raise ValueError("Thermal event table has no event_id key")
    base_features = [column for column in THERMAL_FEATURES + OSM_FEATURES if column in event_columns]
    events = pd.read_parquet(
        EVENTS_PATH,
        columns=["event_id", *base_features],
        filters=[("event_id", "in", labels["event_id"].tolist())],
    )
    if events["event_id"].duplicated().any():
        raise ValueError("Thermal events must contain a unique event_id column")
    missing_thermal = [column for column in ("max_frp", "n_detections", "n_days") if column not in base_features]
    if missing_thermal:
        raise ValueError(f"Thermal event table is missing required fields: {missing_thermal}")

    spectral_columns = _parquet_columns(SPECTRAL_PATH)
    if "event_id" not in spectral_columns:
        raise ValueError("Spectral feature table has no event_id key")
    spectral_features = [c for c in spectral_columns if c not in {"event_id", "has_image"}]
    spectral_columns_to_read = ["event_id", *spectral_features]
    if "has_image" in spectral_columns:
        spectral_columns_to_read.append("has_image")
    spectral = pd.read_parquet(SPECTRAL_PATH, columns=spectral_columns_to_read)
    if "has_image" in spectral.columns:
        spectral = spectral.rename(columns={"has_image": "spectral_has_image"})

    embedding_columns = _parquet_columns(EMBEDDINGS_PATH)
    if "event_id" not in embedding_columns or "fold" not in embedding_columns:
        raise ValueError("Embedding table must contain event_id and fold columns")
    groups = image_feature_groups(embedding_columns)
    if not groups["pca"] or not groups["frozen_pca"] or not groups["probabilities"]:
        raise ValueError("Embedding table is missing pca_*, frozen_pca_* or image probability features")
    selected_embeddings = ["event_id", "fold", *groups["pca"], *groups["frozen_pca"], *groups["probabilities"]]
    embeddings = pd.read_parquet(EMBEDDINGS_PATH, columns=selected_embeddings)
    if embeddings["event_id"].duplicated().any() or spectral["event_id"].duplicated().any():
        raise ValueError("Image feature tables must contain one row per event_id")
    fold_check = folds[["event_id", "fold"]].merge(
        embeddings[["event_id", "fold"]],
        on="event_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_shared", "_embedding"),
    )
    fold_mismatch = fold_check["fold_shared"] != fold_check["fold_embedding"]
    if fold_mismatch.any():
        examples = fold_check.loc[fold_mismatch, "event_id"].head(5).tolist()
        raise ValueError(f"Embedding folds disagree with the shared spatial folds (examples: {examples})")
    embeddings = embeddings.drop(columns="fold")

    frame = labels.merge(folds, on="event_id", how="left", validate="one_to_one", indicator="_fold_merge")
    missing_folds = int((frame["_fold_merge"] != "both").sum())
    if missing_folds:
        raise RuntimeError("Internal alignment error: a selected label has no spatial fold")
    frame = frame.drop(columns="_fold_merge")

    expected_trusted = frame["label_confidence"].isin(["high", "medium"])
    actual_trusted = pd.to_numeric(frame["trusted"], errors="coerce").fillna(0).astype(bool)
    if not expected_trusted.equals(actual_trusted):
        raise ValueError("The handoff trusted flag does not match high/medium label confidence")
    frame = frame.drop(columns=["fold_label", "fold_label_confidence"])
    frame = frame.merge(
        events[["event_id", *base_features]],
        on="event_id",
        how="left",
        validate="one_to_one",
        indicator="_event_merge",
    )
    missing_events = int((frame["_event_merge"] != "both").sum())
    if missing_events:
        raise ValueError(f"{missing_events} labeled events are missing thermal event features")
    frame = frame.drop(columns="_event_merge")
    frame = frame.merge(spectral, on="event_id", how="left", validate="one_to_one")
    frame = frame.merge(embeddings, on="event_id", how="left", validate="one_to_one")

    frame["has_image"] = pd.to_numeric(frame["fold_has_image"], errors="coerce").fillna(0).astype(int)
    if "spectral_has_image" in frame.columns:
        spectral_image = pd.to_numeric(frame["spectral_has_image"], errors="coerce").fillna(0).astype(int)
        image_mismatch = frame["has_image"] != spectral_image
        if image_mismatch.any():
            examples = frame.loc[image_mismatch, "event_id"].head(5).tolist()
            raise ValueError(f"Spectral has_image flags disagree with spatial-fold metadata (examples: {examples})")
    frame = frame.drop(columns=["fold_has_image", "spectral_has_image"], errors="ignore")
    embedding_features = groups["pca"] + groups["frozen_pca"] + groups["probabilities"]
    missing_embeddings = frame["has_image"].eq(1) & frame[embedding_features].isna().all(axis=1)
    if missing_embeddings.any():
        examples = frame.loc[missing_embeddings, "event_id"].head(5).tolist()
        raise ValueError(f"Image-bearing events have no embedding features (examples: {examples})")
    for column in base_features + spectral_features + embedding_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    feature_sets = build_feature_sets(base_features, spectral_features, groups, "has_image")
    return frame, feature_sets, base_features, cohort_alignment


def make_estimator(name: str, seed: int, trees: int) -> Pipeline:
    if name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", classifier),
        ]
    elif name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", classifier),
        ]
    else:
        raise ValueError(f"Unknown estimator: {name}")
    return Pipeline(steps)


def _macro_f1_without_flare(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    keep = np.asarray(y_true) != "gas_flare"
    true_labels = np.asarray(y_true)[keep]
    predicted_labels = np.asarray(y_pred)[keep]
    scores = []
    for label in CLASSES:
        if label == "gas_flare":
            continue
        true_positive = np.sum((true_labels == label) & (predicted_labels == label))
        false_positive = np.sum((true_labels != label) & (predicted_labels == label))
        false_negative = np.sum((true_labels == label) & (predicted_labels != label))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not len(y_true):
        return {
            "n": 0,
            "macro_f1": 0.0,
            "macro_f1_no_flare": 0.0,
            "balanced_accuracy": 0.0,
            "per_class": {},
            "confusion_matrix": np.zeros((len(CLASSES), len(CLASSES)), dtype=int).tolist(),
        }
    no_flare = y_true != "gas_flare"
    return {
        "n": len(y_true),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "macro_f1_no_flare": _macro_f1_without_flare(y_true, y_pred) if no_flare.any() else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "per_class": classification_report(
            y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }


def feature_importance(estimator: Pipeline, feature_names: list[str]) -> dict[str, float]:
    model = estimator.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).mean(axis=0)
    else:
        return {}
    return dict(sorted(zip(feature_names, values.tolist()), key=lambda item: item[1], reverse=True))


def evaluate_candidate(
    frame: pd.DataFrame,
    feature_names: list[str],
    estimator_name: str,
    seed: int,
    trees: int,
    trusted_weight: float,
) -> tuple[dict[str, Any], Pipeline, dict[str, float], pd.DataFrame]:
    y = frame["label"].astype(str).to_numpy()
    folds = frame["fold"].astype(int).to_numpy()
    trusted = pd.to_numeric(frame["trusted"], errors="coerce").fillna(0).astype(bool).to_numpy()
    X = frame[feature_names]
    oof = np.empty(len(frame), dtype=object)
    oof[:] = ""
    oof_probabilities = np.zeros((len(frame), len(CLASSES)), dtype=float)
    fold_metrics: dict[str, Any] = {}
    importances: list[dict[str, float]] = []

    for fold in sorted(np.unique(folds)):
        train_mask = folds != fold
        test_mask = folds == fold
        if not train_mask.any() or not test_mask.any():
            raise ValueError(f"Spatial fold {fold} has an empty train or test split")
        if len(set(y[train_mask])) < 2:
            raise ValueError(f"Spatial fold {fold} training split has fewer than two classes")
        model = make_estimator(estimator_name, seed + int(fold), trees)
        weights = trusted_sample_weights(trusted[train_mask], trusted_weight)
        model.fit(X.loc[train_mask], y[train_mask], model__sample_weight=weights)
        oof[test_mask] = model.predict(X.loc[test_mask])
        fold_probabilities = model.predict_proba(X.loc[test_mask])
        model_classes = model.named_steps["model"].classes_
        for class_index, class_name in enumerate(model_classes):
            oof_probabilities[test_mask, CLASSES.index(str(class_name))] = fold_probabilities[:, class_index]
        test_trusted = test_mask & trusted
        fold_metrics[str(int(fold))] = {
            "trusted": score_predictions(y[test_trusted], oof[test_trusted]),
            "all_events": score_predictions(y[test_mask], oof[test_mask]),
        }
        importances.append(feature_importance(model, feature_names))

    if (oof == "").any():
        raise RuntimeError("Not every event received an out-of-fold prediction")
    trusted_metrics = score_predictions(y[trusted], oof[trusted])
    all_metrics = score_predictions(y, oof)
    importance_mean: dict[str, float] = {}
    if importances:
        for feature in feature_names:
            values = [item[feature] for item in importances if feature in item]
            if values:
                importance_mean[feature] = float(np.mean(values))
    importance_mean = dict(sorted(importance_mean.items(), key=lambda item: item[1], reverse=True))

    # Fit a separate all-label artifact for future unseen events after recording
    # CV scores. The backend's current dashboard overlay uses only OOF outputs.
    final_model = make_estimator(estimator_name, seed, trees)
    final_model.fit(
        X,
        y,
        model__sample_weight=trusted_sample_weights(trusted, trusted_weight),
    )
    oof_predictions = pd.DataFrame({
        "event_id": frame["event_id"].to_numpy(),
        "fold": folds,
        "trusted": trusted.astype(int),
        "has_image": pd.to_numeric(frame["has_image"], errors="coerce").fillna(0).astype(int).to_numpy(),
        "predicted_class": oof,
        "confidence": oof_probabilities.max(axis=1),
    })
    for class_index, class_name in enumerate(CLASSES):
        oof_predictions[f"p_{class_name}"] = oof_probabilities[:, class_index]

    result = {
        "trusted_oof": trusted_metrics,
        "all_events_oof": all_metrics,
        "folds": fold_metrics,
        "trusted_events": int(trusted.sum()),
        "all_events": len(frame),
        "trusted_weight": trusted_weight,
        "feature_importances_mean": importance_mean,
    }
    return result, final_model, importance_mean, oof_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, default=200, help="Random forest trees; ignored for logistic regression")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trusted-weight", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--estimators", nargs="+", choices=["logistic_regression", "random_forest"],
                        default=["logistic_regression", "random_forest"])
    parser.add_argument("--feature-sets", nargs="+",
                        choices=["A_tabular", "B_tabular_spectral", "C_tabular_spectral_frozen",
                                 "D_tabular_spectral_cnn", "E_image_only"],
                        default=["A_tabular", "B_tabular_spectral", "C_tabular_spectral_frozen",
                                 "D_tabular_spectral_cnn", "E_image_only"])
    args = parser.parse_args()
    if args.trees < 1:
        parser.error("--trees must be positive")
    if args.trusted_weight < 1:
        parser.error("--trusted-weight must be at least 1")

    frame, feature_sets, _, cohort_alignment = load_event_dataset()
    if frame["fold"].isna().any():
        raise ValueError("One or more labeled events have no spatial fold assignment")
    frame["fold"] = frame["fold"].astype(int)
    if set(frame["fold"].unique()) != set(range(5)):
        raise ValueError(f"Expected all five spatial folds 0-4; found {sorted(frame['fold'].unique())}")
    print(
        f"Loaded {len(frame):,} labelled events, {int(pd.to_numeric(frame['trusted'], errors='coerce').fillna(0).sum()):,} trusted; "
        f"folds={sorted(frame['fold'].unique())}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "dataset": {
            "source": "person2_fusion_image_features_v1",
            "rows": len(frame),
            "trusted_rows": int(pd.to_numeric(frame["trusted"], errors="coerce").fillna(0).sum()),
            "cohort_alignment": {
                "local_labels_without_fold_count": len(cohort_alignment["local_labels_without_fold"]),
                "local_labels_without_fold_examples": cohort_alignment["local_labels_without_fold"][:10],
                "fold_events_without_local_label_count": len(cohort_alignment["fold_events_without_local_label"]),
                "fold_events_without_local_label_examples": cohort_alignment["fold_events_without_local_label"][:10],
                "local_label_mismatch_count": len(cohort_alignment["local_label_mismatches"]),
                "local_label_mismatch_examples": cohort_alignment["local_label_mismatches"][:10],
                "local_confidence_mismatch_count": len(cohort_alignment["local_confidence_mismatches"]),
                "local_confidence_mismatch_examples": cohort_alignment["local_confidence_mismatches"][:10],
            },
            "folds": sorted(int(value) for value in frame["fold"].unique()),
            "classes": CLASSES,
            "evaluation": "5-fold spatial out-of-fold; trusted labels are the primary metric",
            "caveat": "Most labels are low-confidence rules. Out-of-fold CNN features require these same spatial folds.",
        },
        "candidates": {},
    }
    selection: list[tuple[float, str, str]] = []
    oof_prediction_frames: list[pd.DataFrame] = []

    for feature_set_name in args.feature_sets:
        feature_names = feature_sets[feature_set_name]
        for estimator_name in args.estimators:
            key = f"{feature_set_name}/{estimator_name}"
            print(f"Training {key} ({len(feature_names)} features)", flush=True)
            metrics, model, importances, oof_predictions = evaluate_candidate(
                frame=frame,
                feature_names=feature_names,
                estimator_name=estimator_name,
                seed=args.seed,
                trees=args.trees,
                trusted_weight=args.trusted_weight,
            )
            oof_predictions["feature_set"] = feature_set_name
            oof_predictions["estimator"] = estimator_name
            oof_prediction_frames.append(oof_predictions)
            report["candidates"][key] = {
                "features": feature_names,
                "metrics": metrics,
            }
            selection.append((metrics["trusted_oof"]["macro_f1_no_flare"], feature_set_name, estimator_name))
            out_dir = args.output_dir / feature_set_name / estimator_name
            out_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, out_dir / "model.joblib")
            metadata = {
                "model": f"{estimator_name} event-level image fusion",
                "version": "fusion_v1",
                "feature_set": feature_set_name,
                "features": feature_names,
                "classes": CLASSES,
                "macro_f1": metrics["trusted_oof"]["macro_f1"],
                "trusted_macro_f1_no_flare": metrics["trusted_oof"]["macro_f1_no_flare"],
                "n_train": len(frame),
                "n_trusted": int(metrics["trusted_events"]),
                "split": "5-fold spatial CV using person2_fusion_image_features_v1/data/spatial_folds_v1.csv",
                "feature_importances": importances,
                "metrics": metrics,
                "inference_note": "Requires event-level image features joined to each detection; not a drop-in for the current API model.",
            }
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            print(
                f"  trusted macro-F1={metrics['trusted_oof']['macro_f1']:.3f}, "
                f"without flare={metrics['trusted_oof']['macro_f1_no_flare']:.3f}; saved {out_dir}",
                flush=True,
            )

    report["selection_rule"] = "Highest trusted OOF macro-F1 excluding gas_flare; flare has only 8 trusted events from 2 sites."
    best_candidate = max(selection, key=lambda candidate: candidate[0])
    report["best_candidate_by_selection_rule"] = {
        "feature_set": best_candidate[1],
        "estimator": best_candidate[2],
        "trusted_oof_macro_f1_no_flare": best_candidate[0],
    }
    image_fusion_candidates = [candidate for candidate in selection if candidate[1] in {
        "B_tabular_spectral", "C_tabular_spectral_frozen", "D_tabular_spectral_cnn",
    }]
    if image_fusion_candidates:
        best_image_candidate = max(image_fusion_candidates, key=lambda candidate: candidate[0])
        report["best_image_fusion_candidate"] = {
            "feature_set": best_image_candidate[1],
            "estimator": best_image_candidate[2],
            "trusted_oof_macro_f1_no_flare": best_image_candidate[0],
        }
    else:
        report["best_image_fusion_candidate"] = None
    pd.concat(oof_prediction_frames, ignore_index=True).to_parquet(
        args.output_dir / "fusion_oof_predictions_v1.parquet",
        index=False,
    )
    (args.output_dir / "fusion_metrics_v1.json").write_text(json.dumps(report, indent=2))
    print("\nTop candidates by trusted OOF macro-F1 (excluding gas_flare):", flush=True)
    for score, feature_set_name, estimator_name in sorted(selection, reverse=True):
        print(f"  {score:.4f}  {feature_set_name}/{estimator_name}", flush=True)
    print(f"Report: {args.output_dir / 'fusion_metrics_v1.json'}", flush=True)


if __name__ == "__main__":
    main()
