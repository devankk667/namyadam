"""Exploratory one-fold spatial holdout evaluation for the selected fusion model.

This is not an independent final test: the candidate was previously selected
using the supplied five folds. Reserve a newly labeled, geographically separate
cohort for a genuinely independent estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from train_image_fusion import (
    CLASSES,
    FOLDS_PATH,
    OUTPUT_DIR,
    load_event_dataset,
    make_estimator,
    score_predictions,
    trusted_sample_weights,
)


def bootstrap_block_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    blocks: np.ndarray,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Percentile intervals from resampling spatial blocks, not individual rows."""
    unique_blocks = np.unique(blocks)
    if iterations <= 0 or len(unique_blocks) < 2:
        return {"iterations": 0, "blocks": int(len(unique_blocks)), "macro_f1_95_ci": None, "per_class_f1_95_ci": {}}
    rng = np.random.default_rng(seed)
    macro_scores: list[float] = []
    class_scores: dict[str, list[float]] = {name: [] for name in CLASSES}
    indices_by_block = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
    for _ in range(iterations):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sample = np.concatenate([indices_by_block[block] for block in sampled_blocks])
        true, pred = y_true[sample], y_pred[sample]
        macro_scores.append(float(f1_score(true, pred, labels=CLASSES, average="macro", zero_division=0)))
        per_class = f1_score(true, pred, labels=CLASSES, average=None, zero_division=0)
        for label, value in zip(CLASSES, per_class):
            class_scores[label].append(float(value))
    return {
        "iterations": iterations,
        "blocks": int(len(unique_blocks)),
        "macro_f1_95_ci": [float(v) for v in np.quantile(macro_scores, [0.025, 0.975])],
        "per_class_f1_95_ci": {
            label: [float(v) for v in np.quantile(values, [0.025, 0.975])]
            for label, values in class_scores.items()
        },
    }


def load_selected_candidate(report_path: Path) -> tuple[str, str]:
    default = ("C_tabular_spectral_frozen", "logistic_regression")
    if not report_path.is_file():
        return default
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        candidate = report.get("best_image_fusion_candidate") or {}
        feature_set, estimator = candidate.get("feature_set"), candidate.get("estimator")
        if feature_set and estimator:
            return str(feature_set), str(estimator)
    except (OSError, json.JSONDecodeError):
        pass
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-fold", type=int, default=4, choices=range(5))
    parser.add_argument("--feature-set", choices=["A_tabular", "B_tabular_spectral", "C_tabular_spectral_frozen", "D_tabular_spectral_cnn", "E_image_only"])
    parser.add_argument("--estimator", choices=["logistic_regression", "random_forest"])
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--trusted-weight", type=float, default=5.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trees < 1 or args.trusted_weight < 1 or args.bootstrap_iterations < 0:
        parser.error("trees must be positive, trusted-weight >= 1, and bootstrap iterations nonnegative")

    frame, feature_sets, _, _ = load_event_dataset()
    folds = pd.read_csv(FOLDS_PATH, dtype={"event_id": str, "block_id": str})
    if not {"event_id", "block_id", "fold"}.issubset(folds.columns):
        raise ValueError("Spatial fold file must contain event_id, block_id, and fold")
    if folds["event_id"].duplicated().any() or folds["block_id"].isna().any():
        raise ValueError("Spatial fold event keys must be unique and block IDs populated")
    frame = frame.merge(folds[["event_id", "block_id"]], on="event_id", how="left", validate="one_to_one")
    if frame["block_id"].isna().any():
        raise ValueError("Some evaluation events have no spatial block_id")

    feature_set, estimator_name = load_selected_candidate(OUTPUT_DIR / "fusion_metrics_v1.json")
    feature_set = args.feature_set or feature_set
    estimator_name = args.estimator or estimator_name
    if feature_set not in feature_sets:
        raise ValueError(f"Feature set {feature_set} is unavailable in the current handoff inputs")
    feature_names = feature_sets[feature_set]
    train = frame[frame["fold"] != args.holdout_fold].copy()
    test = frame[frame["fold"] == args.holdout_fold].copy()
    if train.empty or test.empty:
        raise ValueError("Spatial holdout fold produced an empty train or test split")

    trusted_train = pd.to_numeric(train["trusted"], errors="coerce").fillna(0).astype(bool).to_numpy()
    trusted_test = pd.to_numeric(test["trusted"], errors="coerce").fillna(0).astype(bool).to_numpy()
    if not trusted_test.any():
        raise ValueError("Selected holdout fold has no trusted examples")
    model = make_estimator(estimator_name, args.seed, args.trees)
    model.fit(
        train[feature_names], train["label"].astype(str),
        model__sample_weight=trusted_sample_weights(trusted_train, args.trusted_weight),
    )
    predictions = model.predict(test[feature_names])
    y_test = test["label"].astype(str).to_numpy()[trusted_test]
    pred_test = predictions[trusted_test]
    trusted_blocks = test.loc[trusted_test, "block_id"].astype(str).to_numpy()
    trusted_score = score_predictions(y_test, pred_test)
    bootstrap = bootstrap_block_f1(
        y_test, pred_test, trusted_blocks,
        iterations=args.bootstrap_iterations, seed=args.seed,
    )
    output = args.output or OUTPUT_DIR / f"spatial_holdout_fold_{args.holdout_fold}_exploratory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "evaluation_kind": "exploratory_spatial_holdout",
        "independent_final_test": False,
        "caveat": (
            "This fold was supplied with the development package and the candidate was selected using the same five-fold "
            "CV results. Treat this as a spatial holdout diagnostic, not an unbiased final-test estimate."
        ),
        "holdout_fold": args.holdout_fold,
        "candidate": {"feature_set": feature_set, "estimator": estimator_name},
        "rows": {"train": int(len(train)), "holdout": int(len(test)), "trusted_holdout": int(trusted_test.sum())},
        "holdout_spatial_blocks": int(test["block_id"].nunique()),
        "trusted_metrics": trusted_score,
        "trusted_macro_f1_block_bootstrap": bootstrap,
        "classes": CLASSES,
        "output_note": "Use a new reviewed cohort from geographically separate sites for independent final evaluation.",
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"candidate": result["candidate"], "holdout_fold": args.holdout_fold, "trusted_metrics": trusted_score, "bootstrap": bootstrap, "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
