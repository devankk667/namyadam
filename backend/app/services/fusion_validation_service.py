"""Expose spatial out-of-fold image-fusion predictions for known events.

These are validation predictions, not live inference from the all-label model
artifact. The distinction matters because the packaged image features only cover
labeled events and no scene-to-feature inference pipeline is part of the API yet.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from app.core.config import settings

FUSION_CLASSES = [
    "agricultural_burning",
    "gas_flare",
    "industrial_fire",
    "mining_activity",
]
FUSION_ABLATIONS = [
    "A_tabular",
    "B_tabular_spectral",
    "C_tabular_spectral_frozen",
    "D_tabular_spectral_cnn",
]


class FusionValidationService:
    def __init__(self) -> None:
        self._predictions: Dict[str, Dict[str, Any]] = {}
        self._status: Dict[str, Any] = {
            "enabled": False,
            "prediction_kind": "spatial_cv_oof",
            "reason": "Fusion validation predictions have not been generated yet.",
            "feature_set": None,
            "estimator": None,
            "events": 0,
            "image_covered_events": 0,
            "trusted_macro_f1": None,
            "trusted_macro_f1_no_flare": None,
            "ablation_comparison": [],
        }
        self._load()

    def _load(self) -> None:
        predictions_path = settings.FUSION_OOF_PREDICTIONS_PATH
        metrics_path = settings.FUSION_METRICS_PATH
        if not (predictions_path and metrics_path and Path(predictions_path).is_file()
                and Path(metrics_path).is_file()):
            self._status["reason"] = (
                "Train the image fusion experiment to create ml/models/fusion_v1/ "
                "before enabling event-level fusion results."
            )
            return

        try:
            with open(metrics_path, "r", encoding="utf-8") as metrics_file:
                report = json.load(metrics_file)
            selected = report.get("best_image_fusion_candidate")
            if not selected:
                self._status["reason"] = "No tabular-plus-image fusion candidate was selected in the training report."
                return

            feature_set = selected["feature_set"]
            estimator = selected["estimator"]
            candidate_key = f"{feature_set}/{estimator}"
            candidate_metrics = report.get("candidates", {}).get(candidate_key, {}).get("metrics", {})
            comparisons = []
            for key, candidate in report.get("candidates", {}).items():
                try:
                    candidate_feature_set, candidate_estimator = key.rsplit("/", 1)
                except ValueError:
                    continue
                if candidate_feature_set not in FUSION_ABLATIONS:
                    continue
                trusted = candidate.get("metrics", {}).get("trusted_oof", {})
                comparisons.append({
                    "feature_set": candidate_feature_set,
                    "estimator": candidate_estimator,
                    "trusted_macro_f1": trusted.get("macro_f1"),
                    "trusted_macro_f1_no_flare": trusted.get("macro_f1_no_flare"),
                    "trusted_events": trusted.get("n"),
                })
            comparisons.sort(key=lambda item: (
                FUSION_ABLATIONS.index(item["feature_set"]),
                item["estimator"],
            ))
            columns = [
                "event_id", "feature_set", "estimator", "fold", "has_image",
                "predicted_class", "confidence",
                *[f"p_{class_name}" for class_name in FUSION_CLASSES],
            ]
            frame = pd.read_parquet(
                predictions_path,
                columns=columns,
                filters=[("feature_set", "==", feature_set), ("estimator", "==", estimator)],
            )
            if frame["event_id"].duplicated().any():
                raise ValueError("OOF fusion predictions must contain one row per event_id and selected candidate")

            image_rows = frame.loc[pd.to_numeric(frame["has_image"], errors="coerce").eq(1)]
            for row in image_rows.to_dict(orient="records"):
                event_id = str(row["event_id"])
                probabilities = {
                    class_name: float(row[f"p_{class_name}"])
                    for class_name in FUSION_CLASSES
                }
                confidence = float(row["confidence"])
                probability_total = sum(probabilities.values())
                if (
                    not all(math.isfinite(value) for value in probabilities.values())
                    or not math.isfinite(confidence)
                    or probability_total <= 0
                ):
                    continue
                probabilities = {name: value / probability_total for name, value in probabilities.items()}
                self._predictions[event_id] = {
                    "predicted_class": str(row["predicted_class"]),
                    "confidence": confidence,
                    "class_probabilities": probabilities,
                    "prediction_kind": "spatial_cv_oof",
                    "fold": int(row["fold"]),
                    "feature_set": feature_set,
                    "estimator": estimator,
                    "image_features_available": True,
                }

            trusted_metrics = candidate_metrics.get("trusted_oof", {})
            self._status = {
                "enabled": bool(self._predictions),
                "prediction_kind": "spatial_cv_oof",
                "reason": None if self._predictions else "No image-bearing OOF event predictions were found.",
                "feature_set": feature_set,
                "estimator": estimator,
                "events": int(len(frame)),
                "image_covered_events": int(len(self._predictions)),
                "trusted_macro_f1": trusted_metrics.get("macro_f1"),
                "trusted_macro_f1_no_flare": trusted_metrics.get("macro_f1_no_flare"),
                "split": report.get("dataset", {}).get("evaluation"),
                "ablation_comparison": comparisons,
            }
        except Exception as error:
            self._predictions.clear()
            self._status["reason"] = f"Could not load fusion validation predictions: {type(error).__name__}: {error}"
            print(f"[WARNING] {self._status['reason']}")

    def get_by_event_id(self, event_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not event_id:
            return None
        return self._predictions.get(str(event_id))

    def get_status(self) -> Dict[str, Any]:
        return dict(self._status)


fusion_validation_service = FusionValidationService()
