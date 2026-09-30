import json
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np

from app.core.config import settings
from app.schemas.schemas import PredictionExplanation, PredictionInput, PredictionOutput


DEFAULT_CLASSES = [
    "agricultural_burning",
    "gas_flare",
    "industrial_fire",
    "mining_activity",
]
DEFAULT_FEATURES = [
    "bright_ti4",
    "bright_ti5",
    "temp_diff",
    "frp",
    "conf_ord",
    "is_night",
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


class ModelInferenceService:
    def __init__(self):
        self.model = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.is_loaded = False
        self.active_model_key = settings.ACTIVE_MODEL
        self.active_model_display = settings.ACTIVE_MODEL
        self.available_models: Dict[str, Dict[str, Any]] = {}
        self.fallback_reason: Optional[str] = None
        self.load_all_models()
        self.load_model()

    @property
    def inference_mode(self) -> str:
        if not self.is_loaded or self.model is None or self.fallback_reason:
            return "heuristic"
        return "trained"

    def load_all_models(self) -> None:
        model_specs = {
            "self_trained_rf3": (
                settings.MODEL_PATH,
                "Self-Trained Random Forest (Tier 2 Round 3)",
                15,
            ),
            "rf_baseline": (settings.RF_BASELINE_PATH, "Random Forest Baseline", 15),
            "xgb_temporal": (
                settings.XGB_TEMPORAL_PATH,
                "XGBoost with Temporal Features",
                26,
            ),
        }

        for key, (model_dir, display_name, feature_count) in model_specs.items():
            model_path = os.path.join(model_dir, "model.joblib")
            metadata_path = os.path.join(model_dir, "metadata.json")
            if not (os.path.isfile(model_path) and os.path.isfile(metadata_path)):
                print(f"[WARNING] Model artifact unavailable for {key}: {model_dir}")
                continue

            try:
                model = joblib.load(model_path)
                with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                    metadata = json.load(metadata_file)
                features = metadata.get("features") or DEFAULT_FEATURES[:feature_count]
                self.available_models[key] = {
                    "model": model,
                    "metadata": metadata,
                    "name": display_name,
                    "features": features,
                    "expected_features": len(features),
                }
                print(f"[OK] Loaded {display_name}")
            except Exception as error:
                print(f"[ERROR] Could not load {display_name}: {type(error).__name__}: {error}")

    def load_model(self) -> None:
        model_info = self.available_models.get(self.active_model_key)
        if model_info is None:
            self.model = None
            self.metadata = None
            self.is_loaded = False
            self.fallback_reason = f"Configured model '{self.active_model_key}' is unavailable."
            print(f"[WARNING] {self.fallback_reason}")
            if settings.REQUIRE_TRAINED_MODEL:
                raise RuntimeError(self.fallback_reason)
            return

        self.model = model_info["model"]
        self.metadata = model_info["metadata"]
        self.active_model_display = model_info["name"]
        self.is_loaded = True
        self.fallback_reason = None
        print(f"[OK] Active model set to: {self.active_model_display}")

    def switch_model(self, model_name: str) -> Dict[str, Any]:
        model_info = self.available_models.get(model_name)
        if model_info is None:
            return {
                "status": "error",
                "message": f"Model '{model_name}' is not available.",
                "available_models": list(self.available_models),
            }

        self.active_model_key = model_name
        self.load_model()
        return {
            "status": "success",
            "model_name": model_name,
            "description": model_info["name"],
            "macro_f1": model_info["metadata"].get("macro_f1", 0.0),
        }

    @staticmethod
    def _feature_values(input_data: PredictionInput) -> Dict[str, Optional[float]]:
        confidence = float(input_data.confidence)

        return {
            "bright_ti4": float(input_data.brightness),
            "bright_ti5": float(input_data.bright_t31),
            "temp_diff": float(input_data.brightness) - float(input_data.bright_t31),
            "frp": float(input_data.frp),
            "conf_ord": (
                input_data.confidence_ordinal
                if input_data.confidence_ordinal is not None
                else 0 if confidence <= 40 else 1 if confidence <= 80 else 2
            ),
            "is_night": int(input_data.daynight == "N"),
            "nearest_osm_distance_km": float(input_data.dist_to_industrial),
            "industrial_count_1km": input_data.industrial_count_1km,
            "industrial_count_2km": float(input_data.industrial_count_2km),
            "industrial_count_5km": float(input_data.industrial_count_5km),
            "power_plant_count_5km": input_data.power_plant_count_5km,
            "quarry_count_5km": input_data.quarry_count_5km,
            "flare_count_5km": input_data.flare_count_5km,
            "petroleum_well_count_5km": input_data.petroleum_well_count_5km,
            "industrial_landuse_nearby": input_data.industrial_landuse_nearby,
        }

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        if not self.is_loaded or self.model is None:
            reason = self.fallback_reason or "No trained model is active."
            if settings.REQUIRE_TRAINED_MODEL:
                raise RuntimeError(reason)
            return self._heuristic_prediction(input_data, reason)

        model_info = self.available_models[self.active_model_key]
        metadata = model_info["metadata"]
        feature_values = self._feature_values(input_data)
        feature_order = model_info["features"]
        available_features = input_data.available_model_features
        missing = [
            name for name in feature_order
            if feature_values.get(name) is None
            or (available_features is not None and name not in available_features)
        ]
        if missing:
            reason = "Prediction input is missing model features: " + ", ".join(missing) + "."
            print(f"[WARNING] {reason}")
            if settings.REQUIRE_TRAINED_MODEL:
                raise ValueError(reason)
            return self._heuristic_prediction(input_data, reason)

        features = np.asarray([[feature_values[name] for name in feature_order]], dtype=float)
        estimator_features = getattr(self.model, "n_features_in_", len(feature_order))
        if estimator_features != features.shape[1]:
            reason = (
                f"Model '{self.active_model_display}' expects {estimator_features} features; "
                f"metadata supplies {features.shape[1]}."
            )
            print(f"[WARNING] {reason}")
            if settings.REQUIRE_TRAINED_MODEL:
                raise RuntimeError(reason)
            return self._heuristic_prediction(input_data, reason)

        classes = [str(value) for value in getattr(self.model, "classes_", metadata.get("classes", DEFAULT_CLASSES))]
        try:
            probabilities = np.asarray(self.model.predict_proba(features)[0], dtype=float)
            if (
                len(probabilities) != len(classes)
                or not np.isfinite(probabilities).all()
                or float(probabilities.sum()) <= 0
            ):
                raise ValueError("Model returned an invalid class probability vector.")
        except Exception as error:
            reason = f"Trained model inference failed: {type(error).__name__}: {error}"
            print(f"[ERROR] {reason}")
            if settings.REQUIRE_TRAINED_MODEL:
                raise RuntimeError(reason) from error
            return self._heuristic_prediction(input_data, reason)

        best_index = int(np.argmax(probabilities))
        predicted_class = classes[best_index]
        confidence = float(probabilities[best_index])
        class_probabilities = {
            name: round(float(probabilities[index]), 4)
            for index, name in enumerate(classes)
        }
        importances = metadata.get("feature_importances", {})
        top_features = dict(sorted(importances.items(), key=lambda pair: pair[1], reverse=True)[:3])
        if not top_features:
            top_features = {"frp": 0.4, "temp_diff": 0.3, "nearest_osm_distance_km": 0.3}

        return PredictionOutput(
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            class_probabilities=class_probabilities,
            model_version=f"{self.active_model_key} ({metadata.get('macro_f1', 0.0):.3f} F1)",
            explanation=PredictionExplanation(
                top_contributing_features=top_features,
                summary=(
                    f"Classified as '{predicted_class}' with {confidence * 100:.1f}% confidence. "
                    f"Model: {metadata.get('model', self.active_model_display)}."
                ),
            ),
            inference_mode="trained",
        )

    def _heuristic_prediction(self, input_data: PredictionInput, reason: Optional[str] = None) -> PredictionOutput:
        if input_data.dist_to_industrial < 0.5 and input_data.industrial_count_5km > 5:
            predicted_class, confidence = "industrial_fire", 0.85
        elif input_data.frp > 100.0 and input_data.dist_to_industrial > 3.0:
            predicted_class, confidence = "agricultural_burning", 0.80
        elif input_data.industrial_count_5km > 3 and input_data.dist_to_industrial < 2.0:
            predicted_class, confidence = "gas_flare", 0.82
        elif input_data.industrial_count_2km > 2:
            predicted_class, confidence = "mining_activity", 0.75
        else:
            predicted_class, confidence = "agricultural_burning", 0.70

        probabilities = {name: (1.0 - confidence) / 3 for name in DEFAULT_CLASSES}
        probabilities[predicted_class] = confidence
        reason = reason or self.fallback_reason or "Trained model unavailable."
        return PredictionOutput(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=probabilities,
            model_version=f"fallback-heuristic: {reason}",
            explanation=PredictionExplanation(
                top_contributing_features={
                    "dist_to_industrial": 0.40,
                    "industrial_count_5km": 0.35,
                    "frp": 0.25,
                },
                summary=f"Heuristic fallback applied. {reason}",
            ),
            inference_mode="heuristic",
            fallback_reason=reason,
        )

    def get_available_models(self) -> Dict[str, Any]:
        models = []
        for key, info in self.available_models.items():
            metadata = info["metadata"]
            models.append({
                "name": key,
                "description": info["name"],
                "macro_f1": metadata.get("macro_f1", 0.0),
                "train_obs": metadata.get("n_train", metadata.get("train_obs", 0)),
                "feature_count": info["expected_features"],
                "compatible_with_prediction_form": bool(info["features"]) and all(
                    name in DEFAULT_FEATURES for name in info["features"]
                ),
                "active": key == self.active_model_key,
            })
        return {
            "available_models": models,
            "active_model": self.active_model_key,
            "total_models": len(models),
        }


model_service = ModelInferenceService()
