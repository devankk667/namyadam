import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from app.core.config import settings
from app.schemas.schemas import PredictionInput, PredictionOutput, PredictionExplanation

class ModelInferenceService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        model_dir = settings.MODEL_PATH
        model_file = os.path.join(model_dir, "model.joblib")
        scaler_file = os.path.join(model_dir, "scaler.joblib")
        metadata_file = os.path.join(model_dir, "metadata.json")

        if os.path.exists(model_file) and os.path.exists(scaler_file) and os.path.exists(metadata_file):
            try:
                self.model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
                self.is_loaded = True
            except Exception as e:
                print(f"Error loading model: {e}")
                self.is_loaded = False
        else:
            self.is_loaded = False

    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        if not self.is_loaded or self.model is None or self.scaler is None:
            # Fallback heuristic prediction if model not loaded
            return self._heuristic_prediction(input_data)

        # Feature construction matching ml/src/preprocessing.py
        temp_diff = input_data.brightness - input_data.bright_t31

        feature_dict = {
            "brightness": input_data.brightness,
            "bright_t31": input_data.bright_t31,
            "temp_difference": temp_diff,
            "frp": input_data.frp,
            "confidence": input_data.confidence,
            "dist_to_industrial": input_data.dist_to_industrial,
            "industrial_count_2km": input_data.industrial_count_2km,
            "industrial_count_5km": input_data.industrial_count_5km,
            "persistence_score": input_data.persistence_score,
            "detection_count_30d": input_data.detection_count_30d,
        }

        facility_types = ["refinery", "flare_stack", "gas_terminal", "chemical", "power_plant", "steel_works", "factory", "none"]
        for ftype in facility_types:
            feature_dict[f"facility_{ftype}"] = 1.0 if input_data.nearest_facility_type == ftype else 0.0

        feature_names = self.metadata.get("feature_names", list(feature_dict.keys()))
        input_vector = [feature_dict.get(fn, 0.0) for fn in feature_names]

        X_input = np.array([input_vector])
        X_scaled = self.scaler.transform(X_input)

        probs = self.model.predict_proba(X_scaled)[0]
        classes = self.metadata.get("supported_classes", [
            "industrial_thermal_source", "industrial_fire", "wildfire", "agricultural_burning", "other_thermal_anomaly"
        ])

        class_probs = {cls_name: float(probs[i]) for i, cls_name in enumerate(classes)}
        best_idx = np.argmax(probs)
        predicted_class = classes[best_idx]
        conf_score = float(probs[best_idx])

        # Explanation generation
        top_importances = self.metadata.get("feature_importances", {})
        top_3 = dict(sorted(top_importances.items(), key=lambda x: x[1], reverse=True)[:3])

        explanation_summary = (
            f"Classified as '{predicted_class}' with {conf_score*100:.1f}% confidence. "
            f"Primary factors: proximity to industrial entity ({input_data.dist_to_industrial:.2f} km), "
            f"historical persistence ({input_data.persistence_score:.2f}), and FRP ({input_data.frp:.1f} MW)."
        )

        return PredictionOutput(
            predicted_class=predicted_class,
            confidence=round(conf_score, 4),
            class_probabilities={k: round(v, 4) for k, v in class_probs.items()},
            model_version=self.metadata.get("version", "1.0.0"),
            explanation=PredictionExplanation(
                top_contributing_features=top_3,
                summary=explanation_summary
            )
        )

    def _heuristic_prediction(self, input_data: PredictionInput) -> PredictionOutput:
        if input_data.dist_to_industrial < 1.0 and input_data.persistence_score > 0.6:
            pred_class = "industrial_thermal_source"
            conf = 0.91
        elif input_data.dist_to_industrial < 1.5 and input_data.frp > 50.0:
            pred_class = "industrial_fire"
            conf = 0.88
        elif input_data.frp > 80.0 and input_data.dist_to_industrial > 5.0:
            pred_class = "wildfire"
            conf = 0.93
        elif input_data.frp < 30.0 and input_data.dist_to_industrial > 3.0:
            pred_class = "agricultural_burning"
            conf = 0.82
        else:
            pred_class = "other_thermal_anomaly"
            conf = 0.75

        return PredictionOutput(
            predicted_class=pred_class,
            confidence=conf,
            class_probabilities={
                "industrial_thermal_source": 0.2,
                "industrial_fire": 0.2,
                "wildfire": 0.2,
                "agricultural_burning": 0.2,
                "other_thermal_anomaly": 0.2,
                pred_class: conf
            },
            model_version="1.0.0-fallback",
            explanation=PredictionExplanation(
                top_contributing_features={"dist_to_industrial": 0.4, "persistence_score": 0.3, "frp": 0.3},
                summary="Heuristic decision engine rules applied."
            )
        )

model_service = ModelInferenceService()
