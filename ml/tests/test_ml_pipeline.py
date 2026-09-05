"""
Tests for ML preprocessing pipeline, model loading, and training evaluation output.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import pytest

from preprocessing import engineer_features, prepare_features, TARGET_CLASSES

def test_engineer_features():
    raw_data = pd.DataFrame([{
        "brightness": 350.0,
        "bright_t31": 300.0,
        "daynight": "N"
    }])
    df_eng = engineer_features(raw_data)
    assert "temp_difference" in df_eng.columns
    assert df_eng["temp_difference"].iloc[0] == 50.0
    assert df_eng["is_nighttime"].iloc[0] == 1

def test_prepare_features():
    df = pd.DataFrame([{
        "brightness": 350.0,
        "bright_t31": 300.0,
        "frp": 25.0,
        "confidence": 90.0,
        "dist_to_industrial": 0.5,
        "industrial_count_2km": 2,
        "industrial_count_5km": 5,
        "nearest_facility_type": "refinery",
        "persistence_score": 0.8,
        "detection_count_30d": 12,
        "daynight": "D"
    }])

    X_scaled, scaler, feature_names = prepare_features(df, fit=True)
    assert X_scaled.shape[0] == 1
    assert len(feature_names) == 18 # 10 numerical + 8 dummy variables
    assert scaler is not None

def test_trained_model_artifacts_exist():
    model_dir = "ml/models/best_model"
    assert os.path.exists(os.path.join(model_dir, "model.joblib"))
    assert os.path.exists(os.path.join(model_dir, "scaler.joblib"))
    assert os.path.exists(os.path.join(model_dir, "metadata.json"))

    with open(os.path.join(model_dir, "metadata.json")) as f:
        meta = json.load(f)

    assert "model_name" in meta
    assert meta["metrics"]["macro_f1"] > 0.85
    assert len(meta["supported_classes"]) == len(TARGET_CLASSES)
