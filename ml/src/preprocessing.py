"""
Preprocessing and feature transformation utilities for thermal detections.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

NUMERICAL_FEATURES = [
    "brightness",
    "bright_t31",
    "temp_difference",
    "frp",
    "confidence",
    "dist_to_industrial",
    "industrial_count_2km",
    "industrial_count_5km",
    "persistence_score",
    "detection_count_30d"
]

FACILITY_TYPES = ["refinery", "flare_stack", "gas_terminal", "chemical", "power_plant", "steel_works", "factory", "none"]
TARGET_CLASSES = [
    "industrial_thermal_source",
    "industrial_fire",
    "wildfire",
    "agricultural_burning",
    "other_thermal_anomaly"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes derived physics-based and geospatial features."""
    df = df.copy()
    if "temp_difference" not in df.columns:
        df["temp_difference"] = df["brightness"] - df["bright_t31"]
    if "is_nighttime" not in df.columns and "daynight" in df.columns:
        df["is_nighttime"] = (df["daynight"] == "N").astype(int)
    return df

def prepare_features(df: pd.DataFrame, scaler=None, fit=False):
    """
    Extracts numerical feature array and returns transformed X and target y.
    """
    df_eng = engineer_features(df)
    X_num = df_eng[NUMERICAL_FEATURES].copy()

    # Encode categorical nearest_facility_type as dummy variables or mapped ints
    for ftype in FACILITY_TYPES:
        X_num[f"facility_{ftype}"] = (df_eng["nearest_facility_type"] == ftype).astype(float)

    feature_names = list(X_num.columns)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)
    else:
        if fit:
            X_scaled = scaler.fit_transform(X_num)
        else:
            X_scaled = scaler.transform(X_num)

    return X_scaled, scaler, feature_names
