"""
Predict classification labels for the ENTIRE 1.1M NASA FIRMS observations dataset
using trained Random Forest & XGBoost models.

Output:
  data/processed/firms_1.1M_predictions.parquet
  eda/outputs/full_1.1M_classification_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from firms_ids import ensure_observation_ids

ENRICHED = Path("data/processed/firms_osm_enriched.parquet")
TEMPORAL = Path("data/processed/firms_temporal.parquet")
MAP = Path("data/processed/firms_obs_event_map.parquet")
RF_MODEL_DIR = Path("ml/models/rf_baseline")
XGB_MODEL_DIR = Path("ml/models/xgb_temporal")
OUT_PARQUET = Path("data/processed/firms_1.1M_predictions.parquet")
SUMMARY_JSON = Path("eda/outputs/full_1.1M_classification_summary.json")

CONF_ORD = {"l": 0, "n": 1, "h": 2}

def main():
    if not ENRICHED.exists():
        print(f"Error: {ENRICHED} not found.", flush=True)
        return

    print("Loading enriched observations (1.1M)...", flush=True)
    df = ensure_observation_ids(pd.read_parquet(ENRICHED))
    n_total = len(df)
    print(f"Total observations loaded: {n_total}", flush=True)

    rf_meta = json.loads((RF_MODEL_DIR / "metadata.json").read_text())
    rf_features = rf_meta["features"]

    # Preprocess base features
    df["temp_diff"] = (pd.to_numeric(df["bright_ti4"], errors="coerce")
                        - pd.to_numeric(df["bright_ti5"], errors="coerce"))
    df["conf_ord"] = df["confidence"].map(CONF_ORD).fillna(1).astype(int)
    df["is_night"] = (df["daynight"] == "N").astype(int)

    if TEMPORAL.exists():
        print("Joining temporal features...", flush=True)
        tmp = pd.read_parquet(TEMPORAL)
        tmp["obs_id"] = tmp["obs_id"].astype(str)
        extra = [column for column in tmp.columns if column not in df.columns]
        df = df.merge(tmp[["obs_id", *extra]], on="obs_id", how="left", validate="one_to_one", sort=False)

    X_rf = df[rf_features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()

    print("Loading Random Forest model & predicting on entire dataset...", flush=True)
    rf_clf = joblib.load(RF_MODEL_DIR / "model.joblib")
    rf_preds = rf_clf.predict(X_rf)
    rf_probs = rf_clf.predict_proba(X_rf).max(axis=1)

    df["rf_predicted_class"] = rf_preds
    df["rf_confidence"] = np.round(rf_probs, 4)

    if (XGB_MODEL_DIR / "model.joblib").exists():
        xgb_meta = json.loads((XGB_MODEL_DIR / "metadata.json").read_text())
        xgb_features = xgb_meta["features"]
        X_xgb = df[xgb_features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        print("Loading XGBoost model & predicting on entire dataset...", flush=True)
        xgb_clf = joblib.load(XGB_MODEL_DIR / "model.joblib")
        xgb_preds = xgb_clf.predict(X_xgb)
        xgb_probs = xgb_clf.predict_proba(X_xgb).max(axis=1)
        df["xgb_predicted_class"] = xgb_preds
        df["xgb_confidence"] = np.round(xgb_probs, 4)

    print(f"Saving predictions to {OUT_PARQUET}...", flush=True)
    df.to_parquet(OUT_PARQUET, index=False)

    rf_counts = df["rf_predicted_class"].value_counts().to_dict()
    print("RF Prediction Distribution Across 1.1M Records:", rf_counts, flush=True)

    summary = {
        "total_observations": n_total,
        "rf_class_distribution": {k: int(v) for k, v in rf_counts.items()},
        "output_file": str(OUT_PARQUET)
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {SUMMARY_JSON}", flush=True)

if __name__ == "__main__":
    main()
