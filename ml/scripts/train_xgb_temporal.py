"""
XGBoost + temporal/persistence features, same protocol as the RF baseline:
high/medium labels only, GroupShuffleSplit on 0.1-deg spatial blocks
(seed 42, test 0.25). Saves model + metrics to ml/models/xgb_temporal/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

ENRICHED = Path("data/processed/firms_osm_enriched.parquet")
TEMPORAL = Path("data/processed/firms_temporal.parquet")
EVENTS = Path("data/processed/firms_thermal_events.parquet")
LABELS = Path("data/processed/firms_event_labels.parquet")
MAP = Path("data/processed/firms_obs_event_map.parquet")
OUTDIR = Path("ml/models/xgb_temporal")

BASE_FEATURES = ["bright_ti4", "bright_ti5", "temp_diff", "frp",
                 "conf_ord", "is_night", "nearest_osm_distance_km",
                 "industrial_count_1km", "industrial_count_2km", "industrial_count_5km",
                 "power_plant_count_5km", "quarry_count_5km", "flare_count_5km",
                 "petroleum_well_count_5km", "industrial_landuse_nearby"]
TEMP_FEATURES = ["prev_7d", "prev_14d", "prev_30d", "time_since_prev_days",
                 "days_since_first_seen", "cell_prior_count", "recurrence_per_day",
                 "frp_prior_mean", "frp_prior_std", "frp_trend_slope", "event_prev_count"]
FEATURES = BASE_FEATURES + TEMP_FEATURES
CONF_ORD = {"l": 0, "n": 1, "h": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", default="medium", choices=["high", "medium", "low"])
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.min_conf == "high":
        keep_conf = ["high"]
    elif args.min_conf == "medium":
        keep_conf = ["high", "medium"]
    else:  # low — include everything non-unknown
        keep_conf = ["high", "medium", "low"]
    labels = pd.read_parquet(LABELS)
    good = labels[(labels["label"] != "unknown") & (labels["label_confidence"].isin(keep_conf))]
    print(f"labeled events (>{args.min_conf}): {len(good)}", flush=True)

    m = pd.read_parquet(MAP)
    obs_ids = m[m["event_id"].isin(set(good["event_id"]))]["obs_id"].to_numpy()
    df = pd.read_parquet(ENRICHED)
    sub = df.iloc[obs_ids].copy()
    sub["temp_diff"] = (pd.to_numeric(sub["bright_ti4"], errors="coerce")
                        - pd.to_numeric(sub["bright_ti5"], errors="coerce"))
    sub["conf_ord"] = sub["confidence"].map(CONF_ORD).fillna(1).astype(int)
    sub["is_night"] = (sub["daynight"] == "N").astype(int)
    tmp = pd.read_parquet(TEMPORAL).set_index("obs_id")
    sub = sub.join(tmp[TEMP_FEATURES])  # both indexed by obs_id
    lab = good.set_index("event_id")["label"]
    ev_of = m.set_index("obs_id")["event_id"]
    sub["event_id"] = ev_of.loc[sub.index].to_numpy()
    sub["y"] = sub["event_id"].map(lab)
    sub = sub.dropna(subset=["y"])
    events = pd.read_parquet(EVENTS, columns=["event_id", "centroid_lat", "centroid_lon"])
    evc = events.set_index("event_id")
    sub["block"] = ((evc.loc[sub["event_id"], "centroid_lat"].to_numpy() * 10).astype(int).astype(str)
                    + "_" + (evc.loc[sub["event_id"], "centroid_lon"].to_numpy() * 10).astype(int).astype(str))

    X = sub[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    y = sub["y"].to_numpy()
    groups = sub["block"].to_numpy()
    print(f"trainable obs: {len(y)}, classes: {sorted(set(y))}", flush=True)
    if len(y) < 200 or len(set(y)) < 2:
        print("INSUFFICIENT labels. No model written.", flush=True)
        return

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr, te = next(gss.split(X, y, groups=groups))
    _, te_counts = np.unique(y[te], return_counts=True)
    if len(set(y[te])) < 2 or te_counts.min() < 20:
        print(f"INSUFFICIENT test diversity {sorted(te_counts)}. No model written.", flush=True)
        return

    classes = sorted(set(y))
    y_idx = np.array([classes.index(v) for v in y])
    clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        eval_metric="mlogloss", n_jobs=-1, random_state=args.seed)
    clf.fit(X[tr], y_idx[tr])
    pred = np.array(classes)[clf.predict(X[te]).astype(int)]
    macro_f1 = f1_score(y[te], pred, average="macro")
    print(classification_report(y[te], pred), flush=True)
    print(f"macro-F1={macro_f1:.4f} train_blocks={len(set(groups[tr]))} test_blocks={len(set(groups[te]))}", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUTDIR / "model.joblib")
    (OUTDIR / "metadata.json").write_text(json.dumps({
        "model": "XGBClassifier(400, d6, lr0.05, subsample0.8)",
        "features": FEATURES, "classes": sorted(set(y)),
        "macro_f1": macro_f1, "n_train": int(len(tr)), "n_test": int(len(te)),
        "split": "GroupShuffleSplit on 0.1-deg spatial blocks (seed 42)",
        "label_confidence": keep_conf,
    }, indent=2))
    print(f"saved -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
