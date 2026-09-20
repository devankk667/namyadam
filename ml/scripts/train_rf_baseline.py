"""
RF baseline on HIGH/MEDIUM-confidence event labels only (unknown excluded).

Obs <- events via firms_obs_event_map; leakage-safe GroupShuffleSplit on
0.1-degree spatial blocks (test blocks unseen in train). Exits gracefully
(no fake metrics) when labels are insufficient — e.g. before the OSM cache
is full. Saves model + metrics to ml/models/rf_baseline/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

ENRICHED = Path("data/processed/firms_osm_enriched.parquet")
EVENTS = Path("data/processed/firms_thermal_events.parquet")
LABELS = Path("data/processed/firms_event_labels.parquet")
MAP = Path("data/processed/firms_obs_event_map.parquet")
OUTDIR = Path("ml/models/rf_baseline")

FEATURES = ["bright_ti4", "bright_ti5", "temp_diff", "frp",
            "conf_ord", "is_night", "nearest_osm_distance_km",
            "industrial_count_1km", "industrial_count_2km", "industrial_count_5km",
            "power_plant_count_5km", "quarry_count_5km", "flare_count_5km",
            "petroleum_well_count_5km", "industrial_landuse_nearby"]
CONF_ORD = {"l": 0, "n": 1, "h": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", default="medium", choices=["high", "medium"])
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keep_conf = ["high"] if args.min_conf == "high" else ["high", "medium"]
    labels = pd.read_parquet(LABELS)
    events = pd.read_parquet(EVENTS, columns=["event_id", "centroid_lat", "centroid_lon"])
    good = labels[(labels["label"] != "unknown") & (labels["label_confidence"].isin(keep_conf))]
    print(f"labeled events (>{args.min_conf}): {len(good)} of {len(labels)}", flush=True)
    print(good.groupby(["label", "label_confidence"]).size().to_string(), flush=True)

    m = pd.read_parquet(MAP)
    obs_ids = m[m["event_id"].isin(set(good["event_id"]))]["obs_id"].to_numpy()
    df = pd.read_parquet(ENRICHED)
    sub = df.iloc[obs_ids].copy()
    sub["temp_diff"] = (pd.to_numeric(sub["bright_ti4"], errors="coerce")
                        - pd.to_numeric(sub["bright_ti5"], errors="coerce"))
    sub["conf_ord"] = sub["confidence"].map(CONF_ORD).fillna(1).astype(int)
    sub["is_night"] = (sub["daynight"] == "N").astype(int)
    lab = good.set_index("event_id")["label"]
    if "event_id" not in sub.columns:  # enriched lacks event_id -> join via map
        ev_of = m.set_index("obs_id")["event_id"]
        sub["event_id"] = ev_of.loc[sub.index].to_numpy()
    sub["y"] = sub["event_id"].map(lab)
    sub = sub.dropna(subset=["y"])
    evc = events.set_index("event_id")
    sub["block"] = ((evc.loc[sub["event_id"], "centroid_lat"].to_numpy() * 10).astype(int).astype(str)
                    + "_" + (evc.loc[sub["event_id"], "centroid_lon"].to_numpy() * 10).astype(int).astype(str))

    X = sub[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    y = sub["y"].to_numpy()
    groups = sub["block"].to_numpy()
    print(f"trainable obs: {len(y)}, classes: {sorted(set(y))}", flush=True)
    if len(y) < 200 or len(set(y)) < 2:
        print("INSUFFICIENT high/medium labels (need >=200 obs, >=2 classes). "
              "Fill the OSM cache, re-enrich, re-label, then re-run. No model written.", flush=True)
        return

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr, te = next(gss.split(X, y, groups=groups))
    _, te_counts = np.unique(y[te], return_counts=True)
    if len(set(y[te])) < 2 or te_counts.min() < 20:
        print(f"INSUFFICIENT label diversity: test split has classes {sorted(set(y[te]))} "
              f"with supports {sorted(te_counts)}. Need >=2 classes x >=20 obs in test. "
              "Fill the OSM cache, re-enrich, re-label, then re-run. No model written.", flush=True)
        return
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=args.seed)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    macro_f1 = f1_score(y[te], pred, average="macro")
    print(classification_report(y[te], pred), flush=True)
    print(f"macro-F1={macro_f1:.4f} train_blocks={len(set(groups[tr]))} test_blocks={len(set(groups[te]))}", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUTDIR / "model.joblib")
    (OUTDIR / "metadata.json").write_text(json.dumps({
        "model": "RandomForestClassifier(300, balanced_subsample)",
        "features": FEATURES, "classes": sorted(set(y)),
        "macro_f1": macro_f1, "n_train": int(len(tr)), "n_test": int(len(te)),
        "split": "GroupShuffleSplit on 0.1-deg spatial blocks",
        "label_confidence": keep_conf,
    }, indent=2))
    print(f"saved -> {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
