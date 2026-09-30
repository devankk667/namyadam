# -*- coding: utf-8 -*-
"""
Tier 2: Semi-Supervised Self-Training Pipeline
================================================
Strategy:
  Round 0 : Train on HIGH/MEDIUM seed labels only (300 events / 8,765 obs).
  Round 1+ : Predict on ALL 1.89M FIRMS observations (minus already-labeled).
             Accept predictions where max-class probability >= THRESHOLD as
             pseudo-labels.  Merge pseudo-labeled obs with seed obs.  Retrain.
  Repeat for N_ROUNDS iterations.

Key safeguards:
  - Seed test set (spatial GroupShuffleSplit) is held out across ALL rounds
    so evaluation is always on the same clean gold-standard test data.
  - Pseudo-labels are never added to the test set.
  - Per-class pseudo-label cap (MAX_PSEUDO_PER_CLASS) prevents agricultural_burning
    from overwhelming the training set.
  - Threshold starts at 0.95 and decays 0.02 each round.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

ENRICHED = Path("data/processed/firms_osm_enriched.parquet")
TEMPORAL = Path("data/processed/firms_temporal.parquet")
EVENTS   = Path("data/processed/firms_thermal_events.parquet")
LABELS   = Path("data/processed/firms_event_labels.parquet")
MAP      = Path("data/processed/firms_obs_event_map.parquet")
LOG_JSON = Path("eda/outputs/self_train_log.json")

FEATURES = [
    "bright_ti4","bright_ti5","temp_diff","frp","conf_ord","is_night",
    "nearest_osm_distance_km","industrial_count_1km","industrial_count_2km",
    "industrial_count_5km","power_plant_count_5km","quarry_count_5km",
    "flare_count_5km","petroleum_well_count_5km","industrial_landuse_nearby",
]
CONF_ORD = {"l":0,"n":1,"h":2}
SEED, TEST_SIZE = 42, 0.25
MAX_PSEUDO_PER_CLASS = 30_000


def load_all():
    labels = pd.read_parquet(LABELS)
    good   = labels[(labels["label"] != "unknown") & (labels["label_confidence"].isin(["high","medium"]))]
    print(f"Seed labeled events: {len(good)}", flush=True)

    m = pd.read_parquet(MAP)
    seed_event_ids = set(good["event_id"])
    seed_obs_ids   = set(m[m["event_id"].isin(seed_event_ids)]["obs_id"])

    print("Loading 1.89M enriched observations ...", flush=True)
    df = pd.read_parquet(ENRICHED)
    df["temp_diff"] = pd.to_numeric(df["bright_ti4"],errors="coerce") - pd.to_numeric(df["bright_ti5"],errors="coerce")
    df["conf_ord"]  = df["confidence"].map(CONF_ORD).fillna(1).astype(int)
    df["is_night"]  = (df["daynight"]=="N").astype(int)

    if TEMPORAL.exists():
        tmp = pd.read_parquet(TEMPORAL).set_index("obs_id")
        df  = df.join(tmp[[c for c in tmp.columns if c not in df.columns]])

    ev_of = m.set_index("obs_id")["event_id"]
    df["event_id"] = ev_of.reindex(df.index)

    lab_map  = good.set_index("event_id")["label"]
    df["y_seed"] = df["event_id"].map(lab_map)

    evc = pd.read_parquet(EVENTS).set_index("event_id")
    valid_ev = df["event_id"].dropna()
    valid_ev = valid_ev[valid_ev.isin(evc.index)]
    df.loc[valid_ev.index, "block"] = (
        (evc.loc[valid_ev,"centroid_lat"].values*10).astype(int).astype(str) + "_" +
        (evc.loc[valid_ev,"centroid_lon"].values*10).astype(int).astype(str)
    )
    df["block"] = df["block"].fillna("unk")

    X       = df[FEATURES].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(dtype=float)
    is_seed = df.index.isin(seed_obs_ids) & df["y_seed"].notna()

    seed_pos = np.where(is_seed)[0]
    y_all_seed = df["y_seed"].iloc[seed_pos].to_numpy()
    g_all_seed = df["block"].iloc[seed_pos].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_rel, te_rel = next(gss.split(seed_pos, y_all_seed, groups=g_all_seed))

    train_seed_pos = seed_pos[tr_rel]
    test_pos       = seed_pos[te_rel]
    y_train_seed   = y_all_seed[tr_rel]
    y_test         = y_all_seed[te_rel]

    print(f"Seed train obs: {len(train_seed_pos)} | test obs: {len(test_pos)}", flush=True)
    return df, X, train_seed_pos, y_train_seed, test_pos, y_test


def train_clf(X_tr, y_tr):
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=SEED)
    clf.fit(X_tr, y_tr)
    return clf


def pseudo_label(clf, X, exclude_pos, threshold, classes, rng):
    excl = set(exclude_pos.tolist())
    cands = np.array([i for i in range(len(X)) if i not in excl])
    proba = clf.predict_proba(X[cands])
    max_p = proba.max(axis=1)
    pred  = np.array(classes)[proba.argmax(axis=1)]
    mask  = max_p >= threshold
    acc_pos, acc_cls = cands[mask], pred[mask]

    final_pos, final_cls = [], []
    for c in classes:
        idx = np.where(acc_cls == c)[0]
        if len(idx) > MAX_PSEUDO_PER_CLASS:
            idx = rng.choice(idx, MAX_PSEUDO_PER_CLASS, replace=False)
        final_pos.extend(acc_pos[idx].tolist())
        final_cls.extend([c]*len(idx))
    return np.array(final_pos), np.array(final_cls)


def evaluate(clf, X, test_pos, y_test, classes, rnd):
    pred     = clf.predict(X[test_pos])
    macro_f1 = f1_score(y_test, pred, average="macro", labels=classes, zero_division=0)
    print(f"\n=== Round {rnd} Gold Test Evaluation ===", flush=True)
    print(classification_report(y_test, pred, labels=classes, zero_division=0), flush=True)
    print(f"macro-F1 = {macro_f1:.4f}", flush=True)
    return macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds",    type=int,   default=3)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--decay",     type=float, default=0.02)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    df, X, train_seed_pos, y_train_seed, test_pos, y_test = load_all()
    classes = sorted(set(y_train_seed))
    test_excl = set(test_pos.tolist())

    # Round 0: seed-only
    print(f"\n=== Round 0 - Seed training ({len(train_seed_pos)} obs) ===", flush=True)
    clf = train_clf(X[train_seed_pos], y_train_seed)
    m0  = evaluate(clf, X, test_pos, y_test, classes, rnd=0)
    Path("ml/models/self_trained_rf0").mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, "ml/models/self_trained_rf0/model.joblib")

    log = {"rounds":[{"round":0,"train_obs":int(len(train_seed_pos)),"pseudo_obs":0,
                       "threshold":None,"macro_f1":round(m0,4)}]}

    all_pos = train_seed_pos.copy()
    all_y   = y_train_seed.copy()
    thresh  = args.threshold

    for rnd in range(1, args.rounds+1):
        print(f"\n=== Round {rnd} - Pseudo-labeling (threshold={thresh:.3f}) ===", flush=True)
        excl_all = np.concatenate([all_pos, test_pos])
        new_pos, new_cls = pseudo_label(clf, X, excl_all, thresh, classes, rng)

        dist = {c:int((new_cls==c).sum()) for c in classes}
        print(f"  Pseudo-labeled: {len(new_pos):,} obs  {dist}", flush=True)

        if len(new_pos) == 0:
            print("  No new pseudo-labels. Stopping.", flush=True)
            break

        merged_pos = np.concatenate([all_pos, new_pos])
        merged_y   = np.concatenate([all_y,   new_cls])

        print(f"  Retraining on {len(merged_pos):,} total obs ...", flush=True)
        clf = train_clf(X[merged_pos], merged_y)
        mf1 = evaluate(clf, X, test_pos, y_test, classes, rnd=rnd)

        out = Path(f"ml/models/self_trained_rf{rnd}")
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, out/"model.joblib")
        (out/"metadata.json").write_text(json.dumps({
            "model":f"SelfTrainedRF_round{rnd}","features":FEATURES,"classes":classes,
            "macro_f1":round(mf1,4),"train_obs":int(len(merged_pos)),
            "pseudo_obs":int(len(new_pos)),"threshold":thresh
        },indent=2))

        log["rounds"].append({"round":rnd,"train_obs":int(len(merged_pos)),
                               "pseudo_obs":int(len(new_pos)),"threshold":thresh,
                               "macro_f1":round(mf1,4)})
        all_pos = merged_pos
        all_y   = merged_y
        thresh  = max(0.70, thresh - args.decay)

    LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    LOG_JSON.write_text(json.dumps(log, indent=2))
    print(f"\nDone. Log -> {LOG_JSON}", flush=True)


if __name__ == "__main__":
    main()
