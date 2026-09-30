"""
Evaluate the saved RF baseline on its deterministic test split and SAVE everything:

  ml/models/rf_baseline/evaluation/
    evaluation.json            (config, support, per-class P/R/F1, macro-F1)
    confusion_matrix.csv       (counts) + confusion_matrix_norm.csv (recall-normalized)
    confusion_matrix.png
    feature_importance.csv     (rf impurity + test permutation importance)
    feature_importance.png
    events_obs_per_class.csv   (train/test events + obs per class)
    errors_by_geography.csv/.png   (test error rate per 1-deg cell + map)
    errors_by_persistence.csv/.png (error rate by event n_days bins)
    errors_by_osm_type.csv/.png    (error rate by nearest_osm_type)
    errors_by_frp.csv/.png + errors_by_brightness.csv/.png

Split is rebuilt EXACTLY as in train_rf_baseline.py (same seed/order) and the
saved model.joblib is loaded — no retraining.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from train_rf_baseline import CONF_ORD  # noqa: E402
from firms_ids import ensure_observation_ids  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support)  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

ENRICHED = Path("data/processed/firms_osm_enriched.parquet")
TEMPORAL = Path("data/processed/firms_temporal.parquet")
EVENTS = Path("data/processed/firms_thermal_events.parquet")
LABELS = Path("data/processed/firms_event_labels.parquet")
MAP = Path("data/processed/firms_obs_event_map.parquet")
SEED, TEST_SIZE = 42, 0.25


def savefig(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeldir", default="ml/models/rf_baseline")
    args = ap.parse_args()
    global OUT
    MODELDIR = Path(args.modeldir)
    OUT = MODELDIR / "evaluation"

    meta = json.loads((MODELDIR / "metadata.json").read_text())
    FEATURES = meta["features"]
    keep_conf = meta.get("label_confidence", ["high", "medium"])

    labels = pd.read_parquet(LABELS)
    good = labels[(labels["label"] != "unknown") & (labels["label_confidence"].isin(keep_conf))]
    m = pd.read_parquet(MAP)
    m["obs_id"] = m["obs_id"].astype(str)
    df = ensure_observation_ids(pd.read_parquet(ENRICHED))
    labeled_map = m[m["event_id"].isin(set(good["event_id"]))]
    sub = df.merge(labeled_map, on="obs_id", how="inner", validate="one_to_one")
    sub["temp_diff"] = (pd.to_numeric(sub["bright_ti4"], errors="coerce")
                        - pd.to_numeric(sub["bright_ti5"], errors="coerce"))
    sub["conf_ord"] = sub["confidence"].map(CONF_ORD).fillna(1).astype(int)
    sub["is_night"] = (sub["daynight"] == "N").astype(int)
    if TEMPORAL.exists() and any(f not in sub.columns for f in FEATURES):
        tmp = pd.read_parquet(TEMPORAL)
        tmp["obs_id"] = tmp["obs_id"].astype(str)
        extra = [f for f in FEATURES if f not in sub.columns and f in tmp.columns]
        sub = sub.merge(tmp[["obs_id", *extra]], on="obs_id", how="left", validate="one_to_one")
    sub["y"] = sub["event_id"].map(good.set_index("event_id")["label"])
    sub = sub.dropna(subset=["y"]).reset_index(drop=True)

    evc = pd.read_parquet(EVENTS).set_index("event_id")
    sub["block"] = ((evc.loc[sub["event_id"], "centroid_lat"].to_numpy() * 10).astype(int).astype(str)
                    + "_" + (evc.loc[sub["event_id"], "centroid_lon"].to_numpy() * 10).astype(int).astype(str))
    sub["ev_lat"] = evc.loc[sub["event_id"], "centroid_lat"].to_numpy()
    sub["ev_lon"] = evc.loc[sub["event_id"], "centroid_lon"].to_numpy()
    sub["ev_ndays"] = evc.loc[sub["event_id"], "n_days"].to_numpy()
    sub["ev_duration"] = evc.loc[sub["event_id"], "duration_days"].to_numpy()

    X = sub[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    y = sub["y"].to_numpy()
    groups = sub["block"].to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                    random_state=SEED).split(X, y, groups=groups))

    clf = joblib.load(MODELDIR / "model.joblib")
    pred_raw = clf.predict(X[te])
    # XGB was trained on integer-encoded labels; decode via metadata classes
    pred = (np.array(meta["classes"])[np.asarray(pred_raw).astype(int)]
            if np.asarray(pred_raw).dtype.kind in "iu" else np.asarray(pred_raw))
    yt = y[te]
    tst = sub.iloc[te].copy()
    tst["y_pred"] = pred
    tst["correct"] = yt == pred
    classes = sorted(set(y))
    OUT.mkdir(parents=True, exist_ok=True)

    # --- core metrics ---
    p, r, f, s = precision_recall_fscore_support(yt, pred, labels=classes, zero_division=0)
    per_class = {c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                     "f1": round(float(f[i]), 4), "support": int(s[i])}
                 for i, c in enumerate(classes)}
    cm = confusion_matrix(yt, pred, labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(OUT / "confusion_matrix.csv")
    pd.DataFrame(cm / cm.sum(axis=1, keepdims=True), index=classes,
                 columns=classes).round(4).to_csv(OUT / "confusion_matrix_norm.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, norm=LogNorm(vmin=1, vmax=cm.max()))
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color="white")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("RF baseline confusion matrix (test, counts, log scale)")
    savefig(fig, "confusion_matrix.png")

    # --- events/obs per class, train vs test ---
    trd = sub.iloc[tr]
    eo = []
    for split, d in (("train", trd), ("test", tst)):
        for c in classes:
            dd = d[d["y"] == c]
            eo.append({"split": split, "class": c, "obs": len(dd),
                       "events": int(dd["event_id"].nunique())})
    pd.DataFrame(eo).to_csv(OUT / "events_obs_per_class.csv", index=False)

    # --- feature importance: impurity + permutation ---
    y_perm = (np.array([meta["classes"].index(v) for v in yt])
              if np.asarray(pred_raw).dtype.kind in "iu" else yt)
    perm = permutation_importance(clf, X[te], y_perm, n_repeats=5, random_state=SEED, n_jobs=-1)
    fi = pd.DataFrame({"feature": FEATURES, "impurity": clf.feature_importances_,
                       "permutation": perm.importances_mean}).sort_values("permutation", ascending=False)
    fi.to_csv(OUT / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    fi.plot(kind="barh", x="feature", y=["impurity", "permutation"], ax=ax)
    ax.set_title("Feature importance (impurity vs test permutation)")
    savefig(fig, "feature_importance.png")

    # --- errors by geography (1-deg cells) ---
    tst["cell1"] = (tst["ev_lat"].round(0).astype(str) + "," + tst["ev_lon"].round(0).astype(str))
    geo = tst.groupby("cell1").agg(n=("correct", "size"), err=("correct", lambda v: float((~v).mean())),
                                   lat=("ev_lat", "mean"), lon=("ev_lon", "mean")).reset_index()
    geo.to_csv(OUT / "errors_by_geography.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    ok, bad = tst[tst["correct"]], tst[~tst["correct"]]
    ax.scatter(ok["ev_lon"], ok["ev_lat"], s=6, alpha=0.4, label=f"correct {len(ok)}")
    ax.scatter(bad["ev_lon"], bad["ev_lat"], s=18, alpha=0.9, label=f"errors {len(bad)}")
    try:
        import json as _j
        gj = _j.load(open("eda/outputs/india_boundary_hr.geojson"))
        for f in gj["features"]:
            geoms = [f["geometry"]["coordinates"]] if f["geometry"]["type"] == "Polygon" else \
                [p for poly in f["geometry"]["coordinates"] for p in [poly]]
            for poly in (geoms if f["geometry"]["type"] == "MultiPolygon" else geoms):
                ring = np.asarray(poly[0] if f["geometry"]["type"] == "MultiPolygon" else poly[0])
                ax.plot(ring[:, 0], ring[:, 1], color="black", lw=0.8)
    except FileNotFoundError:
        pass
    ax.set_xlim(68, 97); ax.set_ylim(8, 37)
    ax.set_title("Test errors by geography (blue=correct, orange=error)")
    ax.legend(markerscale=3)
    savefig(fig, "errors_by_geography.png")

    # --- errors by persistence ---
    tst["persist_bin"] = pd.cut(tst["ev_ndays"], [-1, 1, 4, 14, 10**6],
                                labels=["1 day", "2-4 days", "5-14 days", "15+ days"])
    pb = tst.groupby(["persist_bin", "y"], observed=True).agg(
        n=("correct", "size"), err=("correct", lambda v: round(float((~v).mean()), 4))).reset_index()
    pb.to_csv(OUT / "errors_by_persistence.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    pb.pivot(index="persist_bin", columns="y", values="err").plot(kind="bar", ax=ax)
    ax.set_title("Test error rate by event persistence"); ax.set_ylabel("error rate")
    savefig(fig, "errors_by_persistence.png")

    # --- errors by OSM evidence type ---
    ob = tst.groupby("nearest_osm_type").agg(n=("correct", "size"),
                                             err=("correct", lambda v: round(float((~v).mean()), 4))).reset_index()
    ob = ob.sort_values("n", ascending=False)
    ob.to_csv(OUT / "errors_by_osm_type.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ob.head(12).plot(kind="bar", x="nearest_osm_type", y="err", ax=ax, legend=False)
    plt.xticks(rotation=30, ha="right")
    ax.set_title("Test error rate by nearest OSM type (top 12 by support)")
    savefig(fig, "errors_by_osm_type.png")

    # --- errors by FRP / brightness ---
    tst["frp_bin"] = pd.cut(pd.to_numeric(tst["frp"], errors="coerce"),
                            [-1, 2, 5, 15, 50, 10**6], labels=["<2", "2-5", "5-15", "15-50", ">50 MW"])
    fb = tst.groupby(["frp_bin", "y"], observed=True).agg(
        n=("correct", "size"), err=("correct", lambda v: round(float((~v).mean()), 4))).reset_index()
    fb.to_csv(OUT / "errors_by_frp.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    fb.pivot(index="frp_bin", columns="y", values="err").plot(kind="bar", ax=ax)
    ax.set_title("Test error rate by FRP range"); ax.set_ylabel("error rate")
    savefig(fig, "errors_by_frp.png")
    tst["ti4_bin"] = pd.cut(pd.to_numeric(tst["bright_ti4"], errors="coerce"),
                            [0, 310, 330, 350, 1000], labels=["<310", "310-330", "330-350", ">350 K"])
    bb = tst.groupby(["ti4_bin", "y"], observed=True).agg(
        n=("correct", "size"), err=("correct", lambda v: round(float((~v).mean()), 4))).reset_index()
    bb.to_csv(OUT / "errors_by_brightness.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    bb.pivot(index="ti4_bin", columns="y", values="err").plot(kind="bar", ax=ax)
    ax.set_title("Test error rate by bright_ti4 range"); ax.set_ylabel("error rate")
    savefig(fig, "errors_by_brightness.png")

    bundle = {"model": str(MODELDIR), "seed": SEED, "test_size": TEST_SIZE,
              "label_confidence": keep_conf, "test_n": int(len(te)),
              "train_blocks": int(len(set(groups[tr]))), "test_blocks": int(len(set(groups[te]))),
              "per_class": per_class,
              "macro_f1": round(float(np.mean([per_class[c]['f1'] for c in classes])), 4),
              "files": sorted(p.name for p in OUT.iterdir())}
    (OUT / "evaluation.json").write_text(json.dumps(bundle, indent=2))
    print(json.dumps({k: v for k, v in bundle.items() if k != "files"}, indent=2), flush=True)
    print(f"saved {len(list(OUT.iterdir()))} files -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
