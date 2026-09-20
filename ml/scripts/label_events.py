"""
Phase B/C — evidence-based event labels (taxonomy docs/LABEL_TAXONOMY.md).

Conjunctions only: facility context AND thermal AND temporal evidence.
Class-specific proximity (flare/quarry/power) is computed here from the OSM
cache against event centroids. industrial_fire is capped at medium (no
independent incident source). wildfire is NOT labeled (needs vegetation mask).
Everything else -> unknown. Expect mostly unknown until the OSM cache is full.

Output: data/processed/firms_event_labels.parquet (event_id + label,
  label_confidence, label_source, label_reason) + eda/outputs/label_audit.json
  + data/processed/label_review.csv (top candidates per class for map checks).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088
EVENTS = Path("data/processed/firms_thermal_events.parquet")
LABELS_OUT = Path("data/processed/firms_event_labels.parquet")
REVIEW_OUT = Path("data/processed/label_review.csv")
AUDIT_OUT = Path("eda/outputs/label_audit.json")

BURN_MONTHS = {10, 11, 3, 4}
CFG = {
    "flare_km": 1.0, "flare_km_high": 0.5, "flare_days": 5,
    "flare_days_high": 10, "flare_span": 30,
    "quarry_km": 2.0, "quarry_days": 3, "quarry_days_high": 15,
    "indust_km": 1.0, "indust_frp": 15.0,
    "agri_rural_km": 5.0, "agri_frp": 30.0, "agri_span": 21,
}


def months_covered(m1: pd.Series, m2: pd.Series) -> pd.Series:
    def rng(a, b):
        a, b = int(a), int(b)
        out = set()
        m = a
        while True:
            out.add(m)
            if m == b:
                break
            m = m % 12 + 1
        return out
    return [bool(rng(a, b) & BURN_MONTHS) for a, b in zip(m1, m2)]


def nearest_km(tree: BallTree | None, pts: np.ndarray) -> np.ndarray:
    if tree is None:
        return np.full(len(pts), np.inf)
    return (tree.query(pts, k=1)[0][:, 0] * EARTH_KM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(EVENTS))
    args = ap.parse_args()

    ev = pd.read_parquet(args.events)
    print(f"events={len(ev)}", flush=True)

    import glob
    ofiles = sorted(glob.glob("data/raw/osm/*.parquet"))
    osm = (pd.concat([pd.read_parquet(f) for f in ofiles], ignore_index=True)
           if ofiles else pd.DataFrame())
    print(f"osm features={len(osm)} from {len(ofiles)} file(s)", flush=True)
    for c in ("industrial", "landuse", "power", "man_made"):
        if c in osm.columns:
            osm[c] = osm[c].fillna("").astype(str)

    pts = np.radians(ev[["centroid_lat", "centroid_lon"]].to_numpy())
    o_rad_all, o_rad = None, None
    t_flare = t_quarry = t_power = None
    if len(osm):
        o_rad_all = np.radians(osm[["lat", "lon"]].to_numpy())
        t_all = BallTree(o_rad_all, metric="haversine")
        sel = osm["man_made"] == "flare"
        t_flare = BallTree(np.radians(osm.loc[sel, ["lat", "lon"]].to_numpy()), metric="haversine") if sel.any() else None
        sel = (osm["landuse"] == "quarry") | (osm["man_made"] == "mineshaft")
        t_quarry = BallTree(np.radians(osm.loc[sel, ["lat", "lon"]].to_numpy()), metric="haversine") if sel.any() else None
        sel = osm["power"].isin(["plant", "generator"])
        t_power = BallTree(np.radians(osm.loc[sel, ["lat", "lon"]].to_numpy()), metric="haversine") if sel.any() else None
    else:
        t_all = None

    d_ind = nearest_km(t_all, pts)
    d_flare = nearest_km(t_flare, pts)
    d_quarry = nearest_km(t_quarry, pts)

    n_days = ev["n_days"].to_numpy()
    dur = ev["duration_days"].to_numpy()
    maxfrp = pd.to_numeric(ev["max_frp"], errors="coerce").to_numpy()
    season = np.array(months_covered(ev["month_first"], ev["month_last"]))

    n = len(ev)
    label = np.full(n, "unknown", dtype=object)
    conf = np.full(n, "", dtype=object)
    source = np.full(n, "insufficient_evidence", dtype=object)
    reason = np.full(n, "", dtype=object)

    def set_rule(mask, lab, cf, src, rsn):
        undecided = label == "unknown"
        m = mask & undecided
        label[m] = lab
        conf[m] = cf
        source[m] = src
        reason[m] = rsn[m] if isinstance(rsn, np.ndarray) else rsn

    # gas_flare first (strongest evidence pattern)
    m = (d_flare <= CFG["flare_km"]) & (n_days >= CFG["flare_days"]) & (dur >= CFG["flare_span"])
    hi = m & (d_flare <= CFG["flare_km_high"]) & (n_days >= CFG["flare_days_high"])
    set_rule(m & ~hi, "gas_flare", "medium", "rule:flare_recurrent",
             np.array([f"flare {d:.2f}km, {dd}d/{du}d span" for d, dd, du in zip(d_flare, n_days, dur)]))
    set_rule(hi, "gas_flare", "high", "rule:flare_persistent",
             np.array([f"flare {d:.2f}km, {dd}d/{du}d span" for d, dd, du in zip(d_flare, n_days, dur)]))
    # mining
    m = (d_quarry <= CFG["quarry_km"]) & (n_days >= CFG["quarry_days"])
    hi = m & (n_days >= CFG["quarry_days_high"])
    set_rule(m & ~hi, "mining_activity", "medium", "rule:quarry_persistent",
             np.array([f"quarry {d:.2f}km, {dd}d" for d, dd in zip(d_quarry, n_days)]))
    set_rule(hi, "mining_activity", "high", "rule:quarry_highly_persistent",
             np.array([f"quarry {d:.2f}km, {dd}d" for d, dd in zip(d_quarry, n_days)]))
    # industrial_fire (capped at medium — no independent incident source)
    m = (d_ind <= CFG["indust_km"]) & (maxfrp >= CFG["indust_frp"]) & (n_days >= 2)
    set_rule(m, "industrial_fire", "medium", "rule:industrial_strong_recurrent",
             np.array([f"industry {d:.2f}km, maxFRP {f:.1f}, {dd}d" for d, f, dd in zip(d_ind, maxfrp, n_days)]))
    m = (d_ind <= CFG["indust_km"]) & (maxfrp >= CFG["indust_frp"]) & (n_days == 1)
    set_rule(m, "industrial_fire", "low", "rule:industrial_strong_single",
             np.array([f"industry {d:.2f}km, maxFRP {f:.1f}, single day" for d, f in zip(d_ind, maxfrp)]))
    # agricultural (rural + seasonal + weak + brief + RECURRENT).
    # Single/double-day detections are insufficient evidence -> stay unknown.
    m = (season & (d_ind > CFG["agri_rural_km"]) & (maxfrp < CFG["agri_frp"])
         & (dur <= CFG["agri_span"]) & (n_days >= 3))
    hi = m & (n_days >= 7)
    set_rule(m & ~hi, "agricultural_burning", "low", "rule:seasonal_rural_brief",
             np.array([f"{dd}d/{du}d span, maxFRP {f:.1f}, rural" for dd, du, f in zip(n_days, dur, maxfrp)]))
    set_rule(hi, "agricultural_burning", "medium", "rule:seasonal_rural_recurrent",
             np.array([f"{dd}d/{du}d span, maxFRP {f:.1f}, rural" for dd, du, f in zip(n_days, dur, maxfrp)]))
    # wildfire: pending vegetation mask -> none assigned
    reason[label == "unknown"] = np.array(
        [f"nearest_ind {d:.1f}km, maxFRP {f:.1f}, {dd}d" for d, f, dd in zip(d_ind, maxfrp, n_days)])[label == "unknown"]

    out = pd.DataFrame({"event_id": ev["event_id"], "label": label,
                        "label_confidence": conf, "label_source": source, "label_reason": reason})
    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LABELS_OUT, index=False)

    dist = out.groupby(["label", "label_confidence"]).size().to_dict()
    audit = {"config": CFG, "burn_months": sorted(BURN_MONTHS), "osm_files": ofiles,
             "n_events": n, "label_x_conf": {f"{k[0]}/{k[1]}": int(v) for k, v in dist.items()},
             "wildfire_labels": 0, "wildfire_note": "pending vegetation/forest mask",
             "usable_high_medium_obs_note": "map via firms_obs_event_map.parquet for training counts"}
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)

    # review sample: strongest candidates per label x confidence
    rev = out[out["label"] != "unknown"].merge(ev, on="event_id", how="left")
    keep = []
    for (lab, cf), grp in rev.groupby(["label", "label_confidence"]):
        grp = grp.copy()
        grp["rank_key"] = grp["n_days"] * 100 + grp["max_frp"].fillna(0)
        keep.append(grp.nlargest(min(50, len(grp)), "rank_key"))
    if keep:
        review = pd.concat(keep)[["event_id", "label", "label_confidence", "label_source",
                                  "label_reason", "centroid_lat", "centroid_lon", "n_detections",
                                  "n_days", "duration_days", "max_frp", "median_frp"]]
        review.to_csv(REVIEW_OUT, index=False)
        print(f"review rows={len(review)} -> {REVIEW_OUT}", flush=True)
    print(f"wrote {LABELS_OUT}", flush=True)


if __name__ == "__main__":
    main()
