"""
Spatial enrichment: FIRMS thermal records x cached OSM industrial features.

  FIRMS (lat/lon/FRP/...) + OSM cache (data/raw/osm/*.parquet)
      -> nearest OSM feature (distance_km, type + provenance id/kind)
      -> local-context counts (1/2/5 km + class-specific 5 km)
      -> data/processed/firms_osm_enriched.parquet (FIRMS columns untouched)

Features (exactly these; no labels yet — label column is "UNKNOWN" for all rows):
  nearest_osm_distance_km, nearest_osm_type,
  industrial_count_1km/2km/5km, power_plant_count_5km, quarry_count_5km,
  flare_count_5km, petroleum_well_count_5km, industrial_landuse_nearby
Provenance (not model features): nearest_osm_id, nearest_osm_kind.

Known limitation (documented, not fixed here): OSM ways are reduced to
centroids, so a hotspot inside a large facility polygon but far from its
centroid reads as "distant". OK for baseline; later upgrade to
point-in-polygon / distance-to-boundary using osm_type+osm_id to re-fetch
polygons. Tags JSON is preserved in the cache for that upgrade.

Usage:
  python ml/scripts/enrich_firms_with_osm.py [--firms PATH] [--chunk 250000]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0088
OUT = Path("data/processed/firms_osm_enriched.parquet")
QA_JSON = Path("eda/outputs/osm_enrichment_qa.json")

DEFAULT_FIRMS = "data/processed/firms_noaa20_sp_bbox_68_8_97_37_2024-06-01_to_2026-05-31.csv"

COUNT_RADII = (1.0, 2.0, 5.0)


def find_firms(path: str | None) -> str:
    if path:
        return path
    cand = Path("data/processed/firms_24months.parquet")
    if cand.exists():
        return str(cand)
    return DEFAULT_FIRMS


def load_osm():
    files = sorted(glob.glob("data/raw/osm/*.parquet"))
    if not files:
        raise FileNotFoundError("No OSM cache found in data/raw/osm/. Run download_osm.py first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)
    print(f"OSM cache: {len(df)} features from {len(files)} file(s): {files}", flush=True)
    return df


def feature_type(row) -> str:
    if row.get("industrial"):
        return f"industrial:{row['industrial']}"
    if row.get("power"):
        return f"power:{row['power']}"
    if row.get("man_made"):
        return f"man_made:{row['man_made']}"
    if row.get("landuse"):
        return f"landuse:{row['landuse']}"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--firms", default=None)
    ap.add_argument("--chunk", type=int, default=250000)
    args = ap.parse_args()

    firms_path = find_firms(args.firms)
    osm = load_osm()
    for c in ("industrial", "landuse", "power", "man_made"):
        osm[c] = osm[c].fillna("").astype(str)

    o_rad = np.radians(osm[["lat", "lon"]].to_numpy())
    tree = BallTree(o_rad, metric="haversine")
    o_type = np.array([feature_type(r) for r in osm.to_dict("records")])
    is_power = (osm["power"] == "plant") | (osm["power"] == "generator")
    is_quarry = (osm["landuse"] == "quarry") | (osm["man_made"] == "mineshaft")
    is_flare = osm["man_made"] == "flare"
    is_well = osm["man_made"] == "petroleum_well"
    is_landuse_ind = osm["landuse"].isin(["industrial", "quarry"])
    m_power = is_power.to_numpy()
    m_quarry = is_quarry.to_numpy()
    m_flare = is_flare.to_numpy()
    m_well = is_well.to_numpy()
    m_land = is_landuse_ind.to_numpy()

    r1, r2, r5 = [r / EARTH_KM for r in COUNT_RADII]
    r2land = 2.0 / EARTH_KM

    read = (pd.read_parquet(firms_path, chunksize=args.chunk) if firms_path.endswith(".parquet")
            else pd.read_csv(firms_path, chunksize=args.chunk))
    parts = []
    total = 0
    for chunk in read:
        lat = pd.to_numeric(chunk["latitude"], errors="coerce").to_numpy()
        lon = pd.to_numeric(chunk["longitude"], errors="coerce").to_numpy()
        pts = np.radians(np.column_stack([lat, lon]))

        dist, idx = tree.query(pts, k=1)
        ind5 = tree.query_radius(pts, r=r5)
        ind1 = tree.query_radius(pts, r=r1)
        ind2 = tree.query_radius(pts, r=r2)

        n = len(chunk)
        c1 = np.empty(n, dtype=np.int32)
        c2 = np.empty(n, dtype=np.int32)
        c5 = np.empty(n, dtype=np.int32)
        p5 = np.empty(n, dtype=np.int32)
        q5 = np.empty(n, dtype=np.int32)
        f5 = np.empty(n, dtype=np.int32)
        w5 = np.empty(n, dtype=np.int32)
        land_near = np.empty(n, dtype=np.int8)
        for i, (a, b, c) in enumerate(zip(ind1, ind2, ind5)):
            c1[i] = len(a)
            c2[i] = len(b)
            c5[i] = len(c)
            p5[i] = int(m_power[c].sum())
            q5[i] = int(m_quarry[c].sum())
            f5[i] = int(m_flare[c].sum())
            w5[i] = int(m_well[c].sum())
            land_near[i] = int(m_land[b].sum() > 0)

        flat = idx[:, 0]
        chunk = chunk.copy()
        chunk["nearest_osm_distance_km"] = (dist[:, 0] * EARTH_KM).round(4)
        chunk["nearest_osm_type"] = o_type[flat]
        chunk["nearest_osm_id"] = osm["osm_id"].to_numpy()[flat]
        chunk["nearest_osm_kind"] = osm["osm_type"].to_numpy()[flat]
        chunk["industrial_count_1km"] = c1
        chunk["industrial_count_2km"] = c2
        chunk["industrial_count_5km"] = c5
        chunk["power_plant_count_5km"] = p5
        chunk["quarry_count_5km"] = q5
        chunk["flare_count_5km"] = f5
        chunk["petroleum_well_count_5km"] = w5
        chunk["industrial_landuse_nearby"] = land_near
        chunk["label"] = "UNKNOWN"  # Step 7 assigns real labels; context is NOT a label
        parts.append(chunk)
        total += n
        print(f"enriched {total} rows ...", flush=True)

    out = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {len(out)} rows x {len(out.columns)} cols -> {OUT}", flush=True)

    qa = {
        "firms_input": firms_path,
        "osm_features_cached": int(len(osm)),
        "total_records": int(len(out)),
        "matched_5km": int((out["industrial_count_5km"] > 0).sum()),
        "no_osm_within_5km": int((out["industrial_count_5km"] == 0).sum()),
        "median_nearest_km": round(float(out["nearest_osm_distance_km"].median()), 3),
        "mean_nearest_km": round(float(out["nearest_osm_distance_km"].mean()), 3),
        "p90_nearest_km": round(float(out["nearest_osm_distance_km"].quantile(0.9)), 3),
        "firms_with_power_plant_5km": int((out["power_plant_count_5km"] > 0).sum()),
        "firms_with_quarry_5km": int((out["quarry_count_5km"] > 0).sum()),
        "firms_with_flare_5km": int((out["flare_count_5km"] > 0).sum()),
        "firms_with_petrowell_5km": int((out["petroleum_well_count_5km"] > 0).sum()),
        "firms_landuse_nearby": int(out["industrial_landuse_nearby"].sum()),
        "nearest_type_top": out["nearest_osm_type"].value_counts().head(10).to_dict(),
        "note": "label=UNKNOWN everywhere. OSM context describes surroundings, never the cause.",
    }
    QA_JSON.parent.mkdir(parents=True, exist_ok=True)
    QA_JSON.write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
