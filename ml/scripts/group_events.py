"""
Phase A — group repeated FIRMS observations into thermal events.

  ~1.1 km spatial cells (0.01 deg) + date-gap split (> --gap-days starts
  a new event) -> one row per physical source episode, not per overpass.

Input : data/processed/firms_osm_enriched.parquet (default; falls back to raw CSV)
Output: data/processed/firms_thermal_events.parquet  (event aggregates)
        data/processed/firms_obs_event_map.parquet   (obs_id -> event_id)

Known limitation: cell borders can split one source (no jitter handling yet);
future upgrade = DBSCAN/haversine clustering. OSM aggregates use max/first
(OSM is static per location, so first == any).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from firms_ids import ensure_observation_ids, fingerprint_observation_ids

ENRICHED = Path("data/processed/firms_osm_enriched.parquet")
RAW = Path("data/processed/firms_noaa20_sp_bbox_68_8_97_37_2024-06-01_to_2026-05-31.csv")
EVENTS_OUT = Path("data/processed/firms_thermal_events.parquet")
MAP_OUT = Path("data/processed/firms_obs_event_map.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--gap-days", type=int, default=14)
    ap.add_argument("--cell", type=float, default=0.01)
    args = ap.parse_args()

    src = args.input or (str(ENRICHED) if ENRICHED.exists() else str(RAW))
    print(f"loading {src} ...", flush=True)
    df = pd.read_parquet(src) if src.endswith(".parquet") else pd.read_csv(src)
    df = ensure_observation_ids(df)
    if df["obs_id"].duplicated().any():
        raise ValueError("FIRMS obs_id values must be unique before event grouping")
    source_fingerprint = fingerprint_observation_ids(df["obs_id"])
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df["cell_lat"] = (df["latitude"] / args.cell).round().astype(int)
    df["cell_lon"] = (df["longitude"] / args.cell).round().astype(int)
    df["temp_diff"] = (pd.to_numeric(df["bright_ti4"], errors="coerce")
                       - pd.to_numeric(df["bright_ti5"], errors="coerce"))
    df = df.sort_values(["cell_lat", "cell_lon", "acq_date"]).reset_index(drop=True)

    prev_cell = (df[["cell_lat", "cell_lon"]] != df[["cell_lat", "cell_lon"]].shift(1)).any(axis=1)
    gap = (df["acq_date"] - df.groupby(["cell_lat", "cell_lon"])["acq_date"].shift(1)).dt.days
    df["new_event"] = prev_cell | (gap > args.gap_days) | gap.isna()
    df["event_seq"] = df.groupby(["cell_lat", "cell_lon"])["new_event"].cumsum().astype(int)
    df["event_id"] = ("E" + (df.groupby(["cell_lat", "cell_lon"], sort=False).ngroup() + 1).astype(str).str.zfill(7)
                      + "_" + df["event_seq"].astype(str))

    g = df.groupby("event_id", sort=False)
    ev = g.agg(
        cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"),
        centroid_lat=("latitude", "mean"), centroid_lon=("longitude", "mean"),
        n_detections=("obs_id", "size"), n_days=("acq_date", "nunique"),
        date_first=("acq_date", "min"), date_last=("acq_date", "max"),
        max_frp=("frp", "max"), median_frp=("frp", "median"), mean_frp=("frp", "mean"),
        median_temp_diff=("temp_diff", "median"),
        frac_night=("daynight", lambda s: float((s == "N").mean())),
    ).reset_index()
    ev["duration_days"] = (ev["date_last"] - ev["date_first"]).dt.days
    ev["month_first"] = ev["date_first"].dt.month
    ev["month_last"] = ev["date_last"].dt.month
    for c in ("nearest_osm_distance_km", "nearest_osm_type", "nearest_osm_id",
              "industrial_count_1km", "industrial_count_2km", "industrial_count_5km",
              "power_plant_count_5km", "quarry_count_5km", "flare_count_5km",
              "petroleum_well_count_5km", "industrial_landuse_nearby"):
        if c in df.columns:
            ev[c] = g[c].max() if df[c].dtype.kind in "iuf" else g[c].first()
        else:
            ev[c] = np.nan

    EVENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(EVENTS_OUT, index=False)
    df[["obs_id", "event_id"]].to_parquet(MAP_OUT, index=False)
    metadata_path = MAP_OUT.with_suffix(MAP_OUT.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps({
        "version": 1,
        "source_file": str(src),
        "source_record_count": int(len(df)),
        "obs_id_fingerprint": source_fingerprint,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "join_key": "stable FIRMS obs_id; row order is not used",
    }, indent=2), encoding="utf-8")

    print(f"events={len(ev)} obs={len(df)} "
          f"singletons={(ev['n_detections'] == 1).mean():.3f} "
          f"median_n={(ev['n_detections']).median()} "
          f"events_ge5d={(ev['n_days'] >= 5).sum()} "
          f"events_ge30d_span={(ev['duration_days'] >= 30).sum()}", flush=True)
    print(f"wrote {EVENTS_OUT} + {MAP_OUT}", flush=True)


if __name__ == "__main__":
    main()
