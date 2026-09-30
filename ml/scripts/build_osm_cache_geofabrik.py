"""
Bulk OSM industrial cache from Geofabrik PBFs (no Overpass hammering).

Reads data/raw/osm_pbf/*-latest.osm.pbf via GDAL, keeps the tag scope of
download_osm.py (industrial=*, landuse industrial/quarry, power plant/generator,
listed man_made=*), reduces polygons/lines to centroids (same baseline caveat),
assigns our 4 region columns + 2-deg tile keys, MERGES with the existing
Overpass cache (same Parquet schema), dedupes by (osm_type, osm_id).

Outputs: data/raw/osm/{west,central,east,north_east}_india.parquet + .metadata.json

Usage:
  python ml/scripts/build_osm_cache_geofabrik.py            # all zones in dir
  python ml/scripts/build_osm_cache_geofabrik.py --zones western-zone
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyogrio import read_arrow
from shapely import from_wkb

PBFDIR = Path("data/raw/osm_pbf")
OUTDIR = Path("data/raw/osm")

REGIONS = {  # same disjoint columns as download_osm.py
    "west_india": (68.0, 8.0, 75.5, 37.0),
    "central_india": (75.5, 8.0, 83.0, 37.0),
    "east_india": (83.0, 8.0, 90.0, 37.0),
    "north_east_india": (90.0, 8.0, 97.0, 37.0),
}
BBOX = (68.0, 8.0, 97.0, 37.0)  # FIRMS bbox parity
MAN_MADE = {"works", "chimney", "flare", "petroleum_well", "mineshaft", "gasometer", "storage_tank"}

LIKES = ["industrial", "plant", "generator", "quarry", "flare", "chimney",
         "works", "petroleum", "mineshaft", "gasometer", "storage_tank", "power", "landuse"]
LIKE_SQL = " OR ".join(f"other_tags LIKE '%{k}%'" for k in LIKES)
WHERE_POINTS = f"man_made IS NOT NULL OR {LIKE_SQL}"
WHERE_POLY = f"landuse IS NOT NULL OR man_made IS NOT NULL OR {LIKE_SQL}"
WHERE_LINES = LIKE_SQL

HSTORE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*=>\s*"((?:[^"\\]|\\.)*)"')


def parse_hstore(s: str) -> dict:
    if not s:
        return {}
    out = {}
    for k, v in HSTORE.findall(s):
        out[k.replace('\\"', '"')] = v.replace('\\"', '"')
    return out


def norm(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def in_scope(tags: dict) -> bool:
    if "industrial" in tags:
        return True
    if tags.get("landuse") in ("industrial", "quarry"):
        return True
    if tags.get("power") in ("plant", "generator"):
        return True
    if tags.get("man_made") in MAN_MADE:
        return True
    return False


def region_of(lon: float) -> str:
    if lon < 75.5:
        return "west_india"
    if lon < 83.0:
        return "central_india"
    if lon < 90.0:
        return "east_india"
    return "north_east_india"


def process_layer(pbf: str, layer: str, where: str, extra_cols: list, osm_type: str) -> pd.DataFrame:
    meta, tbl = read_arrow(pbf, layer=layer, bbox=BBOX, where=where,
                           columns=["osm_id", "name", *extra_cols, "other_tags"])
    df = tbl.to_pandas()
    if not len(df):
        return pd.DataFrame()
    geoms = from_wkb(df.pop("wkb_geometry").to_numpy())
    cent = [g.centroid if g is not None and not g.is_empty else None for g in geoms]
    rows = []
    for i, row in df.iterrows():
        tags = parse_hstore(norm(row.get("other_tags")))
        for c in ("name", *extra_cols):
            v = norm(row.get(c))
            if v and c not in ("osm_id",):
                tags.setdefault(c, v)
        if not in_scope(tags):
            continue
        c = cent[i]
        if c is None:
            continue
        oid = row["osm_id"]
        try:
            oid = int(oid)
        except (ValueError, TypeError):
            continue
        core = {k: tags.get(k, "") for k in ("name", "industrial", "landuse", "power", "man_made", "building")}
        rows.append({"osm_type": osm_type, "osm_id": oid, "lat": float(c.y), "lon": float(c.x),
                     **core, "tags": json.dumps({k: v for k, v in tags.items() if k not in core},
                                                ensure_ascii=False)})
    out = pd.DataFrame(rows)
    if len(out):
        out["region"] = out["lon"].apply(region_of)
        out["tile_key"] = (out["lat"].floordiv(2) * 2).round(1).astype(str) + "_" + \
                          (out["lon"].floordiv(2) * 2).round(1).astype(str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", nargs="*", default=None)
    ap.add_argument("--no-merge", action="store_true", help="skip merging Overpass cache")
    args = ap.parse_args()

    pbfs = sorted(glob.glob(str(PBFDIR / "*-latest.osm.pbf")))
    if args.zones:
        pbfs = [p for p in pbfs if any(z in p for z in args.zones)]
    if not pbfs:
        raise FileNotFoundError(f"No PBFs in {PBFDIR}")
    print(f"pbfs: {[Path(p).name for p in pbfs]}", flush=True)

    frames = []
    for pbf in pbfs:
        t = time.time()
        pts = process_layer(pbf, "points", WHERE_POINTS, ["man_made"], "node")
        lin = process_layer(pbf, "lines", WHERE_LINES, [], "way")
        poly = process_layer(pbf, "multipolygons", WHERE_POLY, ["man_made", "landuse", "osm_way_id"], "way")
        # multipolygon rows keyed by member way where possible (parity w/ Overpass way ids)
        print(f"{Path(pbf).name}: nodes={len(pts)} lines={len(lin)} polys={len(poly)} "
              f"{time.time()-t:.0f}s", flush=True)
        frames.extend([pts, lin, poly])
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"geofabrik rows: {len(new)}", flush=True)

    if not args.no_merge:
        old_files = sorted(glob.glob(str(OUTDIR / "*.parquet")))
        old_files = [f for f in old_files if "_partial_" not in f]
        if old_files:
            old = pd.concat([pd.read_parquet(f) for f in old_files], ignore_index=True)
            print(f"overpass rows: {len(old)}", flush=True)
            new = pd.concat([new, old], ignore_index=True)
    new = new.drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)
    print(f"merged unique: {len(new)}", flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for region in REGIONS:
        sub = new[new["region"] == region].copy()
        sub.to_parquet(OUTDIR / f"{region}.parquet", index=False)
        meta = {"region": region, "rows": len(sub), "bbox_filter": list(BBOX),
                "source": "Geofabrik PBF + Overpass merge" if not args.no_merge else "Geofabrik PBF",
                "pbfs": [Path(p).name for p in pbfs],
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope_note": "same tag scope as download_osm.py; polygons/lines reduced to centroids"}
        (OUTDIR / f"{region}.metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"{region}: {len(sub)} rows", flush=True)


if __name__ == "__main__":
    main()
