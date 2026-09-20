"""
Download + cache OSM industrial features for the India FIRMS bbox — TILED, once.

Never query OSM per FIRMS point. Instead:
  India bbox (68,8,97,37)
    -> 4 disjoint region columns (no gaps, no overlaps)
    -> 2x2-degree sub-tiles per region (auto-split to 1x1 / 0.5x0.5 on timeout)
    -> Overpass API (sequential + sleep, retries, fallback host)
    -> one parquet + metadata JSON per region in data/raw/osm/

Full run:  python ml/scripts/download_osm.py
Test one tile: python ml/scripts/download_osm.py --region west_india --limit-tiles 1 --tile-size 2
Dry run (no network): python ml/scripts/download_osm.py --dry-run

Tag scope (deliberate, documented):
  industrial=* | landuse=industrial|quarry | power=plant|generator
  man_made=works|chimney|flare|petroleum_well|mineshaft|gasometer|storage_tank
  building=industrial
Excluded on purpose: power=pole|tower|line|substation (grid hardware, millions of
features, not thermal sources), man_made=pipeline (linear, not point sources),
building=industrial (every warehouse/shed — huge, low signal; facility-level
sites are covered by industrial=*/landuse=industrial/man_made=works),
relations (covered via member ways/nodes for our feature set).
`out center tags` returns way centroids so every row is a point (lat/lon).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUTDIR = Path("data/raw/osm")
CKPTDIR = OUTDIR / "_checkpoints"

# Disjoint full-height columns over the FIRMS bbox (lon 68-97, lat 8-37).
REGIONS = {
    "west_india": (68.0, 8.0, 75.5, 37.0),
    "central_india": (75.5, 8.0, 83.0, 37.0),
    "east_india": (83.0, 8.0, 90.0, 37.0),
    "north_east_india": (90.0, 8.0, 97.0, 37.0),
}

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

MAN_MADE = "works|chimney|flare|petroleum_well|mineshaft|gasometer|storage_tank"

QUERY_TMPL = """[out:json][timeout:{timeout}];
(
  node["industrial"]({s},{w},{n},{e});
  way["industrial"]({s},{w},{n},{e});
  node["landuse"~"^(industrial|quarry)$"]({s},{w},{n},{e});
  way["landuse"~"^(industrial|quarry)$"]({s},{w},{n},{e});
  node["power"~"^(plant|generator)$"]({s},{w},{n},{e});
  way["power"~"^(plant|generator)$"]({s},{w},{n},{e});
  node["man_made"~"^({mm})$"]({s},{w},{n},{e});
  way["man_made"~"^({mm})$"]({s},{w},{n},{e});
);
out center tags;"""


def build_query(s, w, n, e, timeout=180):
    return QUERY_TMPL.format(s=s, w=w, n=n, e=e, timeout=timeout, mm=MAN_MADE)


def sub_tiles(bbox, size):
    w0, s0, e1, n1 = bbox
    tiles = []
    lat = s0
    while lat < n1:
        lon = w0
        while lon < e1:
            tiles.append((round(lat, 3), round(lon, 3),
                          round(min(lat + size, n1), 3), round(min(lon + size, e1), 3)))
            lon += size
        lat += size
    return tiles


def post_overpass(query, http_timeout=240, retries=5, backoff_base=30.0):
    """POST with retries across endpoints. 400 = bad query (fail fast);
    429/504/timeouts = overloaded server (backoff, honour Retry-After)."""
    last_err = None
    for attempt in range(1, retries + 1):
        for ep in ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": query}, timeout=http_timeout,
                                  headers={"User-Agent": "ThermalGuard-EDA/1.0 (research use)"})
                if r.status_code == 200:
                    payload = r.json()
                    remark = str(payload.get("remark", ""))
                    if "timed out" in remark or "error" in remark.lower():
                        last_err = f"{ep} -> server-side: {remark[:200]}"
                        wait = backoff_base * attempt
                        print(f"  retry {attempt}/{retries} after {wait:.0f}s ({last_err})", flush=True)
                        time.sleep(wait)
                        continue
                    return payload
                if r.status_code == 400:
                    raise RuntimeError(f"Overpass rejected query (400): {r.text[:300]}")
                last_err = f"{ep} -> HTTP {r.status_code}: {r.text[:200]}"
                wait = float(r.headers.get("Retry-After", backoff_base * attempt))
            except RuntimeError:
                raise
            except Exception as ex:  # noqa: BLE001 - network, retry
                last_err = f"{ep} -> {ex}"
                wait = backoff_base * attempt
            print(f"  retry {attempt}/{retries} after {wait:.0f}s ({last_err})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Overpass still failing after {retries} attempts. Last: {last_err}")


def parse_elements(payload, region, tile_key):
    rows = []
    for el in payload.get("elements", []):
        tags = el.get("tags", {}) or {}
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center") or {}
            lat, lon = c.get("lat"), c.get("lon")
        if lat is None or lon is None:
            continue
        core = {k: tags.pop(k, "") for k in
                ("name", "industrial", "landuse", "power", "man_made", "building")}
        rows.append({
            "region": region, "tile_key": tile_key,
            "osm_type": el["type"], "osm_id": el["id"],
            "lat": float(lat), "lon": float(lon),
            "name": core["name"], "industrial": core["industrial"],
            "landuse": core["landuse"], "power": core["power"],
            "man_made": core["man_made"], "building": core["building"],
            "tags": json.dumps(tags, ensure_ascii=False),
        })
    return rows


def fetch_tile(s, w, n, e, region, tile_key, depth=0, max_depth=2, sleep=2.0):
    """Fetch one tile; recursively quarter it on failure (timeout/overload)."""
    try:
        payload = post_overpass(build_query(s, w, n, e))
        time.sleep(sleep)
        return parse_elements(payload, region, tile_key), {"status": "ok", "depth": depth}
    except Exception as ex:  # noqa: BLE001 - retry via split
        if depth >= max_depth:
            return [], {"status": f"failed: {ex}", "depth": depth}
        mid_lat, mid_lon = (s + n) / 2, (w + e) / 2
        rows, info = [], {"status": "split", "depth": depth, "parts": []}
        for i, q in enumerate([(s, w, mid_lat, mid_lon), (s, mid_lon, mid_lat, e),
                               (mid_lat, w, n, mid_lon), (mid_lat, mid_lon, n, e)]):
            sub_rows, sub_info = fetch_tile(*q, region, f"{tile_key}q{i}",
                                            depth + 1, max_depth, sleep)
            rows.extend(sub_rows)
            info["parts"].append(sub_info)
        return rows, info


def load_ckpt(region):
    p = CKPTDIR / f"{region}.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"completed": [], "failed": {}}


def save_ckpt(region, ckpt):
    CKPTDIR.mkdir(parents=True, exist_ok=True)
    (CKPTDIR / f"{region}.json").write_text(json.dumps(ckpt, indent=2))


def run_region(region, tile_size, limit_tiles, sleep, fresh):
    bbox = REGIONS[region]
    tiles = sub_tiles(bbox, tile_size)
    if limit_tiles:
        # pick a tile over industrial Gujarat for the smoke test when possible
        tiles = sorted(tiles, key=lambda t: abs(t[0] - 22) + abs(t[1] - 72))[:limit_tiles]
    part = OUTDIR / f"_partial_{region}.parquet"
    if fresh:
        ckpt = {"completed": [], "failed": {}}
        if part.exists():
            part.unlink()
    else:
        ckpt = load_ckpt(region)
    done = set(ckpt["completed"])
    tile_info = {}
    print(f"[{region}] bbox={bbox} tiles={len(tiles)} resume_skip={len(done)}", flush=True)
    for (s, w, n, e) in tiles:
        key = f"{s}_{w}_{n}_{e}"
        if key in done:
            continue
        rows, info = fetch_tile(s, w, n, e, region, key, sleep=sleep)
        tile_info[key] = info
        if info["status"].startswith("failed"):
            ckpt["failed"][key] = info["status"]
        else:
            # incremental persist: a killed run loses nothing already fetched
            new = pd.DataFrame(rows)
            old = pd.read_parquet(part) if part.exists() else pd.DataFrame()
            combined = (pd.concat([old, new], ignore_index=True)
                        .drop_duplicates(subset=["osm_type", "osm_id"]))
            OUTDIR.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(part, index=False)
            ckpt["completed"].append(key)
        save_ckpt(region, ckpt)
        print(f"[{region}] {key}: {info['status']} rows={len(rows)}", flush=True)
    df = pd.read_parquet(part) if part.exists() else pd.DataFrame()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTDIR / f"{region}.parquet", index=False)
    meta = {
        "region": region, "bbox": bbox, "tile_size_deg": tile_size,
        "tiles_attempted": len(tiles), "tiles_completed": len(ckpt["completed"]),
        "tiles_failed": ckpt["failed"], "rows": len(df),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoints": ENDPOINTS, "query_template": QUERY_TMPL,
        "note": "Raw OSM cache. Train only from these parquets — never live-query per FIRMS point.",
    }
    (OUTDIR / f"{region}.metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[{region}] wrote {len(df)} rows -> {OUTDIR / f'{region}.parquet'}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="Tiled OSM industrial download -> data/raw/osm/")
    ap.add_argument("--region", default="all", choices=["all", *REGIONS])
    ap.add_argument("--tile-size", type=float, default=2.0)
    ap.add_argument("--limit-tiles", type=int, default=0, help="0 = all tiles")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between Overpass calls")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoint, restart region")
    ap.add_argument("--dry-run", action="store_true", help="print plan + one query, no network")
    args = ap.parse_args()

    regions = list(REGIONS) if args.region == "all" else [args.region]
    if args.dry_run:
        for r in regions:
            tiles = sub_tiles(REGIONS[r], args.tile_size)
            print(f"{r}: bbox={REGIONS[r]} tiles={len(tiles)} file=data/raw/osm/{r}.parquet")
        print("--- sample query (first tile of first region) ---")
        s, w, n, e = sub_tiles(REGIONS[regions[0]], args.tile_size)[0]
        print(build_query(s, w, n, e))
        return
    for r in regions:
        run_region(r, args.tile_size, args.limit_tiles or 0, args.sleep, args.fresh)


if __name__ == "__main__":
    main()
