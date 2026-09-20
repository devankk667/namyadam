"""Regenerate spatial maps: detection-density + India boundary overlay + FRP-weighted."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

P = "data/processed/firms_noaa20_sp_bbox_68_8_97_37_2024-06-01_to_2026-05-31.csv"
print("Loading lat/lon/frp ...")
df = pd.read_csv(P, usecols=["latitude", "latitude", "longitude", "frp"])
lon = df["longitude"].to_numpy()
lat = df["latitude"].to_numpy()
frp = pd.to_numeric(df["frp"], errors="coerce").to_numpy()
print(f"n={len(df)}")

import os as _os
_bf = ("eda/outputs/india_boundary_hr.geojson"
       if _os.path.exists("eda/outputs/india_boundary_hr.geojson")
       else "eda/outputs/india_boundary.geojson")
gj = json.load(open(_bf))
print("boundary file:", _bf)
rings = []
for f in gj["features"]:
    geom = f["geometry"]
    if geom["type"] == "Polygon":
        rings.append(np.asarray(geom["coordinates"][0]))
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            rings.append(np.asarray(poly[0]))  # exterior ring only
print(f"boundary rings: {len(rings)}")

def draw_boundary(ax):
    for r in rings:
        ax.plot(r[:, 0], r[:, 1], color="black", lw=2.2, zorder=5)
        ax.plot(r[:, 0], r[:, 1], color="white", lw=1.1, zorder=6)

BINS = [120, 120]
RNG = [[68, 97], [8, 37]]

# 12: detection density + boundary (replaces old plot, corrected title)
counts, xe, ye = np.histogram2d(lon, lat, bins=BINS, range=RNG)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(counts.T[::-1, :], extent=[68, 97, 8, 37], origin="lower",
               aspect="auto", norm=LogNorm(vmin=1, vmax=counts.max()))
draw_boundary(ax)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("FIRMS Detection Density — India (24-Month Dataset)\nVIIRS NOAA-20 SP, 2024-06-01 → 2026-05-31 (counts per bin, log scale)")
cb = fig.colorbar(im, ax=ax); cb.set_label("detections per bin")
fig.savefig("eda/outputs/12_spatial_heatmap.png")
plt.close(fig)

# 16a: mean FRP per bin + boundary
sum_frp, _, _ = np.histogram2d(lon, lat, bins=BINS, range=RNG, weights=np.nan_to_num(frp))
mean_frp = np.divide(sum_frp, counts, out=np.full_like(sum_frp, np.nan), where=counts > 0)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(mean_frp.T[::-1, :], extent=[68, 97, 8, 37], origin="lower",
               aspect="auto", vmin=0, vmax=np.nanquantile(mean_frp, 0.98))
draw_boundary(ax)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Mean FRP per Bin — India (24-Month Thermal Intensity)\nVIIRS NOAA-20 SP (MW; bins with 0 detections masked)")
masked = np.ma.masked_invalid(im.get_array())
im.set_array(masked)
cb = fig.colorbar(im, ax=ax); cb.set_label("mean FRP (MW)")
fig.savefig("eda/outputs/16_frp_mean_weighted.png")
plt.close(fig)

# 16b: total FRP per bin + boundary
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(sum_frp.T[::-1, :], extent=[68, 97, 8, 37], origin="lower",
               aspect="auto", norm=LogNorm(vmin=1, vmax=sum_frp.max()))
draw_boundary(ax)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_title("Total FRP per Bin — India (24-Month Cumulative Fire Intensity)\nVIIRS NOAA-20 SP (MW sum per bin, log scale)")
cb = fig.colorbar(im, ax=ax); cb.set_label("total FRP (MW)")
fig.savefig("eda/outputs/17_frp_sum_weighted.png")
plt.close(fig)
print("DONE: 12_spatial_heatmap.png (overwritten), 16_frp_mean_weighted.png, 17_frp_sum_weighted.png")
