"""QA maps: FIRMS hotspots + cached OSM features (cached-tile extent + zoom)."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("loading enriched + osm ...", flush=True)
e = pd.read_parquet("data/processed/firms_osm_enriched.parquet",
                    columns=["latitude", "longitude", "frp", "industrial_count_5km"])
o = pd.read_parquet("data/raw/osm/west_india.parquet")

# map 1: cached tile extent (22-23N, 72-73E), FIRMS sample + OSM
f = e[(e["latitude"].between(21.5, 23.5)) & (e["longitude"].between(71.5, 73.5))]
s = f.sample(min(20000, len(f)), random_state=7)
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(s["longitude"], s["latitude"], s=1, alpha=0.3, label=f"FIRMS n={len(s)}")
ax.scatter(o["lon"], o["lat"], s=18, marker="^", label=f"OSM n={len(o)}")
ax.set_title("QA: FIRMS hotspots + cached OSM industrial features\nAhmedabad tile 22-23N, 72-73E (only cached tile so far)")
ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend(markerscale=3)
fig.savefig("eda/outputs/18_enrich_qa_tile.png")
plt.close(fig)

# map 2: zoom on densest OSM cluster (Vatva works)
cx, cy = o["lon"].median(), o["lat"].median()
f2 = f[(f["latitude"].between(cy - 0.3, cy + 0.3)) & (f["longitude"].between(cx - 0.3, cx + 0.3))]
s2 = f2.sample(min(8000, len(f2)), random_state=7) if len(f2) else f2
o2 = o[(o["lat"].between(cy - 0.3, cy + 0.3)) & (o["lon"].between(cx - 0.3, cx + 0.3))]
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(s2["longitude"], s2["latitude"], s=4, alpha=0.4, label=f"FIRMS n={len(s2)}")
ax.scatter(o2["lon"], o2["lat"], s=40, marker="^", label=f"OSM n={len(o2)}")
ax.set_title("QA zoom: FIRMS vs OSM around densest cached cluster")
ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend(markerscale=3)
fig.savefig("eda/outputs/19_enrich_qa_zoom.png")
plt.close(fig)
print("DONE: 18_enrich_qa_tile.png, 19_enrich_qa_zoom.png", flush=True)
