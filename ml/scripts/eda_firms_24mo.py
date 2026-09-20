"""
EDA for 24-month NASA FIRMS VIIRS NOAA-20 SP dataset (India bbox).
Data-quality validation + temporal + sensor + geospatial checks with graphs.

Input : data/processed/firms_noaa20_sp_bbox_68_8_97_37_2024-06-01_to_2026-05-31.csv
Output: eda/outputs/*.png + eda/EDA_REPORT.md (tables filled with real values)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED = "data/processed/firms_noaa20_sp_bbox_68_8_97_37_2024-06-01_to_2026-05-31.csv"
OUTDIR = "eda/outputs"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight"})

print("Loading processed CSV ...")
df = pd.read_csv(PROCESSED)
print(f"rows={len(df)} cols={list(df.columns)}")

# ---------- derived ----------
df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
df["hour_utc"] = df["acq_time"].astype(str).str.zfill(4).str[:2].pipe(pd.to_numeric, errors="coerce")
df["temp_diff"] = df["bright_ti4"] - df["bright_ti5"]
df["ym"] = df["acq_date"].dt.to_period("M").astype(str)

audit = {}
audit["rows"] = len(df)
audit["date_min"] = str(df["acq_date"].min())
audit["date_max"] = str(df["acq_date"].max())
audit["n_dates"] = int(df["acq_date"].dt.date.nunique())
audit["expected_days"] = int((df["acq_date"].max() - df["acq_date"].min()).days + 1)
audit["missing_days"] = audit["expected_days"] - audit["n_dates"]
audit["lat_min"], audit["lat_max"] = float(df["latitude"].min()), float(df["latitude"].max())
audit["lon_min"], audit["lon_max"] = float(df["longitude"].min()), float(df["longitude"].max())
audit["bad_lat"] = int((~df["latitude"].between(-90, 90)).sum())
audit["bad_lon"] = int((~df["longitude"].between(-180, 180)).sum())
audit["outside_bbox"] = int(((df["latitude"] < 8) | (df["latitude"] > 37) |
                             (df["longitude"] < 68) | (df["longitude"] > 97)).sum())
audit["bad_date"] = int(df["acq_date"].isna().sum())
audit["neg_frp"] = int((pd.to_numeric(df["frp"], errors="coerce") < 0).sum())
audit["ti4_gt_ti5_frac"] = float((df["bright_ti4"] > df["bright_ti5"]).mean())
audit["scan_le0"] = int((pd.to_numeric(df["scan"], errors="coerce") <= 0).sum())
audit["track_le0"] = int((pd.to_numeric(df["track"], errors="coerce") <= 0).sum())
miss = (df.isna().mean() * 100).sort_values(ascending=False)
audit["missing_pct"] = {k: round(float(v), 4) for k, v in miss.items() if v > 0}

frp = pd.to_numeric(df["frp"], errors="coerce")
audit["frp"] = {k: round(float(v), 3) for k, v in {
    "min": frp.min(), "max": frp.max(), "mean": frp.mean(), "median": frp.median(),
    "std": frp.std(), "q1": frp.quantile(.25), "q3": frp.quantile(.75),
    "p95": frp.quantile(.95), "p99": frp.quantile(.99)}.items()}
for c in ["bright_ti4", "bright_ti5", "temp_diff", "scan", "track"]:
    s = pd.to_numeric(df[c], errors="coerce")
    audit[c] = {k: round(float(v), 3) for k, v in {
        "min": s.min(), "max": s.max(), "median": s.median(),
        "mean": s.mean(), "p99": s.quantile(.99)}.items()}
audit["confidence"] = df["confidence"].value_counts(dropna=False).to_dict()
audit["daynight"] = df["daynight"].value_counts(dropna=False).to_dict()
audit["version"] = df["version"].value_counts(dropna=False).to_dict()
audit["satellite"] = df["satellite"].value_counts(dropna=False).to_dict()

# exact + event-key duplicates (sample-safe: full vectorized, fast enough)
audit["exact_dup_rows"] = int(df.duplicated().sum())
audit["eventkey_dup_rows"] = int(df.duplicated(
    subset=["latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument"]).sum())

# missing acquisition days list
all_days = pd.date_range(df["acq_date"].min(), df["acq_date"].max(), freq="D")
have = set(df["acq_date"].dt.date.astype(str))
missing_days = [str(d.date()) for d in all_days if str(d.date()) not in have]
audit["missing_day_list"] = missing_days[:50]
audit["missing_day_total"] = len(missing_days)

with open(os.path.join(OUTDIR, "audit.json"), "w") as f:
    json.dump(audit, f, indent=2, default=str)
print(json.dumps({k: v for k, v in audit.items() if k != "missing_pct"}, indent=2, default=str))

# ---------- plots ----------
def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, name))
    plt.close(fig)

# 1 monthly counts
m = df.groupby("ym").size()
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(m.index, m.values)
ax.set_title("Monthly detection count (24 months)")
ax.set_ylabel("detections"); plt.xticks(rotation=60); save(fig, "01_monthly_counts.png")

# 2 daily counts
d = df.groupby(df["acq_date"].dt.date).size()
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.plot(list(d.index), d.values, lw=0.7)
ax.set_title("Daily detection count"); ax.set_ylabel("detections"); save(fig, "02_daily_counts.png")

# 3 hour UTC
h = df["hour_utc"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(h.index.astype(int), h.values)
ax.set_title("Hour-of-day distribution (UTC)"); ax.set_xlabel("hour UTC"); ax.set_ylabel("detections"); save(fig, "03_hour_utc.png")

# 4 day/night by month
dn = pd.crosstab(df["ym"], df["daynight"])
fig, ax = plt.subplots(figsize=(14, 4))
dn.plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Day vs Night by month"); plt.xticks(rotation=60); save(fig, "04_daynight_by_month.png")

# 5 FRP hist + log
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(frp.clip(upper=frp.quantile(.999)), bins=80); ax[0].set_title("FRP histogram (clipped p99.9)")
ax[1].hist(frp[frp > 0], bins=80, log=True); ax[1].set_title("FRP histogram (log y)")
fig.suptitle("FRP distribution (MW)"); save(fig, "05_frp_hist.png")

# 6 FRP by confidence (medians + box on sample)
samp = df.sample(min(60000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(8, 4))
samp.boxplot(column="frp", by="confidence", ax=ax, showfliers=False)
ax.set_title("FRP by confidence (sample 60k, no outliers)"); ax.set_ylabel("FRP MW"); save(fig, "06_frp_by_confidence.png")

# 7 brightness
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(df["bright_ti4"], bins=80); ax[0].set_title("bright_ti4 (K)")
ax[1].hist(df["bright_ti5"], bins=80); ax[1].set_title("bright_ti5 (K)")
fig.suptitle("Brightness temperature"); save(fig, "07_brightness_hist.png")

# 8 temp diff
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(df["temp_diff"], bins=100); ax.set_title("temp_difference = bright_ti4 - bright_ti5 (K)")
ax.set_xlabel("K"); save(fig, "08_tempdiff_hist.png")

# 9 FRP vs ti4 density (2D hist, full data binned)
fig, ax = plt.subplots(figsize=(8, 5))
x = df["bright_ti4"].clip(300, 500); y = frp.clip(0, 200)
ax.hist2d(x, y, bins=[120, 120], cmin=1)
ax.set_xlabel("bright_ti4 (K)"); ax.set_ylabel("FRP (MW, clipped 200)")
ax.set_title("FRP vs bright_ti4 density"); save(fig, "09_frp_vs_ti4.png")

# 10 scan/track
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(pd.to_numeric(df["scan"], errors="coerce").clip(upper=2), bins=80); ax[0].set_title("scan (pixel size, clipped 2)")
ax[1].hist(pd.to_numeric(df["track"], errors="coerce").clip(upper=2), bins=80); ax[1].set_title("track (pixel size, clipped 2)")
fig.suptitle("Scan / Track (VIIRS nominal ~0.375 km at nadir)"); save(fig, "10_scan_track.png")

# 11 confidence bars + daynight
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
pd.Series(audit["confidence"]).plot(kind="bar", ax=ax[0], title="Confidence counts (l/n/h)")
pd.Series(audit["daynight"]).plot(kind="bar", ax=ax[1], title="Day/Night counts"); save(fig, "11_confidence_daynight.png")

# 12 spatial hexbin heatmap
fig, ax = plt.subplots(figsize=(8, 7))
ax.hist2d(df["longitude"], df["latitude"], bins=[120, 120], cmin=1)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Spatial heatmap (2D histogram, India bbox)")
ax.set_xlim(68, 97); ax.set_ylim(8, 37); save(fig, "12_spatial_heatmap.png")

# 13 top 1-deg cells
cell = (df["latitude"].round(0).astype(str) + "," + df["longitude"].round(0).astype(str))
top = cell.value_counts().head(20)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top.index[::-1], top.values[::-1]); ax.set_title("Top 20 1-deg cells by detection count")
ax.set_xlabel("detections"); save(fig, "13_top_cells.png")

# 14 missing by month (FRP/conf) — expect ~0
mf = df.assign(mfrp=df["frp"].isna(), mconf=df["confidence"].isna()).groupby("ym")[["mfrp", "mconf"]].mean() * 100
fig, ax = plt.subplots(figsize=(14, 3.5))
mf.plot(ax=ax); ax.set_title("Missing % by month (FRP, confidence)"); ax.set_ylabel("%"); save(fig, "14_missing_by_month.png")

# 15 persistence preview: distinct dates per ~1km cell
df["cell"] = (df["latitude"].round(2).astype(str) + "_" + df["longitude"].round(2).astype(str))
g = df.groupby("cell")["acq_date"].nunique()
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(g.values, bins=np.logspace(0, np.log10(max(2, g.max())), 60), log=True)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_title("Persistence preview: distinct detection-days per ~1km cell (log-log)")
ax.set_xlabel("distinct days"); ax.set_ylabel("cells"); save(fig, "15_persistence.png")
audit["persist_cells"] = int(len(g))
audit["persist_days_describe"] = {k: round(float(v), 2) for k, v in g.describe().to_dict().items()}
audit["persist_ge10days_cells"] = int((g >= 10).sum())
audit["persist_ge30days_cells"] = int((g >= 30).sum())
with open(os.path.join(OUTDIR, "audit.json"), "w") as f:
    json.dump(audit, f, indent=2, default=str)

# correlations
num = df[["latitude", "longitude", "bright_ti4", "bright_ti5", "temp_diff", "scan", "track", "frp"]].apply(pd.to_numeric, errors="coerce")
corr = num.corr(numeric_only=True).round(3)
corr.to_csv(os.path.join(OUTDIR, "correlations.csv"))
print(corr.to_string())
print(f"\nDONE. Graphs + audit.json + correlations.csv in {OUTDIR}/")
