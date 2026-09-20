import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/firms_osm_enriched.parquet",
                     columns=["latitude", "longitude", "acq_date", "acq_time"])
dt = (pd.to_datetime(df["acq_date"], errors="coerce")
      + pd.to_timedelta(pd.to_numeric(df["acq_time"].astype(str).str.zfill(4).str[:2],
                                      errors="coerce").fillna(0), unit="h"))
print("NaT:", int(dt.isna().sum()), "rows:", len(df))
df = df.assign(dt=dt)
df["clat"] = (df["latitude"] / 0.01).round().astype(np.int32)
df["clon"] = (df["longitude"] / 0.01).round().astype(np.int32)
df = df.sort_values(["clat", "clon", "dt"]).reset_index(drop=True)
df["cell"] = df.groupby(["clat", "clon"], sort=False).ngroup()
df["one"] = np.int32(1)
g = df.set_index("dt").groupby("cell", sort=False)["one"]
r = g.rolling("30D").sum()
print("rolling len:", len(r), "frame len:", len(df))
print("index levels:", r.index.names)
# check order: rolling output (cell major) vs frame order
cells_frame = df["cell"].to_numpy()
cells_roll = r.index.get_level_values(0).to_numpy()
print("group order matches frame:", (cells_roll == cells_frame).all())
if not (cells_roll == cells_frame).all():
    mis = np.where(cells_roll != cells_frame)[0][:5]
    print("first mismatched positions:", mis)
    print("frame:", cells_frame[mis])
    print("roll :", cells_roll[mis])
