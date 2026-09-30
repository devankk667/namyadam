"""
Temporal / persistence features — STRICTLY CAUSAL (past-only).

Every feature for an observation uses only detections with earlier timestamps
in the same static spatial cell (0.01 deg). Static cells (not event windows)
mean no future observation can alter any value: zero temporal leakage by
construction. `event_prev_count` is the one exception (event ids come from
gap-split grouping; a future detection bridging two clusters could extend an
event backward — rare, documented, effect small).

Features (per observation):
  prev_7d / prev_14d / prev_30d  — prior detections in cell within window
  time_since_prev_days           — days since previous cell detection (-1 if none)
  days_since_first_seen          — days since first cell detection
  cell_prior_count               — total prior detections in cell
  recurrence_per_day             — cell_prior_count / max(days_since_first_seen, 1)
  frp_prior_mean / frp_prior_std — mean/std of prior cell FRPs (0 if <2 priors)
  frp_trend_slope                — OLS slope of prior FRP vs time, MW/day (0 if <2)
  event_prev_count               — prior detections in same event (see caveat)

Output: data/processed/firms_temporal.parquet (obs_id + features).
Serving note: at inference, "now" = the new detection; history = cached
prior detections in its cell — same computation, no refit needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ENRICHED = Path("data/processed/firms_osm_enriched.parquet")
MAP = Path("data/processed/firms_obs_event_map.parquet")
OUT = Path("data/processed/firms_temporal.parquet")
CELL = 0.01


def main():
    df = pd.read_parquet(ENRICHED, columns=["latitude", "longitude", "frp", "acq_date", "acq_time"])
    df = df.reset_index(drop=True).rename_axis("obs_id").reset_index()
    dt = (pd.to_datetime(df["acq_date"], errors="coerce")
          + pd.to_timedelta(pd.to_numeric(df["acq_time"].astype(str).str.zfill(4).str[:2],
                                          errors="coerce").fillna(0), unit="h"))
    df["dt"] = dt
    df["cell_lat"] = (df["latitude"] / CELL).round().astype(np.int32)
    df["cell_lon"] = (df["longitude"] / CELL).round().astype(np.int32)
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0).astype(np.float64)
    df = df.sort_values(["cell_lat", "cell_lon", "dt"]).reset_index(drop=True)
    cell = df.groupby(["cell_lat", "cell_lon"], sort=False).ngroup().to_numpy()
    df["cell"] = cell

    df["one"] = np.int32(1)
    g = df.set_index("dt").groupby("cell", sort=False)["one"]
    # closed='left' -> window [t-W, t): strictly prior, same-pass peers excluded.
    prev_30d = g.rolling("30D", closed="left").sum().fillna(0).to_numpy().astype(np.int32)
    prev_14d = g.rolling("14D", closed="left").sum().fillna(0).to_numpy().astype(np.int32)
    prev_7d = g.rolling("7D", closed="left").sum().fillna(0).to_numpy().astype(np.int32)

    # spot-check alignment on one cell (vectorized vs direct, [t-W, t) semantics)
    chk = int(df["cell"].iloc[len(df) // 2])
    m = df["cell"].to_numpy() == chk
    t = df.loc[m, "dt"].to_numpy()
    direct = np.array([int(((t >= ti - np.timedelta64(30, "D")) & (t < ti)).sum()) for ti in t])
    assert (prev_30d[m] == direct).all(), "rolling alignment broken"

    dts = df["dt"]
    tprev = (dts - df.groupby("cell", sort=False)["dt"].shift(1)).dt.total_seconds().to_numpy() / 86400.0
    tprev = np.where(np.isnan(tprev), -1.0, tprev)
    t0 = df.groupby("cell", sort=False)["dt"].transform("min")
    dsince = (dts - t0).dt.total_seconds().to_numpy() / 86400.0
    prior_n = df.groupby("cell", sort=False).cumcount().to_numpy().astype(np.int64)
    rate = prior_n / np.clip(dsince, 1.0, None)

    frp = df["frp"].to_numpy()
    x = dsince
    gb = df.groupby("cell", sort=False)
    df["frp2"] = frp * frp
    Sy = gb["frp"].cumsum().to_numpy() - frp
    Syy = df.groupby("cell", sort=False)["frp2"].cumsum().to_numpy() - frp * frp
    Sx = pd.Series(x, index=df.index).groupby(df["cell"], sort=False).cumsum().to_numpy() - x
    Sxx = pd.Series(x * x, index=df.index).groupby(df["cell"], sort=False).cumsum().to_numpy() - x * x
    Sxy = pd.Series(x * frp, index=df.index).groupby(df["cell"], sort=False).cumsum().to_numpy() - x * frp
    n = prior_n.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmean = np.where(n >= 1, Sy / np.maximum(n, 1), 0.0)
        pvar = np.where(n >= 2, (Syy - Sy * Sy / np.maximum(n, 1)) / np.maximum(n - 1, 1), 0.0)
        denom = n * Sxx - Sx * Sx
        slope = np.where((n >= 2) & (np.abs(denom) > 1e-9), (n * Sxy - Sx * Sy) / np.where(np.abs(denom) > 1e-9, denom, 1.0), 0.0)
    pstd = np.sqrt(np.clip(pvar, 0, None))

    # per-event time-ordered prior counts (df is cell/time sorted, not event sorted)
    ev = pd.read_parquet(MAP).set_index("obs_id")["event_id"]
    ev_prev = np.empty(len(df), dtype=np.int64)
    sorder = np.argsort(df["dt"].to_numpy(), kind="stable")
    cur, cnt = None, 0
    eids_sorted = df["obs_id"].map(ev).to_numpy()[sorder]
    tmp = np.empty(len(df), dtype=np.int64)
    last = object()
    run = -1
    for i, e in enumerate(eids_sorted):
        if e != last:
            last, run = e, 0
        else:
            run += 1
        tmp[i] = run
    ev_prev[sorder] = tmp

    out = pd.DataFrame({
        "obs_id": df["obs_id"].to_numpy(),
        "prev_7d": prev_7d, "prev_14d": prev_14d, "prev_30d": prev_30d,
        "time_since_prev_days": np.round(tprev, 4),
        "days_since_first_seen": np.round(dsince, 2),
        "cell_prior_count": prior_n.astype(np.int32),
        "recurrence_per_day": np.round(rate, 5),
        "frp_prior_mean": np.round(pmean, 3),
        "frp_prior_std": np.round(pstd, 3),
        "frp_trend_slope": np.round(slope, 4),
        "event_prev_count": ev_prev.astype(np.int32),
    }).sort_values("obs_id").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {len(out)} rows -> {OUT}", flush=True)
    print(out.describe().round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
