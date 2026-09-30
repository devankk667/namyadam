import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("BUILDING EVENT-LEVEL TRAINING DATASET")
print("=" * 80)

BASE = Path("data/processed")

MAP_PATH = BASE / "firms_obs_event_map.parquet"
TEMPORAL_PATH = BASE / "firms_temporal.parquet"
OSM_PATH = BASE / "firms_osm_enriched.parquet"
LABEL_PATH = BASE / "firms_event_labels.parquet"

OUTPUT_PATH = BASE / "event_training_table.parquet"


# ------------------------------------------------------------------
# 1. LOAD OBSERVATION → EVENT MAPPING
# ------------------------------------------------------------------

print("\n[1/5] Loading observation-event mapping...")

mapping = pd.read_parquet(MAP_PATH)

print(f"Mapping shape: {mapping.shape}")
print(f"Unique observations: {mapping.obs_id.nunique():,}")
print(f"Unique events: {mapping.event_id.nunique():,}")


# ------------------------------------------------------------------
# 2. LOAD TEMPORAL FEATURES
# ------------------------------------------------------------------

print("\n[2/5] Loading temporal features...")

temporal = pd.read_parquet(TEMPORAL_PATH)

print(f"Temporal shape: {temporal.shape}")

temporal_cols = [
    "obs_id",
    "prev_7d",
    "prev_14d",
    "prev_30d",
    "time_since_prev_days",
    "days_since_first_seen",
    "cell_prior_count",
    "recurrence_per_day",
    "frp_prior_mean",
    "frp_prior_std",
    "frp_trend_slope",
    "event_prev_count",
]

temporal = temporal[temporal_cols]


# ------------------------------------------------------------------
# 3. LOAD FIRMS + OSM FEATURES
# ------------------------------------------------------------------

print("\n[3/5] Loading FIRMS + OSM data...")

osm = pd.read_parquet(OSM_PATH)

print(f"OSM shape: {osm.shape}")

# IMPORTANT:
# The OSM dataframe index corresponds exactly to obs_id.
osm = osm.reset_index().rename(columns={"index": "obs_id"})

print(f"OSM obs_id unique: {osm.obs_id.nunique():,}")


numeric_osm = [
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "bright_ti5",
    "frp",
    "nearest_osm_distance_km",
    "industrial_count_1km",
    "industrial_count_2km",
    "industrial_count_5km",
    "power_plant_count_5km",
    "quarry_count_5km",
    "flare_count_5km",
    "petroleum_well_count_5km",
    "industrial_landuse_nearby",
]

categorical_osm = [
    "confidence",
    "daynight",
    "nearest_osm_type",
    "nearest_osm_kind",
]

osm_cols = ["obs_id"] + numeric_osm + categorical_osm

osm = osm[osm_cols]


# ------------------------------------------------------------------
# 4. MERGE OBSERVATION-LEVEL DATA
# ------------------------------------------------------------------

print("\n[4/5] Merging observation-level features...")

data = mapping.merge(
    temporal,
    on="obs_id",
    how="left",
    validate="one_to_one",
)

print(f"After temporal merge: {data.shape}")

data = data.merge(
    osm,
    on="obs_id",
    how="left",
    validate="one_to_one",
)

print(f"After OSM merge: {data.shape}")


# ------------------------------------------------------------------
# 5. AGGREGATE OBSERVATIONS → EVENTS
# ------------------------------------------------------------------

print("\n[5/5] Aggregating observations into events...")


# Basic event information
event_basic = data.groupby("event_id").agg(
    observation_count=("obs_id", "size"),
)


# FIRMS + OSM numerical features
firms_numeric = [
    "bright_ti4",
    "scan",
    "track",
    "bright_ti5",
    "frp",
    "nearest_osm_distance_km",
    "industrial_count_1km",
    "industrial_count_2km",
    "industrial_count_5km",
    "power_plant_count_5km",
    "quarry_count_5km",
    "flare_count_5km",
    "petroleum_well_count_5km",
    "industrial_landuse_nearby",
]

firms_agg = data.groupby("event_id")[firms_numeric].agg(
    ["mean", "max", "std"]
)

firms_agg.columns = [
    f"{col}_{stat}"
    for col, stat in firms_agg.columns
]


# Total FRP over the event
frp_sum = (
    data.groupby("event_id")["frp"]
    .sum()
    .rename("frp_sum")
)


# Temporal features
temporal_mean_max = [
    "prev_7d",
    "prev_14d",
    "prev_30d",
    "cell_prior_count",
    "recurrence_per_day",
    "frp_prior_mean",
    "frp_prior_std",
    "event_prev_count",
]

temporal_agg = data.groupby("event_id")[temporal_mean_max].agg(
    ["mean", "max"]
)

temporal_agg.columns = [
    f"{col}_{stat}"
    for col, stat in temporal_agg.columns
]


# Time gap between observations
data = data.copy()

data["valid_gap"] = data["time_since_prev_days"].where(
    data["time_since_prev_days"] >= 0
)

gap_agg = data.groupby("event_id")["valid_gap"].agg(
    ["mean", "median", "max"]
)

gap_agg.columns = [
    "time_gap_mean",
    "time_gap_median",
    "time_gap_max",
]


# Historical span
history_agg = (
    data.groupby("event_id")["days_since_first_seen"]
    .max()
    .rename("history_span_max")
)


# FRP trend
trend_agg = data.groupby("event_id")["frp_trend_slope"].agg(
    ["mean", "max", "min"]
)

trend_agg.columns = [
    "frp_trend_mean",
    "frp_trend_max",
    "frp_trend_min",
]


# Last temporal state
data_sorted = data.sort_values(["event_id", "obs_id"])

last_temporal = data_sorted.groupby("event_id").agg(
    frp_trend_last=("frp_trend_slope", "last"),
    recurrence_per_day_last=("recurrence_per_day", "last"),
    event_prev_count_last=("event_prev_count", "last"),
)


# Categorical features
def mode_or_unknown(series):
    series = series.dropna()

    if len(series) == 0:
        return "unknown"

    mode = series.mode()

    if len(mode) == 0:
        return "unknown"

    return mode.iloc[0]


categorical_agg = data.groupby("event_id").agg(
    confidence_mode=("confidence", mode_or_unknown),
    daynight_mode=("daynight", mode_or_unknown),
    nearest_osm_type_mode=("nearest_osm_type", mode_or_unknown),
    nearest_osm_kind_mode=("nearest_osm_kind", mode_or_unknown),
)


# ------------------------------------------------------------------
# COMBINE ALL EVENT FEATURES
# ------------------------------------------------------------------

event_data = event_basic.join(firms_agg)
event_data = event_data.join(frp_sum)
event_data = event_data.join(temporal_agg)
event_data = event_data.join(gap_agg)
event_data = event_data.join(history_agg)
event_data = event_data.join(trend_agg)
event_data = event_data.join(last_temporal)
event_data = event_data.join(categorical_agg)

event_data = event_data.reset_index()


# ------------------------------------------------------------------
# ADD EVENT LABELS
# ------------------------------------------------------------------

print("\nLoading event labels...")

labels = pd.read_parquet(LABEL_PATH)

labels = labels[
    [
        "event_id",
        "label",
        "label_confidence",
        "label_source",
    ]
]

# Keep ONLY classified events.
labels = labels[labels["label"] != "unknown"].copy()

print(f"Labelled events: {len(labels):,}")


event_data = event_data.merge(
    labels,
    on="event_id",
    how="inner",
    validate="one_to_one",
)



# ------------------------------------------------------------------
# FINAL CHECKS
# ------------------------------------------------------------------

print("\n" + "=" * 80)
print("FINAL EVENT DATASET")
print("=" * 80)

print(f"Shape: {event_data.shape}")
print(f"Unique events: {event_data.event_id.nunique():,}")


print("\nLabels:")
print(
    event_data["label"]
    .value_counts()
    .to_string()
)


print("\nMissing values:")

print(
    event_data.isna()
    .sum()
    .sort_values(ascending=False)
    .head(20)
    .to_string()
)


print("\nFeature columns:")

for i, col in enumerate(event_data.columns, 1):
    print(f"{i:3d}. {col}")


# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------

event_data.to_parquet(
    OUTPUT_PATH,
    index=False,
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nDone.")