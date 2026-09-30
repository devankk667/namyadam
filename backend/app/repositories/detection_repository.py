import hashlib
import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from app.core.config import settings

def _obs_id_fingerprint(ids) -> str:
    xor_value = 0
    sum_value = 0
    count = 0
    for observation_id in ids:
        digest = int.from_bytes(hashlib.sha256(str(observation_id).encode("utf-8")).digest(), "big")
        xor_value ^= digest
        sum_value = (sum_value + digest) % (1 << 256)
        count += 1
    return f"{count}:{xor_value:064x}:{sum_value:064x}"


def _safe_num(value: Any) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


MODEL_FEATURE_SOURCE_MAP = {
    "bright_ti4": "bright_ti4",
    "bright_ti5": "bright_ti5",
    "frp": "frp",
    "nearest_osm_distance_km": "nearest_osm_distance_km",
    "industrial_count_1km": "industrial_count_1km",
    "industrial_count_2km": "industrial_count_2km",
    "industrial_count_5km": "industrial_count_5km",
    "power_plant_count_5km": "power_plant_count_5km",
    "quarry_count_5km": "quarry_count_5km",
    "flare_count_5km": "flare_count_5km",
    "petroleum_well_count_5km": "petroleum_well_count_5km",
    "industrial_landuse_nearby": "industrial_landuse_nearby",
}


def _representative_sample(
    df: pd.DataFrame,
    sample_size: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Take a proportional class/date-stratified sample; seed for reproducibility in tests."""
    if len(df) <= sample_size:
        return df.copy()

    strata = [name for name in ("rf_predicted_class", "acq_date") if name in df.columns]
    if len(strata) == 2 and df.groupby(strata, dropna=False, sort=False).ngroups > sample_size:
        strata = strata[:1]
    if not strata:
        return df.sample(n=sample_size, random_state=seed).sort_index()

    grouped = df.groupby(strata, dropna=False, sort=False)
    sizes = grouped.size()
    exact_quotas = sizes / len(df) * sample_size
    quotas = exact_quotas.astype(int)
    remainder = sample_size - int(quotas.sum())
    if remainder:
        largest_remainders = (exact_quotas - quotas).sort_values(ascending=False).index[:remainder]
        quotas.loc[largest_remainders] += 1

    samples = [
        group.sample(n=int(quotas.loc[key]), random_state=seed)
        for key, group in grouped
        if int(quotas.loc[key]) > 0
    ]
    sampled = pd.concat(samples) if samples else df.iloc[0:0]
    if len(sampled) < sample_size:
        remaining = df.drop(index=sampled.index)
        sampled = pd.concat([
            sampled,
            remaining.sample(n=sample_size - len(sampled), random_state=seed),
        ])
    return sampled.sort_index()


def _format_acq_time(value: Any) -> str:
    """Normalize FIRMS acq_time (HHMM int/str) to HH:MM."""
    if value is None or pd.isna(value):
        return "00:00"
    if isinstance(value, str):
        value = value.strip()
        if ":" in value:
            return value
        if not value.isdigit():
            return "00:00"
        digits = value.zfill(4)
    else:
        digits = str(int(value)).zfill(4)
    return f"{digits[:2]}:{digits[2:]}"


class ThermalDetectionRepository:
    """
    Decoupled data access repository. Serves detection records from FIRMS predictions
    or demo CSV data based on settings.DEMO_MODE without requiring database dependencies.
    """
    def __init__(self):
        self._cache: List[Dict[str, Any]] = []
        self._live_cache: List[Dict[str, Any]] = []
        self._analytics_snapshot: Optional[Dict[str, Any]] = None
        self.data_source = "unavailable"
        self.data_error: Optional[str] = None
        self.total_source_records = 0
        self._source_obs_id_fingerprint: Optional[str] = None
        self.reload()

    def _event_ids_for_sample(self, sampled: pd.DataFrame) -> List[Optional[str]]:
        """Join only by durable obs_id keys; never infer identity from row position."""
        if "obs_id" not in sampled.columns:
            return [None] * len(sampled)
        if not os.path.isfile(settings.FIRMS_EVENT_MAP_PATH):
            print("[WARNING] Event IDs unavailable: durable FIRMS event map is missing.")
            return [None] * len(sampled)
        try:
            metadata_path = settings.FIRMS_EVENT_MAP_PATH + ".metadata.json"
            if not os.path.isfile(metadata_path):
                raise ValueError("event-map metadata is missing; regenerate the map with the durable-ID pipeline")
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                event_metadata = json.load(metadata_file)
            event_map = pd.read_parquet(
                settings.FIRMS_EVENT_MAP_PATH,
                columns=["obs_id", "event_id"],
            )
            event_map["obs_id"] = event_map["obs_id"].astype("string").str.strip()
            event_map["event_id"] = event_map["event_id"].astype("string").str.strip()
            source_ids = sampled["obs_id"].astype("string").str.strip()
            if (
                self.total_source_records <= 0
                or len(event_map) != self.total_source_records
                or event_map["obs_id"].isna().any()
                or event_map["obs_id"].eq("").any()
                or event_map["obs_id"].duplicated().any()
                or event_map["event_id"].isna().any()
                or event_map["event_id"].eq("").any()
                or self._source_obs_id_fingerprint is None
                or event_metadata.get("version") != 1
                or event_metadata.get("source_record_count") != self.total_source_records
                or event_metadata.get("obs_id_fingerprint") != self._source_obs_id_fingerprint
                or _obs_id_fingerprint(event_map["obs_id"]) != self._source_obs_id_fingerprint
            ):
                raise ValueError("observation map keys/fingerprint do not exactly match the loaded FIRMS source")
            lookup = event_map.set_index("obs_id")["event_id"]
            joined = source_ids.map(lookup)
            if joined.isna().any():
                raise ValueError("observation map does not cover every sampled durable obs_id")
            return joined.astype(str).tolist()
        except Exception as error:
            print(f"[WARNING] Event IDs unavailable for FIRMS sample: {type(error).__name__}: {error}")
            return [None] * len(sampled)

    @staticmethod
    def _normalize_fallback_frame(df: pd.DataFrame) -> pd.DataFrame:
        aliases = {
            "bright_ti4": "brightness",
            "bright_ti5": "bright_t31",
            "rf_predicted_class": "true_class",
            "rf_confidence": "classification_confidence",
            "nearest_osm_distance_km": "dist_to_industrial",
            "nearest_osm_type": "nearest_facility_type",
        }
        df = df.rename(columns={key: value for key, value in aliases.items() if key in df.columns and value not in df.columns}).copy()
        source_fields = set(df.columns)
        if "confidence" in df.columns:
            confidence_codes = {"n": 0, "l": 33, "m": 66, "h": 100}
            mapped = df["confidence"].astype(str).str.lower().map(confidence_codes)
            numeric = pd.to_numeric(df["confidence"], errors="coerce")
            df["confidence"] = mapped.fillna(numeric)
        numeric_defaults = {
            "brightness": 0.0, "bright_t31": 0.0, "frp": 0.0,
            "confidence": 50.0, "classification_confidence": 0.0,
            "dist_to_industrial": 99.0, "latitude": 0.0, "longitude": 0.0,
            "industrial_count_1km": 0, "industrial_count_2km": 0, "industrial_count_5km": 0,
            "power_plant_count_5km": 0, "quarry_count_5km": 0, "flare_count_5km": 0,
            "petroleum_well_count_5km": 0, "industrial_landuse_nearby": 0,
            "persistence_score": 0.0, "detection_count_30d": 0,
        }
        for column, default in numeric_defaults.items():
            if column not in df.columns:
                df[column] = default
            else:
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(default)
        for column, default in {
            "true_class": "other_thermal_anomaly", "nearest_facility_type": "unknown",
            "satellite": "unknown", "instrument": "unknown", "daynight": "D",
            "acq_date": "unknown", "acq_time": "00:00",
        }.items():
            if column not in df.columns:
                df[column] = default
            else:
                df[column] = df[column].fillna(default).astype(str)
        if "id" not in df.columns:
            df["id"] = [f"DET-FALLBACK-{index:04d}" for index in range(len(df))]
        if "region" not in df.columns:
            south_asia = df["latitude"].between(5, 38) & df["longitude"].between(60, 100)
            df["region"] = south_asia.map({True: "South Asia", False: "Unknown"})
        else:
            df["region"] = df["region"].fillna("Unknown").astype(str)
        df["acq_time"] = df["acq_time"].apply(_format_acq_time)
        if "persistence_score" not in source_fields and "cell_prior_count" in source_fields:
            prior = pd.to_numeric(df["cell_prior_count"], errors="coerce").fillna(0)
            df["detection_count_30d"] = prior.astype(int)
            df["persistence_score"] = (prior / 10.0).clip(0, 1)
        elif "persistence_score" not in source_fields and "prev_30d" in source_fields:
            prior = pd.to_numeric(df["prev_30d"], errors="coerce").fillna(0)
            df["detection_count_30d"] = prior.astype(int)
            df["persistence_score"] = (prior / 30.0).clip(0, 1)
        df["persistence_score"] = df["persistence_score"].clip(0, 1)
        available = set()
        field_to_feature = {
            "brightness": "bright_ti4", "bright_t31": "bright_ti5", "frp": "frp",
            "dist_to_industrial": "nearest_osm_distance_km",
            "industrial_count_1km": "industrial_count_1km",
            "industrial_count_2km": "industrial_count_2km",
            "industrial_count_5km": "industrial_count_5km",
            "power_plant_count_5km": "power_plant_count_5km",
            "quarry_count_5km": "quarry_count_5km", "flare_count_5km": "flare_count_5km",
            "petroleum_well_count_5km": "petroleum_well_count_5km",
            "industrial_landuse_nearby": "industrial_landuse_nearby",
        }
        available.update(feature for field, feature in field_to_feature.items() if field in source_fields)
        if {"brightness", "bright_t31"} <= source_fields:
            available.add("temp_diff")
        if "confidence" in source_fields:
            available.add("conf_ord")
        if "daynight" in source_fields:
            available.add("is_night")
        df["_model_features_available"] = [sorted(available)] * len(df)
        return df

    @staticmethod
    def _build_analytics_snapshot(df: pd.DataFrame) -> Dict[str, Any]:
        """Reduce the complete FIRMS frame to compact dashboard aggregates."""
        frame = df
        class_col = "rf_predicted_class" if "rf_predicted_class" in frame.columns else "true_class"
        classes = frame[class_col].fillna("other_thermal_anomaly").astype(str) if class_col in frame else pd.Series("other_thermal_anomaly", index=frame.index)
        frp = pd.to_numeric(frame.get("frp", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0)
        confidence = frame.get("confidence", pd.Series(0.0, index=frame.index))
        confidence_codes = confidence.astype(str).str.lower().map({"n": 0, "l": 33, "m": 66, "h": 100})
        confidence = confidence_codes.fillna(pd.to_numeric(confidence, errors="coerce")).fillna(0)
        persistence = frame.get("persistence_score")
        if persistence is None:
            if "cell_prior_count" in frame:
                persistence = pd.to_numeric(frame["cell_prior_count"], errors="coerce").fillna(0).div(10).clip(0, 1)
            elif "prev_30d" in frame:
                persistence = pd.to_numeric(frame["prev_30d"], errors="coerce").fillna(0).div(30).clip(0, 1)
            else:
                persistence = pd.Series(0.0, index=frame.index)
        else:
            persistence = pd.to_numeric(persistence, errors="coerce").fillna(0).clip(0, 1)
        distance = pd.to_numeric(frame.get("nearest_osm_distance_km", pd.Series(99.0, index=frame.index)), errors="coerce").fillna(99)
        industrial_count = pd.to_numeric(frame.get("industrial_count_5km", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
        rf_conf = pd.to_numeric(frame.get("rf_confidence", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0)
        rf_conf = rf_conf.where(rf_conf <= 1, rf_conf / 100.0)
        industrial_class = classes.eq("industrial_fire")
        high_risk = (frp >= 50) | (industrial_class & (rf_conf >= 0.66))
        summary = {
            "total_detections": int(len(frame)),
            "industrial_associated_detections": int(((distance < 1.0) | (industrial_count > 0)).sum()),
            "active_thermal_sources": int(classes.isin(["industrial_fire", "gas_flare"]).sum()),
            "persistent_sources": int((persistence >= 0.1).sum()),
            "high_confidence_events": int((confidence >= 90).sum()),
            "high_risk_events": int(high_risk.sum()),
        }

        date_col = frame.get("acq_date", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
        temporal_frame = pd.DataFrame({"acq_date": date_col, "frp": frp, "industrial": classes.isin(["industrial_fire", "gas_flare"])})
        temporal = temporal_frame.groupby("acq_date", dropna=False).agg(
            total_events=("acq_date", "size"), avg_frp=("frp", "mean"), industrial_events=("industrial", "sum")
        ).reset_index()
        temporal["avg_frp"] = temporal["avg_frp"].round(2)

        class_counts = classes.replace({
            "agricultural_burning": "agricultural_burning", "industrial_fire": "industrial_fire",
            "gas_flare": "gas_flare", "mining_activity": "mining_activity",
        }).value_counts()
        classification = [
            {"classification": str(name), "count": int(count), "percentage": round(float(count) / len(frame) * 100, 2)}
            for name, count in class_counts.items()
        ] if len(frame) else []

        if "region" in frame:
            regions = frame["region"].fillna("Unknown").astype(str)
        else:
            lat = pd.to_numeric(frame.get("latitude", pd.Series(np.nan, index=frame.index)), errors="coerce")
            lon = pd.to_numeric(frame.get("longitude", pd.Series(np.nan, index=frame.index)), errors="coerce")
            regions = pd.Series(np.where(lat.between(5, 38) & lon.between(60, 100), "South Asia", "Unknown"), index=frame.index)
        region_frame = pd.DataFrame({"region": regions, "persistence": persistence, "high_risk": high_risk})
        regional = region_frame.groupby("region", dropna=False).agg(
            total_detections=("region", "size"), avg_persistence=("persistence", "mean"), high_risk_count=("high_risk", "sum")
        ).reset_index()
        regional["avg_persistence"] = regional["avg_persistence"].round(3)

        persist_mask = persistence >= 0.1
        top_indices = persistence[persist_mask].nlargest(20).index
        persistent_items = []
        for index in top_indices:
            persistent_items.append({
                "detection_id": str(frame.at[index, "obs_id"] if "obs_id" in frame else frame.at[index, "id"] if "id" in frame else index),
                "region": str(regions.loc[index]),
                "latitude": _safe_num(frame.at[index, "latitude"] if "latitude" in frame else 0),
                "longitude": _safe_num(frame.at[index, "longitude"] if "longitude" in frame else 0),
                "persistence_score": float(persistence.loc[index]),
                "nearest_facility": str(frame.at[index, "nearest_osm_type"] if "nearest_osm_type" in frame else "unknown"),
                "detection_count_30d": int(_safe_num(frame.at[index, "cell_prior_count"] if "cell_prior_count" in frame else frame.at[index, "prev_30d"] if "prev_30d" in frame else 1)),
                "status": "ACTIVE_OPERATIONAL_FLARE" if persistence.loc[index] > 0.8 else "RECURRING_INDUSTRIAL",
            })
        return {
            "summary": summary,
            "temporal": temporal.sort_values("acq_date").to_dict(orient="records"),
            "classification": classification,
            "regions": regional.sort_values("total_detections", ascending=False).to_dict(orient="records"),
            "persistence": persistent_items,
            "source_record_count": int(len(frame)),
        }

    def get_analytics_snapshot(self) -> Optional[Dict[str, Any]]:
        return self._analytics_snapshot

    def add_live_records(self, records: List[Dict[str, Any]]) -> None:
        known = {str(row.get("id")) for row in self._cache}
        known.update(str(row.get("id")) for row in self._live_cache)
        for row in records:
            if str(row.get("id")) not in known:
                self._live_cache.append(row)
                known.add(str(row.get("id")))
        self._live_cache.sort(key=lambda row: (str(row.get("acq_date", "")), str(row.get("acq_time", ""))), reverse=True)

    def reload(self):
        """Load a random but class/date-stratified sample from the preferred source."""
        self._cache = []
        self._live_cache = []
        self._analytics_snapshot = None
        self.data_source = "unavailable"
        self.data_error = None
        self.total_source_records = 0
        self._source_obs_id_fingerprint = None

        # Prefer the real FIRMS prediction set when available.
        if os.path.exists(settings.FIRMS_PREDICTIONS_PATH):
            try:
                print(f"Loading FIRMS predictions from {settings.FIRMS_PREDICTIONS_PATH}")
                # Read only the columns needed by UI, analytics, and inference to
                # keep the 1.1M-row parquet footprint modest.
                cols_to_use = [
                    'obs_id', 'bright_ti4', 'bright_ti5', 'frp', 'confidence',
                    'nearest_osm_distance_km', 'nearest_osm_type',
                    'industrial_count_1km', 'industrial_count_2km', 'industrial_count_5km',
                    'power_plant_count_5km', 'quarry_count_5km', 'flare_count_5km',
                    'petroleum_well_count_5km', 'industrial_landuse_nearby',
                    'rf_predicted_class', 'rf_confidence', 'daynight',
                    'acq_date', 'acq_time', 'latitude', 'longitude',
                    'satellite', 'instrument', 'cell_prior_count', 'prev_30d',
                ]
                import pyarrow.parquet as parquet

                parquet_file = parquet.ParquetFile(settings.FIRMS_PREDICTIONS_PATH)
                self.total_source_records = parquet_file.metadata.num_rows
                cols_available = [c for c in cols_to_use if c in parquet_file.schema.names]
                df = pd.read_parquet(settings.FIRMS_PREDICTIONS_PATH, columns=cols_available)
                if df.empty:
                    raise ValueError("FIRMS predictions parquet contains no rows.")
                model_features_present = set(cols_available)
                if "obs_id" in df.columns:
                    source_ids = df["obs_id"].astype("string").str.strip()
                    if source_ids.notna().all() and source_ids.ne("").all() and not source_ids.duplicated().any():
                        self._source_obs_id_fingerprint = _obs_id_fingerprint(source_ids)
                    else:
                        print("[WARNING] Source obs_id values are missing or duplicated; fusion event joins are disabled.")
                else:
                    print("[WARNING] FIRMS prediction parquet has no durable obs_id column; fusion event joins are disabled.")
                self._analytics_snapshot = self._build_analytics_snapshot(df)
                sample_size = min(max(1, settings.FIRMS_SAMPLE_SIZE), len(df))
                sampled = _representative_sample(df, sample_size)
                df_subset = sampled[cols_available].copy()
                df_subset["event_id"] = self._event_ids_for_sample(sampled)

                # Rename columns to match expected schema
                column_mapping = {
                    'bright_ti4': 'brightness',
                    'bright_ti5': 'bright_t31',
                    'rf_predicted_class': 'true_class',
                    'rf_confidence': 'classification_confidence',
                    'nearest_osm_distance_km': 'dist_to_industrial',
                    'nearest_osm_type': 'nearest_facility_type',
                    'cell_prior_count': 'cell_prior_count',
                }
                df_subset = df_subset.rename(columns=column_mapping)
                available_model_features = {
                    MODEL_FEATURE_SOURCE_MAP[source_name]
                    for source_name in model_features_present
                    if source_name in MODEL_FEATURE_SOURCE_MAP
                }
                if {"bright_ti4", "bright_ti5"} <= model_features_present:
                    available_model_features.add("temp_diff")
                if "confidence" in model_features_present:
                    available_model_features.add("conf_ord")
                if "daynight" in model_features_present:
                    available_model_features.add("is_night")
                df_subset["_model_features_available"] = [sorted(available_model_features)] * len(df_subset)

                # Convert FIRMS ordinal confidence codes while also tolerating
                # already-numeric confidence columns from transformed datasets.
                conf_mapping = {'n': 0, 'l': 33, 'm': 66, 'h': 100}
                if 'confidence' in df_subset.columns:
                    mapped_confidence = df_subset['confidence'].astype(str).str.lower().map(conf_mapping)
                    numeric_confidence = pd.to_numeric(df_subset['confidence'], errors='coerce')
                    df_subset['confidence'] = mapped_confidence.fillna(numeric_confidence).fillna(50)

                # Stable source-row IDs make sampled detections identifiable across restarts.
                if 'obs_id' in df_subset.columns and df_subset['obs_id'].notna().all():
                    df_subset['obs_id'] = df_subset['obs_id'].astype(str)
                    df_subset['id'] = df_subset['obs_id']
                else:
                    df_subset['id'] = [f"DET-FIRMS-LEGACY-{int(index):07d}" for index in df_subset.index]


                # Format acquisition time from FIRMS HHMM integer
                if 'acq_time' in df_subset.columns:
                    df_subset['acq_time'] = df_subset['acq_time'].apply(_format_acq_time)
                else:
                    df_subset['acq_time'] = '00:00'

                # Persistence counts remain counts until normalized below.
                if 'cell_prior_count' in df.columns:
                    prior = pd.to_numeric(df.loc[df_subset.index, 'cell_prior_count'], errors='coerce').fillna(0)
                    df_subset['detection_count_30d'] = prior.astype(int)
                    df_subset['persistence_score'] = (prior / 10.0).clip(0, 1)
                elif 'prev_30d' in df_subset.columns:
                    prior = pd.to_numeric(df_subset['prev_30d'], errors='coerce').fillna(0)
                    df_subset['detection_count_30d'] = prior.astype(int)
                    df_subset['persistence_score'] = (prior / 30.0).clip(0, 1)
                else:
                    df_subset['persistence_score'] = 0.0
                    df_subset['detection_count_30d'] = 0

                # Ensure fields used by the UI and model always exist and are numeric.
                if 'region' not in df_subset.columns:
                    latitude = pd.to_numeric(
                        df_subset.get('latitude', pd.Series(index=df_subset.index, dtype=float)),
                        errors='coerce',
                    )
                    longitude = pd.to_numeric(
                        df_subset.get('longitude', pd.Series(index=df_subset.index, dtype=float)),
                        errors='coerce',
                    )
                    in_south_asia = latitude.between(5, 38) & longitude.between(60, 100)
                    df_subset['region'] = in_south_asia.map({True: 'South Asia', False: 'Unknown'})
                defaults = {
                    'brightness': 0.0, 'bright_t31': 0.0, 'frp': 0.0,
                    'confidence': 50.0, 'dist_to_industrial': 99.0,
                    'industrial_count_1km': 0, 'industrial_count_2km': 0, 'industrial_count_5km': 0,
                    'power_plant_count_5km': 0, 'quarry_count_5km': 0, 'flare_count_5km': 0,
                    'petroleum_well_count_5km': 0, 'industrial_landuse_nearby': 0,
                    'classification_confidence': 0.0, 'latitude': 0.0, 'longitude': 0.0,
                    'satellite': 'unknown', 'instrument': 'unknown', 'daynight': 'D',
                    'true_class': 'other_thermal_anomaly', 'nearest_facility_type': 'unknown',
                }
                for col, default in defaults.items():
                    if col not in df_subset.columns:
                        df_subset[col] = default
                    elif isinstance(default, str):
                        df_subset[col] = df_subset[col].fillna(default).astype(str)
                for col in ('brightness', 'bright_t31', 'frp', 'confidence', 'dist_to_industrial',
                            'classification_confidence', 'latitude', 'longitude'):
                    df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce').fillna(defaults[col])

                # Round floats and normalize integer fields for cleaner JSON payloads.
                for col in ('brightness', 'bright_t31', 'frp', 'dist_to_industrial', 'classification_confidence', 'persistence_score'):
                    if col in df_subset.columns:
                        df_subset[col] = df_subset[col].round(4)
                for col in ('industrial_count_1km', 'industrial_count_2km', 'industrial_count_5km',
                            'power_plant_count_5km', 'quarry_count_5km', 'flare_count_5km',
                            'petroleum_well_count_5km', 'industrial_landuse_nearby', 'detection_count_30d'):
                    if col in df_subset.columns:
                        df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce').fillna(0).astype(int)

                # Drop internal columns not needed in API responses
                df_subset = df_subset.drop(columns=['cell_prior_count', 'prev_30d'], errors='ignore')

                self._cache = df_subset.to_dict(orient="records")
                self.data_source = "firms_predictions"
                print(f"[OK] Loaded representative sample of {len(self._cache)} from {self.total_source_records} FIRMS records")
                return
            except Exception as e:
                self.data_error = f"FIRMS predictions: {type(e).__name__}: {e}"
                print(f"Warning: Could not load FIRMS predictions: {self.data_error}")

        # Fallback to demo data or processed data
        if settings.REQUIRE_REAL_DATA:
            self.data_source = "unavailable"
            self.data_error = self.data_error or "Real FIRMS prediction data is required but unavailable."
            print(f"[ERROR] {self.data_error}")
            return

        if os.path.exists(settings.PROCESSED_DATA_PATH):
            try:
                print(f"Loading from processed data: {settings.PROCESSED_DATA_PATH}")
                df = pd.read_csv(settings.PROCESSED_DATA_PATH)
                df = self._normalize_fallback_frame(df)
                self._cache = df.to_dict(orient="records")
                self.total_source_records = len(self._cache)
                self.data_source = "processed_csv"
                print(f"[OK] Loaded {len(self._cache)} records from processed data")
                return
            except Exception as e:
                self.data_error = f"Processed CSV: {type(e).__name__}: {e}"
                print(f"Warning: Could not load processed data: {self.data_error}")

        # Last resort: demo data
        if os.path.exists(settings.DEMO_DATA_PATH):
            try:
                print(f"Loading demo data: {settings.DEMO_DATA_PATH}")
                with open(settings.DEMO_DATA_PATH, "r") as f:
                    demo_data = json.load(f)
                self._cache = self._normalize_fallback_frame(pd.DataFrame(demo_data)).to_dict(orient="records")
                self.total_source_records = len(self._cache)
                self.data_source = "demo_json"
                print(f"[OK] Loaded {len(self._cache)} demo records")
                return
            except Exception as e:
                self.data_error = f"Demo JSON: {type(e).__name__}: {e}"
                print(f"Warning: Could not load demo data: {self.data_error}")

        # No source is silently treated as healthy; health reports unavailable.
        if not self._cache:
            self.data_source = "unavailable"
            self.data_error = self.data_error or "No configured data source could be loaded."
            print(f"[ERROR] No data sources available: {self.data_error}")

    def get_all(self) -> List[Dict[str, Any]]:
        return self._cache

    def get_filtered(
        self,
        min_confidence: Optional[float] = None,
        min_frp: Optional[float] = None,
        classification: Optional[str] = None,
        region: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        records = self._live_cache + self._cache
        def numeric(record: Dict[str, Any], field: str, default: float = 0.0) -> float:
            try:
                return float(record.get(field, default) if record.get(field, default) is not None else default)
            except (TypeError, ValueError):
                return default

        if min_confidence is not None:
            records = [r for r in records if numeric(r, "confidence") >= min_confidence]
        if min_frp is not None:
            records = [r for r in records if numeric(r, "frp") >= min_frp]
        if classification:
            records = [r for r in records if r.get("true_class") == classification]
        if region:
            records = [r for r in records if region.lower() in str(r.get("region", "")).lower()]

        total = len(records)
        start = (page - 1) * page_size
        end = start + page_size
        paged_items = [
            {key: value for key, value in record.items() if key not in {"_model_features_available", "_raw_firms_record"}}
            for record in records[start:end]
        ]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": paged_items
        }

    def get_by_id(self, detection_id: str) -> Optional[Dict[str, Any]]:
        target = str(detection_id)
        for r in self._live_cache + self._cache:
            if str(r.get("obs_id", "")) == target or str(r.get("id", "")) == target:
                return r
        return None

    def get_raw_list(self) -> List[Dict[str, Any]]:
        return self._live_cache + self._cache

detection_repo = ThermalDetectionRepository()
