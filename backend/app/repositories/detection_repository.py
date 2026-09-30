import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from app.core.config import settings

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
        self.data_source = "unavailable"
        self.data_error: Optional[str] = None
        self.total_source_records = 0
        self.reload()

    def _event_ids_for_sample(self, sampled: pd.DataFrame) -> List[Optional[str]]:
        """Join the sampled source row IDs to event IDs only when the map is exact."""
        if not os.path.isfile(settings.FIRMS_EVENT_MAP_PATH):
            return [None] * len(sampled)
        try:
            event_map = pd.read_parquet(
                settings.FIRMS_EVENT_MAP_PATH,
                columns=["obs_id", "event_id"],
            )
            obs_ids = pd.to_numeric(event_map["obs_id"], errors="coerce")
            event_ids = event_map["event_id"]
            if (
                self.total_source_records <= 0
                or len(event_map) != self.total_source_records
                or obs_ids.isna().any()
                or (obs_ids % 1 != 0).any()
                or obs_ids.duplicated().any()
                or int(obs_ids.min()) != 0
                or int(obs_ids.max()) != self.total_source_records - 1
                or event_ids.isna().any()
                or event_ids.astype(str).str.strip().eq("").any()
            ):
                raise ValueError("observation map does not exactly cover the FIRMS source rows")
            event_map["obs_id"] = obs_ids.astype(int)
            event_lookup = event_map.set_index("obs_id")["event_id"]
            event_ids = event_lookup.reindex(sampled.index).tolist()
            if len(event_ids) != len(sampled):
                raise ValueError("observation map did not cover the sampled rows")
            return event_ids
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

    def reload(self):
        """Load a random but class/date-stratified sample from the preferred source."""
        self._cache = []
        self.data_source = "unavailable"
        self.data_error = None
        self.total_source_records = 0

        # Prefer the real FIRMS prediction set when available.
        if os.path.exists(settings.FIRMS_PREDICTIONS_PATH):
            try:
                print(f"Loading FIRMS predictions from {settings.FIRMS_PREDICTIONS_PATH}")
                # Read only the columns needed by UI, analytics, and inference to
                # keep the 1.1M-row parquet footprint modest.
                cols_to_use = [
                    'bright_ti4', 'bright_ti5', 'frp', 'confidence',
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
                df_subset['id'] = [f"DET-FIRMS-{int(index):07d}" for index in df_subset.index]


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
        records = self._cache
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
            {key: value for key, value in record.items() if key != "_model_features_available"}
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
        for r in self._cache:
            if str(r.get("obs_id", "")) == target or str(r.get("id", "")) == target:
                return r
        return None

    def get_raw_list(self) -> List[Dict[str, Any]]:
        return self._cache

detection_repo = ThermalDetectionRepository()
