from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from glob import glob
from io import StringIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

from app.core.config import settings
from app.schemas.schemas import PredictionInput
from app.services.model_service import model_service

EARTH_KM = 6371.0088
MODEL_FEATURES = [
    "bright_ti4", "bright_ti5", "temp_diff", "frp", "conf_ord", "is_night",
    "nearest_osm_distance_km", "industrial_count_1km", "industrial_count_2km",
    "industrial_count_5km", "power_plant_count_5km", "quarry_count_5km",
    "flare_count_5km", "petroleum_well_count_5km", "industrial_landuse_nearby",
]


def stable_observation_id(row: Dict[str, Any], source: str) -> str:
    """Return a content-derived FIRMS identity that does not depend on row order."""
    fields = (
        "satellite", "instrument", "acq_date", "acq_time", "latitude", "longitude",
        "scan", "track", "bright_ti4", "bright_ti5", "frp",
    )
    values = [source.strip().lower()]
    for field in fields:
        value = row.get(field, "")
        try:
            missing = value is None or bool(pd.isna(value))
        except (TypeError, ValueError):
            missing = value is None
        if missing:
            value = ""
        elif field in {"latitude", "longitude", "scan", "track", "bright_ti4", "bright_ti5", "frp"}:
            try:
                value = format(float(str(value)), ".5f")
            except (TypeError, ValueError):
                value = str(value).strip().lower()
        else:
            value = str(value).strip().lower()
        values.append(value)
    digest = hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()[:24]
    return f"FIRMS-{digest}"


def _float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _sensor_confidence(value: Any) -> Optional[float]:
    code = str(value or "").strip().lower()
    return {"n": 0.0, "l": 33.0, "m": 66.0, "h": 100.0}.get(code, _float(value))


def _confidence_ordinal(value: Any) -> Optional[int]:
    code = str(value or "").strip().lower()
    category = {"l": 0, "n": 1, "m": 1, "h": 2}.get(code)
    if category is not None:
        return category
    numeric = _float(value)
    if numeric is None:
        return None
    return 0 if numeric <= 40 else 1 if numeric <= 80 else 2


def _safe_error(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    key = settings.FIRMS_MAP_KEY
    return message.replace(key, "[redacted]") if key else message


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


class _OsmFeatureIndex:
    """Small reusable spatial index over the offline OSM cache."""

    def __init__(self, cache_dir: str):
        paths = [p for p in glob(os.path.join(cache_dir, "*.parquet")) if not os.path.basename(p).startswith("_")]
        if not paths:
            raise FileNotFoundError(f"No OSM parquet cache found in {cache_dir}")
        frames = [pd.read_parquet(path) for path in paths]
        osm = pd.concat(frames, ignore_index=True).drop_duplicates(["osm_type", "osm_id"])
        osm["lat"] = pd.to_numeric(osm["lat"], errors="coerce")
        osm["lon"] = pd.to_numeric(osm["lon"], errors="coerce")
        osm = osm.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        if osm.empty:
            raise ValueError("OSM cache contains no valid coordinates")
        for col in ("industrial", "landuse", "power", "man_made"):
            if col not in osm:
                osm[col] = ""
            osm[col] = osm[col].fillna("").astype(str)
        self.osm = osm
        self.tree = BallTree(np.radians(osm[["lat", "lon"]].to_numpy()), metric="haversine")
        self.types = np.array([
            f"industrial:{r.industrial}" if r.industrial else
            f"power:{r.power}" if r.power else
            f"man_made:{r.man_made}" if r.man_made else
            f"landuse:{r.landuse}" if r.landuse else "other"
            for r in osm.itertuples()
        ])
        self.power = ((osm.power == "plant") | (osm.power == "generator")).to_numpy()
        self.quarry = ((osm.landuse == "quarry") | (osm.man_made == "mineshaft")).to_numpy()
        self.flare = (osm.man_made == "flare").to_numpy()
        self.well = (osm.man_made == "petroleum_well").to_numpy()
        self.industrial_land = osm.landuse.isin(["industrial", "quarry"]).to_numpy()

    def enrich(self, latitude: float, longitude: float) -> Dict[str, Any]:
        point = np.radians([[latitude, longitude]])
        distance, nearest = self.tree.query(point, k=1)
        result = {
            "nearest_osm_distance_km": float(distance[0, 0] * EARTH_KM),
            "nearest_osm_type": str(self.types[nearest[0, 0]]),
        }
        for radius, suffix in ((1, "1km"), (2, "2km"), (5, "5km")):
            indices = self.tree.query_radius(point, r=radius / EARTH_KM)[0]
            result[f"industrial_count_{suffix}"] = int(len(indices))
            if radius == 2:
                result["industrial_landuse_nearby"] = int(self.industrial_land[indices].any())
            if radius == 5:
                result["power_plant_count_5km"] = int(self.power[indices].sum())
                result["quarry_count_5km"] = int(self.quarry[indices].sum())
                result["flare_count_5km"] = int(self.flare[indices].sum())
                result["petroleum_well_count_5km"] = int(self.well[indices].sum())
        return result


class LiveFirmsIngestionService:
    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._osm_index: Optional[_OsmFeatureIndex] = None
        self._osm_error: Optional[str] = None
        self._store_error: Optional[str] = None
        self._status: Dict[str, Any] = {
            "enabled": bool(settings.LIVE_FIRMS_ENABLED and settings.FIRMS_MAP_KEY),
            "configured": bool(settings.LIVE_FIRMS_ENABLED and settings.FIRMS_MAP_KEY),
            "running": False,
            "source": settings.LIVE_FIRMS_SOURCE,
            "bbox": settings.LIVE_FIRMS_BBOX,
            "poll_interval_seconds": settings.LIVE_FIRMS_POLL_INTERVAL_SECONDS,
            "last_poll_at": None,
            "last_success_at": None,
            "last_error": None,
            "records_seen": 0,
            "records_added": 0,
            "stored_records": 0,
            "osm_enrichment_available": False,
            "osm_error": None,
            "model": settings.ACTIVE_MODEL,
            "message": "Live polling is disabled or FIRMS_MAP_KEY is not configured.",
        }
        self._initialize_store()

    def _initialize_store(self) -> None:
        path = settings.LIVE_FIRMS_DATABASE_PATH
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with sqlite3.connect(path) as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS observations ("
                    "observation_id TEXT PRIMARY KEY, acquired_at TEXT, payload TEXT NOT NULL, ingested_at TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_live_observations_acquired_at "
                    "ON observations(acquired_at DESC, ingested_at DESC)"
                )
        except (OSError, sqlite3.Error) as error:
            self._store_error = f"{type(error).__name__}: {error}"

    def _connect(self):
        connection = sqlite3.connect(settings.LIVE_FIRMS_DATABASE_PATH, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _load_osm_index(self) -> None:
        try:
            self._osm_index = _OsmFeatureIndex(settings.OSM_CACHE_DIR)
            self._status["osm_enrichment_available"] = True
            self._status["osm_error"] = None
        except Exception as error:
            self._osm_error = f"{type(error).__name__}: {error}"
            self._status["osm_error"] = self._osm_error
            self._status["osm_enrichment_available"] = False

    async def start(self) -> None:
        if self._store_error:
            self._status["last_error"] = self._store_error
            self._status["message"] = "Live FIRMS storage is unavailable; polling is disabled."
            return
        if not (settings.LIVE_FIRMS_ENABLED and settings.FIRMS_MAP_KEY):
            self._status["stored_records"] = self.count()
            await asyncio.to_thread(self._sync_repository_cache)
            return
        if self._task and not self._task.done():
            return
        await asyncio.to_thread(self._load_osm_index)
        await asyncio.to_thread(self._sync_repository_cache)
        self._status["running"] = True
        self._status["enabled"] = True
        self._status["message"] = "Polling NASA FIRMS and classifying eligible records with the active Tier 2 model."
        self._task = asyncio.create_task(self._poll_loop(), name="live-firms-poller")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._status["running"] = False

    async def _poll_loop(self) -> None:
        try:
            while True:
                try:
                    result = await asyncio.to_thread(self.poll_once)
                    self._status.update(result)
                    await asyncio.to_thread(self._sync_repository_cache)
                except Exception as error:
                    self._status["last_error"] = _safe_error(error)
                    self._status["message"] = "FIRMS poll failed; existing detections remain available."
                await asyncio.sleep(max(60, int(settings.LIVE_FIRMS_POLL_INTERVAL_SECONDS)))
        except asyncio.CancelledError:
            raise
        finally:
            self._status["running"] = False

    def poll_once(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        self._status["last_poll_at"] = now.isoformat()
        try:
            west, south, east, north = [float(part.strip()) for part in settings.LIVE_FIRMS_BBOX.split(",")]
        except Exception as error:
            raise ValueError("LIVE_FIRMS_BBOX must be west,south,east,north") from error
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("LIVE_FIRMS_BBOX coordinates are invalid")
        days = min(5, max(1, int(settings.LIVE_FIRMS_LOOKBACK_DAYS)))
        start = (now.date() - timedelta(days=days - 1)).isoformat()
        bbox = ",".join(str(value) for value in (west, south, east, north))
        url = (
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"{settings.FIRMS_MAP_KEY}/{settings.LIVE_FIRMS_SOURCE}/{bbox}/{days}/{start}"
        )
        response = requests.get(url, timeout=(10, 45))
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        required = {"latitude", "longitude", "acq_date", "acq_time", "frp"}
        if frame.empty:
            if not required.issubset(set(frame.columns)) and "latitude" not in frame.columns:
                # FIRMS returns a header even for a valid empty result; reject API error text.
                raise ValueError(f"FIRMS response missing required columns: {sorted(required)}")
            rows: List[Dict[str, Any]] = []
        else:
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"FIRMS response missing columns: {sorted(missing)}")
            rows = frame.replace({np.nan: None}).to_dict(orient="records")
        added = self._upsert_rows(rows)
        self._status["last_poll_at"] = now.isoformat()
        self._status["last_success_at"] = now.isoformat()
        self._status["last_error"] = None
        self._status["records_seen"] = len(rows)
        self._status["records_added"] = added
        self._status["stored_records"] = self.count()
        self._status["message"] = (
            "New FIRMS records are available. Tier 2 is applied only when all required features are available."
        )
        return dict(self._status)

    def _upsert_rows(self, rows: List[Dict[str, Any]]) -> int:
        added = 0
        with self._connect() as connection:
            for raw in rows:
                latitude, longitude = _float(raw.get("latitude")), _float(raw.get("longitude"))
                if latitude is None or longitude is None or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                    continue
                obs_id = stable_observation_id(raw, settings.LIVE_FIRMS_SOURCE)
                row = self._normalize_live_row(raw, obs_id, latitude, longitude)
                encoded = json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False)
                previous = connection.execute(
                    "SELECT payload FROM observations WHERE observation_id = ?", (obs_id,)
                ).fetchone()
                ingested_at = datetime.now(timezone.utc).isoformat()
                acquired_at = f"{raw.get('acq_date', '')} {raw.get('acq_time', '')}"
                if previous is None:
                    connection.execute(
                        "INSERT INTO observations(observation_id, acquired_at, payload, ingested_at) VALUES (?, ?, ?, ?)",
                        (obs_id, acquired_at, encoded, ingested_at),
                    )
                    added += 1
                else:
                    old_payload = json.loads(previous["payload"])
                    if old_payload.get("prediction_status") != "classified" and row.get("prediction_status") == "classified":
                        connection.execute(
                            "UPDATE observations SET payload = ?, ingested_at = ? WHERE observation_id = ?",
                            (encoded, ingested_at, obs_id),
                        )
            cutoff = (datetime.now(timezone.utc).date() - timedelta(days=max(1, settings.LIVE_FIRMS_RETENTION_DAYS))).isoformat()
            connection.execute("DELETE FROM observations WHERE acquired_at < ?", (cutoff,))
        return added

    def _normalize_live_row(self, raw: Dict[str, Any], obs_id: str, lat: float, lon: float) -> Dict[str, Any]:
        bright_ti4 = _float(raw.get("bright_ti4", raw.get("brightness")))
        bright_ti5 = _float(raw.get("bright_ti5", raw.get("bright_t31")))
        frp = _float(raw.get("frp"), 0.0)
        sensor_conf = _sensor_confidence(raw.get("confidence"))
        confidence_ordinal = _confidence_ordinal(raw.get("confidence"))
        row: Dict[str, Any] = {
            "id": obs_id, "obs_id": obs_id, "event_id": None,
            "latitude": lat, "longitude": lon,
            "acq_date": str(raw.get("acq_date") or "unknown"),
            "acq_time": str(raw.get("acq_time") or "0000"),
            "brightness": bright_ti4, "bright_t31": bright_ti5, "frp": frp,
            "confidence": sensor_conf, "confidence_ordinal": confidence_ordinal,
            "satellite": str(raw.get("satellite") or "unknown"),
            "instrument": str(raw.get("instrument") or "unknown"),
            "daynight": str(raw.get("daynight") or "D"), "region": "South Asia" if 5 <= lat <= 38 and 60 <= lon <= 100 else "Unknown",
            "source": "live_firms", "source_product": settings.LIVE_FIRMS_SOURCE,
            "_raw_firms_record": raw,
            "prediction_status": "awaiting_osm_features", "prediction_mode": None,
            "true_class": "unclassified", "classification_confidence": None,
            "dist_to_industrial": None, "nearest_facility_type": "unavailable",
            "persistence_score": None, "detection_count_30d": None,
            "_model_features_available": [],
        }
        row.update({key: raw.get(key) for key in ("scan", "track", "version", "bright_t31") if key in raw})
        if bright_ti4 is None or bright_ti5 is None or sensor_conf is None or confidence_ordinal is None or self._osm_index is None:
            row["prediction_status"] = "awaiting_required_features"
            return row
        context = self._osm_index.enrich(lat, lon)
        row.update({
            "dist_to_industrial": context["nearest_osm_distance_km"],
            "nearest_facility_type": context["nearest_osm_type"],
            **{key: value for key, value in context.items() if key not in {"nearest_osm_distance_km", "nearest_osm_type"}},
            "persistence_score": 0.0,
            "detection_count_30d": 0,
            "_model_features_available": MODEL_FEATURES,
        })
        model_info = model_service.available_models.get(model_service.active_model_key)
        model_features = set(model_info["features"]) if model_info else set()
        if model_service.inference_mode != "trained" or not model_features.issubset(set(MODEL_FEATURES)):
            row["prediction_status"] = "tier2_required_features_unavailable"
            return row
        prediction_input = PredictionInput(
            brightness=bright_ti4,
            bright_t31=bright_ti5,
            frp=frp or 0.0,
            confidence=sensor_conf,
            confidence_ordinal=confidence_ordinal,
            dist_to_industrial=context["nearest_osm_distance_km"],
            industrial_count_1km=context["industrial_count_1km"],
            industrial_count_2km=context["industrial_count_2km"],
            industrial_count_5km=context["industrial_count_5km"],
            power_plant_count_5km=context["power_plant_count_5km"],
            quarry_count_5km=context["quarry_count_5km"],
            flare_count_5km=context["flare_count_5km"],
            petroleum_well_count_5km=context["petroleum_well_count_5km"],
            industrial_landuse_nearby=context["industrial_landuse_nearby"],
            daynight=row["daynight"],
            available_model_features=MODEL_FEATURES,
        )
        try:
            prediction = model_service.predict(prediction_input)
        except (RuntimeError, ValueError) as error:
            row["prediction_status"] = "tier2_inference_error"
            row["prediction_error"] = f"{type(error).__name__}: {error}"
            return row
        if prediction.inference_mode != "trained":
            row["prediction_status"] = "tier2_inference_fallback"
            row["prediction_error"] = prediction.fallback_reason
            return row
        row.update({
            "true_class": prediction.predicted_class,
            "classification_confidence": prediction.confidence,
            "prediction_status": "classified",
            "prediction_mode": prediction.inference_mode,
            "prediction_model": model_service.active_model_key,
            "class_probabilities": prediction.class_probabilities,
        })
        return row

    def _sync_repository_cache(self) -> None:
        from app.repositories.detection_repository import detection_repo
        detection_repo.add_live_records(self.list_records(limit=5000))

    def list_records(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM observations ORDER BY acquired_at DESC, ingested_at DESC LIMIT ?",
                (max(1, min(int(limit), 5000)),),
            ).fetchall()
        records = [json.loads(row["payload"]) for row in rows]
        return records

    def get_by_id(self, observation_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM observations WHERE observation_id = ?", (observation_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0])

    def get_status(self) -> Dict[str, Any]:
        status = dict(self._status)
        try:
            status["stored_records"] = self.count()
        except (OSError, sqlite3.Error) as error:
            status["stored_records"] = 0
            status["last_error"] = f"{type(error).__name__}: {error}"
        status["active_model_mode"] = model_service.inference_mode
        status["osm_enrichment_available"] = self._osm_index is not None
        status["osm_error"] = self._osm_error
        if status["enabled"] and not status["running"] and not status["last_error"]:
            status["message"] = status.get("message") or "Configured; waiting for poller startup."
        return status


live_firms_service = LiveFirmsIngestionService()
