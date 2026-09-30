import importlib.util
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.services.live_firms_service import (
    LiveFirmsIngestionService,
    _confidence_ordinal,
    stable_observation_id,
)


def firms_row():
    return {
        "satellite": "N20", "instrument": "VIIRS", "acq_date": "2024-06-01", "acq_time": 1234,
        "latitude": 20.123456, "longitude": 73.234567, "scan": 0.4, "track": 0.5,
        "bright_ti4": 330.0, "bright_ti5": 290.0, "frp": 4.2, "confidence": "h", "daynight": "N",
    }


def test_live_firms_id_is_stable_under_input_order_and_matches_pipeline_helper():
    row = firms_row()
    reordered = dict(reversed(list(row.items())))
    script = Path(__file__).resolve().parents[2] / "ml" / "scripts" / "firms_ids.py"
    spec = importlib.util.spec_from_file_location("pipeline_firms_ids", script)
    assert spec and spec.loader
    pipeline_ids = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_ids)
    assert stable_observation_id(row, "VIIRS_NOAA20_SP") == stable_observation_id(reordered, "VIIRS_NOAA20_SP")
    assert stable_observation_id(row, "VIIRS_NOAA20_SP") == pipeline_ids.stable_observation_id(row, "VIIRS_NOAA20_SP")


def test_firms_confidence_categories_match_training_ordinal_encoding():
    assert [_confidence_ordinal(code) for code in ("l", "n", "m", "h")] == [0, 1, 1, 2]


def test_live_firms_store_deduplicates_and_does_not_fabricate_prediction(monkeypatch, tmp_path):
    database = tmp_path / "live.sqlite3"
    monkeypatch.setattr(settings, "LIVE_FIRMS_DATABASE_PATH", str(database))
    monkeypatch.setattr(settings, "OSM_CACHE_DIR", str(tmp_path / "no-osm"))
    service = LiveFirmsIngestionService()
    monkeypatch.setattr(service, "_osm_index", None)

    raw = firms_row()
    raw["acq_date"] = "2099-01-01"
    assert service._upsert_rows([raw]) == 1
    assert service._upsert_rows([raw]) == 0
    stored = service.list_records()
    assert len(stored) == 1
    assert stored[0]["id"] == stable_observation_id(raw, settings.LIVE_FIRMS_SOURCE)
    assert stored[0]["prediction_status"] == "awaiting_required_features"
    assert stored[0]["true_class"] == "unclassified"
    assert stored[0]["classification_confidence"] is None
    assert stored[0]["_raw_firms_record"] == raw


def test_poll_once_parses_and_persists_firms_csv_without_network(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LIVE_FIRMS_DATABASE_PATH", str(tmp_path / "live.sqlite3"))
    monkeypatch.setattr(settings, "FIRMS_MAP_KEY", "private-test-key")
    monkeypatch.setattr(settings, "LIVE_FIRMS_SOURCE", "VIIRS_NOAA20_SP")
    monkeypatch.setattr(settings, "LIVE_FIRMS_BBOX", "68,8,97,37")
    service = LiveFirmsIngestionService()

    class Response:
        text = "latitude,longitude,acq_date,acq_time,frp,bright_ti4,bright_ti5,confidence,satellite,instrument,daynight\n20.1,73.2,2099-01-01,1234,4.2,330,290,h,N20,VIIRS,N\n"

        @staticmethod
        def raise_for_status():
            return None

    def fake_get(url, timeout):
        assert "private-test-key" in url
        assert "VIIRS_NOAA20_SP" in url
        assert timeout == (10, 45)
        return Response()

    monkeypatch.setattr("app.services.live_firms_service.requests.get", fake_get)
    result = service.poll_once()
    assert result["records_seen"] == 1
    assert result["records_added"] == 1
    assert service.list_records()[0]["source"] == "live_firms"


def test_full_source_analytics_snapshot_exceeds_display_sample_count():
    frame = pd.DataFrame({
        "rf_predicted_class": ["agricultural_burning", "industrial_fire", "gas_flare", "mining_activity"],
        "acq_date": ["2024-06-01", "2024-06-01", "2024-06-02", "2024-06-02"],
        "frp": [2.0, 60.0, 4.0, 1.0],
        "confidence": ["h", "m", "l", "n"],
        "nearest_osm_distance_km": [10.0, 0.2, 2.0, 3.0],
        "industrial_count_5km": [0, 2, 0, 0],
        "rf_confidence": [0.7, 0.9, 0.6, 0.8],
        "cell_prior_count": [0, 3, 0, 1],
        "latitude": [20.0] * 4,
        "longitude": [73.0] * 4,
    })
    from app.repositories.detection_repository import ThermalDetectionRepository

    aggregates = ThermalDetectionRepository._build_analytics_snapshot(frame)
    assert aggregates["source_record_count"] == 4
    assert aggregates["summary"]["total_detections"] == 4
    assert sum(item["count"] for item in aggregates["classification"]) == 4
    assert len(aggregates["temporal"]) == 2
