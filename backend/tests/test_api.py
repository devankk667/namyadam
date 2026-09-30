from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app

client = TestClient(app)

def test_configured_firms_data_is_loaded_into_detection_api():
    if not Path(settings.FIRMS_PREDICTIONS_PATH).is_file():
        return

    health = client.get("/api/health")
    assert health.status_code == 200
    health_data = health.json()
    assert health_data["data_source"] == "firms_predictions"
    assert health_data["source_record_count"] >= health_data["map_sample_count"] > 0
    assert health_data["data_record_count"] == health_data["map_sample_count"] + health_data["live_record_count"]

    response = client.get("/api/detections?page=1&page_size=5")
    assert response.status_code == 200
    items = response.json()["items"]
    assert items
    assert all({"id", "latitude", "longitude", "frp", "true_class"} <= item.keys() for item in items)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"healthy", "degraded"}
    assert data["data_source"] in {"firms_predictions", "processed_csv", "demo_json", "unavailable"}
    assert "data_record_count" in data
    assert data["inference_mode"] in {"trained", "heuristic"}

def test_get_detections():
    response = client.get("/api/detections?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert len(data["items"]) <= 10
    assert all("fusion_prediction" in item for item in data["items"])

    zero_filter = client.get("/api/detections?min_confidence=0&min_frp=0&page_size=10")
    assert zero_filter.status_code == 200
    assert len(zero_filter.json()["items"]) <= 10

def test_get_detection_detail_valid():
    # First get a valid detection ID when data is loaded.
    resp_list = client.get("/api/detections?page_size=1")
    assert resp_list.status_code == 200
    items = resp_list.json()["items"]
    if not items:
        assert client.get("/api/detections/NO-DATA-RECORD").status_code == 404
        return
    det_id = items[0]["id"]

    response = client.get(f"/api/detections/{det_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["record"]["id"] == det_id
    assert "fusion_prediction" in data
    assert "_model_features_available" not in data["record"]
    if data["ml_prediction"] is None:
        assert data["prediction_error"]

def test_get_detection_detail_invalid():
    response = client.get("/api/detections/INVALID-ID-9999")
    assert response.status_code == 404

def test_prediction_endpoint():
    payload = {
        "brightness": 370.0,
        "bright_t31": 300.0,
        "frp": 55.0,
        "confidence": 95.0,
        "dist_to_industrial": 0.2,
        "industrial_count_2km": 5,
        "industrial_count_5km": 12,
        "nearest_facility_type": "refinery",
        "persistence_score": 0.90,
        "detection_count_30d": 25,
        "daynight": "N",
        "industrial_count_1km": 3,
        "power_plant_count_5km": 1,
        "quarry_count_5km": 0,
        "flare_count_5km": 2,
        "petroleum_well_count_5km": 0,
        "industrial_landuse_nearby": 1
    }
    response = client.post("/api/predictions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "explanation" in data
    assert data["inference_mode"] in {"trained", "heuristic"}
    assert abs(sum(data["class_probabilities"].values()) - 1.0) < 0.01

def test_analytics_endpoints():
    r_sum = client.get("/api/analytics/summary")
    assert r_sum.status_code == 200
    health = client.get("/api/health").json()
    assert r_sum.json()["total_detections"] == health["analytics_record_count"]

    r_temp = client.get("/api/analytics/temporal")
    assert r_temp.status_code == 200

    r_cls = client.get("/api/analytics/classification")
    assert r_cls.status_code == 200

    r_reg = client.get("/api/analytics/regions")
    assert r_reg.status_code == 200
    regions = r_reg.json()
    assert isinstance(regions, list)
    if regions:
        assert {"region", "total_detections", "avg_persistence", "high_risk_count"} <= regions[0].keys()

    r_persistence = client.get("/api/analytics/persistence")
    assert r_persistence.status_code == 200
    assert isinstance(r_persistence.json(), list)

def test_alerts_endpoint():
    response = client.get("/api/alerts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_ingestion_status_endpoint_is_explicit_when_live_polling_is_not_configured():
    response = client.get("/api/ingestion/status")
    assert response.status_code == 200
    status = response.json()
    assert {"enabled", "configured", "running", "stored_records", "osm_enrichment_available", "last_error"} <= status.keys()
    assert isinstance(status["enabled"], bool)


def test_fusion_status_endpoint():
    response = client.get("/api/fusion/status")
    assert response.status_code == 200
    data = response.json()
    assert data["prediction_kind"] == "spatial_cv_oof"
    assert isinstance(data["enabled"], bool)
    assert "reason" in data
    comparison = data.get("ablation_comparison")
    assert isinstance(comparison, list)
    for candidate in comparison:
        assert {"feature_set", "estimator", "trusted_macro_f1", "trusted_macro_f1_no_flare", "trusted_events"} <= candidate.keys()
    if data["enabled"]:
        assert any(
            candidate["feature_set"] == data["feature_set"]
            and candidate["estimator"] == data["estimator"]
            for candidate in comparison
        )


def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "macro_f1" in data
    assert "supported_classes" in data
    assert data["inference_mode"] in {"trained", "heuristic"}

    available = client.get("/api/models/available")
    assert available.status_code == 200
    available_models = available.json()["available_models"]
    assert isinstance(available_models, list)
    if available_models:
        assert {"name", "description", "active", "feature_count"} <= available_models[0].keys()
        switch = client.post(f"/api/models/switch/{available_models[0]['name']}")
        assert switch.status_code == 200
        assert switch.json()["status"] == "success"

    invalid_switch = client.post("/api/models/switch/not-a-model")
    assert invalid_switch.status_code == 400
