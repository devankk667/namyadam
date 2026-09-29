"""
End-to-end integration smoke test validating full stack data flow:
Frontend API Layer -> FastAPI Backend -> ML Model Inference -> Analytics Engine.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_e2e_full_stack_pipeline():
    # 1. Health check
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    health = health_res.json()
    assert health["status"] in {"healthy", "degraded"}
    assert health["data_source"] in {"firms_predictions", "processed_csv", "demo_json", "unavailable"}
    assert health["inference_mode"] in {"trained", "heuristic"}

    # 2. Query detections
    det_res = client.get("/api/detections?page=1&page_size=10")
    assert det_res.status_code == 200
    detections = det_res.json()["items"]
    assert len(detections) <= 10

    # 3. Retrieve detail when a record is available. Empty/degraded data is a
    # valid application state and should not make the smoke test fail.
    if detections:
        det_id = detections[0]["id"]
        detail_res = client.get(f"/api/detections/{det_id}")
        assert detail_res.status_code == 200
        detail_data = detail_res.json()
        assert detail_data["record"]["id"] == det_id
        assert "ml_prediction" in detail_data
        if detail_data["ml_prediction"] is None:
            assert detail_data["prediction_error"]
        else:
            assert detail_data["ml_prediction"]["confidence"] > 0

    # 4. Custom prediction inference
    pred_payload = {
        "brightness": 372.0,
        "bright_t31": 298.0,
        "frp": 65.0,
        "confidence": 98.0,
        "dist_to_industrial": 0.15,
        "industrial_count_2km": 4,
        "industrial_count_5km": 10,
        "nearest_facility_type": "refinery",
        "persistence_score": 0.88,
        "detection_count_30d": 20,
        "daynight": "N",
        "industrial_count_1km": 2,
        "power_plant_count_5km": 1,
        "quarry_count_5km": 0,
        "flare_count_5km": 1,
        "petroleum_well_count_5km": 0,
        "industrial_landuse_nearby": 1
    }
    pred_res = client.post("/api/predictions", json=pred_payload)
    assert pred_res.status_code == 200
    prediction = pred_res.json()
    assert prediction["predicted_class"] in [
        "agricultural_burning", "gas_flare", "industrial_fire", "mining_activity"
    ]
    assert prediction["inference_mode"] in {"trained", "heuristic"}
    assert abs(sum(prediction["class_probabilities"].values()) - 1.0) < 0.01
    assert "summary" in prediction["explanation"]

    # 5. Analytics summary and alerts engine
    analytics_res = client.get("/api/analytics/summary")
    assert analytics_res.status_code == 200
    assert analytics_res.json()["total_detections"] == health["data_record_count"]

    # Every analytics widget endpoint must remain usable with empty data too.
    for endpoint in (
        "/api/analytics/temporal",
        "/api/analytics/classification",
        "/api/analytics/regions",
        "/api/analytics/persistence",
    ):
        response = client.get(endpoint)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    alerts_res = client.get("/api/alerts")
    assert alerts_res.status_code == 200
    assert isinstance(alerts_res.json(), list)
