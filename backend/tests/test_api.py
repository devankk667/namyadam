import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "data_mode" in data

def test_get_detections():
    response = client.get("/api/detections?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert len(data["items"]) <= 10

def test_get_detection_detail_valid():
    # First get a valid detection ID
    resp_list = client.get("/api/detections?page_size=1")
    det_id = resp_list.json()["items"][0]["id"]

    response = client.get(f"/api/detections/{det_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["record"]["id"] == det_id

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
        "daynight": "N"
    }
    response = client.post("/api/predictions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "explanation" in data

def test_analytics_endpoints():
    r_sum = client.get("/api/analytics/summary")
    assert r_sum.status_code == 200
    assert r_sum.json()["total_detections"] > 0

    r_temp = client.get("/api/analytics/temporal")
    assert r_temp.status_code == 200

    r_cls = client.get("/api/analytics/classification")
    assert r_cls.status_code == 200

    r_reg = client.get("/api/analytics/regions")
    assert r_reg.status_code == 200

def test_alerts_endpoint():
    response = client.get("/api/alerts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_models_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "macro_f1" in data
    assert "supported_classes" in data
