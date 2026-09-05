"""
End-to-end integration smoke test validating full stack data flow:
Frontend API Layer -> FastAPI Backend -> ML Model Inference -> Analytics Engine.
"""

import os
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)

def test_e2e_full_stack_pipeline():
    # 1. Health check
    health_res = client.get("/api/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"
    assert health_res.json()["model_loaded"] is True

    # 2. Query detections
    det_res = client.get("/api/detections?page=1&page_size=10")
    assert det_res.status_code == 200
    detections = det_res.json()["items"]
    assert len(detections) > 0

    first_det = detections[0]
    det_id = first_det["id"]

    # 3. Retrieve detection detail and verify inference binding
    detail_res = client.get(f"/api/detections/{det_id}")
    assert detail_res.status_code == 200
    detail_data = detail_res.json()
    assert detail_data["record"]["id"] == det_id
    assert "ml_prediction" in detail_data
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
        "daynight": "N"
    }
    pred_res = client.post("/api/predictions", json=pred_payload)
    assert pred_res.status_code == 200
    prediction = pred_res.json()
    assert prediction["predicted_class"] in [
        "industrial_thermal_source", "industrial_fire", "wildfire", "agricultural_burning", "other_thermal_anomaly"
    ]
    assert "summary" in prediction["explanation"]

    # 5. Analytics summary and alerts engine
    analytics_res = client.get("/api/analytics/summary")
    assert analytics_res.status_code == 200
    assert analytics_res.json()["total_detections"] >= 1000

    alerts_res = client.get("/api/alerts")
    assert alerts_res.status_code == 200
    assert len(alerts_res.json()) > 0
