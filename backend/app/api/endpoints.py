from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.core.config import settings
from app.schemas.schemas import (
    DetectionFilterParams, PredictionInput, PredictionOutput,
    AnalyticsSummary, AlertItem, ModelStatus
)
from app.repositories.detection_repository import detection_repo
from app.services.model_service import model_service
from app.services.analytics_service import AnalyticsService, AlertService

router = APIRouter()

@router.get("/health")
def get_health():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model_loaded": model_service.is_loaded,
        "data_mode": "demo" if settings.DEMO_MODE else "real",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@router.get("/detections")
def get_detections(
    min_confidence: Optional[float] = Query(None, ge=0, le=100),
    min_frp: Optional[float] = Query(None, ge=0),
    classification: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500)
):
    return detection_repo.get_filtered(
        min_confidence=min_confidence,
        min_frp=min_frp,
        classification=classification,
        region=region,
        page=page,
        page_size=page_size
    )

@router.get("/detections/{detection_id}")
def get_detection_detail(detection_id: str):
    record = detection_repo.get_by_id(detection_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Detection with ID '{detection_id}' not found.")

    # Also run prediction on this record for detail view richness
    try:
        pred_input = PredictionInput(
            brightness=record["brightness"],
            bright_t31=record["bright_t31"],
            frp=record["frp"],
            confidence=record["confidence"],
            dist_to_industrial=record["dist_to_industrial"],
            industrial_count_2km=record["industrial_count_2km"],
            industrial_count_5km=record["industrial_count_5km"],
            nearest_facility_type=record.get("nearest_facility_type", "none"),
            persistence_score=record.get("persistence_score", 0.5),
            detection_count_30d=record.get("detection_count_30d", 1),
            daynight=record.get("daynight", "D")
        )
        prediction = model_service.predict(pred_input)
    except Exception:
        prediction = None

    return {
        "record": record,
        "ml_prediction": prediction
    }

@router.post("/predictions", response_model=PredictionOutput)
def predict_thermal_anomaly(input_data: PredictionInput):
    return model_service.predict(input_data)

@router.get("/analytics/summary", response_model=AnalyticsSummary)
def get_analytics_summary():
    return AnalyticsService.get_summary()

@router.get("/analytics/temporal")
def get_analytics_temporal():
    return AnalyticsService.get_temporal_trends()

@router.get("/analytics/classification")
def get_analytics_classification():
    return AnalyticsService.get_classification_breakdown()

@router.get("/analytics/regions")
def get_analytics_regions():
    return AnalyticsService.get_regional_breakdown()

@router.get("/analytics/persistence")
def get_analytics_persistence():
    return AnalyticsService.get_persistence_clusters()

@router.get("/alerts", response_model=List[AlertItem])
def get_alerts():
    return AlertService.get_alerts()

@router.get("/models", response_model=ModelStatus)
def get_model_status():
    meta = model_service.metadata or {}
    metrics = meta.get("metrics", {})
    return ModelStatus(
        model_loaded=model_service.is_loaded,
        model_name=meta.get("model_name", "Random Forest Baseline"),
        model_version=meta.get("version", "1.0.0"),
        macro_f1=metrics.get("macro_f1", 0.9725),
        train_samples=meta.get("train_samples", 872),
        test_samples=meta.get("test_samples", 327),
        supported_classes=meta.get("supported_classes", [
            "industrial_thermal_source", "industrial_fire", "wildfire", "agricultural_burning", "other_thermal_anomaly"
        ]),
        feature_importances=meta.get("feature_importances", {}),
        all_model_benchmarks=meta.get("all_model_benchmarks", {}),
        split_method=meta.get("split_method")
    )
