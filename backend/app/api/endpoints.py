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
from app.services.fusion_validation_service import fusion_validation_service
from app.services.analytics_service import AnalyticsService, AlertService
from app.services.live_firms_service import live_firms_service

router = APIRouter()

@router.get("/health")
def get_health():
    return {
        "status": "healthy" if detection_repo.data_source != "unavailable" and model_service.inference_mode == "trained" else "degraded",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "model_loaded": model_service.is_loaded,
        "inference_mode": model_service.inference_mode,
        "fallback_reason": model_service.fallback_reason,
        "active_model": model_service.active_model_key,
        "data_mode": detection_repo.data_source,
        "data_source": detection_repo.data_source,
        "data_record_count": len(detection_repo.get_raw_list()),
        "map_sample_count": len(detection_repo.get_raw_list()) - len(detection_repo._live_cache),
        "live_record_count": len(detection_repo._live_cache),
        "source_record_count": detection_repo.total_source_records,
        "analytics_record_count": (detection_repo.get_analytics_snapshot() or {}).get("source_record_count", len(detection_repo.get_raw_list())),
        "analytics_scope": "full_source" if detection_repo.get_analytics_snapshot() else "loaded_records",
        "data_error": detection_repo.data_error,
        "fusion_validation_available": fusion_validation_service.get_status()["enabled"],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@router.get("/ingestion/status")
def get_ingestion_status():
    """Health and provenance for the optional background FIRMS poller."""
    return live_firms_service.get_status()


@router.get("/detections")
def get_detections(
    min_confidence: Optional[float] = Query(None, ge=0, le=100),
    min_frp: Optional[float] = Query(None, ge=0),
    classification: Optional[str] = Query(None),
    region: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500)
):
    result = detection_repo.get_filtered(
        min_confidence=min_confidence,
        min_frp=min_frp,
        classification=classification,
        region=region,
        page=page,
        page_size=page_size
    )
    for item in result["items"]:
        item["fusion_prediction"] = fusion_validation_service.get_by_event_id(item.get("event_id"))
    return result

@router.get("/detections/{detection_id}")
def get_detection_detail(detection_id: str):
    raw_record = detection_repo.get_by_id(detection_id)
    if not raw_record:
        raise HTTPException(status_code=404, detail=f"Detection with ID '{detection_id}' not found.")
    record = {key: value for key, value in raw_record.items() if key != "_model_features_available"}

    if record.get("source") == "live_firms" and record.get("prediction_status") != "classified":
        return {
            "record": record,
            "ml_prediction": None,
            "fusion_prediction": None,
            "prediction_error": (
                "Tier 2 inference was not run: "
                + str(record.get("prediction_status", "required features unavailable"))
            ),
        }

    # Run prediction on this record for detail view
    prediction_error = None
    try:
        pred_input = PredictionInput(
            brightness=record.get("brightness", 300.0),
            bright_t31=record.get("bright_t31", 280.0),
            frp=record.get("frp", 30.0),
            confidence=record.get("confidence", 85.0),
            confidence_ordinal=record.get("confidence_ordinal"),
            dist_to_industrial=record.get("dist_to_industrial", 1.5),
            industrial_count_2km=record.get("industrial_count_2km", 2),
            industrial_count_5km=record.get("industrial_count_5km", 5),
            industrial_count_1km=record.get("industrial_count_1km"),
            power_plant_count_5km=record.get("power_plant_count_5km"),
            quarry_count_5km=record.get("quarry_count_5km"),
            flare_count_5km=record.get("flare_count_5km"),
            petroleum_well_count_5km=record.get("petroleum_well_count_5km"),
            industrial_landuse_nearby=record.get("industrial_landuse_nearby"),
            available_model_features=raw_record.get("_model_features_available"),
            nearest_facility_type=record.get("nearest_facility_type", "none"),
            persistence_score=record.get("persistence_score", 0.0),
            detection_count_30d=record.get("detection_count_30d", 1),
            daynight=record.get("daynight", "D")
        )
        prediction = model_service.predict(pred_input)
    except Exception as e:
        prediction_error = f"{type(e).__name__}: {e}"
        print(f"Detection detail prediction error: {prediction_error}")
        prediction = None

    return {
        "record": record,
        "ml_prediction": prediction,
        "fusion_prediction": fusion_validation_service.get_by_event_id(record.get("event_id")),
        "prediction_error": prediction_error,
    }

@router.post("/predictions", response_model=PredictionOutput)
def predict_thermal_anomaly(input_data: PredictionInput):
    try:
        return model_service.predict(input_data)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

@router.get("/fusion/status")
def get_fusion_status():
    """Status of held-out event-level fusion predictions available to the UI."""
    return fusion_validation_service.get_status()

@router.get("/models/available")
def get_available_models():
    """Get all available trained models"""
    return model_service.get_available_models()

@router.post("/models/switch/{model_name}")
def switch_model(model_name: str):
    """Switch to a different trained model"""
    result = model_service.switch_model(model_name)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

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

    return ModelStatus(
        model_loaded=model_service.is_loaded,
        inference_mode=model_service.inference_mode,
        fallback_reason=model_service.fallback_reason,
        active_model_key=model_service.active_model_key,
        model_name=model_service.active_model_display if model_service.is_loaded else "Heuristic fallback",
        model_version=meta.get("version", "2.0.0"),
        macro_f1=meta.get("macro_f1", 0.0),
        train_samples=meta.get("n_train", meta.get("train_obs", 0)),
        test_samples=meta.get("n_test", meta.get("test_obs", 0)),
        supported_classes=meta.get("classes", [
            "agricultural_burning", "gas_flare", "industrial_fire", "mining_activity"
        ]),
        feature_importances=meta.get("feature_importances", {}),
        all_model_benchmarks=meta.get("all_model_benchmarks", {}),
        split_method=meta.get("split")
    )
