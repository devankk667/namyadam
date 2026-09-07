from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

class DetectionBase(BaseModel):
    id: str
    latitude: float
    longitude: float
    acq_date: str
    acq_time: str
    brightness: float
    bright_t31: float
    frp: float
    confidence: float
    satellite: str
    instrument: str
    daynight: str
    dist_to_industrial: float
    industrial_count_2km: int
    industrial_count_5km: int
    nearest_facility_type: str
    persistence_score: float
    detection_count_30d: int
    region: str
    true_class: Optional[str] = None
    is_industrial: Optional[int] = None

class DetectionFilterParams(BaseModel):
    min_confidence: Optional[float] = None
    min_frp: Optional[float] = None
    classification: Optional[str] = None
    region: Optional[str] = None
    page: int = 1
    page_size: int = 50

class PredictionInput(BaseModel):
    brightness: float = Field(..., json_schema_extra={"example": 365.5})
    bright_t31: float = Field(..., json_schema_extra={"example": 298.2})
    frp: float = Field(..., json_schema_extra={"example": 45.0})
    confidence: float = Field(..., json_schema_extra={"example": 95.0})
    dist_to_industrial: float = Field(..., json_schema_extra={"example": 0.15})
    industrial_count_2km: int = Field(..., json_schema_extra={"example": 4})
    industrial_count_5km: int = Field(..., json_schema_extra={"example": 10})
    nearest_facility_type: str = Field("refinery", json_schema_extra={"example": "refinery"})
    persistence_score: float = Field(0.85, json_schema_extra={"example": 0.85})
    detection_count_30d: int = Field(15, json_schema_extra={"example": 15})
    daynight: str = Field("N", json_schema_extra={"example": "N"})

class PredictionExplanation(BaseModel):
    top_contributing_features: Dict[str, float]
    summary: str

class PredictionOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    model_version: str
    explanation: PredictionExplanation

class AnalyticsSummary(BaseModel):
    total_detections: int
    industrial_associated_detections: int
    active_thermal_sources: int
    persistent_sources: int
    high_confidence_events: int
    high_risk_events: int

class AlertItem(BaseModel):
    id: str
    detection_id: str
    timestamp: str
    latitude: float
    longitude: float
    severity: str # CRITICAL, HIGH, MEDIUM, LOW
    type: str # HIGH_CONFIDENCE_INDUSTRIAL_FIRE, PERSISTENT_THERMAL_SOURCE, UNUSUAL_THERMAL_ACTIVITY, HIGH_FRP_EVENT
    explanation: str
    region: str

class ModelStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_loaded: bool
    model_name: str
    model_version: str
    macro_f1: float
    train_samples: int
    test_samples: int
    supported_classes: List[str]
    feature_importances: Dict[str, float]
    all_model_benchmarks: Dict[str, Dict[str, float]] = {}
    split_method: Optional[str] = None
