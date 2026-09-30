import os
from pydantic_settings import BaseSettings

# Get project root (go up from backend/app/core to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class Settings(BaseSettings):
    APP_NAME: str = "AI Industrial Thermal Anomaly Monitor"
    APP_VERSION: str = "2.0.0"
    APP_ENV: str = "development"

    DEMO_MODE: bool = False
    MODEL_PATH: str = os.path.join(PROJECT_ROOT, "ml", "models", "self_trained_rf3")  # Tier 2 Round 3 (best model)
    RF_BASELINE_PATH: str = os.path.join(PROJECT_ROOT, "ml", "models", "rf_baseline")
    XGB_TEMPORAL_PATH: str = os.path.join(PROJECT_ROOT, "ml", "models", "xgb_temporal")
    DEMO_DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "demo", "thermal_detections_demo.json")
    PROCESSED_DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "processed", "thermal_detections_processed.csv")
    FIRMS_PREDICTIONS_PATH: str = os.path.join(PROJECT_ROOT, "data", "processed", "firms_1.1M_predictions.parquet")
    FIRMS_EVENT_MAP_PATH: str = os.path.join(PROJECT_ROOT, "data", "processed", "firms_obs_event_map.parquet")
    FUSION_OOF_PREDICTIONS_PATH: str = os.path.join(PROJECT_ROOT, "ml", "models", "fusion_v1", "fusion_oof_predictions_v1.parquet")
    FUSION_METRICS_PATH: str = os.path.join(PROJECT_ROOT, "ml", "models", "fusion_v1", "fusion_metrics_v1.json")
    FIRMS_SAMPLE_SIZE: int = 500
    REQUIRE_REAL_DATA: bool = False
    REQUIRE_TRAINED_MODEL: bool = False

    # Model selection: "self_trained_rf3", "rf_baseline", "xgb_temporal"
    ACTIVE_MODEL: str = "self_trained_rf3"

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
