import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "AI Industrial Thermal Anomaly Monitor"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"

    DEMO_MODE: bool = True
    MODEL_PATH: str = "ml/models/best_model"
    DEMO_DATA_PATH: str = "data/demo/thermal_detections_demo.json"
    PROCESSED_DATA_PATH: str = "data/processed/thermal_detections_processed.csv"

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
