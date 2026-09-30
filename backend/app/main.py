from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.endpoints import router as api_router
from app.repositories.detection_repository import detection_repo
from app.services.live_firms_service import live_firms_service

@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.REQUIRE_REAL_DATA and detection_repo.data_source != "firms_predictions":
        raise RuntimeError(detection_repo.data_error or "Required FIRMS data did not load.")
    if detection_repo.data_source == "unavailable":
        print(f"[WARNING] API starting without detection data: {detection_repo.data_error}")
    await live_firms_service.start()
    try:
        yield
    finally:
        await live_firms_service.stop()


app = FastAPI(
    lifespan=lifespan,
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend REST API for AI-based Detection and Classification of Industrial Fires and Persistent Thermal Sources"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router
app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.APP_NAME} REST API",
        "docs_url": "/docs",
        "health_check": "/api/health"
    }
