"""
REST API module using FastAPI for Video Anomaly Detection.
"""

from src.api.app import app
from src.api.database import IncidentDatabase
from src.api.schemas import (
    HealthStatus,
    PredictionRequest,
    PredictionResponse,
    VideoAnalysisResult
)

__all__ = [
    "app",
    "IncidentDatabase",
    "HealthStatus",
    "PredictionRequest",
    "PredictionResponse",
    "VideoAnalysisResult"
]
