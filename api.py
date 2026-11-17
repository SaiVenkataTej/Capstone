"""
FastAPI Server Entry Point for Video Anomaly Detection.
Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Or directly:
    python api.py
"""

from src.api.app import app
from src.api.database import IncidentDatabase
from src.api.schemas import (
    HealthStatus,
    PredictionRequest,
    PredictionResponse,
    VideoAnalysisResult
)

__all__ = ["app", "IncidentDatabase", "HealthStatus", "PredictionRequest", "PredictionResponse", "VideoAnalysisResult"]

if __name__ == "__main__":
    import uvicorn
    print("Starting Video Anomaly Detection API Server on http://localhost:8000...")
    print("Interactive Documentation available at http://localhost:8000/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
