"""
Pydantic schemas for API request and response data models.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    status: str
    device: str
    model_loaded: bool
    checkpoint_path: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class PredictionRequest(BaseModel):
    features: List[List[float]] = Field(
        ...,
        description="Nested list of feature vectors. Shape (T, 512) or (B, T, 512)."
    )
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")
    smooth_sigma: float = Field(1.0, ge=0.0, le=5.0, description="Gaussian smoothing parameter")
    filter_type: str = Field("gaussian", description="Smoothing filter type ('gaussian', 'moving_average', 'none')")


class PredictionResponse(BaseModel):
    video_id: str
    anomaly_scores: List[float]
    is_anomalous: bool
    max_score: float
    threshold: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class VideoAnalysisResult(BaseModel):
    filename: str
    num_snippets: int
    is_anomalous: bool
    max_score: float
    scores: List[float]
    threshold: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class IncidentRecord(BaseModel):
    filename: str
    snippet_idx: int
    score: float
    timestamp: str


class SystemStatistics(BaseModel):
    total_videos: int
    anomalous_videos: int
    total_anomalies: int
    anomaly_rate_percent: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
