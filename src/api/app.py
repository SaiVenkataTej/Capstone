"""
FastAPI Application for Video Anomaly Detection inference and incident logging.
"""

from contextlib import asynccontextmanager
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import torch

from src.api.database import IncidentDatabase
from src.api.schemas import (
    HealthStatus,
    IncidentRecord,
    PredictionRequest,
    PredictionResponse,
    SystemStatistics,
    VideoAnalysisResult
)
from src.models.mil_model import MILAnomalyDetector

# Setup logger
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

# Global instances
device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
model: Optional[MILAnomalyDetector] = None
db: IncidentDatabase = IncidentDatabase()
loaded_checkpoint_path: Optional[str] = None
UPLOAD_DIR = Path("./uploads")


def load_model_weights(checkpoint_path: Path) -> MILAnomalyDetector:
    """Instantiate and load model weights from checkpoint."""
    global loaded_checkpoint_path
    detector = MILAnomalyDetector(input_dim=512, k=5).to(device)
    if checkpoint_path.exists():
        detector.load_checkpoint(checkpoint_path, device=device)
        loaded_checkpoint_path = str(checkpoint_path.resolve())
        logger.info(f"Loaded trained checkpoint from {checkpoint_path}")
    else:
        loaded_checkpoint_path = None
        logger.warning(f"Checkpoint not found at {checkpoint_path}. Using uninitialized model.")
    detector.eval()
    return detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager."""
    global model
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = Path("./experiments/best_model.pt")
    try:
        model = load_model_weights(ckpt_path)
        db.log_event("API Startup", f"Device: {device_name} | Checkpoint: {loaded_checkpoint_path}")
        logger.info("Video Anomaly Detection API initialized successfully.")
    except Exception as e:
        logger.error(f"Error during API startup: {e}")
        model = MILAnomalyDetector(input_dim=512, k=5).to(device).eval()
        
    yield
    
    db.log_event("API Shutdown", "Graceful server termination")
    logger.info("API server shut down gracefully.")


app = FastAPI(
    title="Video Anomaly Detection API",
    description="Production REST API for Weakly Supervised Video Anomaly Detection (WS-MIL)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
async def root():
    """Root metadata endpoint."""
    return {
        "service": "Video Anomaly Detection API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
        "device": device_name,
        "model_loaded": model is not None,
        "checkpoint": loaded_checkpoint_path
    }


@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthStatus(
        status="healthy" if model is not None else "degraded",
        device=device_name,
        model_loaded=model is not None,
        checkpoint_path=loaded_checkpoint_path
    )


@app.get("/info", tags=["Info"])
async def get_info():
    """System and architecture information."""
    return {
        "name": "Video Anomaly Detection System",
        "architecture": "Weakly Supervised Multiple Instance Learning (WS-MIL)",
        "feature_extractor": "CLIP ViT-B/16 (512-D)",
        "device": device_name,
        "checkpoint": loaded_checkpoint_path,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "analyze_video": "/analyze_video",
            "batch_analyze": "/batch_analyze",
            "incidents": "/incidents",
            "statistics": "/statistics",
            "reload_model": "/reload_model"
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_features(request: PredictionRequest):
    """Predict anomaly scores from raw CLIP embeddings."""
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not initialized")
        
    try:
        feats_arr = np.array(request.features, dtype=np.float32)
        if feats_arr.ndim == 2:
            # (T, 512) -> (1, T, 512)
            feats_arr = feats_arr[np.newaxis, ...]
        elif feats_arr.ndim != 3:
            raise ValueError(f"Expected 2D or 3D feature array, got shape {feats_arr.shape}")
            
        tensor_feats = torch.from_numpy(feats_arr).float().to(device)
        with torch.no_grad():
            scores = model.predict(
                tensor_feats,
                smooth_sigma=request.smooth_sigma,
                filter_type=request.filter_type
            )
            
        video_scores = scores[0].tolist()
        max_score = float(np.max(video_scores))
        is_anom = max_score >= request.threshold
        
        db.log_event("Predict Feature Request", f"Max Score: {max_score:.3f} | Anomalous: {is_anom}")
        
        return PredictionResponse(
            video_id=f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            anomaly_scores=video_scores,
            is_anomalous=is_anom,
            max_score=max_score,
            threshold=request.threshold
        )
    except Exception as e:
        logger.error(f"Prediction failure: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/analyze_video", response_model=VideoAnalysisResult, tags=["Inference"])
async def analyze_video(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    smooth_sigma: float = 1.0,
    background_tasks: BackgroundTasks = None
):
    """Upload and analyze a .npy CLIP feature file."""
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not initialized")
        
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only .npy feature files are supported")
        
    save_path = UPLOAD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    try:
        async with aiofiles.open(save_path, "wb") as f:
            content = await file.read()
            await f.write(content)
            
        feats = np.load(save_path).copy()
        if feats.ndim == 1:
            feats = feats[np.newaxis, np.newaxis, :]
        elif feats.ndim == 2:
            feats = feats[np.newaxis, ...]
        elif feats.ndim > 3:
            feats = feats.mean(axis=tuple(range(2, feats.ndim)))
            if feats.ndim == 2:
                feats = feats[np.newaxis, ...]
                
        tensor_feats = torch.from_numpy(feats).float().to(device)
        with torch.no_grad():
            scores = model.predict(tensor_feats, smooth_sigma=smooth_sigma)
            
        video_scores = scores[0].tolist()
        max_score = float(np.max(video_scores))
        is_anom = max_score >= threshold
        
        # Log to database
        db.log_video(
            filename=file.filename,
            num_snippets=len(video_scores),
            is_anomalous=is_anom,
            max_score=max_score
        )
        db.log_anomalies(file.filename, video_scores, threshold=threshold)
        
        if background_tasks is not None:
            background_tasks.add_task(save_path.unlink, missing_ok=True)
            
        return VideoAnalysisResult(
            filename=file.filename,
            num_snippets=len(video_scores),
            is_anomalous=is_anom,
            max_score=max_score,
            scores=video_scores,
            threshold=threshold
        )
    except Exception as e:
        if save_path.exists():
            save_path.unlink(missing_ok=True)
        logger.error(f"Error analyzing video {file.filename}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/batch_analyze", response_model=List[VideoAnalysisResult], tags=["Inference"])
async def batch_analyze(
    files: List[UploadFile] = File(...),
    threshold: float = 0.5,
    smooth_sigma: float = 1.0,
    background_tasks: BackgroundTasks = None
):
    """Analyze a batch of uploaded .npy feature files."""
    results = []
    for upload in files:
        try:
            res = await analyze_video(
                file=upload,
                threshold=threshold,
                smooth_sigma=smooth_sigma,
                background_tasks=background_tasks
            )
            results.append(res)
        except Exception as e:
            logger.warning(f"Batch item failed for {upload.filename}: {e}")
    return results


@app.get("/incidents", tags=["Audit"])
async def get_incidents(limit: int = 20):
    """Retrieve recent anomaly incidents from the database."""
    try:
        df = db.get_recent_incidents(limit=limit)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to fetch incidents: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/statistics", response_model=SystemStatistics, tags=["Audit"])
async def get_statistics():
    """Get system and video classification statistics."""
    try:
        stats = db.get_statistics()
        return SystemStatistics(**stats)
    except Exception as e:
        logger.error(f"Failed to fetch statistics: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/reload_model", tags=["Management"])
async def reload_model(checkpoint_path: str = "./experiments/best_model.pt"):
    """Reload model checkpoint weights dynamically."""
    global model
    path = Path(checkpoint_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Checkpoint not found at {path.resolve()}")
    try:
        model = load_model_weights(path)
        db.log_event("Model Reloaded", str(path.resolve()))
        return {"status": "success", "checkpoint": str(path.resolve())}
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
