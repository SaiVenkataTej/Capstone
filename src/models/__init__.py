"""
Model architectures, loss functions, and temporal smoothing for Video Anomaly Detection.
"""

from src.models.losses import (
    BaseMILLoss,
    FocalTopKMILLoss,
    GEMPoolingLoss,
    LossFactory,
    MaxMarginRankingLoss,
    SoftmaxAttentionMILLoss,
    TopKRankingLoss
)
from src.models.mil_model import MILAnomalyDetector, TemporalScoringNetwork
from src.models.smoothing import apply_temporal_smoothing

__all__ = [
    "MILAnomalyDetector",
    "TemporalScoringNetwork",
    "apply_temporal_smoothing",
    "BaseMILLoss",
    "TopKRankingLoss",
    "MaxMarginRankingLoss",
    "FocalTopKMILLoss",
    "SoftmaxAttentionMILLoss",
    "GEMPoolingLoss",
    "LossFactory"
]
