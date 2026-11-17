"""
Evaluation metrics computation for video-level and snippet-level anomaly detection.
Computes the 6 core metrics:
1. ROC-AUC (Area Under the Receiver Operating Characteristic Curve)
2. Average Precision (AP / PR-AUC)
3. Precision
4. Recall (Sensitivity / True Positive Rate)
5. F1-Score (Harmonic Mean of Precision and Recall)
6. False Alarm Rate (FAR / False Positive Rate on Normal Videos)
"""

from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


class MetricsCalculator:
    """Calculates the 6 primary video-level and snippet-level anomaly classification metrics."""
    
    @staticmethod
    def compute_video_level_metrics(
        all_scores: np.ndarray,
        all_labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute video-level classification metrics.
        
        Args:
            all_scores: Array of shape (N, T) with snippet anomaly scores.
            all_labels: Array of shape (N,) with true video-level binary labels (0=normal, 1=violent).
            threshold: Decision threshold for classification (default: 0.5).
            
        Returns:
            Dictionary of metrics:
            - "auc" / "roc_auc": Area Under ROC Curve
            - "ap" / "average_precision": Average Precision
            - "precision": Precision (TP / (TP + FP))
            - "recall": Recall (TP / (TP + FN))
            - "f1" / "f1_score": F1-Score
            - "far" / "false_alarm_rate": False Alarm Rate (FP / (FP + TN))
            - "accuracy": Overall Accuracy
        """
        # Aggregate snippet scores to video-level prediction via Top-1 max pooling
        video_scores = all_scores.max(axis=1) if all_scores.ndim > 1 else all_scores
        video_scores = np.clip(video_scores, 0.0, 1.0)
        video_preds = (video_scores >= threshold).astype(int)
        
        # 1. ROC-AUC (Area Under ROC Curve)
        try:
            auc = float(roc_auc_score(all_labels, video_scores))
        except ValueError:
            auc = 0.5
            
        # 2. Average Precision (AP / PR-AUC)
        try:
            ap = float(average_precision_score(all_labels, video_scores))
        except ValueError:
            ap = 0.0
            
        # 3. Precision, 4. Recall, 5. F1-score
        precision = float(precision_score(all_labels, video_preds, zero_division=0))
        recall = float(recall_score(all_labels, video_preds, zero_division=0))
        f1 = float(f1_score(all_labels, video_preds, zero_division=0))
        accuracy = float((video_preds == all_labels).mean())
        
        # 6. False Alarm Rate (FAR): False positive rate on normal videos
        normal_mask = (all_labels == 0)
        far = float((video_preds[normal_mask] == 1).mean()) if normal_mask.sum() > 0 else 0.0
            
        return {
            "auc": auc,
            "roc_auc": auc,
            "ap": ap,
            "average_precision": ap,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f1_score": f1,
            "far": far,
            "false_alarm_rate": far,
            "accuracy": accuracy
        }

    @staticmethod
    def compute_snippet_level_metrics(
        all_scores: np.ndarray,
        all_labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute snippet-level metrics using video-level pseudo-labels.
        """
        T = all_scores.shape[1] if all_scores.ndim > 1 else 1
        expanded_labels = np.repeat(all_labels, T)
        flat_scores = np.clip(all_scores.reshape(-1), 0.0, 1.0)
        preds = (flat_scores >= threshold).astype(int)
        
        try:
            auc = float(roc_auc_score(expanded_labels, flat_scores))
        except ValueError:
            auc = 0.5
            
        try:
            ap = float(average_precision_score(expanded_labels, flat_scores))
        except ValueError:
            ap = 0.0
            
        precision = float(precision_score(expanded_labels, preds, zero_division=0))
        recall = float(recall_score(expanded_labels, preds, zero_division=0))
        f1 = float(f1_score(expanded_labels, preds, zero_division=0))
            
        return {
            "auc": auc,
            "roc_auc": auc,
            "ap": ap,
            "average_precision": ap,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f1_score": f1
        }


def compute_metrics(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """Convenience function to compute both video-level and snippet-level metrics."""
    return {
        "video_level": MetricsCalculator.compute_video_level_metrics(all_scores, all_labels, threshold),
        "snippet_level": MetricsCalculator.compute_snippet_level_metrics(all_scores, all_labels, threshold)
    }
