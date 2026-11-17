"""
Temporal smoothing post-processing utilities for anomaly detection scores.
"""

from typing import Union
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch


def apply_temporal_smoothing(
    scores: Union[np.ndarray, torch.Tensor],
    sigma: float = 1.0,
    filter_type: str = "gaussian"
) -> np.ndarray:
    """Apply 1D temporal smoothing filter across snippet anomaly scores.
    
    Args:
        scores: 1D array (T,) or 2D array (B, T) of anomaly scores.
        sigma: Standard deviation for Gaussian kernel.
        filter_type: 'gaussian' or 'moving_average'.
        
    Returns:
        Smoothed numpy array of same shape.
    """
    if isinstance(scores, torch.Tensor):
        scores_arr = scores.detach().cpu().numpy()
    else:
        scores_arr = np.array(scores, copy=True)
        
    if sigma <= 0 or filter_type == "none":
        return scores_arr
        
    if scores_arr.ndim == 1:
        if filter_type == "gaussian":
            return gaussian_filter1d(scores_arr, sigma=sigma, mode="nearest")
        elif filter_type == "moving_average":
            window = max(1, int(2 * sigma + 1))
            return np.convolve(scores_arr, np.ones(window) / window, mode="same")
        return scores_arr
        
    elif scores_arr.ndim == 2:
        smoothed = np.zeros_like(scores_arr)
        for i in range(scores_arr.shape[0]):
            if filter_type == "gaussian":
                smoothed[i] = gaussian_filter1d(scores_arr[i], sigma=sigma, mode="nearest")
            elif filter_type == "moving_average":
                window = max(1, int(2 * sigma + 1))
                smoothed[i] = np.convolve(scores_arr[i], np.ones(window) / window, mode="same")
            else:
                smoothed[i] = scores_arr[i]
        return smoothed
        
    return scores_arr
