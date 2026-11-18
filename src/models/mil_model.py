"""
Multiple Instance Learning (MIL) Model and Loss Formulation.
Implements Weakly Supervised Video Anomaly Detection with Top-K ranking loss.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.losses import LossFactory, TopKRankingLoss
from src.models.smoothing import apply_temporal_smoothing


class TemporalScoringNetwork(nn.Module):
    """
    Lightweight snippet-level anomaly scoring network.
    
    Transforms 512-D CLIP features of each snippet into an anomaly probability in [0, 1].
    Uses GroupNorm for batch-size invariance (preventing crashes on batch_size=1 or edge batches).
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.5
    ):
        super(TemporalScoringNetwork, self).__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]
            
        layers = []
        prev_dim = input_dim
        
        # Hidden layer 1
        layers.append(nn.Linear(prev_dim, hidden_dims[0]))
        num_groups_1 = min(32, hidden_dims[0])
        layers.append(nn.GroupNorm(num_groups=num_groups_1, num_channels=hidden_dims[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))
        prev_dim = hidden_dims[0]
        
        # Hidden layer 2
        if len(hidden_dims) > 1:
            layers.append(nn.Linear(prev_dim, hidden_dims[1]))
            num_groups_2 = min(16, hidden_dims[1])
            layers.append(nn.GroupNorm(num_groups=num_groups_2, num_channels=hidden_dims[1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dims[1]
            
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, D) or (T, D) or (B*T, D)
            
        Returns:
            scores: Anomaly probability scores of shape (B, T, 1) or (B, 1)
        """
        if x.ndim == 2:
            # (T, D) -> (1, T, D)
            x = x.unsqueeze(0)
            
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        scores = self.mlp(x_flat)
        return scores.reshape(B, T, 1)


class MILAnomalyDetector(nn.Module):
    """
    End-to-End Weakly Supervised Anomaly Detector Wrapper.
    Supports modular loss functions: Top-K Ranking, Max-Margin, Focal, Softmax Attention, GEM.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.5,
        k: int = 5,
        smooth_weight: float = 0.01,
        loss_type: str = "topk",
        criterion: Optional[nn.Module] = None,
        **loss_kwargs
    ):
        super(MILAnomalyDetector, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout_rate = dropout_rate
        self.scorer = TemporalScoringNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        if criterion is not None:
            self.criterion = criterion
        else:
            kwargs = {"k": k, "smooth_weight": smooth_weight}
            kwargs.update(loss_kwargs)
            self.criterion = LossFactory.create(loss_type, **kwargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through scoring network."""
        return self.scorer(x)
        
    def compute_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute ranking loss given predicted scores and weak labels."""
        return self.criterion(scores, labels)
        
    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        smooth_sigma: float = 1.0,
        filter_type: str = "gaussian"
    ) -> np.ndarray:
        """
        Run inference and apply temporal smoothing.
        
        Args:
            features: Tensor of shape (B, T, D) or (T, D)
            smooth_sigma: Smoothing standard deviation
            filter_type: 'gaussian' or 'moving_average'
            
        Returns:
            Numpy array of shape (B, T) with smoothed anomaly scores.
        """
        self.eval()
        if features.ndim == 2:
            features = features.unsqueeze(0)
            
        scores = self.forward(features).squeeze(-1).cpu().numpy()  # (B, T)
        return apply_temporal_smoothing(scores, sigma=smooth_sigma, filter_type=filter_type)
        
    def save_checkpoint(
        self,
        save_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model checkpoint and training state."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "metrics": metrics
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu"
    ) -> Dict[str, Any]:
        """Load weights from a checkpoint file, automatically adapting architecture if dimensions differ."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path.resolve()}")
            
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # Auto-detect hidden architecture from checkpoint weights if necessary
        if "scorer.mlp.0.weight" in state_dict:
            ckpt_in_dim = state_dict["scorer.mlp.0.weight"].shape[1]
            h1 = state_dict["scorer.mlp.0.weight"].shape[0]
            detected_hidden_dims = [h1]
            if "scorer.mlp.4.weight" in state_dict:
                h2 = state_dict["scorer.mlp.4.weight"].shape[0]
                detected_hidden_dims.append(h2)
            
            # Check current scorer dimensions
            current_h1 = self.scorer.mlp[0].weight.shape[0]
            current_h2 = self.scorer.mlp[4].weight.shape[0] if len(self.scorer.mlp) > 5 else None
            current_dims = [current_h1] + ([current_h2] if current_h2 is not None else [])
            
            if current_dims != detected_hidden_dims or ckpt_in_dim != self.input_dim:
                self.input_dim = ckpt_in_dim
                self.hidden_dims = detected_hidden_dims
                self.scorer = TemporalScoringNetwork(
                    input_dim=ckpt_in_dim,
                    hidden_dims=detected_hidden_dims,
                    dropout_rate=self.dropout_rate
                )

        self.load_state_dict(state_dict)
        self.to(device)
        self.eval()
        return checkpoint


if __name__ == "__main__":
    print("=" * 60)
    print("MIL Anomaly Detector - Architecture Sanity Check")
    print("=" * 60)
    
    # 1. Test Scorer
    B, T, D = 4, 10, 512
    dummy_x = torch.randn(B, T, D)
    dummy_y = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    
    model = MILAnomalyDetector(input_dim=D, k=5)
    scores = model(dummy_x)
    total_loss, loss_dict = model.compute_loss(scores, dummy_y)
    preds = model.predict(dummy_x, smooth_sigma=1.0)
    
    print(f"  Input Features:     {dummy_x.shape} (Batch=4, Snippets=10, D=512)")
    print(f"  Snippet Scores:     {scores.shape} (Range: [{scores.min():.3f}, {scores.max():.3f}])")
    print(f"  Smoothed Preds:     {preds.shape}")
    print(f"  Computed Loss:      {total_loss.item():.4f}")
    print(f"  Loss Components:    {loss_dict}")
    print("\n[OK] Model Architecture and Loss Functions Verified Successfully!")
