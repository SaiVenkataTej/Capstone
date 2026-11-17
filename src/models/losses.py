"""
Modular Loss Functions for Weakly Supervised Video Anomaly Detection (WS-MIL).

Implements 5 distinct loss formulations for comparative study and research benchmarking:
1. TopKRankingLoss          - Baseline Top-K snippet maximization with normal suppression.
2. MaxMarginRankingLoss     - Sultani et al. (CVPR 2018) hinge ranking loss with sparsity constraint.
3. FocalTopKMILLoss         - Focal loss formulation with dynamic hard-negative modulation.
4. SoftmaxAttentionMILLoss  - Fully differentiable temperature-scaled attention MIL loss.
5. GEMPoolingLoss           - Generalized Mean (p-norm) pooling binary cross-entropy loss.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMILLoss(nn.Module):
    """Base class for all Weakly Supervised Video Anomaly Detection loss functions."""
    
    def __init__(self, smooth_weight: float = 0.01):
        super().__init__()
        self.smooth_weight = smooth_weight
        
    def compute_temporal_smoothness(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute sum of squared consecutive temporal differences."""
        if scores.ndim == 3:
            scores = scores.squeeze(-1)
        if scores.shape[1] > 1:
            return (torch.diff(scores, dim=1) ** 2).mean()
        return torch.tensor(0.0, device=scores.device)


class TopKRankingLoss(BaseMILLoss):
    """
    Top-K Ranking Loss for Weakly Supervised Video Anomaly Detection.
    
    - Anomalous video: Maximizes the top-k highest scoring snippets.
    - Normal video: Minimizes all snippet scores towards zero.
    - Regularization: Penalizes rapid temporal frame-to-frame score oscillations.
    """
    
    def __init__(self, k: int = 5, smooth_weight: float = 0.01):
        super().__init__(smooth_weight=smooth_weight)
        self.k = k
        
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if scores.ndim == 3:
            scores = scores.squeeze(-1)  # (B, T)
            
        B, T = scores.shape
        anomaly_mask = (labels == 1)
        normal_mask = (labels == 0)
        
        loss = torch.tensor(0.0, device=scores.device)
        anomaly_loss_val = 0.0
        normal_loss_val = 0.0
        
        # 1. Anomalous video loss (per-video Top-K snippet maximization)
        if anomaly_mask.sum() > 0:
            anomaly_scores = scores[anomaly_mask]  # (N_anom, T)
            effective_k = min(self.k, anomaly_scores.shape[1])
            topk_scores = torch.topk(anomaly_scores, k=effective_k, dim=1)[0]  # (N_anom, k)
            anomaly_loss = -torch.log(topk_scores.mean(dim=1) + 1e-8).mean()
            loss = loss + anomaly_loss
            anomaly_loss_val = float(anomaly_loss.item())
            
        # 2. Normal video loss (all snippet minimization)
        if normal_mask.sum() > 0:
            normal_scores = scores[normal_mask]  # (N_norm, T)
            normal_loss = -torch.log(1.0 - normal_scores + 1e-8).mean()
            loss = loss + normal_loss
            normal_loss_val = float(normal_loss.item())
            
        # 3. Temporal smoothness
        temporal_loss = self.compute_temporal_smoothness(scores)
        total_loss = loss + self.smooth_weight * temporal_loss
        
        return total_loss, {
            "total_loss": float(total_loss.item()),
            "anomaly_loss": anomaly_loss_val,
            "normal_loss": normal_loss_val,
            "temporal_loss": float(temporal_loss.item())
        }


class MaxMarginRankingLoss(BaseMILLoss):
    """
    Max-Margin Ranking Loss (Sultani et al., CVPR 2018).
    
    Enforces that the maximum anomaly score in an anomalous video exceeds
    the maximum score in a normal video by a margin m:
        max(0, m - max_t(s^a_t) + max_t(s^n_t))
    Also incorporates temporal smoothness and temporal sparsity constraints.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        smooth_weight: float = 0.01,
        sparse_weight: float = 0.001
    ):
        super().__init__(smooth_weight=smooth_weight)
        self.margin = margin
        self.sparse_weight = sparse_weight
        
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if scores.ndim == 3:
            scores = scores.squeeze(-1)
            
        anomaly_mask = (labels == 1)
        normal_mask = (labels == 0)
        
        loss = torch.tensor(0.0, device=scores.device)
        hinge_loss_val = 0.0
        sparse_loss_val = 0.0
        
        if anomaly_mask.sum() > 0 and normal_mask.sum() > 0:
            # Maximum snippet score per video
            max_anom = scores[anomaly_mask].max(dim=1, keepdim=True)[0]  # (N_anom, 1)
            max_norm = scores[normal_mask].max(dim=1, keepdim=True)[0]  # (N_norm, 1)
            
            # Pairwise difference across all anomalous vs normal pairs in batch
            # Shape: (N_anom, N_norm)
            diff_matrix = max_norm.t() - max_anom
            hinge_pairs = torch.clamp(self.margin + diff_matrix, min=0.0)
            hinge_loss = hinge_pairs.mean()
            loss = loss + hinge_loss
            hinge_loss_val = float(hinge_loss.item())
        elif anomaly_mask.sum() > 0:
            # Fallback if batch contains only anomalous videos
            max_anom = scores[anomaly_mask].max(dim=1)[0]
            hinge_loss = -torch.log(max_anom + 1e-8).mean()
            loss = loss + hinge_loss
            hinge_loss_val = float(hinge_loss.item())
        elif normal_mask.sum() > 0:
            # Fallback if batch contains only normal videos
            max_norm = scores[normal_mask].max(dim=1)[0]
            hinge_loss = -torch.log(1.0 - max_norm + 1e-8).mean()
            loss = loss + hinge_loss
            hinge_loss_val = float(hinge_loss.item())
            
        # Sparsity constraint on anomalous videos (anomalies are temporally sparse)
        if anomaly_mask.sum() > 0:
            sparse_loss = scores[anomaly_mask].mean()
            sparse_loss_val = float(sparse_loss.item())
        else:
            sparse_loss = torch.zeros(1, device=scores.device)
            
        # Temporal smoothness
        temporal_loss = self.compute_temporal_smoothness(scores)
        
        total_loss = loss + self.smooth_weight * temporal_loss + self.sparse_weight * sparse_loss
        
        return total_loss, {
            "total_loss": float(total_loss.item()),
            "hinge_loss": hinge_loss_val,
            "sparse_loss": sparse_loss_val,
            "temporal_loss": float(temporal_loss.item())
        }


class FocalTopKMILLoss(BaseMILLoss):
    """
    Focal Top-K MIL Loss.
    
    Extends Top-K ranking with focal modulating factors (1 - p)^gamma to
    prevent easy normal/anomalous videos from dominating gradients and focus on
    ambiguous, hard temporal transitions.
    """
    
    def __init__(
        self,
        k: int = 5,
        alpha: float = 0.5,
        gamma: float = 2.0,
        smooth_weight: float = 0.01
    ):
        super().__init__(smooth_weight=smooth_weight)
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if scores.ndim == 3:
            scores = scores.squeeze(-1)
            
        anomaly_mask = (labels == 1)
        normal_mask = (labels == 0)
        
        loss = torch.tensor(0.0, device=scores.device)
        focal_anom_val = 0.0
        focal_norm_val = 0.0
        
        # 1. Focal loss on anomalous top-k snippets
        if anomaly_mask.sum() > 0:
            anomaly_scores = scores[anomaly_mask]
            effective_k = min(self.k, anomaly_scores.shape[1])
            p_topk = torch.topk(anomaly_scores, k=effective_k, dim=1)[0].mean(dim=1)  # (N_anom,)
            p_topk = torch.clamp(p_topk, 1e-7, 1.0 - 1e-7)
            
            focal_weight = self.alpha * ((1.0 - p_topk) ** self.gamma)
            focal_anom = -(focal_weight * torch.log(p_topk)).mean()
            loss = loss + focal_anom
            focal_anom_val = float(focal_anom.item())
            
        # 2. Focal loss on normal snippets
        if normal_mask.sum() > 0:
            p_norm = scores[normal_mask]  # (N_norm, T)
            p_norm = torch.clamp(p_norm, 1e-7, 1.0 - 1e-7)
            
            focal_weight = (1.0 - self.alpha) * (p_norm ** self.gamma)
            focal_norm = -(focal_weight * torch.log(1.0 - p_norm)).mean()
            loss = loss + focal_norm
            focal_norm_val = float(focal_norm.item())
            
        temporal_loss = self.compute_temporal_smoothness(scores)
        total_loss = loss + self.smooth_weight * temporal_loss
        
        return total_loss, {
            "total_loss": float(total_loss.item()),
            "focal_anomaly_loss": focal_anom_val,
            "focal_normal_loss": focal_norm_val,
            "temporal_loss": float(temporal_loss.item())
        }


class SoftmaxAttentionMILLoss(BaseMILLoss):
    """
    Differentiable Softmax Attention MIL Loss (Deep MIL).
    
    Replaces discrete Top-K selection with a continuous, temperature-scaled
    attention distribution across snippets:
        w_t = exp(tau * s_t) / sum_j exp(tau * s_j)
        y_hat = sum_t (w_t * s_t)
        loss = BinaryCrossEntropy(y_hat, y)
    """
    
    def __init__(self, temperature: float = 2.0, smooth_weight: float = 0.01):
        super().__init__(smooth_weight=smooth_weight)
        self.temperature = temperature
        
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if scores.ndim == 3:
            scores = scores.squeeze(-1)
            
        # Compute temperature-scaled attention weights per video
        weights = F.softmax(scores * self.temperature, dim=1)  # (B, T)
        
        # Bag-level continuous anomaly score
        bag_scores = torch.sum(weights * scores, dim=1)  # (B,)
        bag_scores = torch.clamp(bag_scores, 1e-7, 1.0 - 1e-7)
        
        target = labels.float()
        bce_loss = F.binary_cross_entropy(bag_scores, target)
        
        temporal_loss = self.compute_temporal_smoothness(scores)
        total_loss = bce_loss + self.smooth_weight * temporal_loss
        
        return total_loss, {
            "total_loss": float(total_loss.item()),
            "bce_loss": float(bce_loss.item()),
            "temporal_loss": float(temporal_loss.item())
        }


class GEMPoolingLoss(BaseMILLoss):
    """
    Generalized Mean (GEM / p-norm) Pooling Loss.
    
    Computes a smooth differentiable interpolation between average pooling (p=1)
    and max pooling (p -> inf):
        GEM(s) = ( (1/T) * sum_t s_t^p )^(1/p)
        loss = BinaryCrossEntropy(GEM(s), y)
    """
    
    def __init__(self, p: float = 3.0, smooth_weight: float = 0.01):
        super().__init__(smooth_weight=smooth_weight)
        self.p = p
        
    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if scores.ndim == 3:
            scores = scores.squeeze(-1)
            
        clipped_scores = torch.clamp(scores, min=1e-6)
        # GEM pooling along temporal dimension
        gem_scores = (torch.mean(clipped_scores ** self.p, dim=1)) ** (1.0 / self.p)
        gem_scores = torch.clamp(gem_scores, 1e-7, 1.0 - 1e-7)
        
        target = labels.float()
        bce_loss = F.binary_cross_entropy(gem_scores, target)
        
        temporal_loss = self.compute_temporal_smoothness(scores)
        total_loss = bce_loss + self.smooth_weight * temporal_loss
        
        return total_loss, {
            "total_loss": float(total_loss.item()),
            "bce_loss": float(bce_loss.item()),
            "temporal_loss": float(temporal_loss.item())
        }


class LossFactory:
    """Convenience factory to instantiate loss functions by name."""
    
    AVAILABLE_LOSSES = {
        "topk": TopKRankingLoss,
        "margin": MaxMarginRankingLoss,
        "max_margin": MaxMarginRankingLoss,
        "focal": FocalTopKMILLoss,
        "focal_topk": FocalTopKMILLoss,
        "attention": SoftmaxAttentionMILLoss,
        "softmax_attention": SoftmaxAttentionMILLoss,
        "gem": GEMPoolingLoss,
        "gem_pooling": GEMPoolingLoss
    }
    
    @classmethod
    def create(cls, loss_type: str = "topk", **kwargs) -> BaseMILLoss:
        name = loss_type.lower().strip()
        if name not in cls.AVAILABLE_LOSSES:
            valid = list(cls.AVAILABLE_LOSSES.keys())
            raise ValueError(f"Unknown loss type '{loss_type}'. Available options: {valid}")
            
        loss_cls = cls.AVAILABLE_LOSSES[name]
        
        # Filter kwargs to only those accepted by the chosen class
        import inspect
        sig = inspect.signature(loss_cls.__init__)
        valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        return loss_cls(**valid_args)
