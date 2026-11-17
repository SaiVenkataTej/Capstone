"""
Unit tests for all 5 WS-MIL Loss Formulations and LossFactory.
"""

import unittest
import torch

from src.models.losses import (
    FocalTopKMILLoss,
    GEMPoolingLoss,
    LossFactory,
    MaxMarginRankingLoss,
    SoftmaxAttentionMILLoss,
    TopKRankingLoss
)


class TestLossFunctions(unittest.TestCase):
    
    def setUp(self):
        # Batch of 4 videos, 10 snippets each
        self.B, self.T = 4, 10
        torch.manual_seed(42)
        self.scores = torch.rand(self.B, self.T, requires_grad=True)
        self.labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)
        
    def test_loss_factory(self):
        """Test LossFactory correctly instantiates all loss types."""
        losses = ["topk", "margin", "focal", "attention", "gem"]
        for loss_name in losses:
            loss_fn = LossFactory.create(loss_name, smooth_weight=0.02)
            self.assertIsNotNone(loss_fn)
            
        with self.assertRaises(ValueError):
            LossFactory.create("unsupported_loss")

    def test_topk_ranking_loss(self):
        """Test TopKRankingLoss computation, shape, and gradients."""
        loss_fn = TopKRankingLoss(k=3, smooth_weight=0.01)
        total_loss, loss_dict = loss_fn(self.scores, self.labels)
        
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertEqual(total_loss.ndim, 0)  # Scalar
        self.assertGreater(total_loss.item(), 0.0)
        self.assertIn("anomaly_loss", loss_dict)
        self.assertIn("normal_loss", loss_dict)
        self.assertIn("temporal_loss", loss_dict)
        
        # Test backward pass
        total_loss.backward()
        self.assertIsNotNone(self.scores.grad)

    def test_max_margin_loss(self):
        """Test MaxMarginRankingLoss computation and components."""
        scores = torch.rand(self.B, self.T, requires_grad=True)
        loss_fn = MaxMarginRankingLoss(margin=1.0, smooth_weight=0.01, sparse_weight=0.001)
        total_loss, loss_dict = loss_fn(scores, self.labels)
        
        self.assertGreaterEqual(total_loss.item(), 0.0)
        self.assertIn("hinge_loss", loss_dict)
        self.assertIn("sparse_loss", loss_dict)
        self.assertIn("temporal_loss", loss_dict)
        
        total_loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_focal_topk_loss(self):
        """Test FocalTopKMILLoss computation and components."""
        scores = torch.rand(self.B, self.T, requires_grad=True)
        loss_fn = FocalTopKMILLoss(k=4, alpha=0.5, gamma=2.0, smooth_weight=0.01)
        total_loss, loss_dict = loss_fn(scores, self.labels)
        
        self.assertGreater(total_loss.item(), 0.0)
        self.assertIn("focal_anomaly_loss", loss_dict)
        self.assertIn("focal_normal_loss", loss_dict)
        self.assertIn("temporal_loss", loss_dict)
        
        total_loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_softmax_attention_loss(self):
        """Test SoftmaxAttentionMILLoss computation and components."""
        scores = torch.rand(self.B, self.T, requires_grad=True)
        loss_fn = SoftmaxAttentionMILLoss(temperature=2.0, smooth_weight=0.01)
        total_loss, loss_dict = loss_fn(scores, self.labels)
        
        self.assertGreater(total_loss.item(), 0.0)
        self.assertIn("bce_loss", loss_dict)
        self.assertIn("temporal_loss", loss_dict)
        
        total_loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_gem_pooling_loss(self):
        """Test GEMPoolingLoss computation and components."""
        scores = torch.rand(self.B, self.T, requires_grad=True)
        loss_fn = GEMPoolingLoss(p=3.0, smooth_weight=0.01)
        total_loss, loss_dict = loss_fn(scores, self.labels)
        
        self.assertGreater(total_loss.item(), 0.0)
        self.assertIn("bce_loss", loss_dict)
        self.assertIn("temporal_loss", loss_dict)
        
        total_loss.backward()
        self.assertIsNotNone(scores.grad)


if __name__ == "__main__":
    unittest.main()
