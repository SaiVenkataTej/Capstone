"""
Unit tests for model architecture, forward pass, and ranking loss using standard unittest.
"""

from pathlib import Path
import unittest
import numpy as np
import torch

from src.models.mil_model import MILAnomalyDetector, TemporalScoringNetwork, TopKRankingLoss
from src.models.smoothing import apply_temporal_smoothing


class TestModelArchitecture(unittest.TestCase):
    
    def test_temporal_scoring_network_shapes(self):
        """Verify forward pass output shapes for single and batched tensors."""
        B, T, D = 4, 10, 512
        scorer = TemporalScoringNetwork(input_dim=D, hidden_dims=[256, 128], dropout_rate=0.2)
        scorer.eval()
        
        # Batch input
        x_batch = torch.randn(B, T, D)
        out_batch = scorer(x_batch)
        self.assertEqual(out_batch.shape, (B, T, 1))
        self.assertTrue((out_batch >= 0.0).all() and (out_batch <= 1.0).all())
        
        # Single video input
        x_single = torch.randn(T, D)
        out_single = scorer(x_single)
        self.assertEqual(out_single.shape, (1, T, 1))

    def test_topk_ranking_loss_computation(self):
        """Verify loss calculation and components for mixed batches."""
        criterion = TopKRankingLoss(k=3, smooth_weight=0.02)
        
        scores = torch.tensor([
            [0.1, 0.2, 0.8, 0.9, 0.7],  # Anomaly video
            [0.05, 0.1, 0.08, 0.12, 0.09]  # Normal video
        ], dtype=torch.float32)
        labels = torch.tensor([1, 0], dtype=torch.long)
        
        total_loss, loss_dict = criterion(scores, labels)
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertGreater(total_loss.item(), 0.0)
        self.assertIn("anomaly_loss", loss_dict)
        self.assertGreater(loss_dict["anomaly_loss"], 0.0)
        self.assertIn("normal_loss", loss_dict)
        self.assertGreater(loss_dict["normal_loss"], 0.0)
        self.assertIn("temporal_loss", loss_dict)

    def test_mil_detector_predict_and_checkpoint(self):
        """Test full MIL wrapper prediction and checkpoint saving/loading."""
        model = MILAnomalyDetector(input_dim=512, k=5)
        dummy_feats = torch.randn(2, 10, 512)
        
        preds = model.predict(dummy_feats, smooth_sigma=1.0)
        self.assertEqual(preds.shape, (2, 10))
        
        # Save checkpoint
        ckpt_file = Path("./experiments/test_tmp_model.pt")
        model.save_checkpoint(ckpt_file, epoch=1, metrics={"auc": 0.95})
        self.assertTrue(ckpt_file.exists())
        
        # Load into new instance
        new_model = MILAnomalyDetector(input_dim=512, k=5)
        loaded_data = new_model.load_checkpoint(ckpt_file)
        self.assertEqual(loaded_data["epoch"], 1)
        self.assertEqual(loaded_data["metrics"]["auc"], 0.95)
        
        # Cleanup
        if ckpt_file.exists():
            ckpt_file.unlink()

    def test_temporal_smoothing_functions(self):
        """Test Gaussian and moving average smoothing filters."""
        scores = np.array([0.1, 0.9, 0.1, 0.8, 0.2])
        smoothed_gauss = apply_temporal_smoothing(scores, sigma=1.0, filter_type="gaussian")
        self.assertEqual(smoothed_gauss.shape, scores.shape)
        self.assertLess(smoothed_gauss[1], 0.9)
        
        smoothed_ma = apply_temporal_smoothing(scores, sigma=1.0, filter_type="moving_average")
        self.assertEqual(smoothed_ma.shape, scores.shape)


if __name__ == "__main__":
    unittest.main()
