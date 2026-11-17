"""
Unit test for Optuna hyperparameter optimization.
"""

from pathlib import Path
import unittest
import torch
import optuna

from src.training.tuner import create_objective
from torch.utils.data import DataLoader, TensorDataset


class TestOptunaTuning(unittest.TestCase):
    
    def test_optuna_objective_and_trial(self):
        """Run a 1-trial quick smoke test for the Optuna objective function."""
        # Create dummy dataloaders
        B, T, D = 4, 10, 512
        x_dummy = torch.randn(8, T, D)
        y_dummy = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1], dtype=torch.long)
        
        ds = TensorDataset(x_dummy, y_dummy)
        
        # Format as dictionary batches to match dataset contract
        class DummyDictDataset(torch.utils.data.Dataset):
            def __len__(self): return 8
            def __getitem__(self, idx):
                return {"features": x_dummy[idx], "label": y_dummy[idx], "video_id": f"v_{idx}"}
                
        dummy_ds = DummyDictDataset()
        train_loader = DataLoader(dummy_ds, batch_size=4)
        val_loader = DataLoader(dummy_ds, batch_size=4)
        
        device = torch.device("cpu")
        obj_fn = create_objective(train_loader, val_loader, device=device, epochs_per_trial=2)
        
        study = optuna.create_study(direction="maximize")
        study.optimize(obj_fn, n_trials=1)
        
        self.assertEqual(len(study.trials), 1)
        self.assertGreaterEqual(study.best_value, 0.0)


if __name__ == "__main__":
    unittest.main()
