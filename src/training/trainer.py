"""
Training Pipeline for Video Anomaly Detection model.
Manages training epochs, validation evaluation, checkpointing, and metrics plotting.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.mil_model import MILAnomalyDetector
from src.training.metrics import compute_metrics


class TrainingPipeline:
    """
    End-to-end training and evaluation manager.
    """
    
    def __init__(
        self,
        model: MILAnomalyDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-4,
        learning_rate: Optional[float] = None,
        weight_decay: float = 1e-5,
        step_size: int = 10,
        gamma: float = 0.5,
        grad_clip: float = 1.0,
        output_dir: Union[str, Path] = "./experiments",
        logger: Optional[logging.Logger] = None
    ):
        effective_lr = learning_rate if learning_rate is not None else lr
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("trainer")
        self.grad_clip = grad_clip
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=effective_lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma
        )
        
        self.best_auc = 0.0
        self.best_epoch = 0
        self.history: Dict[str, list] = defaultdict(list)
        
    def train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch over the training dataset."""
        self.model.train()
        epoch_losses: Dict[str, list] = defaultdict(list)
        
        for batch_idx, batch in enumerate(self.train_loader):
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward pass
            scores = self.model(features)
            loss, loss_dict = self.model.compute_loss(scores, labels)
            
            # Backward pass & optimization
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            
            for key, val in loss_dict.items():
                epoch_losses[key].append(val)
                
            # Log progress periodically
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(self.train_loader):
                components = ", ".join(f"{k}: {v:.4f}" for k, v in loss_dict.items() if k != "total_loss")
                self.logger.info(
                    f"  Batch {batch_idx + 1:3d}/{len(self.train_loader):3d} | "
                    f"Loss: {loss.item():.4f} ({components})"
                )
                
        return {key: float(np.mean(vals)) for key, vals in epoch_losses.items()}

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, smooth_sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Evaluate model on a dataset and return scores, labels, and computed metrics."""
        self.model.eval()
        all_scores = []
        all_labels = []
        
        for batch in data_loader:
            features = batch["features"].to(self.device)
            labels = batch["label"].cpu().numpy()
            
            scores = self.model.predict(features, smooth_sigma=smooth_sigma)
            all_scores.append(scores)
            all_labels.append(labels)
            
        all_scores_arr = np.concatenate(all_scores, axis=0)
        all_labels_arr = np.concatenate(all_labels, axis=0)
        
        metrics = compute_metrics(all_scores_arr, all_labels_arr)
        return all_scores_arr, all_labels_arr, metrics

    def fit(self, num_epochs: int = 50, patience: int = 10, smooth_sigma: float = 1.0) -> Dict[str, list]:
        """Execute full training and validation schedule with early stopping."""
        self.logger.info(f"Starting training on device [{self.device}] for {num_epochs} epochs (patience={patience})")
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f"\n--- Epoch {epoch + 1}/{num_epochs} (LR: {self.scheduler.get_last_lr()[0]:.2e}) ---")
            train_losses = self.train_epoch()
            self.scheduler.step()
            
            # Validation
            _, _, val_metrics = self.evaluate(self.val_loader, smooth_sigma=smooth_sigma)
            vl = val_metrics["video_level"]
            val_auc = vl["roc_auc"]
            val_ap = vl["average_precision"]
            val_prec = vl["precision"]
            val_rec = vl["recall"]
            val_f1 = vl["f1_score"]
            val_far = vl["false_alarm_rate"]
            
            self.logger.info(
                f"Epoch {epoch + 1:02d}/{num_epochs:02d} Summary -> "
                f"Loss: {train_losses['total_loss']:.4f} | "
                f"ROC-AUC: {val_auc:.4f} | "
                f"AP: {val_ap:.4f} | "
                f"Precision: {val_prec:.4f} | "
                f"Recall: {val_rec:.4f} | "
                f"F1: {val_f1:.4f} | "
                f"FAR: {val_far:.4f}"
            )
            
            # Track history
            self.history["train_loss"].append(train_losses["total_loss"])
            self.history["val_auc"].append(val_auc)
            self.history["val_ap"].append(val_ap)
            self.history["val_precision"] = self.history.get("val_precision", []) + [val_prec]
            self.history["val_recall"] = self.history.get("val_recall", []) + [val_rec]
            self.history["val_f1"] = self.history.get("val_f1", []) + [val_f1]
            self.history["val_far"].append(val_far)
            
            # Checkpoint on best validation AUC
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                ckpt_path = self.output_dir / "best_model.pt"
                self.model.save_checkpoint(
                    save_path=ckpt_path,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    metrics=val_metrics
                )
                self.logger.info(f"  [SAVED] New best model saved to {ckpt_path} (AUC: {val_auc:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.logger.info(f"\n[EARLY STOPPING] Validation AUC did not improve for {patience} epochs.")
                break
                
        self.logger.info(f"\nBest validation model achieved at Epoch {self.best_epoch} with AUC: {self.best_auc:.4f}")
        
        # Reload best model weights
        best_ckpt = self.output_dir / "best_model.pt"
        if best_ckpt.exists():
            self.model.load_checkpoint(best_ckpt, device=self.device)
            self.logger.info(f"Loaded best checkpoint weights for final evaluation.")
            
        return self.history

    def test(self, smooth_sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Run evaluation on the test set and record results."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EVALUATING ON TEST SET")
        self.logger.info("=" * 60)
        
        scores, labels, metrics = self.evaluate(self.test_loader, smooth_sigma=smooth_sigma)
        
        self.logger.info("\n[VIDEO-LEVEL METRICS]")
        for metric, val in metrics["video_level"].items():
            self.logger.info(f"  {metric.upper():<12}: {val:.4f}")
            
        self.logger.info("\n[SNIPPET-LEVEL METRICS (Approx)]")
        for metric, val in metrics["snippet_level"].items():
            self.logger.info(f"  {metric.upper():<12}: {val:.4f}")
            
        # Save results to JSON
        test_results = {
            "best_epoch": self.best_epoch,
            "best_val_auc": self.best_auc,
            "video_level": metrics["video_level"],
            "snippet_level": metrics["snippet_level"]
        }
        results_path = self.output_dir / "test_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)
        self.logger.info(f"\nTest results saved to: {results_path}")
        
        return scores, labels, metrics

    def plot_history(self, save_filename: Union[str, Path] = "training_history.png") -> Path:
        """Plot and save training loss, validation AUC, and validation FAR curves."""
        if not self.history["train_loss"]:
            return self.output_dir / "training_history.png"
            
        p = Path(save_filename)
        if p.is_absolute() or len(p.parts) > 1:
            plot_path = p
        else:
            plot_path = self.output_dir / p
            
        plot_path.parent.mkdir(parents=True, exist_ok=True)
            
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history["train_loss"], "b-o", label="Train Loss")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()
        
        # AUC
        axes[1].plot(epochs, self.history["val_auc"], "g-s", label="Val AUC")
        axes[1].axhline(y=self.best_auc, color="r", linestyle="--", label=f"Best: {self.best_auc:.4f}")
        axes[1].set_title("Validation ROC-AUC")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()
        
        # FAR
        axes[2].plot(epochs, self.history["val_far"], "m-^", label="Val FAR")
        axes[2].set_title("Validation False Alarm Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("FAR")
        axes[2].grid(True, linestyle="--", alpha=0.6)
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=120)
        plt.close()
        self.logger.info(f"Training history plot saved to: {plot_path}")
        return plot_path
