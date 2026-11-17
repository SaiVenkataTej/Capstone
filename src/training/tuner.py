"""
Optuna Bayesian Hyperparameter Optimization Module for Video Anomaly Detection.
Tunes learning rate, Top-K, regularization weights, dropout, and hidden layer dimensions.
Automatically trains and evaluates the final winning model on the test dataset.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from torch.utils.data import DataLoader

from src.data.dataset import create_dataloaders
from src.models.mil_model import MILAnomalyDetector
from src.training.metrics import compute_metrics
from src.training.trainer import TrainingPipeline


def train_single_epoch(
    model: MILAnomalyDetector,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0
) -> float:
    """Run one training epoch and return average total loss."""
    model.train()
    losses = []
    for batch in train_loader:
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        
        scores = model(features)
        loss, _ = model.compute_loss(scores, labels)
        
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        losses.append(loss.item())
        
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_val_metrics(
    model: MILAnomalyDetector,
    val_loader: DataLoader,
    device: torch.device,
    smooth_sigma: float = 1.0
) -> Dict[str, float]:
    """Evaluate model on validation loader and return video-level classification metrics."""
    model.eval()
    all_scores = []
    all_labels = []
    
    for batch in val_loader:
        features = batch["features"].to(device)
        labels = batch["label"].cpu().numpy()
        scores = model.predict(features, smooth_sigma=smooth_sigma)
        all_scores.append(scores)
        all_labels.append(labels)
        
    scores_arr = np.concatenate(all_scores, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(scores_arr, labels_arr)
    return metrics["video_level"]


def create_objective(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs_per_trial: int = 10,
    loss_type: str = "topk",
    logger: Optional[logging.Logger] = None
):
    """Factory function for Optuna objective supporting all WS-MIL loss formulations."""
    
    def objective(trial: optuna.Trial) -> float:
        # 1. Common Hyperparameter Search Space
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6, step=0.05)
        smooth_weight = trial.suggest_float("smooth_weight", 1e-3, 1e-1, log=True)
        smooth_sigma = trial.suggest_float("smooth_sigma", 0.5, 2.0, step=0.1)
        step_size = trial.suggest_int("step_size", 5, 15)
        gamma = trial.suggest_float("gamma", 0.3, 0.7, step=0.1)
        
        hidden_arch = trial.suggest_categorical("hidden_arch", ["256_128", "512_256", "128_64"])
        if hidden_arch == "256_128":
            hidden_dims = [256, 128]
        elif hidden_arch == "512_256":
            hidden_dims = [512, 256]
        else:
            hidden_dims = [128, 64]
            
        # 2. Loss-Specific Hyperparameters
        k_val = 5
        margin_val = 1.0
        sparse_weight_val = 0.001
        focal_alpha_val = 0.5
        focal_gamma_val = 2.0
        softmax_tau_val = 2.0
        gem_p_val = 3.0
        
        loss_norm = loss_type.lower()
        if loss_norm in ["topk", "focal"]:
            k_val = trial.suggest_int("k", 1, 7)
            
        if loss_norm in ["margin", "max_margin"]:
            margin_val = trial.suggest_float("margin", 0.5, 2.0)
            sparse_weight_val = trial.suggest_float("sparse_weight", 1e-4, 1e-2, log=True)
        elif loss_norm in ["focal", "focal_topk"]:
            focal_alpha_val = trial.suggest_float("focal_alpha", 0.2, 0.8, step=0.1)
            focal_gamma_val = trial.suggest_float("focal_gamma", 1.0, 3.5, step=0.5)
        elif loss_norm in ["attention", "softmax_attention"]:
            softmax_tau_val = trial.suggest_float("softmax_tau", 0.5, 5.0, step=0.5)
        elif loss_norm in ["gem", "gem_pooling"]:
            gem_p_val = trial.suggest_float("gem_p", 1.5, 6.0, step=0.5)
            
        # 3. Build Model & Optimizer
        model = MILAnomalyDetector(
            input_dim=512,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            k=k_val,
            smooth_weight=smooth_weight,
            loss_type=loss_type,
            margin=margin_val,
            sparse_weight=sparse_weight_val,
            alpha=focal_alpha_val,
            gamma=focal_gamma_val,
            temperature=softmax_tau_val,
            p=gem_p_val
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        best_val_auc = 0.0
        
        if logger:
            logger.info(
                f"\n[Optuna] >>> Starting Trial {trial.number:02d} [{loss_type.upper()}] | "
                f"LR={lr:.2e}, Drop={dropout_rate}, Smooth={smooth_weight:.4f}, Arch={hidden_arch}"
            )
        
        # 4. Training Loop with Pruning
        for epoch in range(epochs_per_trial):
            loss_val = train_single_epoch(model, train_loader, optimizer, device)
            scheduler.step()
            
            vl = evaluate_val_metrics(model, val_loader, device, smooth_sigma=smooth_sigma)
            val_auc = vl["roc_auc"]
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                
            if logger is not None and ((epoch + 1) % 2 == 0 or (epoch + 1) == epochs_per_trial):
                logger.info(
                    f"  Trial {trial.number:02d} Epoch {epoch+1:02d}/{epochs_per_trial:02d} -> "
                    f"Loss: {loss_val:.4f} | Val AUC: {val_auc:.4f} | AP: {vl['average_precision']:.4f} | FAR: {vl['false_alarm_rate']:.4f}"
                )
                
            # Report intermediate metric to Optuna for early pruning
            trial.report(val_auc, epoch)
            if trial.should_prune():
                if logger:
                    logger.info(f"  [PRUNED] Trial {trial.number} pruned at epoch {epoch + 1} (Val AUC: {val_auc:.4f})")
                raise optuna.exceptions.TrialPruned()
                
        if logger:
            logger.info(f"[Optuna] <<< Finished Trial {trial.number:02d} | Best Val AUC: {best_val_auc:.4f}")
        return best_val_auc

    return objective


class HyperparameterTuner:
    """
    Optuna Bayesian Hyperparameter Optimization Manager.
    """
    
    def __init__(
        self,
        train_csv: Union[str, Path] = "xd_CLIP_rgb.csv",
        test_csv: Union[str, Path] = "xd_CLIP_rgbtest.csv",
        batch_size: int = 32,
        val_split: float = 0.2,
        loss_type: str = "topk",
        seed: int = 42,
        device: str = "cuda",
        output_dir: Union[str, Path] = "./experiments/optuna_study",
        logger: Optional[logging.Logger] = None
    ):
        self.loss_type = loss_type
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("optuna_tuner")
        
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_csv=train_csv,
            test_csv=test_csv,
            batch_size=batch_size,
            val_split=val_split,
            seed=seed,
            verbose=False
        )
        
    def tune(
        self,
        n_trials: int = 10,
        epochs_per_trial: int = 10,
        final_epochs: int = 30,
        study_name: Optional[str] = None
    ) -> optuna.Study:
        """Run Bayesian optimization study and train the final winning model."""
        if study_name is None:
            study_name = f"mil_anomaly_{self.loss_type}_optuna_study"
            
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING OPTUNA BAYESIAN TUNING FOR [{self.loss_type.upper()}] LOSS")
        self.logger.info("=" * 60)
        self.logger.info(f"Device: {self.device} | Trials: {n_trials} | Epochs/Trial: {epochs_per_trial}")
        
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=2)
        
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        obj_fn = create_objective(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            epochs_per_trial=epochs_per_trial,
            loss_type=self.loss_type,
            logger=self.logger
        )
        
        study.optimize(obj_fn, n_trials=n_trials, show_progress_bar=False)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"OPTUNA TUNING COMPLETED FOR [{self.loss_type.upper()}]!")
        self.logger.info("=" * 60)
        self.logger.info(f"Best Trial #{study.best_trial.number}")
        self.logger.info(f"Best Validation AUC: {study.best_value:.4f}")
        self.logger.info("Optimal Hyperparameters Discovered:")
        for k, v in study.best_params.items():
            self.logger.info(f"  {k:<20}: {v}")
            
        # Save best parameters to JSON
        best_params_path = self.output_dir / "best_hyperparameters.json"
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump({
                "loss_type": self.loss_type,
                "best_trial": study.best_trial.number,
                "best_val_auc": study.best_value,
                "parameters": study.best_params
            }, f, indent=2)
        self.logger.info(f"\nBest hyperparameters saved to: {best_params_path.resolve()}")
        
        # Train Final Optimized Model
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"TRAINING FINAL OPTIMIZED [{self.loss_type.upper()}] MODEL FOR {final_epochs} EPOCHS...")
        self.logger.info("=" * 60)
        
        bp = study.best_params
        hidden_dims = [256, 128] if bp.get("hidden_arch") == "256_128" else ([512, 256] if bp.get("hidden_arch") == "512_256" else [128, 64])
        
        best_model = MILAnomalyDetector(
            input_dim=512,
            hidden_dims=hidden_dims,
            dropout_rate=bp.get("dropout_rate", 0.5),
            k=bp.get("k", 5),
            smooth_weight=bp.get("smooth_weight", 0.01),
            loss_type=self.loss_type,
            margin=bp.get("margin", 1.0),
            sparse_weight=bp.get("sparse_weight", 0.001),
            alpha=bp.get("focal_alpha", 0.5),
            gamma=bp.get("focal_gamma", 2.0),
            temperature=bp.get("softmax_tau", 2.0),
            p=bp.get("gem_p", 3.0)
        )
        
        trainer = TrainingPipeline(
            model=best_model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            device=str(self.device),
            lr=bp.get("learning_rate", 1e-4),
            weight_decay=bp.get("weight_decay", 1e-5),
            step_size=bp.get("step_size", 10),
            gamma=bp.get("gamma", 0.5),
            output_dir=self.output_dir,
            logger=self.logger
        )
        
        # Execute training for final winning model
        patience = max(10, final_epochs // 2)
        trainer.fit(
            num_epochs=final_epochs,
            patience=patience,
            smooth_sigma=bp.get("smooth_sigma", 1.0)
        )
        
        try:
            trainer.plot_history("optuna_training_history.png")
        except Exception as e:
            self.logger.warning(f"Failed to generate optuna_training_history.png plot: {e}")
        
        if self.test_loader is not None:
            # Reload best model weights saved during training
            best_ckpt = self.output_dir / "best_model.pt"
            if best_ckpt.exists():
                best_model.load_checkpoint(best_ckpt, device=self.device)
                self.logger.info(f"Loaded winning checkpoint from {best_ckpt} for final test evaluation.")
                
            _, _, test_metrics = trainer.evaluate(self.test_loader, smooth_sigma=bp.get("smooth_sigma", 1.0))
            with open(self.output_dir / "optuna_test_results.json", "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, indent=2)
            self.logger.info("\n" + "=" * 60)
            self.logger.info("FINAL OPTUNA MODEL TEST RESULTS (6 CORE METRICS):")
            self.logger.info("=" * 60)
            for mk, mv in test_metrics["video_level"].items():
                if isinstance(mv, float):
                    self.logger.info(f"  {mk.replace('_', ' ').title():<25}: {mv:.4f}")
                    
            # Save loss-specific checkpoint to root experiments folder
            root_dir = Path("./experiments")
            root_dir.mkdir(parents=True, exist_ok=True)
            loss_best_path = root_dir / f"best_model_{self.loss_type}.pt"
            study_best = self.output_dir / "best_model.pt"
            
            if study_best.exists():
                shutil.copy(study_best, loss_best_path)
                self.logger.info(f"Saved [{self.loss_type.upper()}] model checkpoint to {loss_best_path.resolve()}")
                
                # If topk or primary run, also update default best_model.pt
                root_default = root_dir / "best_model.pt"
                if self.loss_type == "topk" or not root_default.exists():
                    shutil.copy(study_best, root_default)
                    self.logger.info(f"Updated default checkpoint at {root_default.resolve()}")
                    
        return study, test_metrics
