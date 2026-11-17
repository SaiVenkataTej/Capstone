"""
Train and Evaluate Optimal Optuna Models for Research Paper Publication.

Loads the best hyperparameters discovered by Optuna for each loss function:
  - topk
  - margin
  - gem
  - attention
  - focal

Trains each winning model with full validation checkpointing (trainer.fit),
evaluates on the test set, and updates the publication-grade LaTeX, CSV, and JSON comparison tables.
"""

import argparse
import json
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from src.data.dataset import create_dataloaders
from src.models.mil_model import MILAnomalyDetector
from src.training.trainer import TrainingPipeline
from src.utils.logger import setup_logger

ALL_LOSSES = ["topk", "margin", "gem", "attention", "focal"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and benchmark optimal models using Optuna parameters")
    parser.add_argument(
        "--loss", nargs="+", default=["all"],
        help="Losses to train (choices: topk, margin, gem, attention, focal, all). Default: all"
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience (default: 12)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--train_csv", type=str, default="xd_CLIP_rgb.csv", help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, default="xd_CLIP_rgbtest.csv", help="Path to test CSV")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output_dir", type=str, default="./experiments")
    return parser.parse_args()


def train_single_optimal(
    loss_name: str,
    base_output_dir: Path,
    train_loader,
    val_loader,
    test_loader,
    epochs: int = 25,
    patience: int = 12,
    device: str = "cpu",
    seed: int = 42
) -> Optional[Dict]:
    study_dir = base_output_dir / f"optuna_study_{loss_name}"
    hp_path = study_dir / "best_hyperparameters.json"
    
    if not hp_path.exists():
        print(f"[WARN] Hyperparameter file not found: {hp_path}. Skipping [{loss_name}].")
        return None
        
    with open(hp_path, "r", encoding="utf-8") as f:
        hp_info = json.load(f)
        
    p = hp_info["parameters"]
    best_val_auc = hp_info.get("best_val_auc", 0.0)
    
    # Resolve architecture
    arch_str = p.get("hidden_arch", "256_128")
    if arch_str == "512_256":
        hidden_dims = [512, 256]
    elif arch_str == "128_64":
        hidden_dims = [128, 64]
    else:
        hidden_dims = [256, 128]
        
    logger = setup_logger(f"optimal_train_{loss_name}", log_dir=study_dir)
    logger.info("=" * 65)
    logger.info(f"TRAINING OPTIMAL FINAL [{loss_name.upper()}] MODEL FOR {epochs} EPOCHS")
    logger.info("=" * 65)
    logger.info(f"Optuna Best Val ROC-AUC: {best_val_auc:.4f}")
    logger.info(f"Parameters: {p}")
    
    # Initialize model
    model = MILAnomalyDetector(
        input_dim=512,
        hidden_dims=hidden_dims,
        dropout_rate=p.get("dropout_rate", 0.5),
        k=p.get("k", 5),
        smooth_weight=p.get("smooth_weight", 0.01),
        loss_type=loss_name,
        margin=p.get("margin", 1.0),
        sparse_weight=p.get("sparse_weight", 0.001),
        alpha=p.get("focal_alpha", 0.5),
        gamma=p.get("focal_gamma", 2.0),
        temperature=p.get("softmax_tau", 2.0),
        p=p.get("gem_p", 3.0)
    )
    
    pipeline = TrainingPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=p.get("learning_rate", 1e-4),
        weight_decay=p.get("weight_decay", 1e-5),
        step_size=p.get("step_size", 10),
        gamma=p.get("gamma", 0.5),
        output_dir=study_dir,
        logger=logger
    )
    
    smooth_sigma = p.get("smooth_sigma", 1.0)
    
    # Train model
    pipeline.fit(
        num_epochs=epochs,
        patience=patience,
        smooth_sigma=smooth_sigma
    )
    
    # Plot training history curves
    try:
        pipeline.plot_history("optuna_training_history.png")
    except Exception as e:
        logger.warning(f"Could not save training plot: {e}")
        
    # Reload best checkpoint from training
    best_ckpt = study_dir / "best_model.pt"
    if best_ckpt.exists():
        model.load_checkpoint(best_ckpt, device=torch.device(device))
        logger.info(f"Loaded best checkpoint from {best_ckpt} for testing.")
        
    # Test evaluation
    _, _, metrics = pipeline.evaluate(test_loader, smooth_sigma=smooth_sigma)
    
    # Save test results
    with open(study_dir / "optuna_test_results.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    # Copy checkpoint to experiments/best_model_<loss>.pt
    root_ckpt = base_output_dir / f"best_model_{loss_name}.pt"
    if best_ckpt.exists():
        shutil.copy(best_ckpt, root_ckpt)
        logger.info(f"Saved primary checkpoint to {root_ckpt}")
        
    vl = metrics.get("video_level", {})
    row = {
        "Loss Function": loss_name.upper(),
        "Val ROC-AUC (%)": round(best_val_auc * 100.0, 2),
        "Test PR-AUC / AP (%)": round(vl.get("average_precision", 0.0) * 100.0, 2),
        "Test ROC-AUC (%)": round(vl.get("roc_auc", 0.0) * 100.0, 2),
        "Test F1-Score (%)": round(vl.get("f1_score", 0.0) * 100.0, 2),
        "Test Precision (%)": round(vl.get("precision", 0.0) * 100.0, 2),
        "Test Recall (%)": round(vl.get("recall", 0.0) * 100.0, 2),
        "Test FAR (%)": round(vl.get("false_alarm_rate", 0.0) * 100.0, 2),
        "Best Model Checkpoint": f"experiments/best_model_{loss_name}.pt",
        "Study Folder": str(study_dir)
    }
    
    logger.info("-" * 60)
    logger.info(f"FINAL METRICS FOR [{loss_name.upper()}]:")
    logger.info(f"  Test ROC-AUC            : {vl.get('roc_auc', 0.0):.4f}")
    logger.info(f"  Test Average Precision  : {vl.get('average_precision', 0.0):.4f}")
    logger.info(f"  Test F1-Score           : {vl.get('f1_score', 0.0):.4f}")
    logger.info(f"  Test FAR                : {vl.get('false_alarm_rate', 0.0):.4f}")
    logger.info("-" * 60)
    return row


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    losses_to_run = ALL_LOSSES if "all" in [l.lower() for l in args.loss] else [l.lower() for l in args.loss]
    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"TRAINING OPTIMAL MODELS FOR RESEARCH PAPER: {', '.join([l.upper() for l in losses_to_run])}")
    print(f"Device: {args.device} | Epochs: {args.epochs} | Patience: {args.patience}")
    print("=" * 70)
    
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        val_split=0.2,
        seed=args.seed,
        verbose=False
    )
    
    results = []
    for loss_name in losses_to_run:
        res = train_single_optimal(
            loss_name=loss_name,
            base_output_dir=base_out,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            seed=args.seed
        )
        if res:
            results.append(res)
            
    if results:
        df = pd.DataFrame(results).sort_values(by="Test PR-AUC / AP (%)", ascending=False)
        print("\n" + "=" * 80)
        print("          FINAL RESEARCH PAPER COMPARISON LEADERBOARD")
        print("=" * 80)
        print(df[["Loss Function", "Val ROC-AUC (%)", "Test ROC-AUC (%)", "Test PR-AUC / AP (%)", "Test F1-Score (%)", "Test FAR (%)"]].to_string(index=False))
        
        # Save comparison files
        csv_path = base_out / "optuna_tuned_loss_comparison.csv"
        json_path = base_out / "optuna_tuned_loss_comparison.json"
        tex_path = base_out / "optuna_tuned_loss_comparison.tex"
        
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        
        # Format clean LaTeX table for publication
        display_cols = ["Loss Function", "Val ROC-AUC (%)", "Test ROC-AUC (%)", "Test PR-AUC / AP (%)", "Test F1-Score (%)", "Test FAR (%)"]
        tex_df = df[display_cols]
        tex_content = tex_df.to_latex(index=False, float_format="%.2f", column_format="lccccc")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)
            
        print(f"\n[OK] Updated comparison table saved to:")
        print(f"  - CSV  : {csv_path}")
        print(f"  - JSON : {json_path}")
        print(f"  - LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
