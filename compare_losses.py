"""
Comparative Benchmarking CLI for WS-MIL Loss Formulations.
Automates training, evaluation, comparison table generation, and plotting across:
1. TopKRankingLoss          (Baseline Top-K)
2. MaxMarginRankingLoss     (Sultani et al., CVPR 2018)
3. FocalTopKMILLoss         (Focal Hard-Negative Modulation)
4. SoftmaxAttentionMILLoss  (Differentiable Attention MIL)
5. GEMPoolingLoss           (Generalized Mean Pooling MIL)

Usage:
    python compare_losses.py --epochs 15 --batch_size 32 --device cuda
    python compare_losses.py --losses topk margin focal --epochs 10
"""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
import torch

from src.data.dataset import create_dataloaders
from src.models.mil_model import MILAnomalyDetector
from src.training.trainer import TrainingPipeline
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark and Compare Multiple WS-MIL Loss Functions")
    parser.add_argument("--losses", nargs="+", default=["topk", "margin", "focal", "attention", "gem"],
                        help="List of loss functions to benchmark (choices: topk, margin, focal, attention, gem)")
    parser.add_argument("--train_csv", type=str, default="xd_CLIP_rgb.csv", help="Path to train CSV")
    parser.add_argument("--test_csv", type=str, default="xd_CLIP_rgbtest.csv", help="Path to test CSV")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs per loss function")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--k", type=int, default=5, help="Top-K parameter for Top-K / Focal losses")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin for MaxMarginRankingLoss")
    parser.add_argument("--smooth_weight", type=float, default=0.01, help="Smoothness regularization weight")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./experiments/loss_benchmark", help="Output comparison directory")
    return parser.parse_args()


def plot_comparative_curves(
    results: Dict[str, dict],
    output_dir: Path
):
    """Plot comparative Training Loss curves, ROC Curves, and PR Curves for all loss functions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training Loss History Comparison
    plt.figure(figsize=(10, 6))
    for loss_name, data in results.items():
        history = data["history"]
        if "train_loss" in history and history["train_loss"]:
            epochs = range(1, len(history["train_loss"]) + 1)
            plt.plot(epochs, history["train_loss"], marker="o", label=f"{loss_name.upper()}")
            
    plt.title("Training Loss Convergence Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Total Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "comparative_loss_convergence.png", dpi=300)
    plt.close()
    
    # 2. Validation ROC-AUC History Comparison
    plt.figure(figsize=(10, 6))
    for loss_name, data in results.items():
        history = data["history"]
        if "val_auc" in history and history["val_auc"]:
            epochs = range(1, len(history["val_auc"]) + 1)
            plt.plot(epochs, history["val_auc"], marker="s", label=f"{loss_name.upper()} (Best: {data['best_val_auc']:.4f})")
            
    plt.title("Validation ROC-AUC Across Training Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation ROC-AUC", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "comparative_val_auc_history.png", dpi=300)
    plt.close()
    
    # 3. Test ROC and PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for loss_name, data in results.items():
        scores = data["test_scores"]
        labels = data["test_labels"]
        # Max-pool to video level
        video_scores = scores.max(axis=1) if scores.ndim > 1 else scores
        video_scores = np.clip(video_scores, 0.0, 1.0)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(labels, video_scores)
        auc_val = data["metrics"]["video_level"]["roc_auc"]
        axes[0].plot(fpr, tpr, label=f"{loss_name.upper()} (AUC = {auc_val:.4f})", linewidth=2)
        
        # PR Curve
        prec, rec, _ = precision_recall_curve(labels, video_scores)
        ap_val = data["metrics"]["video_level"]["average_precision"]
        axes[1].plot(rec, prec, label=f"{loss_name.upper()} (AP = {ap_val:.4f})", linewidth=2)
        
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_title("Test Set ROC Curves", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (TPR)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)
    
    axes[1].set_title("Test Set Precision-Recall Curves", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparative_roc_pr_curves.png", dpi=300)
    plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("loss_benchmark", log_dir=out_dir)
    logger.info("=" * 70)
    logger.info("     WEAKLY SUPERVISED MIL LOSS FUNCTION COMPARATIVE STUDY")
    logger.info("=" * 70)
    logger.info(f"Losses to Benchmark : {', '.join([l.upper() for l in args.losses])}")
    logger.info(f"Training Epochs     : {args.epochs} per loss")
    logger.info(f"Batch Size          : {args.batch_size} | LR: {args.lr} | Device: {args.device}")
    logger.info(f"Output Directory    : {out_dir.resolve()}")
    
    # 1. Load DataLoaders (shared across all experiments for fairness)
    logger.info("\nLoading dataset and initializing DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=False
    )
    
    benchmark_results: Dict[str, dict] = {}
    summary_rows = []
    
    # 2. Iterate through each loss function
    for idx, loss_name in enumerate(args.losses, 1):
        logger.info("\n" + "=" * 70)
        logger.info(f"[{idx}/{len(args.losses)}] TRAINING & EVALUATING: {loss_name.upper()} LOSS")
        logger.info("=" * 70)
        
        # Fixed seed per experiment for exact reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        loss_output_dir = out_dir / f"loss_{loss_name}"
        
        # Instantiate model with chosen loss
        model = MILAnomalyDetector(
            input_dim=512,
            hidden_dims=[256, 128],
            dropout_rate=0.5,
            k=args.k,
            smooth_weight=args.smooth_weight,
            loss_type=loss_name,
            margin=args.margin
        )
        
        trainer = TrainingPipeline(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=args.device,
            lr=args.lr,
            output_dir=loss_output_dir,
            logger=logger
        )
        
        start_t = time.time()
        history = trainer.fit(num_epochs=args.epochs, patience=10)
        train_duration = time.time() - start_t
        
        # Test evaluation
        test_scores, test_labels, test_metrics = trainer.test()
        
        # Copy to root experiments/best_model_<loss_name>.pt
        root_dir = Path("./experiments")
        root_dir.mkdir(parents=True, exist_ok=True)
        loss_best_path = root_dir / f"best_model_{loss_name}.pt"
        if (loss_output_dir / "best_model.pt").exists():
            shutil.copy(loss_output_dir / "best_model.pt", loss_best_path)
            logger.info(f"Saved [{loss_name.upper()}] model checkpoint to {loss_best_path.resolve()}")
        
        vl = test_metrics["video_level"]
        summary_rows.append({
            "Loss Function": loss_name.upper(),
            "ROC-AUC (%)": round(vl["roc_auc"] * 100.0, 2),
            "PR-AUC / AP (%)": round(vl["average_precision"] * 100.0, 2),
            "Precision (%)": round(vl["precision"] * 100.0, 2),
            "Recall (%)": round(vl["recall"] * 100.0, 2),
            "F1-Score (%)": round(vl["f1_score"] * 100.0, 2),
            "FAR (%)": round(vl["false_alarm_rate"] * 100.0, 2),
            "Best Val AUC (%)": round(trainer.best_auc * 100.0, 2),
            "Train Time (s)": round(train_duration, 1)
        })
        
        benchmark_results[loss_name] = {
            "best_epoch": trainer.best_epoch,
            "best_val_auc": trainer.best_auc,
            "history": history,
            "metrics": test_metrics,
            "test_scores": test_scores,
            "test_labels": test_labels,
            "duration_sec": train_duration
        }
        
    # 3. Create Comparison Table
    summary_df = pd.DataFrame(summary_rows)
    # Sort by PR-AUC (official XD-Violence benchmark primary ranking metric)
    summary_df = summary_df.sort_values(by="PR-AUC / AP (%)", ascending=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("                FINAL BENCHMARK COMPARISON TABLE")
    logger.info("=" * 70)
    print("\n" + summary_df.to_string(index=False))
    
    # Save CSV & JSON table
    summary_csv = out_dir / "loss_comparison_summary.csv"
    summary_json = out_dir / "loss_comparison_summary.json"
    summary_latex = out_dir / "loss_comparison_table.tex"
    
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_json(summary_json, orient="records", indent=2)
    summary_df.to_latex(summary_latex, index=False)
    
    # 4. Generate Publication Plots
    logger.info("\nGenerating publication-quality comparison plots...")
    plot_comparative_curves(benchmark_results, out_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"BENCHMARK COMPLETED SUCCESSFULLY! Artifacts saved to {out_dir.resolve()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
