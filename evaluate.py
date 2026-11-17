"""
Standalone Model Evaluation CLI Entry Point.
Usage:
    python evaluate.py --checkpoint experiments/best_model.pt --test_csv xd_CLIP_rgbtest.csv
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import XDViolenceClipDataset
from src.models.mil_model import MILAnomalyDetector
from src.training.metrics import compute_metrics
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained WS-MIL Anomaly Detector")
    parser.add_argument("--checkpoint", type=str, default="./experiments/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--test_csv", type=str, default="xd_CLIP_rgbtest.csv", help="Path to test CSV")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold")
    parser.add_argument("--smooth_sigma", type=float, default=1.0, help="Temporal Gaussian smoothing sigma")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_file", type=str, default="./experiments/eval_summary.json", help="Path to save results")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("evaluate")
    
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at: {ckpt_path.resolve()}")
        return
        
    logger.info("=" * 60)
    logger.info("EVALUATION ON TEST DATASET")
    logger.info("=" * 60)
    logger.info(f"Loading checkpoint from: {ckpt_path}")
    logger.info(f"Test CSV: {args.test_csv} | Device: {args.device}")
    
    # 1. Load Dataset
    test_dataset = XDViolenceClipDataset(args.test_csv, verbose=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 2. Instantiate and load model
    model = MILAnomalyDetector(input_dim=512, k=5)
    model.load_checkpoint(ckpt_path, device=args.device)
    model.eval()
    
    # 3. Predict & Compute Metrics
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"].to(args.device)
            labels = batch["label"].cpu().numpy()
            
            scores = model.predict(feats, smooth_sigma=args.smooth_sigma)
            all_scores.append(scores)
            all_labels.append(labels)
            
    scores_arr = np.concatenate(all_scores, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    
    # Compute the 6 core evaluation metrics
    metrics = compute_metrics(scores_arr, labels_arr, threshold=args.threshold)
    
    logger.info("\n" + "=" * 55)
    logger.info("  EVALUATION RESULTS (6 CORE METRICS)")
    logger.info("=" * 55)
    vl = metrics["video_level"]
    logger.info(f"  1. ROC-AUC (Area Under ROC)       : {vl['roc_auc']:.4f}")
    logger.info(f"  2. PR-AUC (Average Precision / AP): {vl['average_precision']:.4f}")
    logger.info(f"  3. Precision (Positive Accuracy)  : {vl['precision']:.4f}")
    logger.info(f"  4. Recall (Sensitivity / TPR)     : {vl['recall']:.4f}")
    logger.info(f"  5. F1-Score (Harmonic Mean)       : {vl['f1_score']:.4f}")
    logger.info(f"  6. False Alarm Rate (FAR / FPR)   : {vl['false_alarm_rate']:.4f}")
    logger.info("-" * 55)
    logger.info(f"  Overall Accuracy                  : {vl['accuracy']:.4f}")
    logger.info("=" * 55)
        
    # Save output
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nEvaluation summary saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
