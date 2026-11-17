"""
Training CLI Entry Point for Video Anomaly Detection (WS-MIL).
Usage:
    python train.py --config configs/config.yaml --epochs 50 --lr 1e-4 --device cuda
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from src.data.dataset import create_dataloaders
from src.models.mil_model import MILAnomalyDetector
from src.training.trainer import TrainingPipeline
from src.utils.config import load_config, merge_configs
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train Weakly Supervised MIL Video Anomaly Detector")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--train_csv", type=str, default=None, help="Override path to train CSV")
    parser.add_argument("--test_csv", type=str, default=None, help="Override path to test CSV")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--k", type=int, default=None, help="Override Top-K parameter")
    parser.add_argument("--loss", type=str, default=None, choices=["topk", "margin", "focal", "attention", "gem"], help="Loss function (topk, margin, focal, attention, gem)")
    parser.add_argument("--smooth_weight", type=float, default=None, help="Smoothness regularization weight")
    parser.add_argument("--margin", type=float, default=1.0, help="Margin parameter for MaxMarginRankingLoss")
    parser.add_argument("--sparse_weight", type=float, default=0.001, help="Sparsity weight for MaxMarginRankingLoss")
    parser.add_argument("--focal_alpha", type=float, default=0.5, help="Alpha parameter for FocalTopKMILLoss")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for FocalTopKMILLoss")
    parser.add_argument("--softmax_tau", type=float, default=2.0, help="Temperature parameter for SoftmaxAttentionMILLoss")
    parser.add_argument("--gem_p", type=float, default=3.0, help="p-norm parameter for GEMPoolingLoss")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cuda' or 'cpu')")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1. Load Configuration
    config_path = Path(args.config)
    if not config_path.exists():
        fallback_path = Path("config.yaml")
        if fallback_path.exists():
            config_path = fallback_path
            
    config = load_config(config_path) if config_path.exists() else {}
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    infer_cfg = config.get("inference", {})
    log_cfg = config.get("logging", {})
    
    # 2. Resolve Parameters with CLI overrides
    train_csv = args.train_csv or data_cfg.get("train_csv", "xd_CLIP_rgb.csv")
    test_csv = args.test_csv or data_cfg.get("test_csv", "xd_CLIP_rgbtest.csv")
    batch_size = args.batch_size or data_cfg.get("batch_size", 32)
    val_split = data_cfg.get("val_split", 0.2)
    expected_snippets = data_cfg.get("expected_snippets", 10)
    feature_dim = model_cfg.get("input_dim", 512)
    hidden_dims = model_cfg.get("hidden_dims", [256, 128])
    dropout_rate = model_cfg.get("dropout_rate", 0.5)
    k_val = args.k or model_cfg.get("k", 5)
    loss_type = args.loss or model_cfg.get("loss_type", "topk")
    smooth_weight = args.smooth_weight if args.smooth_weight is not None else model_cfg.get("smooth_weight", 0.01)
    
    epochs = args.epochs or train_cfg.get("num_epochs", 50)
    lr = args.lr or float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    step_size = train_cfg.get("step_size", 10)
    gamma = train_cfg.get("gamma", 0.5)
    patience = args.patience or train_cfg.get("patience", 10)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    seed = args.seed or train_cfg.get("seed", 42)
    device = args.device or train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # Default output dir organized by loss function if not explicitly specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base_log_dir = log_cfg.get("output_dir", "./experiments")
        output_dir = Path(base_log_dir) / f"loss_{loss_type}"
        
    # 3. Setup Seed & Logger
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger = setup_logger(name=f"train_{loss_type}", log_dir=output_dir)
    logger.info("=" * 60)
    logger.info("WEAKLY SUPERVISED VIDEO ANOMALY DETECTION TRAINING")
    logger.info("=" * 60)
    logger.info(f"Loaded config from: {config_path}")
    logger.info(f"Loss Formulation: {loss_type.upper()} | Device: {device} | Seed: {seed}")
    logger.info(f"Output Directory: {output_dir}")
    
    # 4. Create DataLoaders
    logger.info("\nInitializing datasets and DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=train_csv,
        test_csv=test_csv,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
        verbose=True
    )
    
    # 5. Initialize Model with Selected Loss Formulation
    logger.info(f"\nInstantiating MIL Scoring Network with [{loss_type.upper()}] Loss...")
    model = MILAnomalyDetector(
        input_dim=feature_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        k=k_val,
        smooth_weight=smooth_weight,
        loss_type=loss_type,
        margin=args.margin,
        sparse_weight=args.sparse_weight,
        alpha=args.focal_alpha,
        gamma=args.focal_gamma,
        temperature=args.softmax_tau,
        p=args.gem_p
    )
    
    # 6. Initialize Training Pipeline
    pipeline = TrainingPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        step_size=step_size,
        gamma=gamma,
        grad_clip=grad_clip,
        output_dir=output_dir,
        logger=logger
    )
    
    # 7. Execute Training
    pipeline.fit(num_epochs=epochs, patience=patience)
    
    # 8. Plot Curves & Test
    pipeline.plot_history()
    pipeline.test()
    
    # Save copy as experiments/best_model_<loss>.pt
    root_dir = Path("./experiments")
    root_dir.mkdir(parents=True, exist_ok=True)
    loss_best = root_dir / f"best_model_{loss_type}.pt"
    if (output_dir / "best_model.pt").exists():
        import shutil
        shutil.copy(output_dir / "best_model.pt", loss_best)
        logger.info(f"Saved primary [{loss_type.upper()}] model checkpoint to: {loss_best.resolve()}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
