"""
Optuna Bayesian Hyperparameter Optimization CLI Entry Point.
Usage:
    python tune.py --n_trials 20 --epochs_per_trial 15 --device cuda
"""

import argparse
import json
from pathlib import Path
from typing import List
import pandas as pd
import torch

from src.training.tuner import HyperparameterTuner
from src.utils.logger import setup_logger


ALL_LOSSES = ["topk", "margin", "focal", "attention", "gem"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Optuna Bayesian Hyperparameter Search for WS-MIL")
    parser.add_argument(
        "--loss", "--losses", nargs="+", default=["topk"],
        help="Loss function(s) to optimize (choices: topk, margin, focal, attention, gem, all). Default: topk"
    )
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna optimization trials per loss")
    parser.add_argument("--epochs_per_trial", type=int, default=8, help="Number of training epochs per trial")
    parser.add_argument("--final_epochs", type=int, default=25, help="Number of training epochs for final winning model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--train_csv", type=str, default="xd_CLIP_rgb.csv", help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, default="xd_CLIP_rgbtest.csv", help="Path to test CSV")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device ('cuda' or 'cpu')")
    parser.add_argument("--output_dir", type=str, default=None, help="Base directory to save results")
    parser.add_argument("--study_name", type=str, default=None, help="Name of the Optuna study")
    parser.add_argument("--parallel", action="store_true", help="Run tuning for multiple losses concurrently in parallel processes")
    parser.add_argument("--max_workers", type=int, default=None, help="Max parallel worker processes (default: number of losses)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip losses that already have completed optuna_test_results.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def run_tuning_for_loss(loss_name: str, args_dict: dict) -> dict:
    """Run hyperparameter tuning for a single loss formulation."""
    base_dir = Path(args_dict["output_dir"]) if args_dict.get("output_dir") else Path("./experiments")
    loss_output_dir = base_dir / f"optuna_study_{loss_name}"
    loss_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if existing results should be reused
    hp_file = loss_output_dir / "best_hyperparameters.json"
    res_file = loss_output_dir / "optuna_test_results.json"
    
    if args_dict.get("skip_existing") and hp_file.exists() and res_file.exists():
        with open(hp_file, "r") as f:
            hp_data = json.load(f)
        with open(res_file, "r") as f:
            res_data = json.load(f)
        vl = res_data.get("video_level", {})
        print(f"[CACHE] Found existing results for [{loss_name.upper()}]. Skipping re-tuning.")
        return {
            "Loss Function": loss_name.upper(),
            "Best Val ROC-AUC (%)": round(hp_data.get("best_val_auc", 0.0) * 100.0, 2),
            "Test PR-AUC / AP (%)": round(vl.get("average_precision", 0.0) * 100.0, 2),
            "Test ROC-AUC (%)": round(vl.get("roc_auc", 0.0) * 100.0, 2),
            "Test F1-Score (%)": round(vl.get("f1_score", 0.0) * 100.0, 2),
            "Test FAR (%)": round(vl.get("false_alarm_rate", 0.0) * 100.0, 2),
            "Best Model Checkpoint": f"experiments/best_model_{loss_name}.pt",
            "Study Folder": str(loss_output_dir)
        }
        
    logger = setup_logger(f"optuna_tune_{loss_name}", log_dir=loss_output_dir)
    print("\n" + "=" * 70)
    print(f"OPTUNA BAYESIAN HYPERPARAMETER TUNING: [{loss_name.upper()}] LOSS")
    print("=" * 70)
    
    tuner = HyperparameterTuner(
        train_csv=args_dict["train_csv"],
        test_csv=args_dict["test_csv"],
        batch_size=args_dict["batch_size"],
        val_split=args_dict["val_split"],
        loss_type=loss_name,
        seed=args_dict["seed"],
        device=args_dict["device"],
        output_dir=loss_output_dir,
        logger=logger
    )
    
    study_name = args_dict.get("study_name") or f"mil_optuna_{loss_name}"
    
    study, test_metrics = tuner.tune(
        n_trials=args_dict["n_trials"],
        epochs_per_trial=args_dict["epochs_per_trial"],
        final_epochs=args_dict["final_epochs"],
        study_name=study_name
    )
    
    vl = test_metrics.get("video_level", {})
    row = {
        "Loss Function": loss_name.upper(),
        "Best Val ROC-AUC (%)": round(study.best_value * 100.0, 2),
        "Test PR-AUC / AP (%)": round(vl.get("average_precision", 0.0) * 100.0, 2),
        "Test ROC-AUC (%)": round(vl.get("roc_auc", 0.0) * 100.0, 2),
        "Test F1-Score (%)": round(vl.get("f1_score", 0.0) * 100.0, 2),
        "Test FAR (%)": round(vl.get("false_alarm_rate", 0.0) * 100.0, 2),
        "Best Model Checkpoint": f"experiments/best_model_{loss_name}.pt",
        "Study Folder": str(loss_output_dir)
    }
    
    print("\n" + "-" * 60)
    print(f"RESULTS FOR [{loss_name.upper()}]:")
    print(f"  Best Validation ROC-AUC : {study.best_value:.4f}")
    print(f"  Test Average Precision  : {vl.get('average_precision', 0.0):.4f}")
    print(f"  Test ROC-AUC            : {vl.get('roc_auc', 0.0):.4f}")
    print(f"  Saved Winning Model     : ./experiments/best_model_{loss_name}.pt")
    print(f"  Saved Hyperparameters   : {loss_output_dir / 'best_hyperparameters.json'}")
    print(f"  Saved Test Results      : {loss_output_dir / 'optuna_test_results.json'}")
    print("-" * 60)
    return row


def main():
    args = parse_args()
    
    # Resolve losses list
    losses_to_tune: List[str] = []
    for l in args.loss:
        if l.lower() == "all":
            losses_to_tune = list(ALL_LOSSES)
            break
        elif l.lower() in ALL_LOSSES:
            if l.lower() not in losses_to_tune:
                losses_to_tune.append(l.lower())
        else:
            raise ValueError(f"Unknown loss type '{l}'. Available options: {ALL_LOSSES + ['all']}")
            
    summary_rows = []
    args_dict = vars(args)
    
    if args.parallel and len(losses_to_tune) > 1:
        from concurrent.futures import ProcessPoolExecutor
        workers = args.max_workers or min(len(losses_to_tune), 4)
        print("\n" + "=" * 70)
        print(f"RUNNING PARALLEL TUNING FOR {len(losses_to_tune)} LOSSES WITH {workers} WORKERS")
        print(f"Target Losses: {', '.join([l.upper() for l in losses_to_tune])}")
        print("=" * 70)
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_tuning_for_loss, loss, args_dict) for loss in losses_to_tune]
            for f in futures:
                try:
                    summary_rows.append(f.result())
                except Exception as e:
                    print(f"[ERROR in parallel tuning]: {e}")
    else:
        for loss_name in losses_to_tune:
            row = run_tuning_for_loss(loss_name, args_dict)
            summary_rows.append(row)
        
    if len(summary_rows) > 1:
        df = pd.DataFrame(summary_rows).sort_values(by="Test PR-AUC / AP (%)", ascending=False)
        print("\n" + "=" * 80)
        print("          FINAL OPTUNA HYPERPARAMETER TUNING LEADERBOARD")
        print("=" * 80)
        print(df.to_string(index=False))
        
        base_out = Path(args.output_dir) if args.output_dir else Path("./experiments")
        base_out.mkdir(parents=True, exist_ok=True)
        df.to_csv(base_out / "optuna_tuned_loss_comparison.csv", index=False)
        df.to_json(base_out / "optuna_tuned_loss_comparison.json", orient="records", indent=2)
        df.to_latex(base_out / "optuna_tuned_loss_comparison.tex", index=False)
        print(f"\nSaved cross-loss summary table to: {base_out / 'optuna_tuned_loss_comparison.csv'}")


if __name__ == "__main__":
    main()
