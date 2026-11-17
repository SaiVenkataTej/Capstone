"""
Utility script to finalize and evaluate an already-trained Optuna study model.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import XDViolenceClipDataset
from src.models.mil_model import MILAnomalyDetector
from src.training.metrics import compute_metrics


def finalize_study(loss_name: str = "topk", test_csv: str = "xd_CLIP_rgbtest.csv"):
    study_dir = Path(f"experiments/optuna_study_{loss_name}")
    hp_file = study_dir / "best_hyperparameters.json"
    ckpt_file = study_dir / "best_model.pt"
    
    if not ckpt_file.exists() or not hp_file.exists():
        print(f"No checkpoint found at {ckpt_file}")
        return
        
    with open(hp_file, "r") as f:
        hp_data = json.load(f)
    p = hp_data["parameters"]
    print(f"Loaded hyperparameters for [{loss_name.upper()}]:", p)
    
    hidden_dims = [256, 128] if p.get("hidden_arch") == "256_128" else ([512, 256] if p.get("hidden_arch") == "512_256" else [128, 64])
    
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
    
    model.load_checkpoint(ckpt_file, device=torch.device("cpu"))
    model.eval()
    
    print(f"Loading test dataset from: {test_csv}")
    test_dataset = XDViolenceClipDataset(test_csv, verbose=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            feats = batch["features"]
            labels = batch["label"].numpy()
            scores = model.predict(feats, smooth_sigma=p.get("smooth_sigma", 0.6))
            all_scores.append(scores)
            all_labels.append(labels)
            
    scores_arr = np.concatenate(all_scores, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(scores_arr, labels_arr)
    
    res_path = study_dir / "optuna_test_results.json"
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    root_dir = Path("./experiments")
    root_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ckpt_file, root_dir / f"best_model_{loss_name}.pt")
    shutil.copy(ckpt_file, root_dir / "best_model.pt")
    
    vl = metrics["video_level"]
    print("\n" + "=" * 60)
    print(f"FINAL TEST METRICS FOR OPTUNA [{loss_name.upper()}]:")
    print("=" * 60)
    print(f"  1. ROC-AUC           : {vl['roc_auc']:.4f}  ({vl['roc_auc']*100:.2f}%)")
    print(f"  2. Avg Precision (AP): {vl['average_precision']:.4f}  ({vl['average_precision']*100:.2f}%)")
    print(f"  3. Precision         : {vl['precision']:.4f}  ({vl['precision']*100:.2f}%)")
    print(f"  4. Recall            : {vl['recall']:.4f}  ({vl['recall']*100:.2f}%)")
    print(f"  5. F1-Score          : {vl['f1_score']:.4f}  ({vl['f1_score']*100:.2f}%)")
    print(f"  6. False Alarm Rate  : {vl['false_alarm_rate']:.4f}  ({vl['false_alarm_rate']*100:.2f}%)")
    print(f"  7. Accuracy          : {vl['accuracy']:.4f}  ({vl['accuracy']*100:.2f}%)")
    print("=" * 60)
    print(f"\nSaved test results  : {res_path.resolve()}")
    print(f"Saved model ckpt    : {(root_dir / f'best_model_{loss_name}.pt').resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finalize and evaluate Optuna study model.")
    parser.add_argument("--loss", type=str, default="topk", help="Loss function name")
    parser.add_argument("--test_csv", type=str, default="xd_CLIP_rgbtest.csv", help="Path to test CSV file")
    args = parser.parse_args()
    
    finalize_study(loss_name=args.loss, test_csv=args.test_csv)
