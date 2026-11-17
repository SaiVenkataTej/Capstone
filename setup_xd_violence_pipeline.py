"""
XD-Violence CLIP Features Dataset Pipeline Setup.
Scans XDTrainClipFeatures and XDTestClipFeatures directories, extracts video labels,
and generates standardized CSVs (xd_CLIP_rgb.csv and xd_CLIP_rgbtest.csv).

XD-Violence Dataset Taxonomy:
    - label_A -> 0 (Normal / Non-violent activity)
    - label_B1, label_B2, label_B4, label_B5, label_B6, label_G -> 1 (Violent / Anomalous)
"""

from pathlib import Path
import pandas as pd
from src.data.pipeline import (
    generate_vadclip_csv,
    scan_features_directory,
    validate_features
)


def main():
    train_dir = Path("XDTrainClipFeatures")
    test_dir = Path("XDTestClipFeatures")
    
    print("=" * 65)
    print("      XD-VIOLENCE CLIP FEATURE DATASET PIPELINE SETUP")
    print("=" * 65)
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir.resolve()}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir.resolve()}")
        
    print(f"\nScanning Training Features: {train_dir.resolve()}")
    train_labels, train_total, train_unique = scan_features_directory(train_dir)
    train_violent = sum(1 for v in train_labels.values() if v == 1)
    train_normal = len(train_labels) - train_violent
    print(f"   Total snippet files: {train_total:,}")
    print(f"   Unique videos:       {train_unique:,}")
    print(f"   Violent videos (1):  {train_violent:,} ({train_violent/train_unique*100:.1f}%)")
    print(f"   Normal videos (0):   {train_normal:,} ({train_normal/train_unique*100:.1f}%)")
    
    print(f"\nScanning Test Features: {test_dir.resolve()}")
    test_labels, test_total, test_unique = scan_features_directory(test_dir)
    test_violent = sum(1 for v in test_labels.values() if v == 1)
    test_normal = len(test_labels) - test_violent
    print(f"   Total snippet files: {test_total:,}")
    print(f"   Unique videos:       {test_unique:,}")
    print(f"   Violent videos (1):  {test_violent:,} ({test_violent/test_unique*100:.1f}%)")
    print(f"   Normal videos (0):   {test_normal:,} ({test_normal/test_unique*100:.1f}%)")
    
    # Generate CSV files in root and in data/
    out_train_csv = Path("xd_CLIP_rgb.csv")
    out_test_csv = Path("xd_CLIP_rgbtest.csv")
    
    print("\nGenerating dataset CSV files...")
    df_train = generate_vadclip_csv(train_dir, train_labels, out_train_csv)
    df_test = generate_vadclip_csv(test_dir, test_labels, out_test_csv)
    
    print(f"   [SAVED] {out_train_csv} ({len(df_train):,} records)")
    print(f"   [SAVED] {out_test_csv} ({len(df_test):,} records)")
    
    # Sanity validation
    print("\nValidating feature file integrity...")
    train_val = validate_features(train_dir, sample_size=5)
    test_val = validate_features(test_dir, sample_size=5)
    print(f"   Train Features Valid: {train_val['valid']} (Shapes: {train_val['shapes']}, Dtypes: {train_val['dtypes']})")
    print(f"   Test Features Valid:  {test_val['valid']} (Shapes: {test_val['shapes']}, Dtypes: {test_val['dtypes']})")
    
    print("\n" + "=" * 65)
    print(" PIPELINE INITIALIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
