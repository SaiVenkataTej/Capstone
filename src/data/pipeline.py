"""
Data preparation pipeline for scanning directories, extracting labels, and generating CSVs.
"""

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def get_base_video_name(filename: str) -> str:
    """Extract base video identifier from .npy filename.
    
    Example:
        'A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A__0.npy'
        -> 'A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A'
    """
    stem = Path(filename).stem
    if "__" in stem:
        parts = stem.rsplit("__", 1)
        if parts[-1].isdigit():
            return parts[0]
    elif "_" in stem:
        parts = stem.rsplit("_", 1)
        if parts[-1].isdigit():
            return parts[0]
    return stem


def extract_label_from_filename(filename: str) -> Optional[int]:
    """Extract binary anomaly label from XD-Violence filename.
    
    XD-Violence Dataset Specification:
        - 'label_A': Normal activities (Non-violent / Drama / Romance) -> 0
        - 'label_B*' / 'label_G*' (e.g. B1=Fighting, B2=Shooting, B4=Riot, B5=Abuse, 
          B6=Explosion/Accident, G=Gunfire, and multi-label combinations) -> 1 (Violent / Anomalous)
          
    Returns:
        0 for normal, 1 for violent/anomalous, or None if unparseable.
    """
    match = re.search(r"label_([^\._]+)", filename)
    if not match:
        return None
        
    tag = match.group(1).strip()
    
    # Class A or A-0-0 is Normal (0)
    if tag == "A" or tag.startswith("A-"):
        return 0
        
    # Any B-series (B1, B2, B4, B5, B6) or G-series (G) is Violent (1)
    if any(k in tag for k in ["B1", "B2", "B4", "B5", "B6", "G", "B"]):
        return 1
        
    return None


def scan_features_directory(feature_dir: Union[str, Path]) -> Tuple[Dict[str, int], int, int]:
    """Scan a directory of .npy CLIP feature files and extract labels.
    
    Args:
        feature_dir: Path to directory containing .npy files.
        
    Returns:
        (label_dict, total_files, unique_videos)
    """
    dir_path = Path(feature_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Feature directory not found: {dir_path.resolve()}")
        
    npy_files = list(dir_path.glob("*.npy"))
    if not npy_files:
        raise ValueError(f"No .npy files found in: {dir_path.resolve()}")
        
    label_dict: Dict[str, int] = {}
    total_files = len(npy_files)
    
    for file_path in npy_files:
        base_name = get_base_video_name(file_path.name)
        label = extract_label_from_filename(file_path.name)
        if label is not None:
            label_dict[base_name] = label
            
    return label_dict, total_files, len(label_dict)


def generate_vadclip_csv(
    feature_dir: Union[str, Path],
    label_dict: Dict[str, int],
    output_csv: Union[str, Path],
    use_relative_path: bool = False
) -> pd.DataFrame:
    """Generate space-separated CSV file compatible with VADClip format.
    
    Format per row:
        path/to/feature.npy <label>
    """
    dir_path = Path(feature_dir)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    npy_files = sorted(dir_path.glob("*.npy"))
    rows = []
    
    for npy_file in npy_files:
        base_name = get_base_video_name(npy_file.name)
        label = label_dict.get(base_name)
        if label is not None:
            path_str = str(npy_file.relative_to(Path.cwd())) if use_relative_path else str(npy_file.resolve())
            rows.append({"video_path": path_str, "label": label})
            
    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep=" ", index=False, header=False)
    return df


def validate_features(feature_dir: Union[str, Path], sample_size: int = 5) -> Dict[str, Any]:
    """Inspect random feature samples in a directory to verify integrity.
    
    Args:
        feature_dir: Directory containing .npy files.
        sample_size: Number of files to sample.
        
    Returns:
        Summary dict of sampled feature shapes and data types.
    """
    dir_path = Path(feature_dir)
    npy_files = list(dir_path.glob("*.npy"))
    if not npy_files:
        return {"valid": False, "error": "No files found"}
        
    sample_indices = np.random.choice(len(npy_files), size=min(sample_size, len(npy_files)), replace=False)
    shapes = []
    dtypes = []
    has_nan = False
    has_inf = False
    
    for idx in sample_indices:
        file_path = npy_files[idx]
        arr = np.load(file_path)
        shapes.append(arr.shape)
        dtypes.append(str(arr.dtype))
        if np.any(np.isnan(arr)):
            has_nan = True
        if np.any(np.isinf(arr)):
            has_inf = True
            
    return {
        "valid": not (has_nan or has_inf),
        "shapes": list(set(shapes)),
        "dtypes": list(set(dtypes)),
        "has_nan": has_nan,
        "has_inf": has_inf
    }
