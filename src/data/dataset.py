"""
PyTorch Dataset and DataLoader implementation for XD-Violence CLIP features.
Optimized with O(1) in-memory pre-indexing for fast multi-epoch training.
"""

from collections import defaultdict
from pathlib import Path
import platform
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def default_base_name_fn(filename: str) -> str:
    """Extract base video identifier from snippet filename.
    
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


class XDViolenceClipDataset(Dataset):
    """
    Weakly Supervised Multiple Instance Learning (WS-MIL) Dataset.
    
    Groups all snippet feature files belonging to a single video together.
    Loads 10 CLIP embeddings (512-D each) and the shared video-level binary label.
    """
    
    def __init__(
        self,
        csv_file: Union[str, Path],
        expected_snippets: int = 10,
        feature_dim: int = 512,
        transform: Optional[Callable] = None,
        verbose: bool = False
    ):
        """
        Args:
            csv_file: Path to space-separated CSV (video_path label).
            expected_snippets: Number of snippets expected per video (default: 10).
            feature_dim: Dimension of CLIP visual feature (default: 512).
            transform: Optional transformation callable.
            verbose: Whether to print dataset initialization info.
        """
        self.csv_path = Path(csv_file)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found at: {self.csv_path.resolve()}")
            
        self.expected_snippets = expected_snippets
        self.feature_dim = feature_dim
        self.transform = transform
        
        # Read CSV: handle paths that may contain spaces on Windows
        lines = self.csv_path.read_text(encoding="utf-8").strip().splitlines()
        records = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if " " in line:
                path_part, label_part = line.rsplit(" ", 1)
                path_clean = path_part.strip().strip('"\'')
                records.append({"video_path": path_clean, "label": int(label_part)})
            elif "," in line:
                path_part, label_part = line.rsplit(",", 1)
                path_clean = path_part.strip().strip('"\'')
                records.append({"video_path": path_clean, "label": int(label_part)})
                
        raw_df = pd.DataFrame(records)
        
        # Pre-index video files by base_name in O(N) once during initialization
        self.video_snippets_map: Dict[str, List[Path]] = defaultdict(list)
        self.video_labels_map: Dict[str, int] = {}
        
        for _, row in raw_df.iterrows():
            file_path = Path(row["video_path"])
            base_name = default_base_name_fn(file_path.name)
            self.video_snippets_map[base_name].append(file_path)
            self.video_labels_map[base_name] = int(row["label"])
            
        # Unique video list with sorted snippet paths for O(1) item retrieval
        self.video_keys: List[str] = list(self.video_snippets_map.keys())
        for base_name in self.video_keys:
            self.video_snippets_map[base_name] = sorted(self.video_snippets_map[base_name])
            
        if verbose:
            violent_count = sum(1 for label in self.video_labels_map.values() if label == 1)
            normal_count = len(self.video_labels_map) - violent_count
            print(f"Loaded dataset from: {self.csv_path.name}")
            print(f"  Total records:  {len(raw_df):,}")
            print(f"  Unique videos:  {len(self.video_keys):,}")
            print(f"  Class balance:  {violent_count:,} violent ({violent_count/len(self.video_keys)*100:.1f}%), "
                  f"{normal_count:,} normal ({normal_count/len(self.video_keys)*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.video_keys)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """
        Fetch all snippet embeddings for a video in O(1) time.
        
        Returns:
            dict containing:
                features: torch.FloatTensor of shape (expected_snippets, feature_dim)
                label: torch.LongTensor scalar (0 or 1)
                video_id: str
                num_snippets: int
        """
        video_id = self.video_keys[idx]
        snippet_paths = self.video_snippets_map[video_id]
        label = self.video_labels_map[video_id]
        
        features_list = []
        for path in snippet_paths:
            if not path.exists():
                # Fallback if path stored in CSV was from another machine or relative
                resolved = None
                for candidate_dir in [Path("XDTrainClipFeatures"), Path("XDTestClipFeatures"), Path("data"), Path(".")]:
                    candidate = candidate_dir / path.name
                    if candidate.exists():
                        resolved = candidate
                        break
                if resolved is not None:
                    path = resolved
                    
            # Use .copy() to ensure contiguous, resizable array for PyTorch DataLoader
            feat = np.load(path).copy()
            
            # If feature has frame dimension (e.g. (16, 512)), average pool to (512,)
            if feat.ndim > 1:
                feat = feat.mean(axis=0)
            features_list.append(feat)
            
        # Handle cases where snippet count differs from expected_snippets
        if len(features_list) == 0:
            # Empty fallback
            stacked = np.zeros((self.expected_snippets, self.feature_dim), dtype=np.float32)
        else:
            stacked = np.stack(features_list, axis=0).astype(np.float32)
            
        # If video has more or fewer snippets, interpolate or pad/slice to expected_snippets
        actual_snippets = stacked.shape[0]
        if actual_snippets < self.expected_snippets:
            pad = np.repeat(stacked[-1:], self.expected_snippets - actual_snippets, axis=0)
            stacked = np.concatenate([stacked, pad], axis=0)
        elif actual_snippets > self.expected_snippets:
            # Sample evenly spaced snippets
            indices = np.linspace(0, actual_snippets - 1, self.expected_snippets, dtype=int)
            stacked = stacked[indices]
            
        features_tensor = torch.from_numpy(stacked).float()
        
        if self.transform is not None:
            features_tensor = self.transform(features_tensor)
            
        return {
            "features": features_tensor,  # (expected_snippets, feature_dim)
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": video_id,
            "num_snippets": self.expected_snippets
        }


def create_dataloaders(
    train_csv: Union[str, Path],
    test_csv: Union[str, Path],
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    val_split: float = 0.2,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create Train, Validation, and Test DataLoaders.
    
    Args:
        train_csv: Path to train CSV file.
        test_csv: Path to test CSV file.
        batch_size: Batch size for training/testing.
        num_workers: Number of workers for DataLoader (auto-detected if None).
        pin_memory: Pin memory for GPU transfer (auto-detected if None).
        val_split: Validation split fraction (e.g. 0.2).
        seed: Random seed for split reproducibility.
        verbose: Verbose logging flag.
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
        
    # Windows spawn multiprocessing crashes with PyTorch shared memory if num_workers > 0
    if num_workers is None:
        num_workers = 0 if platform.system() == "Windows" else 4
        
    full_train_dataset = XDViolenceClipDataset(train_csv, verbose=verbose)
    test_dataset = XDViolenceClipDataset(test_csv, verbose=verbose)
    
    # Split train dataset into train and validation splits
    generator = torch.Generator().manual_seed(seed)
    train_len = int((1.0 - val_split) * len(full_train_dataset))
    val_len = len(full_train_dataset) - train_len
    
    train_subset, val_subset = torch.utils.data.random_split(
        full_train_dataset, [train_len, val_len], generator=generator
    )
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
