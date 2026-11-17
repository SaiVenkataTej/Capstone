"""
Unit tests for data pipeline and dataset utilities using standard unittest.
"""

from pathlib import Path
import shutil
import unittest
import numpy as np
import torch

from src.data.dataset import XDViolenceClipDataset, default_base_name_fn
from src.data.pipeline import (
    extract_label_from_filename,
    generate_vadclip_csv,
    get_base_video_name
)


class TestDataPipeline(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = Path("./test_tmp_data")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_label_and_name_extraction(self):
        """Verify regex extraction from standard XD-Violence naming patterns."""
        fn_normal = "A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A__3.npy"
        fn_violent_b = "Street_Fight_Scene__#00-10-00_00-10-30_label_B1-0-0__9.npy"
        fn_violent_g = "Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6__0.npy"
        
        self.assertEqual(extract_label_from_filename(fn_normal), 0)
        self.assertEqual(extract_label_from_filename(fn_violent_b), 1)
        self.assertEqual(extract_label_from_filename(fn_violent_g), 1)
        self.assertIsNone(extract_label_from_filename("unlabeled_video.npy"))
        
        self.assertEqual(get_base_video_name(fn_normal), "A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A")
        self.assertEqual(default_base_name_fn(fn_normal), "A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A")

    def test_dataset_preindexing_and_get_item(self):
        """Create dummy snippet .npy files and verify O(1) loading and shapes."""
        feat_dir = self.test_dir / "dummy_features"
        feat_dir.mkdir()
        
        video_names = ["video_test_label_A", "video_test_label_B1-0-0"]
        csv_rows = []
        
        for v_name in video_names:
            label = 0 if "label_A" in v_name else 1
            for s in range(10):
                fname = f"{v_name}__{s}.npy"
                fpath = feat_dir / fname
                np.save(fpath, np.random.randn(512).astype(np.float32))
                csv_rows.append(f"{fpath.resolve()} {label}")
                
        csv_file = self.test_dir / "test.csv"
        with open(csv_file, "w") as f:
            f.write("\n".join(csv_rows))
            
        dataset = XDViolenceClipDataset(csv_file, expected_snippets=10, feature_dim=512)
        self.assertEqual(len(dataset), 2)
        
        sample_0 = dataset[0]
        self.assertEqual(sample_0["features"].shape, (10, 512))
        self.assertIsInstance(sample_0["label"], torch.Tensor)
        self.assertIn(sample_0["label"].item(), [0, 1])


if __name__ == "__main__":
    unittest.main()
