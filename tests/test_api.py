"""
Unit tests for FastAPI components and SQLite database using standard unittest.
"""

from pathlib import Path
import unittest
import numpy as np

from src.api.database import IncidentDatabase
from src.api.schemas import (
    HealthStatus,
    PredictionRequest,
    PredictionResponse,
    VideoAnalysisResult
)


class TestAPIAndDatabase(unittest.TestCase):
    
    def setUp(self):
        self.db_path = Path("./test_incidents.db")
        self.db = IncidentDatabase(db_path=self.db_path)
        
    def tearDown(self):
        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except PermissionError:
                pass

    def test_database_logging_and_retrieval(self):
        """Test video logging, anomaly logging, and incident querying."""
        # 1. Log Video
        video_id = self.db.log_video("test_cctv_01.npy", num_snippets=10, is_anomalous=True, max_score=0.88)
        self.assertIsInstance(video_id, int)
        
        # 2. Log Anomalies
        dummy_scores = [0.1, 0.2, 0.85, 0.92, 0.15, 0.1, 0.05, 0.08, 0.12, 0.1]
        self.db.log_anomalies("test_cctv_01.npy", dummy_scores, threshold=0.5)
        
        # 3. Retrieve Incidents
        df = self.db.get_recent_incidents(limit=5)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)  # 2 snippets > 0.5
        
        # 4. Statistics
        stats = self.db.get_statistics()
        self.assertEqual(stats["total_videos"], 1)
        self.assertEqual(stats["anomalous_videos"], 1)
        self.assertEqual(stats["total_anomalies"], 2)
        self.assertEqual(stats["anomaly_rate_percent"], 100.0)

    def test_pydantic_schemas(self):
        """Test schema parsing and validation."""
        req = PredictionRequest(
            features=[[0.1] * 512 for _ in range(10)],
            threshold=0.6,
            smooth_sigma=1.2
        )
        self.assertEqual(req.threshold, 0.6)
        self.assertEqual(len(req.features), 10)
        
        res = PredictionResponse(
            video_id="pred_123",
            anomaly_scores=[0.1, 0.8],
            is_anomalous=True,
            max_score=0.8,
            threshold=0.5
        )
        self.assertTrue(res.is_anomalous)
        self.assertEqual(res.max_score, 0.8)


if __name__ == "__main__":
    unittest.main()
