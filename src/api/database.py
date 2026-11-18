"""
SQLite Incident and Audit Database Manager.
Thread-safe helper for recording analyzed videos, incident anomalies, and system events.
"""

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class IncidentDatabase:
    """SQLite Database manager with WAL mode and robust error handling."""
    
    def __init__(self, db_path: Union[str, Path] = "incidents.db"):
        self.db_path = str(Path(db_path).resolve())
        self._init_db()
        
    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with timeout and WAL mode."""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize required database tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Videos table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE,
                    upload_time TIMESTAMP,
                    num_snippets INTEGER,
                    is_anomalous BOOLEAN,
                    max_score REAL
                )
            """)
            
            # Anomalies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER,
                    snippet_idx INTEGER,
                    score REAL,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (id) ON DELETE CASCADE
                )
            """)
            
            # System logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    event TEXT,
                    details TEXT
                )
            """)
            
            # Auto-migrate schema: ensure max_score column exists in videos table
            cursor.execute("PRAGMA table_info(videos)")
            existing_cols = [c[1] for c in cursor.fetchall()]
            if "max_score" not in existing_cols:
                cursor.execute("ALTER TABLE videos ADD COLUMN max_score REAL DEFAULT 0.0")
                
            conn.commit()

    def log_video(
        self,
        filename: str,
        num_snippets: int,
        is_anomalous: bool,
        max_score: float = 0.0
    ) -> int:
        """Insert or update video record and return video ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO videos (filename, upload_time, num_snippets, is_anomalous, max_score)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    upload_time = excluded.upload_time,
                    num_snippets = excluded.num_snippets,
                    is_anomalous = excluded.is_anomalous,
                    max_score = excluded.max_score
            """, (filename, datetime.now().isoformat(), num_snippets, int(is_anomalous), float(max_score)))
            
            cursor.execute("SELECT id FROM videos WHERE filename = ?", (filename,))
            video_id = cursor.fetchone()[0]
            conn.commit()
            return video_id

    def log_anomalies(
        self,
        video_filename: str,
        anomaly_scores: Union[np.ndarray, List[float]],
        threshold: float = 0.5
    ) -> None:
        """Log anomalous snippets for a video."""
        scores_list = [float(s) for s in anomaly_scores]
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM videos WHERE filename = ?", (video_filename,))
            res = cursor.fetchone()
            if not res:
                return
            video_id = res[0]
            
            # Clear previous anomalies for this video to prevent duplicates
            cursor.execute("DELETE FROM anomalies WHERE video_id = ?", (video_id,))
            
            for idx, score in enumerate(scores_list):
                if score >= threshold:
                    cursor.execute("""
                        INSERT INTO anomalies (video_id, snippet_idx, score, timestamp)
                        VALUES (?, ?, ?, ?)
                    """, (video_id, idx, score, datetime.now().isoformat()))
            conn.commit()

    def log_event(self, event: str, details: str = "") -> None:
        """Log system or audit event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_logs (timestamp, event, details)
                VALUES (?, ?, ?)
            """, (datetime.now().isoformat(), event, details))
            conn.commit()

    def get_recent_incidents(self, limit: int = 20) -> pd.DataFrame:
        """Retrieve recent anomaly incidents as a pandas DataFrame."""
        with self._get_connection() as conn:
            query = """
                SELECT v.filename, a.snippet_idx, a.score, a.timestamp
                FROM anomalies a
                JOIN videos v ON a.video_id = v.id
                ORDER BY a.timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
            return df

    def get_statistics(self) -> Dict[str, Any]:
        """Compute database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM videos")
            total_videos = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM videos WHERE is_anomalous = 1")
            anomalous_videos = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM anomalies")
            total_anomalies = cursor.fetchone()[0]
            
            rate = (anomalous_videos / total_videos * 100.0) if total_videos > 0 else 0.0
            
            return {
                "total_videos": total_videos,
                "anomalous_videos": anomalous_videos,
                "total_anomalies": total_anomalies,
                "anomaly_rate_percent": round(rate, 2)
            }

    def clear_all(self) -> None:
        """Clear all records from database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM anomalies")
            cursor.execute("DELETE FROM videos")
            cursor.execute("DELETE FROM system_logs")
            conn.commit()
