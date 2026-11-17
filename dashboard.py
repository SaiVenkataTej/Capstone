"""
Streamlit Dashboard Entry Point for Video Anomaly Detection.
Run with:
    streamlit run dashboard.py
"""

from src.api.database import IncidentDatabase
from src.dashboard.app import render_dashboard
from src.models.mil_model import MILAnomalyDetector

if __name__ == "__main__":
    render_dashboard()
