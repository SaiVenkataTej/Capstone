"""
Streamlit Web Dashboard for Real-time Video Anomaly Detection.
Provides interactive model inference, temporal score visualization, and incident logging.
"""

from datetime import datetime
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.api.database import IncidentDatabase
from src.models.mil_model import MILAnomalyDetector


@st.cache_resource
def get_model(checkpoint_path: str = "./experiments/best_model.pt") -> Tuple[MILAnomalyDetector, bool, str]:
    """Load trained model weights with caching."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MILAnomalyDetector(input_dim=512, k=5)
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        model.load_checkpoint(ckpt, device=device)
        model_loaded = True
        msg = f"Loaded checkpoint: {ckpt.name}"
    else:
        model = model.to(device)
        model_loaded = False
        msg = f"No checkpoint found at {ckpt}. Running with uninitialized weights."
    model.eval()
    return model, model_loaded, msg


@st.cache_resource
def get_db(db_path: str = "incidents.db") -> IncidentDatabase:
    """Initialize database connection."""
    return IncidentDatabase(db_path=db_path)


def render_dashboard():
    """Main rendering entry point for Streamlit UI."""
    st.set_page_config(
        page_title="Surveillance Anomaly Detection",
        page_icon="📹",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    db = get_db()
    model, is_loaded, status_msg = get_model()
    
    # Custom CSS Styling
    st.markdown("""
    <style>
        .metric-card {
            background-color: #1e293b;
            color: #f8fafc;
            padding: 1.25rem;
            border-radius: 0.75rem;
            border: 1px solid #334155;
        }
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.875rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("📹 Video Anomaly System")
    if is_loaded:
        st.sidebar.success(f"🟢 {status_msg}")
    else:
        st.sidebar.warning(f"🟡 {status_msg}")
        
    page = st.sidebar.radio(
        "Navigation Menu",
        ["🏠 System Overview", "📤 Upload & Inference", "📊 Analytics & Audit", "📝 Incident Records", "⚙️ Model Settings"]
    )
    
    if page == "🏠 System Overview":
        show_overview(db, is_loaded)
    elif page == "📤 Upload & Inference":
        show_inference(model, db)
    elif page == "📊 Analytics & Audit":
        show_analytics(db)
    elif page == "📝 Incident Records":
        show_incidents(db)
    elif page == "⚙️ Model Settings":
        show_settings(db)


def show_overview(db: IncidentDatabase, is_loaded: bool):
    st.title("🎥 CCTV Video Anomaly Detection Platform")
    st.markdown(
        "Weakly Supervised Multiple Instance Learning (WS-MIL) for fine-grained temporal anomaly localization."
    )
    
    stats = db.get_statistics()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Videos Analyzed", f"{stats['total_videos']:,}")
    with c2:
        st.metric("Anomalous Videos", f"{stats['anomalous_videos']:,}", delta=f"{stats['anomaly_rate_percent']}% rate")
    with c3:
        st.metric("Incident Snippets Logged", f"{stats['total_anomalies']:,}")
    with c4:
        st.metric("Model Checkpoint Status", "Loaded" if is_loaded else "Uninitialized")
        
    st.divider()
    
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("📋 Recent Incidents")
        recent_df = db.get_recent_incidents(limit=6)
        if not recent_df.empty:
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No anomalies detected in the database yet. Upload feature embeddings to begin analysis.")
            
    with col_right:
        st.subheader("🔬 Architecture Highlights")
        st.markdown(r"""
        - **Visual Backbone**: CLIP ViT-B/16 (512-D spatial-temporal embeddings)
        - **Scoring Engine**: 3-layer MLP with GroupNorm & Regularized Dropout
        - **Supervision**: Top-K Ranking Loss with Temporal Smoothness Constraint
        - **Post-Processing**: 1D Gaussian Temporal Filter ($\sigma = 1.0$)
        """)


def show_inference(model: MILAnomalyDetector, db: IncidentDatabase):
    st.title("📤 Video Feature Upload & Inference")
    st.markdown("Upload pre-extracted CLIP visual embeddings (`.npy`) for real-time anomaly localization.")
    
    c_left, c_right = st.columns([2, 1])
    with c_left:
        uploaded_file = st.file_uploader("Select CLIP Feature File (.npy)", type=["npy"])
    with c_right:
        threshold = st.slider("Anomaly Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        sigma = st.slider(r"Temporal Smoothing ($\sigma$)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        
    if uploaded_file is not None:
        try:
            feats = np.load(uploaded_file).copy()
            if feats.ndim == 1:
                feats = feats[np.newaxis, np.newaxis, :]
            elif feats.ndim == 2:
                feats = feats[np.newaxis, ...]
            elif feats.ndim > 3:
                feats = feats.mean(axis=tuple(range(2, feats.ndim)))
                if feats.ndim == 2:
                    feats = feats[np.newaxis, ...]
                    
            device = next(model.parameters()).device
            feats_tensor = torch.from_numpy(feats).float().to(device)
            
            with torch.no_grad():
                scores = model.predict(feats_tensor, smooth_sigma=sigma)
                
            video_scores = scores[0]
            max_score = float(np.max(video_scores))
            is_anom = max_score >= threshold
            
            st.divider()
            st.subheader("📈 Temporal Anomaly Profile")
            
            fig = go.Figure()
            x_vals = [f"Snippet {i+1}" for i in range(len(video_scores))]
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=video_scores,
                mode="lines+markers",
                name="Anomaly Probability",
                line=dict(color="#3b82f6", width=3),
                marker=dict(size=8, color="#2563eb"),
                hovertemplate="Snippet: %{x}<br>Score: %{y:.3f}<extra></extra>"
            ))
            
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Threshold ({threshold:.2f})",
                annotation_position="top right"
            )
            
            fig.update_layout(
                yaxis=dict(range=[0, 1.05], title="Anomaly Probability Score"),
                xaxis=dict(title="Temporal Video Snippets (16 frames each)"),
                height=420,
                margin=dict(l=40, r=40, t=30, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification Card
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if is_anom:
                    st.error(f"🚨 **ANOMALOUS EVENT DETECTED** (Peak Score: **{max_score:.3f}**)")
                else:
                    st.success(f"✅ **NORMAL VIDEO ACTIVITY** (Peak Score: **{max_score:.3f}**)")
                    
            with res_col2:
                if st.button("💾 Save & Log to Audit Database", use_container_width=True):
                    db.log_video(
                        filename=uploaded_file.name,
                        num_snippets=len(video_scores),
                        is_anomalous=is_anom,
                        max_score=max_score
                    )
                    db.log_anomalies(uploaded_file.name, video_scores, threshold=threshold)
                    st.success("Successfully logged incident record to database!")
                    
        except Exception as e:
            st.error(f"Error processing feature file: {e}")


def show_analytics(db: IncidentDatabase):
    st.title("📊 Analytics & Detection Statistics")
    stats = db.get_statistics()
    
    if stats["total_videos"] == 0:
        st.info("No data recorded yet. Upload and analyze videos to populate metrics.")
        return
        
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Video Distribution")
        fig = go.Figure(data=[go.Pie(
            labels=["Anomalous", "Normal"],
            values=[stats["anomalous_videos"], stats["total_videos"] - stats["anomalous_videos"]],
            hole=0.45,
            marker_colors=["#ef4444", "#10b981"]
        )])
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Audit Summary")
        st.markdown(f"""
        - **Total Videos Processed**: `{stats['total_videos']}`
        - **Confirmed Anomalies**: `{stats['anomalous_videos']}`
        - **Total Incident Snippets**: `{stats['total_anomalies']}`
        - **Anomaly Detection Rate**: `{stats['anomaly_rate_percent']}%`
        """)


def show_incidents(db: IncidentDatabase):
    st.title("📝 Incident Audit Log")
    limit = st.slider("Number of records to fetch", min_value=5, max_value=100, value=25)
    df = db.get_recent_incidents(limit=limit)
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Audit Log as CSV",
            data=csv_bytes,
            file_name=f"incident_audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No recorded incidents found.")


def show_settings(db: IncidentDatabase):
    st.title("⚙️ System & Model Configuration")
    st.markdown("Manage model checkpoints and purge database records.")
    
    st.subheader("Model Weights")
    ckpt_path = st.text_input("Model Checkpoint Path", value="./experiments/best_model.pt")
    if st.button("Reload Model Weights"):
        st.cache_resource.clear()
        st.success(f"Cleared cache. Reloading from {ckpt_path}")
        st.rerun()
        
    st.divider()
    st.subheader("Database Maintenance")
    if st.button("🗑️ Clear All Incident Records", type="secondary"):
        db.clear_all()
        st.warning("All database records and incident logs have been cleared.")
        st.rerun()


if __name__ == "__main__":
    render_dashboard()
