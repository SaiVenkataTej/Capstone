"""
Streamlit Web Dashboard for Real-time Video Anomaly Detection.
Provides interactive model inference, temporal score visualization, and incident logging.
"""

from datetime import datetime
from pathlib import Path
import tempfile
from typing import Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
import torch

from src.api.database import IncidentDatabase
from src.models.mil_model import MILAnomalyDetector


@st.cache_resource
def get_cached_clip_extractor():
    """Load and cache the CLIP ViT-B/16 visual feature extractor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from analyze_custom_video import get_clip_extractor
        extractor = get_clip_extractor(device)
        return extractor, True, f"CLIP ViT-B/16 ready ({device})"
    except Exception as e:
        return None, False, f"CLIP extractor unavailable: {e}"


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
    st.title("📤 Video & Feature Inference")
    st.markdown("Upload a raw video file (`.mp4`, `.avi`, `.mov`) or pre-extracted CLIP visual embeddings (`.npy`) for real-time anomaly localization. Uploaded videos and detection results are **automatically archived and logged**.")
    
    c_left, c_right = st.columns([2, 1])
    with c_left:
        uploaded_file = st.file_uploader(
            "Select Video or Feature File",
            type=["mp4", "avi", "mov", "mkv", "npy"]
        )
    with c_right:
        threshold = st.slider("Anomaly Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        sigma = st.slider(r"Temporal Smoothing ($\sigma$)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        
    if uploaded_file is not None:
        filename = uploaded_file.name
        is_video = any(filename.lower().endswith(ext) for ext in [".mp4", ".avi", ".mov", ".mkv"])
        
        try:
            feats = None
            fps = 30.0
            snippet_len = 16
            
            # Ensure permanent uploads directory exists
            upload_dir = Path("./uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Unique archived file name
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_filename = f"{timestamp_str}_{filename}"
            archived_path = upload_dir / archived_filename
            
            if is_video:
                # 1. Permanently save uploaded video to uploads directory
                file_bytes = uploaded_file.read()
                archived_path.write_bytes(file_bytes)
                
                st.subheader("🎥 Video Preview")
                st.video(str(archived_path))
                st.caption(f"📁 Archived video path: `{archived_path.resolve()}`")
                
                # 2. Extract CLIP features
                clip_fn, clip_ok, clip_msg = get_cached_clip_extractor()
                if not clip_ok or clip_fn is None:
                    st.error(f"Failed to initialize CLIP feature extractor: {clip_msg}")
                    return
                    
                cap = cv2.VideoCapture(str(archived_path))
                if not cap.isOpened():
                    st.error(f"Could not open video file: {filename}")
                    return
                    
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                progress_bar = st.progress(0, text="Extracting CLIP ViT-B/16 snippet features...")
                frames_buffer = []
                snippet_features = []
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames_buffer.append(Image.fromarray(frame_rgb))
                    frame_count += 1
                    
                    if len(frames_buffer) == snippet_len:
                        feat = clip_fn(frames_buffer)
                        snippet_features.append(feat)
                        frames_buffer = []
                        if total_frames > 0:
                            pct = min(1.0, frame_count / total_frames)
                            progress_bar.progress(pct, text=f"Processing snippet {len(snippet_features)} ({frame_count}/{total_frames} frames)...")
                            
                if len(frames_buffer) > 0:
                    feat = clip_fn(frames_buffer)
                    snippet_features.append(feat)
                    
                cap.release()
                progress_bar.progress(1.0, text="Feature extraction completed!")
                
                if snippet_features:
                    feats = np.stack(snippet_features)[np.newaxis, ...]  # (1, num_snippets, 512)
                else:
                    st.error("No valid frames could be decoded from this video.")
                    return
            else:
                # Pre-extracted .npy features - also archive a copy
                file_bytes = uploaded_file.read()
                archived_path.write_bytes(file_bytes)
                feats = np.load(str(archived_path)).copy()
                if feats.ndim == 1:
                    feats = feats[np.newaxis, np.newaxis, :]
                elif feats.ndim == 2:
                    feats = feats[np.newaxis, ...]
                elif feats.ndim > 3:
                    feats = feats.mean(axis=tuple(range(2, feats.ndim)))
                    if feats.ndim == 2:
                        feats = feats[np.newaxis, ...]

            # 3. Model Inference
            device = next(model.parameters()).device
            feats_tensor = torch.from_numpy(feats).float().to(device)
            
            with torch.no_grad():
                scores = model.predict(feats_tensor, smooth_sigma=sigma)
                
            video_scores = scores[0]
            max_score = float(np.max(video_scores))
            is_anom = max_score >= threshold
            
            # 4. Automatically Log Video and Anomalies to Database
            db_record_name = archived_filename
            db.log_video(
                filename=db_record_name,
                num_snippets=len(video_scores),
                is_anomalous=is_anom,
                max_score=max_score
            )
            db.log_anomalies(db_record_name, video_scores, threshold=threshold)
            
            # 5. Interactive Plotly Anomaly Timeline
            st.divider()
            st.subheader("📈 Temporal Anomaly Risk Profile")
            
            x_labels = []
            hover_labels = []
            flagged_intervals = []
            
            for i, score in enumerate(video_scores):
                start_sec = (i * snippet_len) / fps
                end_sec = ((i + 1) * snippet_len) / fps
                time_range = f"{int(start_sec // 60):02d}:{int(start_sec % 60):02d} - {int(end_sec // 60):02d}:{int(end_sec % 60):02d}"
                
                if is_video:
                    x_labels.append(f"{int(start_sec // 60):02d}:{int(start_sec % 60):02d}")
                    hover_labels.append(f"Snippet #{i+1}<br>Time: {time_range}<br>Score: {score:.3f}")
                else:
                    x_labels.append(f"Snippet {i+1}")
                    hover_labels.append(f"Snippet #{i+1}<br>Score: {score:.3f}")
                    
                if score >= threshold:
                    flagged_intervals.append({
                        "Snippet #": i + 1,
                        "Time Window": time_range if is_video else f"Snippet {i+1}",
                        "Anomaly Score": f"{score:.4f}",
                        "Risk Level": "CRITICAL" if score >= 0.75 else "HIGH"
                    })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=video_scores,
                mode="lines+markers",
                name="Anomaly Probability",
                line=dict(color="#3b82f6", width=3),
                marker=dict(size=8, color="#2563eb"),
                hoverinfo="text",
                hovertext=hover_labels
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
                xaxis=dict(title="Video Timeline (MM:SS)" if is_video else "Temporal Video Snippets"),
                height=420,
                margin=dict(l=40, r=40, t=30, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 6. Classification Status & Auto-Save Confirmation
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                if is_anom:
                    st.error(f"🚨 **ANOMALOUS EVENT DETECTED** (Peak Score: **{max_score:.3f}**)")
                else:
                    st.success(f"✅ **NORMAL VIDEO ACTIVITY** (Peak Score: **{max_score:.3f}**)")
                    
            with res_col2:
                st.info(f"💾 **Auto-Saved**: Stored in `uploads/{archived_filename}` and logged {len(video_scores)} snippets to `incidents.db`.")
                    
            # 7. Flagged Intervals Table
            if flagged_intervals:
                st.subheader("⚠️ Flagged Anomaly Intervals")
                st.markdown(f"Detected **{len(flagged_intervals)}** suspicious snippet window(s) exceeding threshold `{threshold:.2f}`:")
                df_intervals = pd.DataFrame(flagged_intervals)
                st.dataframe(df_intervals, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error during video/feature analysis: {e}")


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
