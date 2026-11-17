"""
Interactive CLI to check a raw video (.mp4, .avi, .mov) or feature file (.npy) for anomalies.
Usage:
    python analyze_custom_video.py --features XDTestClipFeatures/sample.npy
    python analyze_custom_video.py --video my_cctv_footage.mp4
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.models.mil_model import MILAnomalyDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a video or feature file for anomalies/violence.")
    parser.add_argument("--features", type=str, default=None, help="Path to pre-extracted .npy feature file")
    parser.add_argument("--video", type=str, default=None, help="Path to raw video file (.mp4, .avi, .mov, etc.)")
    parser.add_argument("--checkpoint", type=str, default="experiments/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly detection threshold (0.0 - 1.0)")
    parser.add_argument("--smooth_sigma", type=float, default=1.0, help="Temporal Gaussian smoothing sigma")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> MILAnomalyDetector:
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt.resolve()}")
    model = MILAnomalyDetector(input_dim=512, k=5)
    model.load_checkpoint(ckpt, device=device)
    model.eval()
    return model


def analyze_features(features_path: str, model: MILAnomalyDetector, device: str, threshold: float, sigma: float):
    print(f"\n[INFO] Loading feature file: {features_path}")
    feats = np.load(features_path).copy()
    
    if feats.ndim == 1:
        feats = feats[np.newaxis, np.newaxis, :]
    elif feats.ndim == 2:
        feats = feats[np.newaxis, ...]
    elif feats.ndim > 3:
        feats = feats.mean(axis=tuple(range(2, feats.ndim)))
        if feats.ndim == 2:
            feats = feats[np.newaxis, ...]
            
    tensor_feats = torch.from_numpy(feats).float().to(device)
    with torch.no_grad():
        scores = model.predict(tensor_feats, smooth_sigma=sigma)[0]
        
    print_results(scores, threshold)


def get_clip_extractor(device: str):
    """Initialize OpenAI CLIP ViT-B/16 visual feature extractor."""
    # Method 1: Try open_clip
    try:
        import open_clip
        print("[INFO] Loading CLIP ViT-B/16 model (via open_clip)...")
        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai", device=device)
        clip_model.eval()

        def extract_fn(pil_images: List[Any]) -> np.ndarray:
            tensors = torch.stack([preprocess(img) for img in pil_images]).to(device)
            with torch.no_grad():
                feats = clip_model.encode_image(tensors)
                # L2-normalize to unit hypersphere
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.mean(dim=0).cpu().numpy().astype(np.float32)

        return extract_fn
    except ImportError:
        pass

    # Method 2: Try official OpenAI clip package
    try:
        import clip
        print("[INFO] Loading CLIP ViT-B/16 model (via openai/clip)...")
        clip_model, preprocess = clip.load("ViT-B/16", device=device)
        clip_model.eval()

        def extract_fn(pil_images: List[Any]) -> np.ndarray:
            tensors = torch.stack([preprocess(img) for img in pil_images]).to(device)
            with torch.no_grad():
                feats = clip_model.encode_image(tensors)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.mean(dim=0).cpu().numpy().astype(np.float32)

        return extract_fn
    except ImportError:
        pass

    # Method 3: Try HuggingFace transformers
    try:
        from transformers import CLIPVisionModelWithProjection, AutoProcessor
        print("[INFO] Loading CLIP ViT-B/16 model (via transformers)...")
        clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16").to(device)
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
        clip_model.eval()

        def extract_fn(pil_images: List[Any]) -> np.ndarray:
            inputs = processor(images=pil_images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = clip_model(**inputs)
                feats = outputs.image_embeds
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.mean(dim=0).cpu().numpy().astype(np.float32)

        return extract_fn
    except ImportError:
        pass

    # Fallback message
    raise RuntimeError(
        "\n[ERROR] CLIP is required to extract features from raw video files (.mp4).\n"
        "Please install open-clip-torch in your virtual environment:\n\n"
        "    pip install open-clip-torch\n"
    )


def extract_features_from_video(video_path: str, device: str = "cpu", snippet_len: int = 16) -> Tuple[np.ndarray, float, int]:
    """Read video frames and extract L2-normalized 512-D CLIP ViT-B/16 embeddings for 16-frame snippets."""
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video Info: FPS={fps:.1f}, Total Frames={total_frames}, Duration={total_frames/fps:.2f}s")

    extract_fn = get_clip_extractor(device)

    frames_buffer = []
    snippet_features = []

    print("[INFO] Extracting CLIP ViT-B/16 snippet features from video frames...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames_buffer.append(pil_img)
        frame_idx += 1

        if len(frames_buffer) == snippet_len:
            feat = extract_fn(frames_buffer)  # (512,)
            snippet_features.append(feat)
            frames_buffer = []
            if len(snippet_features) % 10 == 0 or frame_idx >= total_frames:
                print(f"  Processed {len(snippet_features)} snippets ({frame_idx}/{total_frames} frames)...")

    # Handle trailing frames
    if len(frames_buffer) > 0:
        feat = extract_fn(frames_buffer)
        snippet_features.append(feat)

    cap.release()

    if snippet_features:
        features_arr = np.stack(snippet_features)[np.newaxis, ...]  # (1, num_snippets, 512)
    else:
        features_arr = np.zeros((1, 1, 512), dtype=np.float32)

    return features_arr, fps, snippet_len


def print_results(scores: np.ndarray, threshold: float, fps: float = 30.0, snippet_len: int = 16):
    max_score = float(np.max(scores))
    mean_score = float(np.mean(scores))
    is_anom = max_score >= threshold
    
    print("\n" + "=" * 60)
    print("                    ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total Snippets Evaluated : {len(scores)}")
    print(f"Peak Anomaly Score       : {max_score:.4f} (Threshold: {threshold:.2f})")
    print(f"Average Anomaly Score    : {mean_score:.4f}")
    
    if is_anom:
        print(f"[ALERT] Status: [ANOMALOUS / VIOLENCE DETECTED]")
        print("\nFlagged Snippet Time Intervals:")
        anom_indices = np.where(scores >= threshold)[0]
        for idx in anom_indices:
            start_sec = (idx * snippet_len) / fps
            end_sec = ((idx + 1) * snippet_len) / fps
            print(f"  - Snippet #{idx+1:02d} | Score: {scores[idx]:.3f} | Approx Time: {int(start_sec//60):02d}:{int(start_sec%60):02d} - {int(end_sec//60):02d}:{int(end_sec%60):02d}")
    else:
        print(f"[OK] Status: [NORMAL ACTIVITY - NO ANOMALY DETECTED]")
    print("=" * 60)


def main():
    args = parse_args()
    
    if not args.features and not args.video:
        print("[ERROR] Please specify either --features <file.npy> or --video <file.mp4>")
        print("Example:")
        print("  python analyze_custom_video.py --features XDTestClipFeatures/Bad.Boys.1995__#01-11-55_01-12-40_label_G-B2-B6__0.npy")
        return
        
    model = load_model(args.checkpoint, args.device)
    
    if args.features:
        analyze_features(args.features, model, args.device, args.threshold, args.smooth_sigma)
    elif args.video:
        features_arr, fps, snippet_len = extract_features_from_video(args.video, device=args.device)
        tensor_feats = torch.from_numpy(features_arr).float().to(args.device)
        with torch.no_grad():
            scores = model.predict(tensor_feats, smooth_sigma=args.smooth_sigma)[0]
        print_results(scores, args.threshold, fps=fps, snippet_len=snippet_len)


if __name__ == "__main__":
    main()
