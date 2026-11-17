# 🎥 Weakly Supervised Video Anomaly Detection (WS-MIL)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-ready system for fine-grained temporal anomaly detection in surveillance video feeds using **Weakly Supervised Multiple Instance Learning (WS-MIL)** and **CLIP ViT-B/16** visual representations.

---

## 📑 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Project Structure](#-project-structure)
- [Modular Blocks & Swapping Guide (MODULAR_BLOCKS_GUIDE.md)](MODULAR_BLOCKS_GUIDE.md)
- [Quick Start Guide](#-quick-start-guide)
- [Configuration & Experimentation](#-configuration--experimentation)
- [Hyperparameter Optimization with Optuna](#-hyperparameter-optimization-with-optuna)
- [Training & Evaluation](#-training--evaluation)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [FastAPI REST API](#-fastapi-rest-api)
- [Docker Deployment](#-docker-deployment)
- [Test Suite](#-test-suite)

---

## 🧠 Architecture Overview

Surveillance feeds only come with coarse video-level labels (normal vs. anomalous). This project leverages **Multiple Instance Learning (MIL)** with a **Top-K Ranking Loss** to localize specific anomalous snippets within a video:

```
[Video Feed] ──► [16-Frame Snippets] ──► [CLIP ViT-B/16 (512-D)]
                                                  │
                                                  ▼
                                      [TemporalScoringNetwork (MLP)]
                                                  │
                                                  ▼
                                      [Anomaly Scores s_t in (0, 1)]
                                                  │
                                                  ▼
                                      [1D Gaussian Temporal Filter]
                                                  │
                                                  ▼
                                      [Fine-Grained Incident Output]
```

### Core Components
1. **Backbone**: Pre-extracted CLIP ViT-B/16 embeddings (512-D).
2. **Temporal Scoring Network**: 3-layer MLP (`512 -> 256 -> 128 -> 1`) with `GroupNorm` and `Dropout(0.5)`.
3. **Loss Function**: Top-$K$ Ranking Loss ($k=5$) + Temporal Smoothness Constraint ($\sum (\Delta s)^2$).
4. **Post-Processing**: 1D Gaussian temporal smoothing ($\sigma = 1.0$).

---

## 📂 Project Structure

```text
Capstone/
├── configs/
│   └── config.yaml                     # Centralized experiment & production configuration
├── data/                               # Generated CSVs, metadata, and data cache
├── experiments/                        # Trained models (best_model.pt), logs & history plots
├── src/                                # Core modular Python package
│   ├── data/                           # O(1) PyTorch dataset & data preparation pipeline
│   ├── models/                         # MIL scoring network, ranking loss & smoothing
│   ├── training/                       # Training pipeline, learning rate scheduling & metrics
│   ├── api/                            # FastAPI REST service, schemas & SQLite database
│   ├── dashboard/                      # Streamlit interactive UI application
│   └── utils/                          # YAML config loaders & structured logging
├── tests/                              # Automated Pytest suite
├── Dockerfile                          # Multi-stage production container
├── requirements.txt                    # Standardized dependencies
├── train.py                            # CLI entry point for training
├── evaluate.py                         # CLI entry point for evaluation
├── api.py                              # Entry point for FastAPI server
├── dashboard.py                        # Entry point for Streamlit dashboard
├── setup_xd_violence_pipeline.py       # Data pipeline & CSV generation
└── FILE_MANIFEST.md                    # Complete file registry
```

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
# Clone the repository
git clone <repo-url>
cd Capstone

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data Pipeline
```bash
python setup_xd_violence_pipeline.py
```

### 3. Launch Dashboard
```bash
streamlit run dashboard.py
```
Access the dashboard at `http://localhost:8501`.

### 4. Launch REST API Server
```bash
python api.py
# Or with uvicorn directly:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## ⚙️ Configuration & Experimentation

All hyperparameters are centralized in [configs/config.yaml](file:///c:/Users/SAI%20VENKAT%20TEJA%20S/Desktop/Capstone/configs/config.yaml).

```yaml
data:
  train_csv: "xd_CLIP_rgb.csv"
  test_csv: "xd_CLIP_rgbtest.csv"
  batch_size: 32
  val_split: 0.2

model:
  input_dim: 512
  hidden_dims: [256, 128]
  dropout_rate: 0.5
  k: 5
  smooth_weight: 0.01

training:
  num_epochs: 50
  learning_rate: 0.0001
  patience: 10
  device: "cuda"
```

You can run experiments with different configuration files or command-line overrides:
```bash
python train.py --config configs/config.yaml --epochs 50 --lr 0.0001 --k 5
```

### 4. Hyperparameter Optimization with Optuna
Find mathematically optimal hyperparameters automatically using Bayesian optimization (TPE Sampler & Median Pruner):

```bash
# Run 20 trials of Bayesian hyperparameter search
python tune.py --n_trials 20 --epochs_per_trial 15

# Customize search parameters
python tune.py --n_trials 30 --epochs_per_trial 20 --batch_size 32 --output_dir ./experiments/optuna_study
```

The tuner searches over:
- Learning rate ($10^{-5}$ to $10^{-3}$, log-scale)
- Weight decay ($10^{-6}$ to $10^{-4}$, log-scale)
- Top-K ranking ($3$ to $7$)
- Smooth weight ($\lambda_{smooth} \in [10^{-3}, 10^{-1}]$)
- Dropout rate ($0.1$ to $0.6$)
- Hidden layer dimensions (`[256, 128]`, `[512, 256]`, `[128, 64]`)
- Gaussian smoothing ($\sigma \in [0.5, 2.0]$)

Best parameters are automatically saved to `experiments/optuna_study/best_hyperparameters.json`.

---

## 🏋️ Training & Evaluation

### 1. Train with Any Loss Formulation
```bash
# Baseline: Top-K Ranking Loss
python train.py --loss topk --k 5 --epochs 50 --batch_size 32

# Max-Margin Ranking Loss (Sultani et al., CVPR 2018)
python train.py --loss margin --margin 1.0 --epochs 50 --batch_size 32

# Focal Top-K Loss (Hard-Negative Modulation)
python train.py --loss focal --focal_gamma 2.0 --epochs 50 --batch_size 32

# Softmax Attention MIL Loss (Differentiable Deep MIL)
python train.py --loss attention --softmax_tau 2.0 --epochs 50 --batch_size 32

# Generalized Mean (GEM) Pooling Loss
python train.py --loss gem --gem_p 3.0 --epochs 50 --batch_size 32
```
Training progress is logged to console and saved under `experiments/loss_<name>/`. The best checkpoint is automatically saved to `experiments/loss_<name>/best_model.pt`, and loss curves are plotted to `experiments/loss_<name>/training_history.png`.

### 2. Automated Multi-Loss Benchmark Comparison
Train and compare all loss formulations side-by-side with publication plots:
```bash
python compare_losses.py --epochs 15 --batch_size 32 --device cuda
```
Outputs:
- `experiments/loss_benchmark/loss_comparison_summary.csv`
- `experiments/loss_benchmark/loss_comparison_table.tex` (LaTeX table for papers)
- `experiments/loss_benchmark/comparative_roc_pr_curves.png`
- `experiments/loss_benchmark/comparative_loss_convergence.png`

### 3. Evaluate on Test Set
```bash
python evaluate.py --checkpoint experiments/loss_topk/best_model.pt --test_csv xd_CLIP_rgbtest.csv
```

---

## 📊 Streamlit Dashboard

The Streamlit UI provides:
- **System Overview**: High-level incident statistics and database health.
- **Upload & Inference**: Upload `.npy` feature files, tune decision threshold and smoothing parameters on the fly, and view interactive Plotly temporal score curves.
- **Analytics & Audit**: Class distribution charts and historical audit summary.
- **Incident Records**: Filter and export logged anomalies as CSV.

---

## 🌐 FastAPI REST API

| Endpoint | Method | Description |
|---|:---:|---|
| `/health` | `GET` | Health check & model status |
| `/info` | `GET` | System architecture details |
| `/predict` | `POST` | Predict anomaly scores for raw feature arrays |
| `/analyze_video` | `POST` | Upload and analyze a `.npy` feature file |
| `/batch_analyze` | `POST` | Batch analyze multiple `.npy` feature files |
| `/incidents` | `GET` | Retrieve logged incidents from SQLite |
| `/statistics` | `GET` | System anomaly detection statistics |
| `/reload_model` | `POST` | Dynamically reload model weights |

---

## 🐳 Docker Deployment

Build and run the containerized application:
```bash
# Build Docker image
docker build -t video-anomaly-detection:latest .

# Run Streamlit Dashboard (Port 8501)
docker run -p 8501:8501 -v $(pwd)/experiments:/app/experiments video-anomaly-detection:latest

# Run FastAPI Server (Port 8000)
docker run -p 8000:8000 -v $(pwd)/experiments:/app/experiments video-anomaly-detection:latest uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 🧪 Test Suite

Run the automated test suite with pytest:
```bash
pytest tests/ -v
```
