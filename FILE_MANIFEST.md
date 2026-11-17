# 📂 Video Anomaly Detection Project - File Manifest

## 📦 Active Project Structure

### 🚀 Root Entry Points & Configs
| File | Command | Purpose |
|---|---|---|
| `train.py` | `python train.py` | Main CLI for model training, validation, and checkpoint saving |
| `tune.py` | `python tune.py` | CLI for Optuna Bayesian hyperparameter optimization |
| `evaluate.py` | `python evaluate.py` | Standalone evaluation on the 800 test videos (`xd_CLIP_rgbtest.csv`) |
| `dashboard.py` | `streamlit run dashboard.py` | Interactive Streamlit web application & Plotly anomaly curves |
| `api.py` | `python api.py` / `uvicorn api:app` | FastAPI production REST server |
| `config.yaml` | - | Hyperparameter and configuration file |
| `requirements.txt` | `pip install -r requirements.txt` | Python dependency specifications |
| `Dockerfile` | `docker build -t vad .` | Multi-stage container image definition |
| `README.md` | - | Comprehensive project documentation and guide |
| `MODULAR_BLOCKS_GUIDE.md` | - | Plug-and-Play architecture guide for easily replacing blocks |
| `incidents.db` | - | SQLite audit database for video incidents and logs |

### 🛠️ Core Source Package (`src/`)
| Subpackage | File | Description |
|---|---|---|
| `src.data` | `src/data/dataset.py` | $O(1)$ pre-indexed `XDViolenceClipDataset` and DataLoader builder |
| `src.data` | `src/data/pipeline.py` | Directory scanner, label extractor, and CSV generator |
| `src.models` | `src/models/mil_model.py` | `TemporalScoringNetwork`, `TopKRankingLoss`, `MILAnomalyDetector` |
| `src.models` | `src/models/smoothing.py` | 1D Gaussian & moving-average temporal smoothing filters |
| `src.training` | `src/training/trainer.py` | `TrainingPipeline` with learning rate scheduling and early stopping |
| `src.training` | `src/training/tuner.py` | `HyperparameterTuner` using Optuna TPE sampler and median pruning |
| `src.training` | `src/training/metrics.py` | Video-level (AUC, AP, F1, FAR) and snippet-level metrics |
| `src.api` | `src/api/app.py` | FastAPI application with prediction routes and lifecycle management |
| `src.api` | `src/api/schemas.py` | Pydantic request and response schemas |
| `src.api` | `src/api/database.py` | Thread-safe SQLite incident and audit database layer |
| `src.dashboard` | `src/dashboard/app.py` | Streamlit interactive user interface |
| `src.utils` | `src/utils/config.py` | YAML configuration loader and merger |
| `src.utils` | `src/utils/logger.py` | Unified console and file logging setup |

### 🧪 Automated Test Suite (`tests/`)
| Test File | Description |
|---|---|
| `tests/test_model.py` | Forward pass shapes, Top-K loss, temporal smoothing, checkpoint saving/loading |
| `tests/test_data.py` | Filename regex parsing, $O(1)$ snippet pre-indexing, dataset item fetching |
| `tests/test_api.py` | Pydantic schemas, SQLite video logging, anomaly recording, and statistics |

### 📦 Archived Legacy Files (`archive/`)
All original standalone scripts and drafts are preserved in [archive/](file:///c:/Users/SAI%20VENKAT%20TEJA%20S/Desktop/Capstone/archive) with a detailed registry in [archive/README.md](file:///c:/Users/SAI%20VENKAT%20TEJA%20S/Desktop/Capstone/archive/README.md).
