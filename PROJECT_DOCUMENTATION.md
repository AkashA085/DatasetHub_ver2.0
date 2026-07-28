# DatasetHub — Knowledge Transfer Document

> Complete project documentation for the DatasetHub YOLO Training Platform.
> Last updated: July 2026

---

## 1. Project Overview

DatasetHub is a web-based platform for managing computer vision datasets and training YOLO object detection models. It handles the full lifecycle: upload image/label ZIPs → validate and analyze → augment data → train YOLO models with GPU acceleration → run inference with trained models.

**Core value:** Users upload drone/aerial imagery datasets (images + YOLO labels), and the system trains custom YOLOv8/YOLO11/YOLO26 models using an NVIDIA GPU, tracking everything through MLflow.

**Architecture:** React frontend (port 5173) → FastAPI backend (port 8001) → PostgreSQL database (port 5432, Docker) + local filesystem storage + NVIDIA GPU for training.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│                     http://localhost:5173                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP (React + Vite)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Vite Dev Server)                    │
│                     Port 5173                                    │
│                                                                  │
│  React 18 + Vite 5 + Chart.js + Axios + React Router 6         │
│                                                                  │
│  Pages: Dashboard | Datasets | Upload | Augment | Train |       │
│         Models | Images | Dataset Details                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ /api/* proxied to localhost:8001
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI + Uvicorn)                   │
│                     Port 8001                                    │
│                                                                  │
│  Python 3.10 + FastAPI 0.109 + SQLAlchemy + Ultralytics        │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Datasets │ │  Upload  │ │ Training │ │  Export  │           │
│  │  Routes  │ │  Routes  │ │  Routes  │ │  Routes  │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       │             │             │             │                 │
│  ┌────▼─────────────▼─────────────▼─────────────▼─────┐        │
│  │              Services Layer                          │        │
│  │  Validator | Analyzer | Augmentation | Export        │        │
│  │  Splitter | FormatConverter | Zipper                 │        │
│  └───────────────────┬─────────────────────────────────┘        │
│                       │                                          │
│  ┌───────────────────▼─────────────────────────────────┐        │
│  │              Training Engine (Background Thread)      │        │
│  │  YOLO Dataset Prep → Albumentations → YOLO Train     │        │
│  │  → EarlyStopping → MLflow Tracking → Metrics Save    │        │
│  └───────────────────┬─────────────────────────────────┘        │
└──────────────────────┼──────────────────────────────────────────┘
                       │
          ┌────────────┼────────────────┐
          ▼            ▼                ▼
┌──────────────┐ ┌──────────┐ ┌──────────────────┐
│  PostgreSQL  │ │ Storage  │ │   NVIDIA GPU     │
│  (Docker)    │ │  Root    │ │  RTX 3060 12GB   │
│  Port 5432   │ │ Local FS │ │  CUDA 12.1       │
│              │ │ 174 GB   │ │  PyTorch 2.3     │
│  DB: dataset │ │          │ │                  │
│  _management │ │          │ │  YOLOv8/11/26    │
└──────────────┘ └──────────┘ └──────────────────┘
```

---

## 3. Tech Stack

### Backend
| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10 | Runtime |
| FastAPI | 0.109.0 | Web framework |
| Uvicorn | 0.27.0 | ASGI server |
| SQLAlchemy | 2.0.49 | ORM + database |
| psycopg2-binary | 2.9.11 | PostgreSQL driver |
| Pydantic | 2.9.2 | Data validation |
| Ultralytics | 8.4.37 | YOLO training |
| PyTorch | 2.3.0+cu121 | GPU compute |
| MLflow | 3.11.1 | Experiment tracking |
| Albumentations | 2.0.8 | Image augmentation |
| OpenCV | 4.13.0 | Image processing |
| Pillow | 12.2.0 | Image loading |
| NumPy | 2.2.0 | Numerical ops |
| Pandas | 2.2.3 | Data analysis |
| Matplotlib | 3.10.8 | Plotting |

### Frontend
| Component | Version | Purpose |
|-----------|---------|---------|
| React | 18.2.0 | UI framework |
| Vite | 5.0.8 | Build tool + dev server |
| React Router | 6.20.0 | Client-side routing |
| Axios | 1.6.2 | HTTP client |
| Chart.js | 4.4.0 | Data visualization |
| react-icons | 4.12.0 | Icon library |

### Infrastructure
| Component | Version | Purpose |
|-----------|---------|---------|
| PostgreSQL | 16 (Alpine) | Database (Docker) |
| NVIDIA CUDA | 12.1 | GPU compute |
| NVIDIA Driver | Latest | GPU access |
| Docker | Latest | Container runtime |

### Hardware
| Resource | Specification |
|----------|--------------|
| GPU | NVIDIA GeForce RTX 3060 (12 GB VRAM) |
| CPU | 16 cores |
| RAM | 30 GB |
| Storage | 916 GB NVMe SSD |

---

## 4. How to Run

### Prerequisites
- NVIDIA GPU with CUDA drivers installed
- Docker installed (for PostgreSQL)
- Python 3.10+
- Node.js 20+

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/AkashA085/DatasetHub_ver2.0.git
cd DatasetHub_ver2.0

# 2. Start everything (PostgreSQL + Backend + Frontend)
bash start_local.sh
```

That's it. `start_local.sh` handles:
- Starting PostgreSQL Docker container (`local-postgres`)
- Starting backend Uvicorn on port 8001
- Starting frontend Vite on port 5173
- Health checks for all services
- Ctrl+C cleanup (kills all processes)

### Manual Start (if needed)

```bash
# PostgreSQL
docker start local-postgres 2>/dev/null || docker run -d \
  --name local-postgres \
  -e POSTGRES_DB=dataset_management \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -v model_trainig_agents_postgres_data:/var/lib/postgresql/data \
  --restart unless-stopped \
  postgres:16-alpine

# Backend
cd backend/Dataset_Management_tool
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001

# Frontend
cd frontend/Dataset_Management_tool_frontend
npm run dev -- --host 0.0.0.0
```

### Access Points
| Service | URL |
|---------|-----|
| Frontend UI | http://localhost:5173 |
| Backend API | http://localhost:8001 |
| API Documentation | http://localhost:8001/docs |
| Health Check | http://localhost:8001/health |
| Database | localhost:5432 |

### Stopping Services
```bash
# Clean shutdown (if using start_local.sh)
# Press Ctrl+C — it kills all children automatically

# Manual shutdown
pkill -f "uvicorn app.main:app"
pkill -f "vite.*--host"
```

---

## 5. Directory Structure

```
MODEL_TRAINIG WEBSITE_FINAL/
├── start_local.sh                    # All-in-one startup script
├── run_backend.sh                    # Backend-only startup
├── run_frontend.sh                   # Frontend-only startup
├── docker-compose.yml                # Docker config (legacy, not used in local mode)
├── PROJECT_DOCUMENTATION.md          # This file
│
├── backend/
│   └── Dataset_Management_tool/
│       ├── .env                      # Environment config (DATABASE_URL, STORAGE_ROOT)
│       ├── requirements.txt          # Python dependencies
│       ├── app/
│       │   ├── main.py               # FastAPI app, middleware, routes, /health
│       │   ├── core/
│       │   │   ├── __init__.py       # ORM models, engine, ensure_additional_columns
│       │   │   └── database.py       # SQLAlchemy engine, all ORM models (8 tables)
│       │   ├── api/
│       │   │   └── routes/
│       │   │       ├── datasets.py   # Dataset CRUD, images, labels, statistics
│       │   │       ├── upload.py     # Upload pipeline (ZIP extract + validate + store)
│       │   │       ├── training.py   # YOLO training + inference + MLflow (2200 lines)
│       │   │       ├── export.py     # Dataset download URL
│       │   │       ├── augmentation.py  # Albumentations augmentation
│       │   │       └── labeling.py   # Legacy JSON label update
│       │   ├── models/
│       │   │   └── schemas.py        # Pydantic request/response schemas
│       │   ├── services/
│       │   │   ├── validator.py      # Image/label validation
│       │   │   ├── analyzer.py       # Dataset statistics
│       │   │   ├── augmentation.py   # Augmentation pipeline
│       │   │   ├── export_service.py # Permanent storage + ZIP creation
│       │   │   ├── splitter.py       # Train/val/test splitting
│       │   │   └── zipper.py         # ZIP utilities
│       │   ├── utils/
│       │   │   └── file_utils.py     # Storage paths, ZIP extraction, cleanup
│       │   └── train_endpoint.py     # Legacy subprocess training endpoint
│       └── tests/                    # Unit + integration tests
│
├── frontend/
│   └── Dataset_Management_tool_frontend/
│       ├── package.json              # Node dependencies
│       ├── package-lock.json         # Locked dependency versions
│       ├── vite.config.js            # Vite config + proxy /api -> localhost:8001
│       ├── index.html                # SPA entry point
│       ├── public/
│       │   └── FWD_only_logo.png     # Sidebar logo
│       └── src/
│           ├── main.jsx              # React root with ErrorBoundary
│           ├── App.jsx               # Router configuration (9 routes)
│           ├── config.js             # API_BASE_URL = http://localhost:8001/api/v1
│           ├── index.css             # Global styles (blue/white theme)
│           ├── api/
│           │   └── datasetApi.js     # Axios client (all API methods)
│           ├── components/
│           │   ├── Layout/
│           │   │   ├── Layout.jsx    # Sidebar + main content wrapper
│           │   │   └── Layout.css    # Sidebar styles (dark blue/slate)
│           │   └── Common/
│           │       ├── AppErrorBoundary.jsx
│           │       ├── ErrorMessage.jsx/css
│           │       └── LoadingSpinner.jsx/css
│           └── pages/
│               ├── HomePage.jsx/css          # Dashboard with stats
│               ├── DatasetsPage.jsx/css      # Dataset listing with filters
│               ├── DatasetDetailsPage.jsx/css # Dataset detail + label editor
│               ├── UploadPage.jsx/css        # Upload images + labels ZIPs
│               ├── ImagesPage.jsx/css        # Global image gallery
│               ├── AugmentationPage.jsx/css  # Augmentation config
│               ├── TrainingPage.jsx/css      # Training config + job monitoring
│               └── ModelPreviewPage.jsx/css  # Inference preview
│
├── scripts/
│   ├── setup_nvidia_runtime.sh       # Install nvidia-docker2
│   └── setup_and_run_backend.sh      # Full NVIDIA setup + Docker build
│
└── datasethub_storage/               # All persistent data (174 GB)
    ├── uploads/                      # Uploaded dataset ZIPs + extracted files
    ├── exports/                      # Processed dataset ZIPs for download
    ├── training/                     # Training job artifacts
    │   └── jobs/{job_id}/            # Per-job: weights, logs, metrics, datasets
    ├── processed/                    # Internal annotations.json per dataset
    ├── analysis/                     # CSV statistics per dataset
    └── mlflow.db                     # MLflow tracking database
```

---

## 6. Database Schema

Database: PostgreSQL 16 (`dataset_management`)
Connection: `postgresql+psycopg2://postgres:postgres@localhost:5432/dataset_management`

### Entity Relationship Diagram

```
User 1──* Project 1──* Dataset 1──* Image 1──* Label
                     │                │
                     │                └── DatasetValidation (1:1)
                     │
                     ├── ClassDistribution (1:*)
                     └── TrainingJob (1:*)
```

### Table: users
| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(36) | PK, UUID |
| email | VARCHAR(256) | NOT NULL |
| created_at | TIMESTAMP | Default: utcnow |

### Table: projects
| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(36) | PK, UUID |
| name | VARCHAR(200) | NOT NULL |
| user_id | VARCHAR(36) | FK → users.id |
| created_at | TIMESTAMP | Default: utcnow |

### Table: datasets
| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(36) | PK, UUID |
| project_id | VARCHAR(36) | FK → projects.id |
| format_type | VARCHAR(50) | e.g., "yolo" |
| total_images | INTEGER | Default: 0 |
| total_labels | INTEGER | Default: 0 |
| total_classes | INTEGER | Default: 0 |
| total_objects | INTEGER | Default: 0 |
| avg_objects_per_image | DOUBLE | Default: 0.0 |
| missing_label_count | INTEGER | Default: 0 |
| corrupted_image_count | INTEGER | Default: 0 |
| csv_file_path | VARCHAR(1024) | Path to analysis CSV |
| zip_file_path | VARCHAR(1024) | Path to export ZIP |
| analysis_summary | JSON | Full analysis data |
| created_at | TIMESTAMP | Default: utcnow |

### Table: images
| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(36) | PK, UUID |
| dataset_id | VARCHAR(36) | FK → datasets.id |
| file_name | VARCHAR(1024) | Original filename |
| file_path | VARCHAR(4096) | Absolute path on disk |
| has_label | BOOLEAN | Default: false |

### Table: labels
| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(36) | PK, UUID |
| image_id | VARCHAR(36) | FK → images.id (nullable) |
| class_id | VARCHAR(50) | Class identifier |
| bbox_data | JSON | Format: `{"yolo": [cx, cy, w, h]}` |

### Table: dataset_validations
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER | PK, auto-increment |
| dataset_id | VARCHAR(36) | FK → datasets.id, UNIQUE |
| total_images | INTEGER | |
| total_labels | INTEGER | |
| missing_labels | INTEGER | Images without labels |
| orphan_labels | INTEGER | Labels without images |
| empty_labels | INTEGER | Empty label files |
| corrupted_images | INTEGER | Unreadable images |
| class_ids_found | JSON | List of class IDs |
| missing_label_images | JSON | List of filenames |
| orphan_label_files | JSON | List of filenames |
| empty_label_files | JSON | List of filenames |
| corrupted_image_files | JSON | List of filenames |

### Table: class_distributions
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER | PK, auto-increment |
| dataset_id | VARCHAR(36) | FK → datasets.id |
| class_id | VARCHAR(50) | Class identifier |
| object_count | INTEGER | Number of objects |

### Table: training_jobs
| Column | Type | Notes |
|--------|------|-------|
| id | VARCHAR(36) | PK, job_id |
| dataset_id | VARCHAR(36) | FK → datasets.id |
| status | VARCHAR(50) | queued/preparing/running/completed/failed/cancelled |
| params | JSON | Full training configuration |
| created_at | TIMESTAMP | Job creation time |
| started_at | TIMESTAMP | Nullable |
| finished_at | TIMESTAMP | Nullable |
| metrics | JSON | precision, recall, mAP, losses, training_time |
| artifacts | JSON | weight paths, run_dir, plots |
| mlflow | JSON | tracking URI, run_id, experiment info |
| error | VARCHAR(2048) | Error message if failed |
| logs | JSON | List of timestamped log strings |

---

## 7. Storage Layout

All persistent data lives under `datasethub_storage/` (configured via `DATASET_STORAGE_ROOT` env var).

```
datasethub_storage/                          # 174 GB total
│
├── uploads/                                 # 3.7 GB — Uploaded datasets
│   └── {session_uuid}/
│       ├── images/                          # Extracted source images
│       ├── labels/                          # Extracted YOLO .txt label files
│       ├── images.zip                       # Original upload
│       └── labels.zip                       # Original upload
│
├── processed/                               # Internal annotations
│   └── {session_uuid}/
│       └── annotations.json                 # Pascal VOC format annotations
│
├── analysis/                                # CSV analysis per dataset
│   └── {session_uuid}/
│       └── statistics.csv
│
├── exports/                                 # 3.5 GB — Downloadable ZIPs
│   └── {session_uuid}/
│       └── {session_uuid}.zip               # Processed dataset ZIP
│
├── training/                                # Training artifacts
│   └── jobs/
│       └── {job_id}/
│           ├── job_meta.json                # Dataset ID + params
│           ├── early_stop_best.pt           # Early stopping checkpoint
│           ├── run/                         # YOLO run output
│           │   └── train/
│           │       ├── weights/
│           │       │   ├── best.pt          # Best model weights
│           │       │   └── last.pt          # Last epoch weights
│           │       ├── results.csv          # Per-epoch metrics
│           │       ├── confusion_matrix.png
│           │       ├── results.png          # Loss/mAP curves
│           │       └── ...                  # Other YOLO artifacts
│           └── dataset/                     # Symlinked dataset for training
│               ├── train/
│               ├── val/
│               ├── test/
│               └── data.yaml
│
├── mlflow.db                                # 3 MB — MLflow SQLite tracking
└── 00000000-0000-0000-0000-000000000001/    # Default user directory
```

---

## 8. API Reference

Base URL: `http://localhost:8001/api/v1`

### Health & System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health (DB, GPU, disk) |

### Dataset Management

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload-dataset` | Upload images ZIP + labels ZIP (multipart) |
| GET | `/datasets` | List datasets (paginated, filterable, sortable) |
| GET | `/datasets/{id}` | Dataset detail + validation + class distribution |
| GET | `/datasets/{id}/images` | List images with YOLO labels (paginated) |
| GET | `/datasets/{id}/images/issues` | Flagged images (missing labels, blur, invalid bbox) |
| GET | `/datasets/{id}/statistics` | Dataset statistics for charts |
| PUT | `/datasets/{id}/images/{image_id}/labels` | Update image labels (DB + disk) |
| DELETE | `/datasets/{id}` | Delete dataset and all files |
| GET | `/images` | Global image gallery (all datasets) |

### Augmentation

| Method | Path | Description |
|--------|------|-------------|
| POST | `/augment` | Apply augmentations to dataset |

### Labeling

| Method | Path | Description |
|--------|------|-------------|
| POST | `/label` | Update labels (legacy JSON-based) |

### Export

| Method | Path | Description |
|--------|------|-------------|
| GET | `/download/{dataset_id}` | Get download URL for dataset ZIP |

### Training

| Method | Path | Description |
|--------|------|-------------|
| GET | `/train/detect-devices` | Detect GPU/CPU devices |
| GET | `/train/gpu-debug` | Raw CUDA environment info |
| POST | `/train/start` | Start YOLO training job |
| GET | `/train/jobs` | List all training jobs |
| GET | `/train/jobs/{job_id}` | Single job detail |
| GET | `/train/jobs/{job_id}/download` | Download training results ZIP |
| POST | `/train/jobs/{job_id}/stop` | Cancel a running job |
| DELETE | `/train/jobs/{job_id}` | Delete a training job |
| POST | `/train/predict` | Run inference on an image |

### Static Files

| Method | Path | Description |
|--------|------|-------------|
| GET | `/storage/{path}` | Serve files from storage root |
| GET | `/files/{path}` | Serve exported ZIP files |

---

## 9. Frontend Pages

### Dashboard (`/`)
- Shows total datasets, images, classes, objects, training time
- Lists recent datasets with quick-delete
- Quick action cards (Upload, Browse, Augment)

### Datasets (`/datasets`)
- Paginated dataset listing with format filter
- Sort by creation date, image count, or class count
- Cards show format, date, image/class/object counts

### Dataset Details (`/datasets/:id`)
- Overview tab: stats + class distribution charts (bar + pie)
- Images tab: browse images with labels, pagination
- Issues tab: flagged images (missing labels, blur, invalid bboxes)
- Label editor: click images to view/edit YOLO bounding boxes
- Download button for dataset ZIP

### Upload (`/upload`)
- Drag-and-drop or file picker for images ZIP + labels ZIP
- Format selection (YOLO, COCO, Pascal VOC)
- Upload progress bar
- Validation report after upload

### Images (`/images`)
- Global gallery across all datasets
- Pagination, label status filter
- Click to navigate to dataset details

### Augment (`/augment`)
- Select dataset from dropdown
- Configure: flip, rotation, brightness, contrast, blur, noise
- Runs server-side Albumentations pipeline

### Train (`/train`)
- **Left panel:** Training configuration
  - Dataset selector, base model (YOLOv8n/s/m, YOLO11n/s/m, YOLO26s/m/l)
  - Epochs, batch size, image size, learning rate
  - Device selector (GPU auto-detection)
  - Validation/test split, patience (early stopping)
  - Augmentation toggle, experiment name
- **Right panel:** Active training jobs
  - Real-time status polling (every 3 seconds)
  - Progress bar with epoch count
  - Live metrics: mAP, precision, recall
  - Stop/delete buttons
  - Download results button (completed jobs)

### Models (`/models`)
- Select dataset → loads trained model
- Browse dataset images with inference overlay
- Run inference on any image (upload, URL, or webcam)
- Adjust confidence/overlap thresholds
- View predictions as YOLO or Pascal VOC format

---

## 10. Training Pipeline

### Full Flow

```
1. USER CONFIGURES TRAINING
   └─ Selects dataset, model, epochs, batch size, etc.
   └─ Clicks "Start Training"
   
2. JOB CREATION (training.py: start_training)
   └─ Creates job record in DB (status: queued)
   └─ Spawns background thread

3. DATASET PREPARATION (_prepare_yolo_dataset)
   └─ Queries images + labels from DB
   └─ Validates all files exist on disk
   └─ Creates train/val/test split (symlink-based)
   └─ Writes data.yaml for YOLO
   └─ Status: preparing

4. OPTIONAL AUGMENTATION (build_augmented_dataset)
   └─ Albumentations pipeline:
      - Motion blur, low-light, sensor noise
      - Atmospheric effects, geometric transforms
      - Occlusion, color jitter, histogram equalization
   └─ Creates augmented images + labels

5. YOLO TRAINING (advanced_train)
   └─ Loads YOLO model (yolov8n.pt or user choice)
   └─ Configures: AdamW optimizer, cosine LR, warmup
   └─ Attaches callbacks:
      - EarlyStopping (patience-based on mAP50-95)
      - MLflow tracking (params, metrics, artifacts)
      - Job status updates (every epoch)
   └─ Runs YOLO training loop
   └─ Status: running

6. EARLY STOPPING (EarlyStopping class)
   └─ Monitors mAP50-95 each epoch
   └─ If no improvement for N epochs (patience) → stop
   └─ Saves best model on each improvement

7. COMPLETION
   └─ Saves best.pt + last.pt weights
   └─ Generates loss/mAP curves (results.png)
   └─ Generates confusion matrix
   └─ Computes final F1 score
   └─ Logs everything to MLflow
   └─ Status: completed
   └─ Metrics + artifact paths saved to DB

8. INFERENCE (predict_image)
   └─ Loads best.pt from most recent training job
   └─ Runs YOLO prediction on input image
   └─ Returns bounding boxes + classes + confidence
```

### Supported Models
| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| YOLOv8n | Nano | Fastest | Good |
| YOLOv8s | Small | Fast | Better |
| YOLOv8m | Medium | Medium | Best |
| YOLO11n | Nano | Fastest | Good |
| YOLO11s | Small | Fast | Better |
| YOLO11m | Medium | Medium | Best |
| YOLO26s | Small | Fast | Better |
| YOLO26m | Medium | Medium | Best |
| YOLO26l | Large | Slow | Best |

### Training Time Estimates (50K images, batch 16, 640px)
| Model | Per Epoch | 100 Epochs |
|-------|-----------|------------|
| YOLOv8n | ~5 min | ~8 hrs |
| YOLOv8s | ~12 min | ~20 hrs |
| YOLOv8m | ~40 min | ~67 hrs |

---

## 11. Environment Variables

### Backend (`.env` file at `backend/Dataset_Management_tool/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+psycopg2://postgres:postgres@localhost:5432/dataset_management` | PostgreSQL connection string |
| `DATASET_STORAGE_ROOT` | `<project>/datasethub_storage` | Root directory for all stored data |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed CORS origins |

### Frontend (`vite.config.js`)

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_PROXY_TARGET` | `http://localhost:8001` | Backend URL for dev proxy |

### Hardcoded in `config.js`

| Constant | Value | Description |
|----------|-------|-------------|
| `API_BASE_URL` | `http://localhost:8001/api/v1` | Base URL for all API calls |

---

## 12. Production Fixes Applied

16 security and reliability fixes were applied and pushed to `main` (commit `b423f27`):

### Security Fixes
| # | Fix | File |
|---|-----|------|
| 1 | `/health` endpoint — returns DB status, GPU info, disk space | `main.py` |
| 5 | SSRF protection — blocks private IPs, loopback, non-HTTP schemes on predict endpoint | `training.py` |
| 6 | CORS locked to configurable origins (env var `CORS_ORIGINS`) | `main.py` |
| 4 | Upload size limit reduced from 100 GB to 10 GB (backend + frontend) | `main.py`, `UploadPage.jsx` |

### Reliability Fixes
| # | Fix | File |
|---|-----|------|
| 8 | Thread-safe job state — `_get_job()` / `_update_job()` with lock-protected mutations | `training.py` |
| 9 | Label update atomicity — writes file first, then commits DB (was reversed) | `datasets.py` |
| 10 | LRU cache on `_load_annotations_yolo_map` (20 entries, invalidates on label update) | `datasets.py` |
| 11 | Export download checks multiple storage paths (exports, uploads, root) | `export.py` |
| 12 | Null-safety on all API responses (`|| []`, `?? {}` guards) | All page JSX files |
| 13 | AbortController on data-fetching effects (prevents stale state updates) | HomePage, DatasetsPage, DatasetDetailsPage |

### Frontend Fixes
| # | Fix | File |
|---|-----|------|
| 3 | `avg_objects_per_image.toFixed(2)` null crash — guarded with `?? 0` | `DatasetDetailsPage.jsx` |
| 7 | `.gitignore` merge conflict resolved | Frontend `.gitignore` |
| 14 | Navigation active state uses `startsWith` for sub-routes | `Layout.jsx` |
| 15 | AugmentationPage `setTimeout` cleanup on unmount via `useRef` | `AugmentationPage.jsx` |
| 16 | ModelPreviewPage removed `previewImageSize` from inference effect deps | `ModelPreviewPage.jsx` |

### Infrastructure Fixes
| # | Fix | File |
|---|-----|------|
| 2 | `start_local.sh` — process cleanup trap (Ctrl+C kills all children) | `start_local.sh` |

---

## 13. Troubleshooting

### Backend won't start
```bash
# Check if port 8001 is already in use
lsof -i :8001
# Kill existing process
pkill -f "uvicorn app.main:app"
# Check logs
tail -50 /tmp/backend.log
```

### PostgreSQL not connecting
```bash
# Check if container is running
docker ps | grep local-postgres
# Start it
docker start local-postgres
# Check health
pg_isready -h localhost -p 5432 -U postgres
```

### GPU not detected
```bash
# Check NVIDIA driver
nvidia-smi
# Check CUDA in Python
python -c "import torch; print(torch.cuda.is_available())"
# If False, torch may need CUDA reinstall:
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Training fails immediately
```bash
# Check training logs
tail -100 /tmp/backend.log | grep -i "error\|fail\|exception"
# Common causes:
# - Out of GPU memory → reduce batch_size (try 8 or 4)
# - Dataset too small → need at least a few images
# - Missing ultralytics → pip install ultralytics
```

### Frontend not loading
```bash
# Check if running
curl -s http://localhost:5173 | head -5
# Check for errors
tail -20 /tmp/frontend.log
# Rebuild if needed
cd frontend/Dataset_Management_tool_frontend && rm -rf node_modules && npm install && npm run dev
```

### Out of disk space
```bash
# Check usage
du -sh datasethub_storage/*/
# Clean old training runs (biggest consumer)
rm -rf datasethub_storage/training/jobs/<old_job_id>
# Clean Docker
docker system prune
# Check free space
df -h /
```

### Broken image symlinks
```bash
# Find broken symlinks
find datasethub_storage/ -maxdepth 5 -type l ! -exec test -e {} \; -print
# Fix: re-upload the dataset, or manually recreate symlinks
```

### Reset everything
```bash
# Nuclear option — start fresh
docker stop local-postgres && docker rm local-postgres
rm -rf datasethub_storage/*
rm -f backend/Dataset_Management_tool/dataset_management.db
bash start_local.sh
```

---

*Document generated for DatasetHub v1.0 — Production-ready as of July 2026.*
*Repository: https://github.com/AkashA085/DatasetHub_ver2.0*
