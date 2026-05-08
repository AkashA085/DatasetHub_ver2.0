from __future__ import annotations

import csv
import json
import random
import shutil
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import tempfile
import requests
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi import Form, File, UploadFile
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session

from app.core.database import get_db, Dataset, Image, ClassDistribution, TrainingJob
from app.utils.file_utils import STORAGE_ROOT, PROCESSED_DIR

router = APIRouter(prefix="/train", tags=
                   ["Training"])

TRAINING_ROOT = STORAGE_ROOT / "training"
TRAINING_JOBS_DIR = TRAINING_ROOT / "jobs"
TRAINING_JOBS_DIR.mkdir(parents=True, exist_ok=True)

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


class TrainingStartRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    dataset_id: str
    model: str = "yolov8n.pt"
    model_architecture: str = "YOLOv8"
    pretrained_weights_used: bool = True
    epochs: int = Field(50, ge=1, le=1000)
    batch_size: int = Field(16, ge=1, le=256)
    image_size: int = Field(640, ge=128, le=2048)
    learning_rate: float = Field(0.01, gt=0.0, le=1.0)
    optimizer: str = "auto"
    device: str = "cpu"
    val_split: float = Field(0.2, ge=0.05, le=0.4)
    test_split: float = Field(0.1, ge=0.0, le=0.4)
    seed: int = 42
    augmentation_enabled: bool = False
    augmentation_pipeline_name: str = "none"
    flip_enabled: bool = False
    rotation_angle: float = 0.0
    brightness_range: str = "0.0-0.0"
    noise_level: float = 0.0
    blur_enabled: bool = False
    augmented_images_count: int = Field(0, ge=0)
    experiment_name: str = "dataset_training"
    run_name: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    register_best_model: bool = False
    model_version: Optional[str] = None
    model_stage: str = "Staging"
    model_description: Optional[str] = None


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    dataset_id: str
    params: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    artifacts: Optional[Dict[str, str]] = None
    mlflow: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = []


class TrainingJobListResponse(BaseModel):
    jobs: List[TrainingJobResponse]


def _append_log(job: Dict[str, Any], message: str) -> None:
    ts = datetime.utcnow().strftime("%H:%M:%S")
    job["logs"].append(f"[{ts}] {message}")
    if len(job["logs"]) > 200:
        job["logs"] = job["logs"][-200:]


def _save_job_to_db(job: Dict[str, Any], db: Session) -> None:
    """Persist job status to database."""
    try:
        existing = db.query(TrainingJob).filter(TrainingJob.id == job["job_id"]).first()
        if existing:
            # Update existing job
            existing.status = job["status"]
            existing.started_at = job.get("started_at")
            existing.finished_at = job.get("finished_at")
            existing.metrics = job.get("metrics")
            existing.artifacts = job.get("artifacts")
            existing.mlflow = job.get("mlflow")
            existing.error = job.get("error")
            existing.logs = job.get("logs", [])
        else:
            # Create new job record
            training_job = TrainingJob(
                id=job["job_id"],
                dataset_id=job["dataset_id"],
                status=job["status"],
                params=job["params"],
                created_at=datetime.fromisoformat(job["created_at"]),
                started_at=datetime.fromisoformat(job["started_at"]) if job.get("started_at") else None,
                finished_at=datetime.fromisoformat(job["finished_at"]) if job.get("finished_at") else None,
                metrics=job.get("metrics"),
                artifacts=job.get("artifacts"),
                mlflow=job.get("mlflow"),
                error=job.get("error"),
                logs=job.get("logs", []),
            )
            db.add(training_job)
        db.commit()
    except Exception as e:
        print(f"Failed to save job to DB: {e}")
        db.rollback()


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _read_last_row(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    last_row: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row
    return last_row


def _build_loss_metrics(results_csv: Path) -> Dict[str, float]:
    row = _read_last_row(results_csv)
    if not row:
        return {}
    train_loss = 0.0
    val_loss = 0.0
    train_keys = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    val_keys = ["val/box_loss", "val/cls_loss", "val/dfl_loss"]
    train_found = False
    val_found = False
    for key in train_keys:
        v = _safe_float(row.get(key))
        if v is not None:
            train_loss += v
            train_found = True
    for key in val_keys:
        v = _safe_float(row.get(key))
        if v is not None:
            val_loss += v
            val_found = True
    out: Dict[str, float] = {}
    if train_found:
        out["training_loss"] = train_loss
    if val_found:
        out["validation_loss"] = val_loss
    return out


def _plot_curves(results_csv: Path, loss_curve_path: Path, accuracy_curve_path: Path) -> Dict[str, str]:
    created: Dict[str, str] = {}
    if not results_csv.exists():
        return created
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return created

    try:
        df = pd.read_csv(results_csv)
    except Exception:
        return created
    if df.empty:
        return created

    if "epoch" in df.columns:
        x = df["epoch"]
    else:
        x = list(range(len(df)))

    train_cols = [c for c in ["train/box_loss", "train/cls_loss", "train/dfl_loss"] if c in df.columns]
    val_cols = [c for c in ["val/box_loss", "val/cls_loss", "val/dfl_loss"] if c in df.columns]
    if train_cols or val_cols:
        plt.figure(figsize=(8, 4))
        if train_cols:
            plt.plot(x, df[train_cols].sum(axis=1), label="train_loss")
        if val_cols:
            plt.plot(x, df[val_cols].sum(axis=1), label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_curve_path)
        plt.close()
        created["loss_curve.png"] = str(loss_curve_path)

    map_col = "metrics/mAP50(B)"
    if map_col in df.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(x, df[map_col], label="mAP50")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(accuracy_curve_path)
        plt.close()
        created["accuracy_curve.png"] = str(accuracy_curve_path)

    return created


def _plot_class_distribution(class_distribution: Dict[str, int], output_path: Path) -> bool:
    if not class_distribution:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    labels = list(class_distribution.keys())
    values = [class_distribution[k] for k in labels]

    plt.figure(figsize=(10, 4))
    plt.bar(labels, values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return True


def _write_training_logs(job: Dict[str, Any], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(job.get("logs", [])))


def _to_mlflow_params(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value)
        else:
            out[key] = value
    return out


def _log_params_chunked(mlflow_module: Any, params: Dict[str, Any], chunk_size: int = 100) -> None:
    items = list(params.items())
    for i in range(0, len(items), chunk_size):
        mlflow_module.log_params(dict(items[i:i + chunk_size]))


def _resolve_device(device_value: Any) -> str:
    """Resolve device with CUDA validation and optimization."""
    raw = str(device_value or "").strip().lower()
    if raw in {"", "none"}:
        return "cpu"
    if raw != "auto":
        return str(device_value)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            # Validate CUDA is properly configured
            try:
                torch.cuda.init()
                torch.cuda.empty_cache()
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                device_props = torch.cuda.get_device_properties(0)
                total_memory = device_props.total_memory / (1024**3)  # Convert to GB
                return "0"
            except Exception as cuda_error:
                print(f"⚠️  CUDA error: {cuda_error}. Falling back to CPU.")
                return "cpu"
        return "cpu"
    except Exception:
        return "cpu"


def _validate_and_setup_gpu(device: str, job: Dict[str, Any]) -> None:
    """Validate GPU setup and log device information."""
    if device.lower() == "cpu":
        _append_log(job, "⚠️  Training will run on CPU - this will be VERY SLOW")
        return
    
    try:
        import torch  # type: ignore
        _append_log(job, f"✓ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            _append_log(job, f"✓ GPU Device Count: {device_count}")
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)
                _append_log(job, f"  GPU {i}: {props.name} ({total_mem:.1f}GB)")
            torch.cuda.empty_cache()
            _append_log(job, "✓ GPU memory cleared and ready")
    except Exception as e:
        _append_log(job, f"⚠️  GPU validation warning: {str(e)}")


def _get_optimal_batch_size(device: str, image_size: int = 640) -> int:
    """Recommend optimal batch size based on GPU memory."""
    if device.lower() == "cpu":
        return 8  # Conservative for CPU
    
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return 16
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        # Batch size recommendations based on VRAM and image size
        if image_size >= 1280:
            if total_memory >= 24:
                return 64
            elif total_memory >= 16:
                return 32
            elif total_memory >= 8:
                return 16
            else:
                return 8
        elif image_size >= 640:
            if total_memory >= 24:
                return 128
            elif total_memory >= 16:
                return 64
            elif total_memory >= 8:
                return 32
            else:
                return 16
        else:  # <= 416
            if total_memory >= 24:
                return 256
            elif total_memory >= 16:
                return 128
            elif total_memory >= 8:
                return 64
            else:
                return 32
    except Exception:
        return 16  # Default fallback




    """Prepare YOLO dataset from uploaded real data only.
    
    CRITICAL: This function MUST use only images with labels from uploaded datasets.
    No mock data, no fallback data - ONLY real uploaded labeled images.
    """
    _append_log(job, f"Preparing dataset from uploaded data for dataset_id={dataset_id}")
    
    # Query ONLY images that have labels (real labeled data)
    # has_label field was set during upload based on matching label files
    images = db.query(Image).filter(
        Image.dataset_id == dataset_id,
        Image.has_label == True  # CRITICAL: Only real labeled images
    ).all()
    
    if not images:
        all_images = db.query(Image).filter(Image.dataset_id == dataset_id).all()
        _append_log(job, f"ERROR: No labeled images found. Total images: {len(all_images)}")
        if all_images:
            unlabeled = [img for img in all_images if not img.file_name or not Path(img.file_path).exists()]
            if unlabeled:
                _append_log(job, f"Found {len(unlabeled)} images without labels or inaccessible files")
        raise ValueError(f"No labeled images found for dataset {dataset_id}. Cannot train without labels.")

    _append_log(job, f"Found {len(images)} labeled images for dataset")
    
    # Validate all images and labels exist and are accessible
    pairs = []
    invalid_images = []
    
    for img in images:
        try:
            img_path = Path(img.file_path)
            
            # Validate image file exists and is readable  
            if not img_path.exists():
                invalid_images.append((img.file_name, f"Image file not found: {img_path}"))
                continue
            
            if not img_path.is_file():
                invalid_images.append((img.file_name, f"Image path is not a file: {img_path}"))
                continue
            
            # Construct label path using reliable method
            # Labels stored in same parent dir as images, just under "labels" instead of "images"
            parts = list(img_path.parts)
            
            if "images" not in parts:
                invalid_images.append((img.file_name, f"Image not in 'images' directory: {img_path}"))
                continue
            
            # Replace "images" with "labels" in the path
            idx = parts.index("images")
            label_parts = parts[:idx] + ["labels"] + parts[idx+1:]
            
            # Create label path with .txt extension (using stem to preserve name without extension)
            label_path = Path(*label_parts).parent / (Path(img_path).stem + ".txt")
            
            # Validate label file exists and is readable
            if not label_path.exists():
                invalid_images.append((img.file_name, f"Label file not found: {label_path}"))
                continue
            
            if not label_path.is_file():
                invalid_images.append((img.file_name, f"Label path is not a file: {label_path}"))
                continue
            
            # Verify label file is not empty (must have annotations)
            if label_path.stat().st_size == 0:
                invalid_images.append((img.file_name, f"Label file is empty: {label_path}"))
                continue
            
            # All checks passed - add to pairs
            pairs.append((img_path, label_path, img.file_name))
            
        except Exception as e:
            invalid_images.append((img.file_name, f"Error processing image: {str(e)}"))
            continue
    
    # Log any issues for debugging
    if invalid_images:
        _append_log(job, f"WARNING: {len(invalid_images)} images have issues and were skipped:")
        for fname, reason in invalid_images[:5]:  # Log first 5 for debugging
            _append_log(job, f"  - {fname}: {reason}")
        if len(invalid_images) > 5:
            _append_log(job, f"  ... and {len(invalid_images) - 5} more")
    
    if not pairs:
        raise ValueError(f"No valid image/label pairs found for training dataset {dataset_id}. "
                        f"Checked {len(images)} labeled images. "
                        f"{len(invalid_images)} had issues. "
                        f"Ensure all uploaded images have corresponding label files.")

    _append_log(job, f"Validated {len(pairs)} image/label pairs - all real uploaded data")
    
    # Shuffle and split using deterministic seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(pairs)

    total = len(pairs)
    n_test = int(total * test_split)
    n_val = int(total * val_split)
    n_train = total - n_test - n_val
    
    if n_train <= 0:
        raise ValueError(f"Invalid split sizes for {total} images. Train split became empty. "
                        f"Consider reducing val_split ({val_split}) and test_split ({test_split}).")
    
    _append_log(job, f"Split data: train={n_train}, val={n_val}, test={n_test}")

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val] if n_val > 0 else []
    test_pairs = pairs[n_train + n_val:] if n_test > 0 else []

    # Create training directory structure
    job_dir = TRAINING_JOBS_DIR / job["job_id"]
    dataset_dir = job_dir / "dataset"
    
    for split in ["train", "val", "test"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def copy_pairs(split_name: str, split_pairs: List[Any]) -> None:
        """Copy image/label pairs to training directory."""
        copy_count = 0
        for img_path, lbl_path, file_name in split_pairs:
            try:
                # Copy image
                dest_img = dataset_dir / split_name / "images" / file_name
                shutil.copy2(img_path, dest_img)
                
                # Copy label with correct name
                label_name = f"{Path(file_name).stem}.txt"
                dest_lbl = dataset_dir / split_name / "labels" / label_name
                shutil.copy2(lbl_path, dest_lbl)
                
                copy_count += 1
            except Exception as e:
                _append_log(job, f"ERROR copying {file_name}: {str(e)}")
                raise
        
        _append_log(job, f"Copied {copy_count} image/label pairs to {split_name} split")

    # Copy all splits
    copy_pairs("train", train_pairs)
    if val_pairs:
        copy_pairs("val", val_pairs)
    if test_pairs:
        copy_pairs("test", test_pairs)

    # Get class distribution from database
    class_rows = db.query(ClassDistribution).filter(ClassDistribution.dataset_id == dataset_id).all()
    
    if not class_rows:
        _append_log(job, "WARNING: No class distribution data found in database")
        class_ids = ["0"]
        names = ["class_0"]
    else:
        class_ids = sorted({str(c.class_id) for c in class_rows}, 
                          key=lambda x: int(x) if str(x).isdigit() else x)
        max_id = max(int(x) for x in class_ids if str(x).isdigit()) if any(str(x).isdigit() for x in class_ids) else len(class_ids) - 1
        names = [f"class_{i}" for i in range(max_id + 1)]
    
    _append_log(job, f"Dataset classes: {len(names)} -> {names}")

    # Create YOLO data.yaml file
    data_yaml = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "val/images" if val_pairs else "train/images",
        "test": "test/images" if test_pairs else "val/images" if val_pairs else "train/images",
        "names": names,
        "nc": len(names),
    }
    
    yaml_path = job_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    _append_log(job, f"Created YOLO config at {yaml_path}")
    _append_log(job, f"✓ Dataset prepared successfully from {total} real uploaded labeled images")
    return {
        "job_dir": job_dir,
        "yaml_path": yaml_path,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "n_test": len(test_pairs),
    }


def _run_training(job_id: str) -> None:
    from app.core.database import SessionLocal
    db = SessionLocal()
    
    try:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job:
            return
 
        job["status"] = "preparing"
        training_start_dt = datetime.utcnow()
        job["started_at"] = training_start_dt.isoformat()
        _append_log(job, "Starting training job.")

        dataset = db.query(Dataset).filter(Dataset.id == job["dataset_id"]).first()
        class_rows = db.query(ClassDistribution).filter(ClassDistribution.dataset_id == job["dataset_id"]).all()
        prepared = _prepare_yolo_dataset(
            dataset_id=job["dataset_id"],
            seed=job["params"]["seed"],
            val_split=job["params"]["val_split"],
            test_split=job["params"]["test_split"],
            db=db,
            job=job,
        )

        # Save job to DB after dataset preparation
        _save_job_to_db(job, db)

        class_distribution = {str(c.class_id): int(c.object_count or 0) for c in class_rows}
        dataset_name = f"dataset_{job['dataset_id'][:8]}"
        dataset_version = "1"
        dataset_format = "YOLO"
        if dataset is not None:
            dataset_name = f"dataset_{dataset.id[:8]}"
            dataset_version = "1"
            dataset_format = (dataset.format_type or "YOLO").upper()
            if isinstance(dataset.analysis_summary, dict):
                dataset_name = dataset.analysis_summary.get("dataset_name", dataset_name)
                dataset_version = str(dataset.analysis_summary.get("dataset_version", dataset_version))

        if job["params"]["augmented_images_count"] <= 0:
            guessed_aug_count = 0
            aug_dir = PROCESSED_DIR / job["dataset_id"] / "augmented_images"
            if aug_dir.exists():
                guessed_aug_count = len([p for p in aug_dir.rglob("*") if p.is_file()])
            job["params"]["augmented_images_count"] = guessed_aug_count

        run_dir = prepared["job_dir"] / "runs"
        run_name = job["params"].get("run_name") or f"train_{job['job_id'][:8]}"
        experiment_name = job["params"].get("experiment_name", "dataset_training")
        mlflow_tracking_uri = job["params"].get("mlflow_tracking_uri")

        mlflow = None
        mlflow_client = None
        mlflow_active = False
        mlflow_run_id = None
        try:
            import mlflow  # type: ignore
            from mlflow.tracking import MlflowClient  # type: ignore
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name, nested=False)
            mlflow_active = True
            mlflow_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
            mlflow_client = MlflowClient()
            job["mlflow"] = {
                "enabled": True,
                "tracking_uri": mlflow.get_tracking_uri(),
                "experiment_name": experiment_name,
                "run_id": mlflow_run_id or "",
            }
            _append_log(job, f"MLflow run started (experiment='{experiment_name}', run_id='{mlflow_run_id}')")
        except Exception as mlflow_init_error:
            job["mlflow"] = {
                "enabled": False,
                "error": f"{type(mlflow_init_error).__name__}: {mlflow_init_error}",
            }
            _append_log(job, f"MLflow disabled: {job['mlflow']['error']}")

        requested_device = job["params"].get("device")
        resolved_device = _resolve_device(requested_device)
        if str(requested_device).strip().lower() == "auto":
            _append_log(job, f"Resolved device='auto' to '{resolved_device}'.")

        # Validate and setup GPU
        _validate_and_setup_gpu(resolved_device, job)

        tracking_params: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "dataset_format": dataset_format,
            "total_images": int(dataset.total_images) if dataset and dataset.total_images is not None else prepared["n_train"] + prepared["n_val"] + prepared["n_test"],
            "total_labels": int(dataset.total_labels) if dataset and dataset.total_labels is not None else prepared["n_train"] + prepared["n_val"] + prepared["n_test"],
            "number_of_classes": int(dataset.total_classes) if dataset and dataset.total_classes is not None else len(class_distribution),
            "class_distribution": class_distribution,
            "train_val_split_ratio": f"{prepared['n_train']}:{prepared['n_val']}",
            "augmentation_enabled": job["params"]["augmentation_enabled"],
            "augmentation_pipeline_name": job["params"]["augmentation_pipeline_name"],
            "flip_enabled": job["params"]["flip_enabled"],
            "rotation_angle": job["params"]["rotation_angle"],
            "brightness_range": job["params"]["brightness_range"],
            "noise_level": job["params"]["noise_level"],
            "blur_enabled": job["params"]["blur_enabled"],
            "augmented_images_count": job["params"]["augmented_images_count"],
            "model_architecture": job["params"]["model_architecture"],
            "pretrained_weights_used": job["params"]["pretrained_weights_used"],
            "image_size": job["params"]["image_size"],
            "batch_size": job["params"]["batch_size"],
            "learning_rate": job["params"]["learning_rate"],
            "optimizer": job["params"]["optimizer"],
            "epochs": job["params"]["epochs"],
            "device_used": resolved_device,
            "run_name": run_name,
            "experiment_name": experiment_name,
            "training_start_time": job["started_at"],
            "register_best_model": job["params"]["register_best_model"],
            "model_version": job["params"].get("model_version", ""),
            "model_stage": job["params"]["model_stage"],
            "model_description": job["params"].get("model_description", ""),
        }
        if mlflow_active:
            _log_params_chunked(mlflow, _to_mlflow_params(tracking_params))

        job["status"] = "running"
        _append_log(job, "Running Ultralytics YOLO training.")
        _save_job_to_db(job, db)

        try:
             from ultralytics import YOLO, settings
             settings.update({"mlflow": False})
        except Exception as e:
            raise RuntimeError(
                "Ultralytics is not installed in backend environment. "
                "Install with: pip install ultralytics"
            ) from e

        model = YOLO(job["params"]["model"])

        # Clear GPU memory before training
        if resolved_device != "cpu":
            try:
                import torch  # type: ignore
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                _append_log(job, "✓ GPU memory cleared and optimized")
            except Exception as e:
                _append_log(job, f"⚠️  GPU memory clear warning: {str(e)}")

        # capture stdout/stderr from Ultralytics training so we can surface it
        # in the job log and make it visible to the frontend.  Ultralytics
        # prints progress directly to stdout, so we temporarily replace
        # sys.stdout/sys.stderr with a small wrapper that echoes to the
        # original stream and also appends each line to our job record.
        import sys

        class _StreamInterceptor:
            def __init__(self, job: Dict[str, Any], orig_stream):
                self.job = job
                self.orig = orig_stream

            def write(self, s: str) -> None:
                # send non-empty lines to logs
                if s:
                    for line in s.splitlines():
                        if line.strip():
                            _append_log(self.job, line)
                try:
                    self.orig.write(s)
                except Exception:
                    pass

            def flush(self) -> None:
                try:
                    self.orig.flush()
                except Exception:
                    pass

        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        interceptor = _StreamInterceptor(job, orig_stdout)
        sys.stdout = interceptor
        sys.stderr = interceptor
        try:
            result = model.train(
                data=str(prepared["yaml_path"]),
                epochs=job["params"]["epochs"],
                imgsz=job["params"]["image_size"],
                batch=job["params"]["batch_size"],
                lr0=job["params"]["learning_rate"],
                optimizer=job["params"]["optimizer"],
                device=resolved_device,
                project=str(run_dir),
                name="train",
                exist_ok=True,
                seed=job["params"]["seed"],
                # GPU Optimization Parameters
                workers=8 if resolved_device != "cpu" else 2,  # Increase workers for GPU training
                cache=True if resolved_device != "cpu" else False,  # Cache images in RAM for faster loading on GPU
                amp=True,  # Automatic Mixed Precision - faster GPU training
                patience=20,  # Early stopping patience (epochs without improvement)
                # Data Augmentation - enhanced for better model robustness
                hsv_h=0.015,  # Image HSV-Hue augmentation
                hsv_s=0.7,    # Image HSV-Saturation augmentation
                hsv_v=0.4,    # Image HSV-Value augmentation
                degrees=10.0,  # Image rotation (+/- deg)
                translate=0.1, # Image translation (+/- fraction)
                scale=0.5,     # Image scale (+/- gain)
                flipud=0.0,    # Image flip up-down (probability)
                fliplr=0.5,    # Image flip left-right (probability)
                mosaic=1.0,    # Image mosaic (probability)
                mixup=0.0,     # Image mixup (probability)
                # Performance optimizations
                close_mosaic=10,  # Disables mosaic augmentation for final 10 epochs
                # Validation
                val=True,
                save_period=1,
                # Memory optimization
                max_det=300,  # Maximum detections per image
            )
        finally:
            # restore original streams in all cases
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

        save_dir = Path(str(result.save_dir)) if hasattr(result, "save_dir") else (run_dir / "train")
        best_pt = save_dir / "weights" / "best.pt"
        last_pt = save_dir / "weights" / "last.pt"

        results_dict = result.results_dict if hasattr(result, "results_dict") and isinstance(result.results_dict, dict) else {}
        precision = _safe_float(results_dict.get("metrics/precision(B)"))
        recall = _safe_float(results_dict.get("metrics/recall(B)"))
        map50 = _safe_float(results_dict.get("metrics/mAP50(B)"))
        map50_95 = _safe_float(results_dict.get("metrics/mAP50-95(B)"))

        metrics: Dict[str, float] = {}
        if precision is not None:
            metrics["precision"] = precision
        if recall is not None:
            metrics["recall"] = recall
        if map50 is not None:
            metrics["mAP"] = map50
            metrics["accuracy"] = map50
        if map50_95 is not None:
            metrics["mAP50_95"] = map50_95
        if precision is not None and recall is not None and (precision + recall) > 0:
            metrics["F1_score"] = (2 * precision * recall) / (precision + recall)

        results_csv = save_dir / "results.csv"
        metrics.update(_build_loss_metrics(results_csv))

        inference_time = None
        if hasattr(result, "speed") and isinstance(result.speed, dict):
            inference_time = _safe_float(result.speed.get("inference"))
        if inference_time is not None:
            metrics["inference_time"] = inference_time

        job["metrics"] = metrics or None

        artifact_dir = prepared["job_dir"] / "mlflow_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        tracked_artifacts: Dict[str, str] = {}

        if best_pt.exists():
            best_target = artifact_dir / "best_model.pt"
            shutil.copy2(best_pt, best_target)
            tracked_artifacts["best_model.pt"] = str(best_target)
        if last_pt.exists():
            last_target = artifact_dir / "last_model.pt"
            shutil.copy2(last_pt, last_target)
            tracked_artifacts["last_model.pt"] = str(last_target)

        confusion_matrix = save_dir / "confusion_matrix.png"
        if confusion_matrix.exists():
            cm_target = artifact_dir / "confusion_matrix.png"
            shutil.copy2(confusion_matrix, cm_target)
            tracked_artifacts["confusion_matrix.png"] = str(cm_target)

        curves = _plot_curves(
            results_csv=results_csv,
            loss_curve_path=artifact_dir / "loss_curve.png",
            accuracy_curve_path=artifact_dir / "accuracy_curve.png",
        )
        tracked_artifacts.update(curves)

        class_dist_path = artifact_dir / "class_distribution_chart.png"
        if _plot_class_distribution(class_distribution, class_dist_path):
            tracked_artifacts["class_distribution_chart.png"] = str(class_dist_path)

        logs_path = artifact_dir / "training_logs.txt"
        _write_training_logs(job, logs_path)
        tracked_artifacts["training_logs.txt"] = str(logs_path)

        if mlflow_active:
            if metrics:
                mlflow.log_metrics(metrics)
            for artifact_name, artifact_path in tracked_artifacts.items():
                if Path(artifact_path).exists():
                    mlflow.log_artifact(artifact_path)
            if results_csv.exists():
                mlflow.log_artifact(str(results_csv))
            if save_dir.exists():
                mlflow.log_artifact(str(save_dir / "results.png")) if (save_dir / "results.png").exists() else None

            if job["params"]["register_best_model"] and mlflow_client:
                registry_info: Dict[str, Any] = {
                    "requested": True,
                    "model_stage": job["params"]["model_stage"],
                }
                try:
                    mlflow.pytorch.log_model(model.model, artifact_path="model")
                    registered_name = f"{dataset_name}_detector".replace("-", "_")
                    model_uri = f"runs:/{mlflow_run_id}/model"
                    model_version = mlflow.register_model(model_uri=model_uri, name=registered_name)
                    if job["params"].get("model_description"):
                        mlflow_client.update_model_version(
                            name=registered_name,
                            version=model_version.version,
                            description=job["params"]["model_description"],
                        )
                    mlflow_client.transition_model_version_stage(
                        name=registered_name,
                        version=model_version.version,
                        stage=job["params"]["model_stage"],
                    )
                    registry_info.update(
                        {
                            "status": "registered",
                            "registered_model_name": registered_name,
                            "model_version": str(model_version.version),
                        }
                    )
                    _append_log(job, f"Model registered to MLflow Model Registry as '{registered_name}' v{model_version.version}.")
                except Exception as registry_error:
                    registry_info.update(
                        {
                            "status": "failed",
                            "error": f"{type(registry_error).__name__}: {registry_error}",
                        }
                    )
                    _append_log(job, f"Model registry step failed: {registry_info['error']}")
                if isinstance(job.get("mlflow"), dict):
                    job["mlflow"]["registry"] = registry_info

        job["artifacts"] = {
            "run_dir": str(save_dir),
            "best_weights": str(best_pt) if best_pt.exists() else "",
            "last_weights": str(last_pt) if last_pt.exists() else "",
            **tracked_artifacts,
        }
        _save_job_to_db(job, db)
        
        training_end_dt = datetime.utcnow()
        total_training_time = (training_end_dt - training_start_dt).total_seconds()
        job["finished_at"] = training_end_dt.isoformat()
        if job.get("metrics") is None:
            job["metrics"] = {}
        if isinstance(job["metrics"], dict):
            job["metrics"]["total_training_time"] = total_training_time
        if mlflow_active:
            mlflow.log_param("training_end_time", job["finished_at"])
            mlflow.log_metric("total_training_time", total_training_time)
            mlflow.end_run()
        if isinstance(job.get("mlflow"), dict):
            job["mlflow"]["training_start_time"] = job["started_at"]
            job["mlflow"]["training_end_time"] = job["finished_at"]
            job["mlflow"]["total_training_time"] = total_training_time
        job["status"] = "completed"
        _append_log(job, "Training completed successfully.")

    except Exception as e:
        try:
            import mlflow  # type: ignore
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
        except Exception:
            pass
        job["status"] = "failed"
        job["error"] = f"{type(e).__name__}: {str(e)}"
        _append_log(job, f"Training failed: {job['error']}")
        _append_log(job, traceback.format_exc())
    finally:
        if not job.get("finished_at"):
            job["finished_at"] = datetime.utcnow().isoformat()
        _save_job_to_db(job, db)
        db.close()


@router.post("/start", response_model=TrainingJobResponse)
async def start_training(request: TrainingStartRequest, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if request.val_split + request.test_split >= 0.9:
        raise HTTPException(status_code=400, detail="val_split + test_split is too large")

    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "queued",
        "dataset_id": request.dataset_id,
        "params": request.model_dump(),
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "finished_at": None,
        "metrics": None,
        "artifacts": None,
        "error": None,
        "logs": [],
    }
    _append_log(job, "Job queued.")

    # persist minimal metadata so we can later look up models by dataset id
    job_dir = TRAINING_JOBS_DIR / job_id
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        meta_path = job_dir / "job_meta.json"
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump({
                "dataset_id": request.dataset_id,
                "params": request.model_dump(),
                "created_at": job["created_at"],
            }, mf)
    except Exception:
        # non‑fatal; just log and continue
        _append_log(job, "Failed writing job metadata file")

    with _jobs_lock:
        _jobs[job_id] = job

    _save_job_to_db(job, db)

    t = threading.Thread(target=_run_training, args=(job_id,), daemon=True)
    t.start()

    return TrainingJobResponse(**job)


@router.get("/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(db: Session = Depends(get_db)):
    # Get all jobs from database (most recent first)
    jobs_db = db.query(TrainingJob).order_by(TrainingJob.created_at.desc()).all()
    
    # Convert to response format
    jobs_response = []
    for job_record in jobs_db:
        job_dict = {
            "job_id": job_record.id,
            "status": job_record.status,
            "dataset_id": job_record.dataset_id,
            "params": job_record.params or {},
            "created_at": job_record.created_at.isoformat() if job_record.created_at else "",
            "started_at": job_record.started_at.isoformat() if job_record.started_at else None,
            "finished_at": job_record.finished_at.isoformat() if job_record.finished_at else None,
            "metrics": job_record.metrics,
            "artifacts": job_record.artifacts,
            "mlflow": job_record.mlflow,
            "error": job_record.error,
            "logs": job_record.logs or [],
        }
        jobs_response.append(TrainingJobResponse(**job_dict))
    
    return TrainingJobListResponse(jobs=jobs_response)


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str, db: Session = Depends(get_db)):
    # Try to get from database first (persistent)
    job_record = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if job_record:
        job_dict = {
            "job_id": job_record.id,
            "status": job_record.status,
            "dataset_id": job_record.dataset_id,
            "params": job_record.params or {},
            "created_at": job_record.created_at.isoformat() if job_record.created_at else "",
            "started_at": job_record.started_at.isoformat() if job_record.started_at else None,
            "finished_at": job_record.finished_at.isoformat() if job_record.finished_at else None,
            "metrics": job_record.metrics,
            "artifacts": job_record.artifacts,
            "mlflow": job_record.mlflow,
            "error": job_record.error,
            "logs": job_record.logs or [],
        }
        return TrainingJobResponse(**job_dict)
    
    # Fallback to in-memory (for jobs just started)
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return TrainingJobResponse(**job)


class PredictionResponse(BaseModel): 
    predictions: List[Dict[str, Any]]
    inference_time_ms: Optional[float] = None


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Run inference using the most recent trained model for a dataset",
    description="Provide a dataset_id and either an image_url or an uploaded file."
)
async def predict_image(
    request: Request,
    dataset_id: str = Form(...),
    image_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
):
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    
    if not image_url and not image_file:
        raise HTTPException(status_code=400, detail="Either image_url or image_file must be provided")
    
    if image_url and image_file:
        raise HTTPException(status_code=400, detail="Provide either image_url or image_file, not both")

    if image_url:
        image_url = image_url.strip()
        if image_url.startswith("/api/"):
            image_url = image_url[len("/api"):]
        if image_url.startswith("/"):
            base = str(request.base_url).rstrip("/")
            image_url = f"{base}{image_url}"

    # find the most recent job directory for this dataset that contains a best.pt
    def _find_latest_weights(dataset_id: str) -> Optional[Path]:
        # first look for jobs that have explicit metadata for this dataset
        matched = []  # type: List[Path]
        all_weights: List[Path] = []
        for job_dir in TRAINING_JOBS_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            weight_file = job_dir / "runs" / "train" / "weights" / "best.pt"
            if weight_file.exists():
                all_weights.append(weight_file)
            meta_path = job_dir / "job_meta.json"
            if not meta_path.exists():
                continue
            try:
                m = json.load(open(meta_path, "r", encoding="utf-8"))
            except Exception:
                continue
            if m.get("dataset_id") == dataset_id and weight_file.exists():
                matched.append(weight_file)
        if matched:
            # return newest of the matched set
            return max(matched, key=lambda p: p.stat().st_mtime)
        # no matching metadata, fall back to latest available weights
        if all_weights:
            chosen = max(all_weights, key=lambda p: p.stat().st_mtime)
            # log warning to stdout so developer can see potential mismatch
            print(f"WARNING: no metadata for dataset '{dataset_id}'; using '{chosen}'")
            return chosen
        return None

    weight_path = _find_latest_weights(dataset_id)
    if not weight_path:
        raise HTTPException(status_code=404, detail="No trained model weights found for this dataset")

    # get image path from url or uploaded file
    temp_path: Optional[Path] = None
    try:
        if image_file:
            if not image_file.filename:
                raise HTTPException(status_code=400, detail="Invalid file uploaded")
            # Check file size (limit to 10MB)
            file_size = 0
            contents = await image_file.read()
            file_size = len(contents)
            if file_size > 10 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
            suffix = Path(image_file.filename).suffix.lower()
            if suffix not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                raise HTTPException(status_code=400, detail="Unsupported file type. Supported: jpg, png, bmp, tiff")
            # Use temp dir on D drive
            import tempfile
            temp_dir = TRAINING_JOBS_DIR / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            tmp = temp_dir / f"upload_{uuid.uuid4()}{suffix}"
            tmp.write_bytes(contents)
            temp_path = tmp
        elif image_url:
            # Validate URL
            try:
                parsed = urlparse(image_url)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError("Invalid URL")
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid image_url")
            # download
            resp = requests.get(image_url, timeout=10)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download image from URL")
            if len(resp.content) > 10 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Downloaded image too large. Maximum size is 10MB")
            suffix = Path(parsed.path).suffix.lower() or ".jpg"
            if suffix not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                raise HTTPException(status_code=400, detail="Unsupported image type from URL")
            # Use temp dir on D drive
            temp_dir = TRAINING_JOBS_DIR / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            tmp = temp_dir / f"download_{uuid.uuid4()}{suffix}"
            tmp.write_bytes(resp.content)
            temp_path = tmp

        # perform inference
        from ultralytics import YOLO
        import time
        model = YOLO(str(weight_path))
        start_time = time.perf_counter()
        results = model.predict(source=str(temp_path), conf=0.001, device=_resolve_device("cpu"))
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        preds: List[Dict[str, Any]] = []
        # open image to get dimensions
        from PIL import Image as PILImage
        img = PILImage.open(str(temp_path))
        img_w, img_h = img.size
        for r in results:
            # results comes as list of Results, usually one element when single image
            if not hasattr(r, "boxes"):
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                x1, y1, x2, y2 = xyxy
                x_center = ((x1 + x2) / 2.0) / img_w
                y_center = ((y1 + y2) / 2.0) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                preds.append({
                    "x": x_center,
                    "y": y_center,
                    "width": width,
                    "height": height,
                    "confidence": conf,
                    "class_id": str(cls),
                    "class": f"class_{cls}",
                    "detection_id": str(uuid.uuid4()),
                })
        return PredictionResponse(predictions=preds, inference_time_ms=inference_time_ms)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
