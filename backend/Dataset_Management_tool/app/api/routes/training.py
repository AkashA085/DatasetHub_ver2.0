from __future__ import annotations

import csv
import json
import os
import random
import shutil
import threading
import traceback
import time
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

# ── 3rd-party ──────────────────────────────────────────────────────────────────
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
#  ADVANCED TRAINING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def get_training_cfg(params: Dict[str, Any], dataset_path: str, data_yaml: str) -> Dict[str, Any]:
    """Generate CFG dict from API parameters."""
    return dict(
        # ── Camera / Input ────────────────────────────────────────────────────────
        cam_w           = 1280,
        cam_h           = 720,
        imgsz           = params.get("image_size", 640),
        # ── Model ─────────────────────────────────────────────────────────────────
        model           = params.get("model", "yolov8n.pt"),
        nc              = 1,  # Will be updated based on dataset
        class_names     = ["drone"],  # Will be updated based on dataset
        # ── Dataset ───────────────────────────────────────────────────────────────
        dataset_root    = dataset_path,
        data_yaml       = data_yaml,
        # ── Training — tuned for RTX 5070 Ti ─────────────────────────────────────
        epochs          = params.get("epochs", 100),
        batch           = params.get("batch_size", 16),
        workers         = params.get("workers", 0),
        device          = params.get("device", "0"),
        optimizer       = params.get("optimizer", "AdamW"),
        lr0             = params.get("learning_rate", 0.0001),
        lrf             = 0.005,
        momentum        = 0.937,
        weight_decay    = 5e-4,
        warmup_epochs   = 5,
        warmup_bias_lr  = 0.1,
        cos_lr          = True,
        amp             = True,
        cache           = "disk",
        patience        = params.get("patience", 0),
        save_period     = -1,  # save only best + last (no intermediate checkpoints)
        # ── Loss weights (small fast objects need strong box loss) ────────────────
        box             = 9.5,
        cls             = 0.3,
        dfl             = 1.5,
        # ── Small-object tricks ───────────────────────────────────────────────────
        multi_scale     = False,
        overlap_mask    = False,
        # ── Output ────────────────────────────────────────────────────────────────
        project         = str(TRAINING_JOBS_DIR / params.get("job_id", "unknown") / "runs"),
        name            = params.get("run_name", f"exp_{datetime.now().strftime('%Y%m%d_%H%M')}"),
        # ── MLflow ────────────────────────────────────────────────────────────────
        mlflow_uri      = params.get("mlflow_tracking_uri", str(STORAGE_ROOT.parent / "mlruns")),
        mlflow_exp      = params.get("experiment_name", "Drone_vs_Drone_Detection"),
        # ── Jetson Export ─────────────────────────────────────────────────────────
        jetson_format   = "engine",
        jetson_fp16     = True,
        jetson_imgsz    = 640,
        jetson_workspace= 4,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  ALBUMENTATIONS AUGMENTATION PIPELINES
# ══════════════════════════════════════════════════════════════════════════════

def build_train_transform() -> A.Compose:
    bp = A.BboxParams(
        format         = "yolo",
        label_fields   = ["class_labels"],
        min_area       = 9,
        min_visibility = 0.20,
    )
    return A.Compose([
        # ── A. MOTION BLUR ─────────────────────────────────────────────────
        A.OneOf([
            A.MotionBlur(blur_limit=(7, 25), p=1.0),
            A.ZoomBlur(max_factor=(1.0, 1.10), p=1.0),
            A.Blur(blur_limit=(3, 7), p=1.0),
        ], p=0.65),
        # ── B. NIGHT / LOW-LIGHT ───────────────────────────────────────────
        A.OneOf([
            A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.65, -0.25),
                    contrast_limit=(0.1, 0.5), p=1.0),
                A.RandomGamma(gamma_limit=(20, 70), p=1.0),
            ]),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.35, -0.05),
                contrast_limit=(-0.1, 0.3), p=1.0),
            A.Compose([
                A.ToGray(p=1.0),
                A.CLAHE(clip_limit=6.0, tile_grid_size=(4, 4), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.4, 0.0),
                    contrast_limit=(0.2, 0.5), p=1.0),
            ]),
            A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.5, -0.1),
                    contrast_limit=(0.0, 0.4), p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=(-40, -10),
                    val_shift_limit=(-20, 10), p=1.0),
            ]),
        ], p=0.55),
        # ── C. SENSOR NOISE ────────────────────────────────────────────────
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0, 0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.06), intensity=(0.15, 0.55), p=1.0),
            A.MultiplicativeNoise(
                multiplier=(0.80, 1.20), per_channel=True,
                elementwise=True, p=1.0),
        ], p=0.55),
        # ── D. ATMOSPHERIC (altitude weather) ─────────────────────────────
        A.OneOf([
            A.RandomFog(
                fog_coef_range=(0.04, 0.25),
                alpha_coef=0.10, p=1.0),
            A.RandomRain(
                slant_range=(-15, 15),
                drop_length=15, drop_width=1,
                drop_color=(180, 180, 180),
                blur_value=3, brightness_coefficient=0.85,
                rain_type="drizzle", p=1.0),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.4),
                angle_range=(0.0, 1.0),
                num_flare_circles_range=(2, 5),
                src_radius=100, src_color=(255, 220, 160), p=1.0),
            A.RandomSnow(
                snow_point_range=(0.02, 0.12),
                brightness_coeff=1.5, p=1.0),
        ], p=0.25),
        # ── E. GEOMETRIC (moving platform) ────────────────────────────────
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,
                 fill=114, p=0.45),
        A.Affine(
            translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            shear=(-6, 6), p=0.35),
        A.Perspective(scale=(0.02, 0.07), p=0.30),
        A.HorizontalFlip(p=0.50),
        A.VerticalFlip(p=0.10),
        # ── F. STREAM / ENCODING ARTEFACTS ────────────────────────────────
        A.ImageCompression(quality_range=(40, 90), p=0.30),
        A.Defocus(radius=(1, 4), alias_blur=(0.1, 0.5), p=0.15),
        # ── G. OCCLUSION (partial cloud, bird, rotor wash) ─────────────────
        A.CoarseDropout(
            num_holes_range=(1, 8), hole_height_range=(8, 40), hole_width_range=(8, 40),
            fill=114, p=0.20),
        # ── H. COLOUR JITTER ───────────────────────────────────────────────
        A.HueSaturationValue(
            hue_shift_limit=12, sat_shift_limit=35, val_shift_limit=25,
            p=0.35),
        A.RGBShift(r_shift_limit=15, g_shift_limit=10,
                   b_shift_limit=15, p=0.25),
    ], bbox_params=bp)


def build_val_transform() -> A.Compose:
    bp = A.BboxParams(
        format       = "yolo",
        label_fields = ["class_labels"],
        min_area     = 4,
        min_visibility = 0.1,
    )
    return A.Compose([
        A.LongestMaxSize(max_size=640),
        A.PadIfNeeded(
            min_height=640, min_width=640,
            border_mode=cv2.BORDER_CONSTANT, fill=114),
    ], bbox_params=bp)


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTED DATASET WRITER
# ══════════════════════════════════════════════════════════════════════════════

def _process_single_image(args):
    img_path, src_lbl_dir, dst_img_dir, dst_lbl_dir, multiplier, mode = args
    try:
        import cv2
        from pathlib import Path
        import albumentations as A
        
        lbl_path = Path(src_lbl_dir) / (img_path.stem + ".txt")
        if not lbl_path.exists():
            return 0

        image = cv2.imread(str(img_path))
        if image is None:
            return 0
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, classes = [], []
        for line in lbl_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            classes.append(int(p[0]))
            bboxes.append([min(max(float(x), 0.0), 1.0) for x in p[1:5]])

        # Rebuild transform in the worker
        transform = build_train_transform() if mode == "train" else build_val_transform()
        versions = [("orig", image, bboxes, classes)]
        for i in range(multiplier):
            try:
                aug = transform(image=image, bboxes=bboxes, class_labels=classes)
                versions.append((f"a{i:02d}", aug["image"], aug["bboxes"], aug["class_labels"]))
            except Exception:
                pass

        written = 0
        for tag, aug_img, aug_boxes, aug_cls in versions:
            stem = f"{img_path.stem}_{tag}"
            out_im = Path(dst_img_dir) / f"{stem}.jpg"
            out_lb = Path(dst_lbl_dir) / f"{stem}.txt"
            cv2.imwrite(str(out_im), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(out_lb, "w") as f:
                for cls_id, box in zip(aug_cls, aug_boxes):
                    f.write(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
            written += 1
        return written
    except Exception:
        return 0

def write_augmented_split(
    src_img_dir : str,
    src_lbl_dir : str,
    dst_img_dir : str,
    dst_lbl_dir : str,
    multiplier  : int = 4,
    mode        : str = "train",
    job         : dict = None,
) -> int:
    imgs = sorted(Path(src_img_dir).glob("*.*"))
    Path(dst_img_dir).mkdir(parents=True, exist_ok=True)
    Path(dst_lbl_dir).mkdir(parents=True, exist_ok=True)

    total_imgs = len(imgs)
    if total_imgs == 0:
        return 0

    args_list = [(img, src_lbl_dir, dst_img_dir, dst_lbl_dir, multiplier, mode) for img in imgs]
    written = 0
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing
    
    # Use threads — IO-bound (file copies) and safe after CUDA init (no fork)
    workers = min(multiprocessing.cpu_count(), 8)
    
    if job:
        _append_log(job, f"Starting parallel augmentation for {mode} split ({total_imgs} images) using {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_single_image, args): args for args in args_list}
        for i, future in enumerate(as_completed(futures)):
            written += future.result()
            
            # Log progress every 10%
            if job and total_imgs > 0 and (i + 1) % max(1, total_imgs // 10) == 0:
                percent = int(((i + 1) / total_imgs) * 100)
                _append_log(job, f"Augmentation progress ({mode}): {percent}% completed ({i+1}/{total_imgs} images)")

    if job:
        _append_log(job, f"Finished augmentation for {mode} split. Created {written} total files.")
        
    return written


def build_augmented_dataset(cfg: dict, multiplier: int = 4, job: dict = None) -> str:
    root     = Path(cfg["dataset_root"])
    aug_root = root / "augmented"

    # train: augment
    write_augmented_split(
        src_img_dir = str(root / "train" / "images"),
        src_lbl_dir = str(root / "train" / "labels"),
        dst_img_dir = str(aug_root / "train" / "images"),
        dst_lbl_dir = str(aug_root / "train" / "labels"),
        multiplier  = multiplier,
        mode        = "train",
        job         = job,
    )
    # val/test: copy only
    import os
    for split in ["val", "test"]:
        for sub in ["images", "labels"]:
            src = root / split / sub
            dst = aug_root / split / sub
            if src.exists():
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True, copy_function=os.symlink)

    # write new data.yaml
    new_yaml = aug_root / "drone_data.yaml"
    with open(cfg["data_yaml"]) as f:
        orig = yaml.safe_load(f)
    orig["path"] = str(aug_root.resolve())
    with open(new_yaml, "w") as f:
        yaml.dump(orig, f, default_flow_style=False)

    return str(new_yaml)


# ══════════════════════════════════════════════════════════════════════════════
#  EARLY STOPPING LOGIC
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(
        self,
        patience=0,           # 0 disables early stopping (default)
        min_delta=0.001,      # minimum improvement required
        save_path="best_early_stop.pt"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path

        self.best_metric = None
        self.no_improve_count = 0

    def step(self, current_metric, model):
        # First epoch
        if self.best_metric is None:
            self.best_metric = current_metric
            try:
                torch.save(model.state_dict(), self.save_path)
                print(f"Initial best metric: {current_metric:.6f}")
            except Exception as e:
                print(f"Failed to save initial best model: {e}")
            return False

        # Check improvement
        if current_metric is None:
            return False

        if current_metric > self.best_metric + self.min_delta:
            print(
                f"Improved: "
                f"{self.best_metric:.6f} -> {current_metric:.6f}"
            )
            self.best_metric = current_metric
            self.no_improve_count = 0

            # Save best model
            try:
                torch.save(model.state_dict(), self.save_path)
                print(f"Best model saved: {self.save_path}")
            except Exception as e:
                print(f"Failed to save best model: {e}")
            return False
        if self.patience <= 0:
            # Patience <= 0 means no early stopping.
            return False

        else:
            self.no_improve_count += 1
            print(
                f"No improvement for "
                f"{self.no_improve_count} epoch(s)"
            )

        # Stop condition
        if self.no_improve_count >= self.patience:
            print("\nStopping training...")
            print(
                f"No improvement in last "
                f"{self.patience} epochs."
            )
            return True

        return False


# ══════════════════════════════════════════════════════════════════════════════
#  MLflow CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

class DroneMLflowCallback:
    def __init__(self, run, cfg: dict):
        self.run = run
        self.cfg = cfg
        self.best_map50 = 0.0

    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch
        # losses
        for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            try: mlflow.log_metric(f"loss/{k}", float(v), step=epoch)
            except Exception: pass
        # val metrics
        for k, v in trainer.metrics.items():
            try:
                key = k.replace("(B)", "").strip()
                mlflow.log_metric(f"val/{key}", float(v), step=epoch)
                if "mAP50" in key and float(v) > self.best_map50:
                    self.best_map50 = float(v)
                    mlflow.log_metric("best_mAP50", self.best_map50, step=epoch)
            except Exception: pass
        # learning rate
        for i, lr in enumerate(trainer.scheduler.get_last_lr()):
            mlflow.log_metric(f"lr/pg{i}", lr, step=epoch)

    def on_train_end(self, trainer):
        save_dir = Path(trainer.save_dir)
        # weights
        for w in ["best.pt", "last.pt"]:
            p = save_dir / "weights" / w
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="weights")
        # plots
        for f in save_dir.glob("*.png"):
            mlflow.log_artifact(str(f), artifact_path="plots")
        for f in save_dir.glob("*.csv"):
            mlflow.log_artifact(str(f), artifact_path="results")
        # final metrics
        for k, v in trainer.metrics.items():
            try: mlflow.log_metric(f"final/{k.replace('(B)','').strip()}", float(v))
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  ADVANCED TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def advanced_train(cfg: dict = None, use_albumentations: bool = True, aug_mult: int = 4, job: Dict[str, Any] = None, existing_mlflow_run=None):
    """Advanced YOLO training with Albumentations and MLflow."""
    if cfg is None:
        cfg = get_training_cfg({}, "", "")

    # ── optional: pre-generate augmented dataset on disk ─────────────────────
    active_yaml = cfg["data_yaml"]
    if use_albumentations:
        _append_log(job, f"Building Albumentations dataset (×{aug_mult})…")
        active_yaml = build_augmented_dataset(cfg, multiplier=aug_mult, job=job)
    else:
        _append_log(job, "Skipping Albumentations pre-augmentation (using YOLO built-in)")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_run = None
    mlflow_run_context = None
    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri(cfg["mlflow_uri"])
        mlflow.set_experiment(cfg["mlflow_exp"])
        mlflow_run_context = mlflow.start_run(run_name=cfg["name"])
        mlflow_run = mlflow_run_context.__enter__()
        mlflow_enabled = True
        _append_log(job, f"MLflow run started (experiment='{cfg['mlflow_exp']}', run_id='{mlflow_run.info.run_id}')")
    except Exception as mlflow_err:
        _append_log(job, f"MLflow init failed (non-fatal): {mlflow_err}")

    try:
        if mlflow_enabled and mlflow_run:
            try:
                loggable = {k: str(v) for k, v in cfg.items()
                            if not isinstance(v, (list, dict))}
                loggable["aug_multiplier"]   = str(aug_mult)
                loggable["use_albumentations"] = str(use_albumentations)
                loggable["cuda_device"]      = torch.cuda.get_device_name(0) \
                                               if torch.cuda.is_available() else "cpu"
                loggable["torch_version"]    = torch.__version__
                mlflow.log_params(loggable)
                mlflow.log_artifact(active_yaml, artifact_path="dataset")
            except Exception as log_err:
                _append_log(job, f"MLflow log warning (non-fatal): {log_err}")

        # ── load model ────────────────────────────────────────────────────────
        model = YOLO(cfg["model"])












        # ── attach MLflow callback ────────────────────────────────────────────
        if mlflow_enabled and mlflow_run:
            cb = DroneMLflowCallback(mlflow_run, cfg)
            model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
            model.add_callback("on_train_end",       cb.on_train_end)

        # ── attach Early Stopping callback ────────────────────────────────────
        # Save early stopping weights in the run directory
        job_id = cfg.get("job_id", "unknown")
        early_stop_save_path = TRAINING_JOBS_DIR / job_id / "early_stop_best.pt"
        early_stop_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        early_stopper = EarlyStopping(
            patience=cfg.get("patience", 0),
            min_delta=0.001,
            save_path=str(early_stop_save_path)
        )

        def on_train_epoch_end_early_stop(trainer):
            if job and job.get("stop_requested"):
                print("Stopping YOLO training loop due to user cancellation.")
                trainer.stop = True
                return

            # Extract mAP50-95 from trainer metrics
            # Note: YOLOv8 uses specific keys in trainer.metrics
            metrics = trainer.metrics
            # Try multiple common keys for mAP50-95; stay None if not available yet
            current_map = metrics.get('metrics/mAP50-95(B)')
            if current_map is None:
                current_map = metrics.get('mAP50-95')
            
            # Update job metrics for frontend real-time tracking
            if job:
                from app.core.database import SessionLocal
                if job.get("metrics") is None:
                    job["metrics"] = {}
                job["metrics"]["epoch"] = trainer.epoch + 1
                job["metrics"]["total_epochs"] = trainer.epochs
                if metrics:
                    job["metrics"]["mAP50"] = metrics.get('metrics/mAP50(B)', 0.0)
                    job["metrics"]["mAP50_95"] = current_map
                    job["metrics"]["precision"] = metrics.get('metrics/precision(B)', 0.0)
                    job["metrics"]["recall"] = metrics.get('metrics/recall(B)', 0.0)
                try:
                    db_session = SessionLocal()
                    _save_job_to_db(job, db_session)
                    db_session.close()
                except Exception:
                    pass

            if early_stopper.step(current_map, trainer.model):
                trainer.stop = True

        model.add_callback("on_train_epoch_end", on_train_epoch_end_early_stop)

        # ── YOLO train ────────────────────────────────────────────────────────
        # Disable YOLO's built-in early stopping when custom EarlyStopping is active
        yolo_patience = 9999 if cfg.get("patience", 0) > 0 else cfg.get("patience", 9999)

        results = model.train(
            data            = active_yaml,
            epochs          = cfg["epochs"],
            imgsz           = cfg["imgsz"],
            batch           = cfg["batch"],
            device          = cfg["device"],
            optimizer       = cfg["optimizer"],
            lr0             = cfg["lr0"],
            lrf             = cfg["lrf"],
            momentum        = cfg["momentum"],
            weight_decay    = cfg["weight_decay"],
            warmup_epochs   = cfg["warmup_epochs"],
            warmup_bias_lr  = cfg["warmup_bias_lr"],
            cos_lr          = cfg["cos_lr"],
            amp             = cfg["amp"],
            cache           = cfg["cache"],
            workers         = cfg["workers"],
            box             = cfg["box"],
            cls             = cfg["cls"],
            dfl             = cfg["dfl"],
            multi_scale     = cfg["multi_scale"],
            patience        = yolo_patience,
            save_period     = cfg["save_period"],
            project         = cfg["project"],
            name            = cfg["name"],
            exist_ok        = True,
            verbose         = True,
            # ── YOLO built-in augmentation (adds diversity on top of Albumentations)
            mosaic          = 1.0,
            mixup           = 0.15,
            copy_paste      = 0.10,
            hsv_h           = 0.015,
            hsv_s           = 0.4,
            hsv_v           = 0.3,
            degrees         = 10.0,
            translate       = 0.1,
            scale           = 0.5,
            flipud          = 0.1,
            fliplr          = 0.5,
            perspective    = 0.0005,
        )

    finally:
        if mlflow_run is not None:
            try:
                mlflow_run_context.__exit__(None, None, None)
            except Exception:
                pass

    return results


router = APIRouter(prefix="/train", tags=
                   ["Training"])

TRAINING_ROOT = STORAGE_ROOT / "training"
TRAINING_JOBS_DIR = TRAINING_ROOT / "jobs"
TRAINING_JOBS_DIR.mkdir(parents=True, exist_ok=True)

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_training_threads: Dict[str, threading.Thread] = {}
_thread_start_times: Dict[str, float] = {}
_threads_lock = threading.Lock()


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Thread-safe read of a job from _jobs."""
    with _jobs_lock:
        return _jobs.get(job_id)


def _update_job(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Thread-safe update of a job dict. Returns the updated job or None."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        job.update(updates)
        return job


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
    device: str = "0"
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
    # Set to 0 to disable early stopping; any positive value enables early‑stop after that many epochs.
    patience: int = Field(0, ge=0, le=1000)


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


class GPUDeviceInfo(BaseModel):
    """Information about a single GPU device."""
    device_id: str
    device_name: str
    total_memory_gb: float
    available: bool


class DeviceDetectionResponse(BaseModel):
    """Response with available devices and recommended device."""
    cuda_available: bool
    device_count: int
    devices: List[GPUDeviceInfo]
    recommended_device: str  # "auto", "cpu", or GPU ID like "0"
    message: str


def _cleanup_dead_threads():
    """Check running jobs; if a job's thread is dead but the job is still 'running',
    mark it as failed (the thread crashed silently)."""
    import traceback
    now = time.time()
    with _threads_lock:
        dead_job_ids = [
            jid for jid, t in list(_training_threads.items())
            if not t.is_alive() and now - _thread_start_times.get(jid, now) > 2
        ]
        for jid in dead_job_ids:
            del _training_threads[jid]
            _thread_start_times.pop(jid, None)

    for jid in dead_job_ids:
        with _jobs_lock:
            job = _jobs.get(jid)
        if job and job.get("status") in ("queued", "preparing", "running"):
            _update_job(jid, {
                "status": "failed",
                "error": "Training thread died unexpectedly (likely CUDA or DataLoader crash). Please check system resources and try again.",
                "finished_at": datetime.utcnow().isoformat(),
            })
            from app.core.database import SessionLocal
            try:
                db_session = SessionLocal()
                _save_job_to_db(job, db_session)
                db_session.close()
            except Exception:
                pass


def _append_log(job: Dict[str, Any], message: str) -> None:
    ts = datetime.utcnow().strftime("%H:%M:%S")
    job["logs"].append(f"[{ts}] {message}")
    if len(job["logs"]) > 200:
        job["logs"] = job["logs"][-200:]


def _save_job_to_db(job: Dict[str, Any], db: Session) -> None:
    """Persist job status to database."""
    try:
        # Helper to convert ISO string to datetime if needed
        def to_datetime(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except (ValueError, TypeError):
                    return None
            return None
        
        existing = db.query(TrainingJob).filter(TrainingJob.id == job["job_id"]).first()
        if existing:
            # Update existing job - convert ISO strings to datetime objects
            existing.status = job["status"]
            existing.started_at = to_datetime(job.get("started_at"))
            existing.finished_at = to_datetime(job.get("finished_at"))
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
                created_at=to_datetime(job["created_at"]),
                started_at=to_datetime(job.get("started_at")),
                finished_at=to_datetime(job.get("finished_at")),
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


def _count_job_dataset_images(job_id: str) -> int:
    job_dir = TRAINING_JOBS_DIR / job_id / "dataset"
    if not job_dir.exists():
        return 0

    image_count = 0
    for split in ["train", "val", "test"]:
        split_images = job_dir / split / "images"
        if split_images.exists() and split_images.is_dir():
            image_count += sum(1 for item in split_images.iterdir() if item.is_file())
    return image_count


def _compute_job_metrics_fallback(job_dict: Dict[str, Any]) -> Dict[str, Any]:
    metrics = job_dict.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}

    # Total training time fallback from timestamps
    if metrics.get("total_training_time") is None and job_dict.get("started_at"):
        try:
            start = datetime.fromisoformat(job_dict["started_at"])
            if job_dict.get("finished_at"):
                end = datetime.fromisoformat(job_dict["finished_at"])
            else:
                end = datetime.utcnow()
            metrics["total_training_time"] = (end - start).total_seconds()
        except Exception:
            pass

    # Image count fallback from prepared training directory
    if metrics.get("images_trained") is None:
        image_count = _count_job_dataset_images(job_dict["job_id"])
        if image_count > 0:
            metrics["images_trained"] = image_count
            if metrics.get("total_images") is None:
                metrics["total_images"] = image_count
        elif isinstance(metrics.get("total_images"), (int, float)):
            metrics["images_trained"] = int(metrics["total_images"])

    return metrics


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
    try:
        import torch  # type: ignore
        cuda_available = torch.cuda.is_available()
        if raw in {"", "none"}:
            return "0" if cuda_available else "cpu"
        if raw == "auto":
            return "0" if cuda_available else "cpu"
        if raw in {"cpu"}:
            return "cpu"

        # Support explicit device IDs and cuda-style values.
        if raw.startswith("cuda"):
            if not cuda_available:
                return "cpu"
            if raw == "cuda":
                return "0" if torch.cuda.device_count() > 0 else "cpu"
            try:
                _, idx = raw.split(":", 1)
                idx = idx.strip()
                if idx.isdigit() and int(idx) < torch.cuda.device_count():
                    return str(int(idx))
            except Exception:
                return "cpu"
        if raw.isdigit():
            if not cuda_available:
                return "cpu"
            idx = int(raw)
            if idx >= 0 and idx < torch.cuda.device_count():
                return str(idx)
            return "cpu"

        return "cpu"
    except Exception:
        return "cpu"


def _validate_and_setup_gpu(device: str, job: Dict[str, Any]) -> None:
    """Validate GPU setup and log device information. Enforces GPU usage when available."""
    if device.lower() == "cpu":
        _append_log(job, "⚠️  ⚠️  ⚠️  WARNING: Training will run on CPU - this will be VERY SLOW ⚠️  ⚠️  ⚠️")
        return
    
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            _append_log(job, f"✓✓✓ USING GPU FOR TRAINING ✓✓✓")
            _append_log(job, f"GPU count: {device_count}")
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)
                _append_log(job, f"  GPU {i}: {props.name} ({total_mem:.1f}GB)")
        else:
            _append_log(job, "⚠️  CUDA not available – falling back to CPU (training will be slow)")
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


def _prepare_yolo_dataset(
    dataset_id: str,
    seed: int,
    val_split: float,
    test_split: float,
    db: Session,
    job: Dict[str, Any],
) -> Dict[str, Any]:
    """Prepare YOLO dataset from uploaded data, including images without labels by creating empty placeholder label files.""",
  
    _append_log(job, f"Preparing dataset from uploaded data for dataset_id={dataset_id}")
    
    # Query ONLY required columns for images to save RAM
    images = db.query(Image.file_path, Image.file_name).filter(
        Image.dataset_id == dataset_id
    ).all()
    
    if not images:
        all_images_count = db.query(Image.id).filter(Image.dataset_id == dataset_id).count()
        _append_log(job, f"ERROR: No labeled images found. Total images: {all_images_count}")
        raise ValueError(f"No labeled images found for dataset {dataset_id}. Cannot train without labels.")

    _append_log(job, f"Found {len(images)} images for dataset")
    
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
            
            # Ensure label directory exists
            label_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If label file missing or empty, create empty placeholder for negative sample
            if not label_path.exists() or label_path.stat().st_size == 0:
                label_path.touch()
                _append_log(job, f"INFO: Created/Ensured empty label file for image {img.file_name}")
            
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
        import os
        import shutil
        for img_path, lbl_path, file_name in split_pairs:
            try:
                # Prefer symlink to save disk space; fall back to hard-copy
                dest_img = dataset_dir / split_name / "images" / file_name
                if not dest_img.exists():
                    try:
                        os.symlink(img_path, dest_img)
                    except OSError:
                        shutil.copy2(img_path, dest_img)
                
                # Symlink/copy label with correct name
                label_name = f"{Path(file_name).stem}.txt"
                dest_lbl = dataset_dir / split_name / "labels" / label_name
                if not dest_lbl.exists():
                    try:
                        os.symlink(lbl_path, dest_lbl)
                    except OSError:
                        shutil.copy2(lbl_path, dest_lbl)
                copy_count += 1
            except Exception as e:
                _append_log(job, f"ERROR linking {file_name}: {str(e)}")
                raise
        
        _append_log(job, f"Copied {copy_count} image/label pairs to {split_name} split")

    # Copy all splits
    copy_pairs("train", train_pairs)
    if val_pairs:
        copy_pairs("val", val_pairs)
    if test_pairs:
        copy_pairs("test", test_pairs)

    # Get class distribution from database and stored dataset metadata
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    class_rows = db.query(ClassDistribution).filter(ClassDistribution.dataset_id == dataset_id).all()
    custom_names = []
    if dataset and isinstance(dataset.analysis_summary, dict):
        custom_names = dataset.analysis_summary.get("class_names") or []

    if not class_rows:
        _append_log(job, "WARNING: No class distribution data found in database")
        class_ids = ["0"]
        names = custom_names if custom_names else ["class_0"]
    else:
        class_ids = sorted({str(c.class_id) for c in class_rows}, 
                          key=lambda x: int(x) if str(x).isdigit() else x)
        max_id = max(int(x) for x in class_ids if str(x).isdigit()) if any(str(x).isdigit() for x in class_ids) else len(class_ids) - 1
        if custom_names and len(custom_names) >= max_id + 1:
            names = [custom_names[i] if i < len(custom_names) else f"class_{i}" for i in range(max_id + 1)]
        else:
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
 
        if job.get("stop_requested"):
            _update_job(job_id, {
                "status": "cancelled",
                "finished_at": datetime.utcnow().isoformat(),
            })
            _append_log(job, "Training job cancelled before start.")
            _save_job_to_db(job, db)
            return
 
        training_start_dt = datetime.utcnow()
        _update_job(job_id, {
            "status": "preparing",
            "started_at": training_start_dt.isoformat(),
        })
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
        # Check for cancellation before starting heavy training step
        if job.get("stop_requested"):
            _update_job(job_id, {
                "status": "cancelled",
                "finished_at": datetime.utcnow().isoformat(),
            })
            _append_log(job, "Training cancelled before execution.")
            _save_job_to_db(job, db)
            return

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
        if not mlflow_tracking_uri:
            mlflow_tracking_uri = str(STORAGE_ROOT.parent / "mlruns")

        mlflow = None
        mlflow_client = None
        mlflow_active = False
        mlflow_run_id = None
        try:
            from ultralytics import YOLO, settings
            settings.update({"mlflow": False})
        except Exception as e:
            _append_log(job, f"Ultralytics import failed: {e}. Attempting pip install...")
            # Attempt on‑the‑fly installation (best‑effort, may require network)
            try:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                from ultralytics import YOLO, settings
                settings.update({"mlflow": False})
                _append_log(job, "Ultralytics installed successfully at runtime.")
            except Exception as install_err:
                raise RuntimeError(
                    f"Ultralytics is not installed and automatic installation failed: {install_err}. "
                    "Please add 'ultralytics' to backend requirements and rebuild the image."
                ) from install_err
        mlflow_active = False

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

        _update_job(job_id, {"status": "running"})
        if job.get("stop_requested"):
            _update_job(job_id, {
                "status": "cancelled",
                "finished_at": datetime.utcnow().isoformat(),
            })
            _append_log(job, "Training job cancelled before execution.")
            _save_job_to_db(job, db)
            return

        _append_log(job, "Running advanced YOLO training with Albumentations and MLflow.")
        _save_job_to_db(job, db)

        # Generate CFG for advanced training
        cfg = get_training_cfg(
            params=job["params"],
            dataset_path=str(prepared["job_dir"] / "dataset"),
            data_yaml=str(prepared["yaml_path"])
        )
        # Update class names from dataset
        if dataset and isinstance(dataset.analysis_summary, dict):
            custom_names = dataset.analysis_summary.get("class_names") or []
            if custom_names:
                cfg["class_names"] = custom_names
                cfg["nc"] = len(custom_names)

        # Clear GPU memory before training
        if resolved_device != "cpu":
            try:
                import torch  # type: ignore
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                _append_log(job, "✓ GPU memory cleared and optimized")
            except Exception as e:
                _append_log(job, f"  GPU memory clear warning: {str(e)}")

        existing_mlflow_run = None

        # capture stdout/stderr from training so we can surface it
        # in the job log and make it visible to the frontend.
        import sys

        class _StreamInterceptor:
            def __init__(self, job: Dict[str, Any], orig_stream):
                self.job = job
                self.orig = orig_stream

            def write(self, s: str) -> None:
                # Capture all output, including progress bars. Replace carriage returns with newlines.
                if s:
                    cleaned = s.replace('\r', '\n')
                    for line in cleaned.splitlines():
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
            # Use advanced training with Albumentations
            result = advanced_train(
                cfg=cfg,
                use_albumentations=job["params"].get("augmentation_enabled", False),
                aug_mult=4,  # Default multiplier
                job=job,
                existing_mlflow_run=existing_mlflow_run
            )
        finally:
            # restore original streams in all cases
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            _append_log(job, "Stdout/err streams restored after training.")

        # After training, check if cancellation was requested during training
        if job.get("stop_requested"):
            _update_job(job_id, {
                "status": "cancelled",
                "finished_at": datetime.utcnow().isoformat(),
            })
            _append_log(job, "Training cancelled after execution.")
            _save_job_to_db(job, db)
            return

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

        metrics["train_images"] = prepared["n_train"]
        metrics["validation_images"] = prepared["n_val"]
        metrics["test_images"] = prepared["n_test"]
        metrics["images_trained"] = prepared["n_train"] + prepared["n_val"] + prepared["n_test"]
        metrics["total_images"] = prepared["n_train"] + prepared["n_val"] + prepared["n_test"]
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
                    from ultralytics import YOLO
                    registered_name = f"{dataset_name}_detector".replace("-", "_")
                    if best_pt.exists():
                        trained_model = YOLO(str(best_pt))
                        mlflow.pytorch.log_model(trained_model.model, artifact_path="model")
                    else:
                        _append_log(job, "WARNING: best.pt not found, skipping model registration.")
                        raise FileNotFoundError("best.pt not found for MLflow model registration")
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
        _update_job(job_id, {"status": "completed"})
        _append_log(job, "Training completed successfully.")

    except Exception as e:
        try:
            import mlflow  # type: ignore
            if mlflow.active_run():
                mlflow.end_run(status="FAILED")
        except Exception:
            pass
        job["error"] = f"{type(e).__name__}: {str(e)}"
        _update_job(job_id, {"status": "failed", "error": job["error"]})
        _append_log(job, f"Training failed: {job['error']}")
        _append_log(job, traceback.format_exc())
    finally:
        if job is not None:
            if not job.get("finished_at"):
                job["finished_at"] = datetime.utcnow().isoformat()
            _save_job_to_db(job, db)
        db.close()


@router.get("/detect-devices", response_model=DeviceDetectionResponse)
async def get_available_devices():
    """Detect and return available computing devices."""
    try:
        import torch  # type: ignore
        
        devices = []
        cuda_available = False
        device_count = 0
        recommended = "cpu"
        
        # Check CUDA availability
        if torch.cuda.is_available():
            cuda_available = True
            try:
                torch.cuda.init()
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        total_memory = props.total_memory / (1024**3)  # Convert to GB
                        devices.append(GPUDeviceInfo(
                            device_id=str(i),
                            device_name=props.name,
                            total_memory_gb=round(total_memory, 2),
                            available=True
                        ))
                    except Exception:
                        devices.append(GPUDeviceInfo(
                            device_id=str(i),
                            device_name=f"GPU {i}",
                            total_memory_gb=0.0,
                            available=False
                        ))
                
                # Recommend first available GPU
                if device_count > 0:
                    recommended = "0"
                    message = f"GPU detected: Using GPU 0 by default. {device_count} GPU(s) available."
                else:
                    message = "CUDA available but no GPU devices detected. Using CPU."
            except Exception as e:
                message = f"CUDA available but error detecting devices: {str(e)}. Using CPU."
        else:
            message = "No CUDA/GPU support detected. Training will use CPU."
            recommended = "cpu"
        
        return DeviceDetectionResponse(
            cuda_available=cuda_available,
            device_count=device_count,
            devices=devices,
            recommended_device=recommended,
            message=message
        )
    except Exception as e:
        return DeviceDetectionResponse(
            cuda_available=False,
            device_count=0,
            devices=[],
            recommended_device="cpu",
            message=f"Error detecting devices: {str(e)}. Using CPU for training."
        )


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
        "stop_requested": False,
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
    with _threads_lock:
        _training_threads[job_id] = t
        _thread_start_times[job_id] = time.time()

    return TrainingJobResponse(**job)


@router.get("/jobs", response_model=TrainingJobListResponse)
async def list_training_jobs(dataset_id: Optional[str] = None, db: Session = Depends(get_db)):
    # Clean up any threads that died silently
    _cleanup_dead_threads()

    # Get all jobs from database (most recent first)
    query = db.query(TrainingJob)
    if dataset_id:
        query = query.filter(TrainingJob.dataset_id == dataset_id)
    jobs_db = query.order_by(TrainingJob.created_at.desc()).all()
    
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
        job_dict["metrics"] = _compute_job_metrics_fallback(job_dict)
        jobs_response.append(TrainingJobResponse(**job_dict))
    
    return TrainingJobListResponse(jobs=jobs_response)


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str, db: Session = Depends(get_db)):
    # Clean up any threads that died silently
    _cleanup_dead_threads()

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
        job_dict["metrics"] = _compute_job_metrics_fallback(job_dict)
        return TrainingJobResponse(**job_dict)
    
    # Fallback to in-memory (for jobs just started)
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return TrainingJobResponse(**job)


@router.get("/jobs/{job_id}/download")
async def download_training_folder(job_id: str, db: Session = Depends(get_db)):
    # Verify job exists
    job_record = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job_record:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        artifacts = job.get("artifacts") or {}
    else:
        artifacts = job_record.artifacts or {}

    run_dir_str = artifacts.get("run_dir")
    if not run_dir_str:
        raise HTTPException(status_code=404, detail="Run directory not found. Job may not be completed.")

    run_dir = Path(run_dir_str)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run directory does not exist on disk.")

    from app.utils.file_utils import EXPORTS_DIR, create_zip_archive
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = EXPORTS_DIR / f"training_job_{job_id}.zip"

    if not zip_path.exists():
        # Zip the train directory
        try:
            create_zip_archive(run_dir.parent, run_dir.name, zip_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to zip training folder: {str(e)}")

    from fastapi.responses import FileResponse
    return FileResponse(
        path=zip_path,
        filename=f"training_{job_id}.zip",
        media_type="application/zip"
    )


@router.delete("/jobs/{job_id}", response_model=TrainingJobResponse)
async def delete_training_job(job_id: str, db: Session = Depends(get_db)):
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
        if job:
            if job["status"] not in {"completed", "failed", "cancelled"}:
                job["stop_requested"] = True
                job["status"] = "cancelled"
                job["finished_at"] = datetime.utcnow().isoformat()
                _append_log(job, "Job cancellation requested by user before delete. Deleting job.")
                _save_job_to_db(job, db)
                # Remove job from in‑memory dict (already popped)
                # Delete job files on disk
                job_dir = TRAINING_JOBS_DIR / job_id
                if job_dir.exists():
                    import shutil
                    shutil.rmtree(job_dir)
                # Delete persistent DB record if exists
                job_record = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                if job_record:
                    db.delete(job_record)
                    db.commit()
                return TrainingJobResponse(**job)
            return TrainingJobResponse(**job)

    job_record = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job_record:
        raise HTTPException(status_code=404, detail="Training job not found")

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
    db.delete(job_record)
    db.commit()
    return TrainingJobResponse(**job_dict)


@router.post("/jobs/{job_id}/stop", response_model=TrainingJobResponse)
async def stop_training_job(job_id: str, db: Session = Depends(get_db)):
    _cleanup_dead_threads()
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            if job["status"] not in {"completed", "failed", "cancelled"}:
                job["stop_requested"] = True
                job["status"] = "cancelled"
                job["finished_at"] = datetime.utcnow().isoformat()
                _append_log(job, "Training stopped by user.")
                _save_job_to_db(job, db)
            return TrainingJobResponse(**job)

    job_record = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job_record:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job_record.status not in {"completed", "failed", "cancelled"}:
        job_record.status = "cancelled"
        job_record.error = "Training job cancelled by user."
        job_record.finished_at = datetime.utcnow()
        db.commit()

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


class PredictionResponse(BaseModel): 
    predictions: List[Dict[str, Any]]
    inference_time_ms: Optional[float] = None


def find_latest_weights(dataset_id: str) -> Optional[Path]:
    """Find the most recent job directory for this dataset that contains a best.pt."""
    # first look for jobs that have explicit metadata for this dataset
    matched = []  # type: List[Path]
    all_weights: List[Path] = []
    if not TRAINING_JOBS_DIR.exists():
        return None
        
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
            with open(meta_path, "r", encoding="utf-8") as f:
                m = json.load(f)
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
    db: Session = Depends(get_db),
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

    weight_path = find_latest_weights(dataset_id)
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
            # Validate URL — block SSRF
            try:
                parsed = urlparse(image_url)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError("Invalid URL")
                if parsed.scheme not in ("http", "https"):
                    raise ValueError("Only http/https URLs are allowed")
                import ipaddress
                hostname = parsed.hostname or ""
                if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
                    raise ValueError("Internal URLs are not allowed")
                try:
                    ip = ipaddress.ip_address(hostname)
                    if ip.is_private or ip.is_loopback or ip.is_link_local:
                        raise ValueError("Private/internal IPs are not allowed")
                except ValueError as e:
                    if "not allowed" in str(e):
                        raise
                    pass  # hostname is a domain name, not an IP — OK
            except ValueError as e:
                if "not allowed" in str(e):
                    raise HTTPException(status_code=400, detail=f"URL rejected: {e}")
                raise HTTPException(status_code=400, detail="Invalid image_url")
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
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        custom_names = []
        
        if dataset and isinstance(dataset.analysis_summary, dict):
            custom_names = dataset.analysis_summary.get("class_names") or []

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
                class_name = None
                if 0 <= cls < len(custom_names):
                    class_name = custom_names[cls]
                else:
                    class_name = model.names.get(cls, f"class_{cls}")
                preds.append({
                    "x": x_center,
                    "y": y_center,
                    "width": width,
                    "height": height,
                    "confidence": conf,
                    "class_id": str(cls),
                    "class": class_name,
                    "class_name": class_name,
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
