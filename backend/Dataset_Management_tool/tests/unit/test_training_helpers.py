import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from app.api.routes.training import _resolve_device, _compute_job_metrics_fallback, _append_log


def test_resolve_device():
    import torch
    cuda_available = torch.cuda.is_available()
    expected_gpu = "0" if cuda_available else "cpu"
    
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == expected_gpu
    assert _resolve_device("auto") == expected_gpu

def test_compute_job_metrics_fallback():
    now = datetime.utcnow()
    # Function expects ISO format strings
    job_dict = {
        "job_id": "test",
        "metrics": None,
        "started_at": (now - timedelta(minutes=1)).isoformat(),
        "finished_at": now.isoformat()
    }
    metrics = _compute_job_metrics_fallback(job_dict)
    assert metrics["total_training_time"] >= 59 # Allow for tiny rounding

def test_append_log():
    job = {"logs": []}
    _append_log(job, "Test message")
    assert len(job["logs"]) == 1
    assert "Test message" in job["logs"][0]

from app.api.routes.training import build_train_transform, build_val_transform, write_augmented_split, _prepare_yolo_dataset
import albumentations as A

def test_build_transforms():
    train_tf = build_train_transform()
    val_tf = build_val_transform()
    assert isinstance(train_tf, A.Compose)
    assert isinstance(val_tf, A.Compose)

@patch("app.api.routes.training.cv2")
def test_write_augmented_split(mock_cv2, tmp_path):
    # Use real paths for globbing
    src_img = tmp_path / "src_img"
    src_lbl = tmp_path / "src_lbl"
    src_img.mkdir()
    src_lbl.mkdir()
    
    (src_img / "img1.jpg").touch()
    (src_lbl / "img1.txt").write_text("0 0.5 0.5 0.1 0.1")
    
    # Mock cv2 constants for Pydantic validation in Albumentations
    mock_cv2.BORDER_CONSTANT = 0
    mock_cv2.COLOR_BGR2RGB = 4
    mock_cv2.COLOR_RGB2BGR = 4
    mock_cv2.IMWRITE_JPEG_QUALITY = 1
    
    # Mock cv2.imread
    mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    
    dst_img = tmp_path / "dst_img"
    dst_lbl = tmp_path / "dst_lbl"
    
    count = write_augmented_split(
        src_img_dir=str(src_img),
        src_lbl_dir=str(src_lbl),
        dst_img_dir=str(dst_img),
        dst_lbl_dir=str(dst_lbl),
        mode="train",
        multiplier=1
    )
    
    # multiplier=1 means 1 original + 1 augmented = 2 versions per image
    assert count == 2

from app.api.routes.training import _plot_curves, _plot_class_distribution

@patch("matplotlib.pyplot")
def test_plot_curves(mock_plt, tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("epoch,metrics/mAP50(B)\n0,0.8\n1,0.9\n")
    p1 = tmp_path / "p1.png"
    p2 = tmp_path / "p2.png"
    
    # Patch sys.modules to handle the local import in training.py
    with patch.dict("sys.modules", {"matplotlib.pyplot": mock_plt}):
        plots = _plot_curves(csv_path, p1, p2)
    
    assert "accuracy_curve.png" in plots
    assert mock_plt.savefig.called



@patch("matplotlib.pyplot")
def test_plot_class_distribution(mock_plt, tmp_path):
    dist = {"drone": 10}
    save_path = tmp_path / "dist.png"
    
    # Patch sys.modules to handle the local import in training.py
    with patch.dict("sys.modules", {"matplotlib.pyplot": mock_plt}):
        _plot_class_distribution(dist, str(save_path))
    
    assert mock_plt.savefig.called


def test_prepare_yolo_dataset_success(db_session, tmp_path):
    from app.core.database import Dataset, Image, User, Project
    from app.api.routes.training import _prepare_yolo_dataset, TRAINING_JOBS_DIR
    import os
    
    # Setup directories
    ds_root = tmp_path / "ds"
    img_dir = ds_root / "images"
    lbl_dir = ds_root / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    
    # Create User and Project
    user = User(id="user1", email="test@example.com")
    db_session.add(user)
    db_session.flush()
    project = Project(id="proj1", name="test project", user_id="user1")
    db_session.add(project)
    db_session.flush()

    # Create dataset in DB
    ds = Dataset(id="test_ds", format_type="yolo", project_id="proj1")
    db_session.add(ds)
    db_session.commit()
    
    # Create 10 images and labels
    for i in range(10):
        img_path = img_dir / f"img{i}.jpg"
        img_path.touch()
        lbl_path = lbl_dir / f"img{i}.txt"
        lbl_path.write_text("0 0.5 0.5 0.1 0.1")
        
        img_rec = Image(
            id=f"img_id_{i}",
            dataset_id=ds.id,
            file_path=str(img_path),
            file_name=f"img{i}.jpg",
            has_label=True
        )
        db_session.add(img_rec)
    db_session.commit()
    
    job = {"job_id": "test_job_123", "logs": []}
    
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path / "jobs"):
        prepared = _prepare_yolo_dataset(
            dataset_id=ds.id,
            seed=42,
            val_split=0.2,
            test_split=0.1,
            db=db_session,
            job=job
        )
    
    assert prepared["n_train"] == 7
    assert prepared["n_val"] == 2
    assert prepared["n_test"] == 1
    assert "job_dir" in prepared
    assert Path(prepared["job_dir"]).exists()
    assert (Path(prepared["job_dir"]) / "data.yaml").exists()

def test_prepare_yolo_dataset_empty(db_session):
    job = {"logs": []}
    with pytest.raises(ValueError, match="No labeled images found"):
        _prepare_yolo_dataset(
            dataset_id="empty_ds",
            seed=42,
            val_split=0.2,
            test_split=0.1,
            db=db_session,
            job=job
        )


def test_build_augmented_dataset(tmp_path):
    from app.api.routes.training import build_augmented_dataset
    
    # Setup source and dest
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "train").mkdir()
    (src / "train" / "images").mkdir()
    (src / "train" / "labels").mkdir()
    
    # Create one image/label
    (src / "train" / "images" / "img1.jpg").touch()
    (src / "train" / "labels" / "img1.txt").write_text("0 0.5 0.5 0.1 0.1")
    
    split_info = {
        "job_dir": str(src),
        "n_train": 1,
        "n_val": 0,
        "n_test": 0
    }
    
    # Mock cv2 and albumentations
    with patch("app.api.routes.training.cv2.imread", return_value=np.zeros((100,100,3), dtype=np.uint8)), \
         patch("app.api.routes.training.cv2.imwrite"), \
         patch("app.api.routes.training.build_train_transform") as mock_build:
        
        mock_transform = MagicMock(return_value={"image": np.zeros((100,100,3), dtype=np.uint8), 
                                                "bboxes": [[0.5, 0.5, 0.1, 0.1]], 
                                                "class_labels": [0]})
        mock_build.return_value = mock_transform
        
        cfg = {"dataset_root": str(src), "data_yaml": str(src / "data.yaml")}
        (src / "data.yaml").write_text("path: .\nnames: [test]")
        result = build_augmented_dataset(
            cfg=cfg,
            multiplier=2
        )
        
    assert "drone_data.yaml" in result

def test_append_log():

    from app.api.routes.training import _append_log
    job = {"logs": []}
    _append_log(job, "test message")
    assert len(job["logs"]) == 1
    assert "test message" in job["logs"][0]

def test_prepare_yolo_dataset_error_paths(db_session, tmp_path):
    from app.core.database import Dataset, Image, User, Project
    from app.api.routes.training import _prepare_yolo_dataset, TRAINING_JOBS_DIR
    
    # Create setup
    user = User(id="user2", email="u2@ex.com")
    db_session.add(user)
    project = Project(id="p2", name="p2", user_id="user2")
    db_session.add(project)
    ds = Dataset(id="ds_err", format_type="yolo", project_id="p2")
    db_session.add(ds)
    
    # One valid image but missing label
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "img1.jpg").touch()
    
    img_rec = Image(id="i1", dataset_id="ds_err", file_path=str(img_dir / "img1.jpg"), 
                    file_name="img1.jpg", has_label=True)
    db_session.add(img_rec)
    db_session.commit()
    
    job = {"job_id": "j_err", "logs": []}
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path / "jobs"):
        # Should raise ValueError because no VALID pairs found
        with pytest.raises(ValueError, match="No valid image/label pairs found"):
            _prepare_yolo_dataset("ds_err", 42, 0.2, 0.1, db_session, job)

def test_drone_mlflow_callback():
    from app.api.routes.training import DroneMLflowCallback
    mock_run = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.epoch = 1
    mock_trainer.label_loss_items.return_value = {"box": 0.1}
    mock_trainer.metrics = {"mAP50": 0.8}
    mock_trainer.scheduler.get_last_lr.return_value = [0.001]
    mock_trainer.save_dir = "/tmp/save"
    
    with patch("app.api.routes.training.mlflow.log_metric") as mock_log, \
         patch("app.api.routes.training.mlflow.log_artifact"), \
         patch("app.api.routes.training.Path.exists", return_value=True), \
         patch("app.api.routes.training.Path.glob", return_value=[]):
        
        cb = DroneMLflowCallback(mock_run, {})
        cb.on_train_epoch_end(mock_trainer)
        assert mock_log.called
        
        cb.on_train_end(mock_trainer)

def test_resolve_device_explicit():
    from app.api.routes.training import _resolve_device
    # Requested cpu
    assert _resolve_device("cpu") == "cpu"
    # Requested invalid
    assert _resolve_device("invalid") == "cpu"

def test_get_recommended_batch_size():
    from app.api.routes.training import _get_optimal_batch_size
    
    # CPU
    assert _get_optimal_batch_size("cpu", 640) == 8
    
    # GPU mock
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties") as mock_props:
        
        mock_props.return_value.total_memory = 24 * (1024**3) # 24GB
        
        assert _get_optimal_batch_size("cuda", 1280) == 64
        assert _get_optimal_batch_size("cuda", 640) == 128
        assert _get_optimal_batch_size("cuda", 320) == 256
        
        mock_props.return_value.total_memory = 4 * (1024**3) # 4GB
        assert _get_optimal_batch_size("cuda", 640) == 16

def test_drone_mlflow_callback_methods():
    from app.api.routes.training import DroneMLflowCallback
    mock_run = MagicMock()
    callback = DroneMLflowCallback(mock_run, cfg={})
    
    mock_trainer = MagicMock()
    mock_trainer.epoch = 1
    mock_trainer.loss_items = [0.1, 0.2, 0.3]
    mock_trainer.label_loss_items.return_value = {"box_loss": 0.1, "cls_loss": 0.2}
    mock_trainer.metrics = {"metrics/mAP50(B)": 0.5}
    mock_trainer.tloss = [0.1, 0.2, 0.3]
    mock_trainer.scheduler.get_last_lr.return_value = [0.001]
    
    with patch("app.api.routes.training.mlflow") as mock_mlflow:
        callback.on_train_epoch_end(mock_trainer)
        assert mock_mlflow.log_metric.called
        
        mock_trainer.save_dir = "/tmp/fake_save_dir"
        callback.on_train_end(mock_trainer)
        assert mock_mlflow.log_artifact.called or mock_mlflow.log_metric.called

