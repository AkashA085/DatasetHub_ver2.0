import pytest
from unittest.mock import patch, MagicMock
from app.api.routes.training import _run_training, _jobs
from app.core.database import TrainingJob, Dataset
from datetime import datetime
from pathlib import Path

@pytest.fixture(autouse=True)
def clear_jobs():
    _jobs.clear()
    yield
    _jobs.clear()


def get_full_params():
    return {
        "model": "yolov8n.pt",
        "model_architecture": "YOLOv8",
        "pretrained_weights_used": True,
        "epochs": 1,
        "batch_size": 1,
        "image_size": 640,
        "learning_rate": 0.01,
        "optimizer": "auto",
        "device": "cpu",
        "val_split": 0.2,
        "test_split": 0.1,
        "seed": 42,
        "augmentation_enabled": False,
        "augmentation_pipeline_name": "none",
        "flip_enabled": False,
        "rotation_angle": 0.0,
        "brightness_range": "0.0-0.0",
        "noise_level": 0.0,
        "blur_enabled": False,
        "augmented_images_count": 0,
        "experiment_name": "test_exp",
        "register_best_model": False,
        "model_stage": "Staging"
    }

@patch("app.api.routes.training.YOLO")
@patch("app.api.routes.training.mlflow")
def test_run_training_mock(mock_mlflow, mock_yolo, db_session, sample_dataset):
    job_id = "test_job_id"
    job_params = get_full_params()
    
    # DB setup
    job_record = TrainingJob(id=job_id, dataset_id=sample_dataset.id, status="preparing", params=job_params)
    db_session.add(job_record)
    db_session.commit()
    
    # Global dict setup
    _jobs[job_id] = {
        "job_id": job_id,
        "dataset_id": sample_dataset.id,
        "status": "preparing",
        "params": job_params,
        "logs": [],
        "started_at": datetime.utcnow().isoformat()
    }
    
    # Mocks
    mock_model = MagicMock()
    mock_yolo.return_value = mock_model
    
    mock_results = MagicMock()
    mock_results.results_dict = {"metrics/mAP50(B)": 0.8}
    # Mock save_dir to return a Path-like object that supports / operator
    mock_save_dir = MagicMock(spec=Path)
    mock_results.save_dir = mock_save_dir
    mock_model.train.return_value = mock_results
    
    mock_job_dir = MagicMock(spec=Path)
    prepared = {
        "job_dir": mock_job_dir,
        "yaml_path": mock_job_dir / "data.yaml",
        "n_train": 1, "n_val": 1, "n_test": 0
    }
    
    with patch("app.api.routes.training._prepare_yolo_dataset", return_value=prepared), \
         patch("app.api.routes.training._plot_curves", return_value={}), \
         patch("app.api.routes.training._plot_class_distribution", return_value=True), \
         patch("app.api.routes.training._write_training_logs"), \
         patch("app.api.routes.training._save_job_to_db"), \
         patch("app.api.routes.training.Path.exists", return_value=False), \
         patch("app.api.routes.training.Path.mkdir"), \
         patch("app.core.database.SessionLocal", return_value=db_session):
            _run_training(job_id=job_id)
    
    assert _jobs[job_id]["status"] == "completed", f"Error: {_jobs[job_id].get('error')}"

@patch("app.api.routes.training.YOLO")
@patch("app.api.routes.training.mlflow")
def test_run_training_failure(mock_mlflow, mock_yolo, db_session, sample_dataset):
    job_id = "fail_job_id"
    job_params = get_full_params()
    
    # DB setup
    job_record = TrainingJob(id=job_id, dataset_id=sample_dataset.id, status="preparing", params=job_params)
    db_session.add(job_record)
    db_session.commit()
    
    # Global dict setup
    _jobs[job_id] = {
        "job_id": job_id,
        "dataset_id": sample_dataset.id,
        "status": "preparing",
        "params": job_params,
        "logs": [],
        "started_at": datetime.utcnow().isoformat()
    }
    
    # Force failure
    mock_yolo.side_effect = Exception("Boom!")
    
    prepared = {"job_dir": Path("/tmp"), "yaml_path": Path("/tmp/data.yaml"), "n_train": 1, "n_val": 1, "n_test": 0}
    
    with patch("app.api.routes.training._prepare_yolo_dataset", return_value=prepared):
        with patch("app.core.database.SessionLocal", return_value=db_session):
            _run_training(job_id=job_id)
    
    assert _jobs[job_id]["status"] == "failed"
    assert "Boom!" in _jobs[job_id]["error"]

@patch("app.api.routes.training.YOLO")
@patch("ultralytics.YOLO")
@patch("app.api.routes.training.mlflow")
def test_run_training_registration_success(mock_mlflow, mock_ultralytics_yolo, mock_yolo, db_session, sample_dataset, tmp_path):
    job_id = "reg_job_id"
    job_params = get_full_params()
    job_params["register_best_model"] = True
    
    # DB setup
    job_record = TrainingJob(id=job_id, dataset_id=sample_dataset.id, status="preparing", params=job_params)
    db_session.add(job_record)
    db_session.commit()
    
    # Global dict setup
    _jobs[job_id] = {
        "job_id": job_id,
        "dataset_id": sample_dataset.id,
        "status": "preparing",
        "params": job_params,
        "logs": [],
        "started_at": datetime.utcnow().isoformat(),
        "mlflow": {"run_id": "fake_run_id"}
    }
    
    # Mocks
    import torch
    mock_model = MagicMock()
    mock_model.model = MagicMock(spec=torch.nn.Module)
    mock_yolo.return_value = mock_model
    mock_ultralytics_yolo.return_value = mock_model
    mock_results = MagicMock()
    mock_results.results_dict = {"metrics/mAP50(B)": 0.8}
    
    # Real temp dir for save_dir to handle filesystem ops
    save_dir = Path(tmp_path) / "runs" / "train" / "exp"
    save_dir.mkdir(parents=True, exist_ok=True)
    mock_results.save_dir = save_dir
    mock_model.train.return_value = mock_results
    
    # Create dummy results.csv
    (save_dir / "results.csv").write_text("epoch,metrics/mAP50(B)\n0,0.8\n")
    # And weights/best.pt
    (save_dir / "weights").mkdir(exist_ok=True)
    (save_dir / "weights" / "best.pt").touch()
    
    # Mock registration
    mock_version = MagicMock()
    mock_version.version = "1"
    mock_mlflow.register_model.return_value = mock_version
    mock_mlflow.active_run.return_value = MagicMock(info=MagicMock(run_id="fake_run_id"))
    
    prepared = {"job_dir": Path("/tmp"), "yaml_path": Path("/tmp/data.yaml"), "n_train": 1, "n_val": 1, "n_test": 0}
    
    with patch("app.api.routes.training._prepare_yolo_dataset", return_value=prepared), \
         patch("app.api.routes.training._plot_curves", return_value={}), \
         patch("app.api.routes.training._plot_class_distribution", return_value=True), \
         patch("app.api.routes.training._write_training_logs"), \
         patch("app.api.routes.training._save_job_to_db"), \
         patch("app.api.routes.training.Path.exists", return_value=True), \
         patch("app.api.routes.training.Path.mkdir"), \
         patch("app.api.routes.training.shutil.copy2"), \
         patch("app.api.routes.training.mlflow") as mock_mlflow_mod, \
         patch.dict("sys.modules", {"mlflow": mock_mlflow_mod, "mlflow.pytorch": MagicMock(), "mlflow.tracking": MagicMock()}), \
         patch("app.core.database.SessionLocal", return_value=db_session):



            
            # Deep mock for mlflow.pytorch.log_model
            mock_mlflow_mod.pytorch = MagicMock()
            mock_mlflow_mod.active_run.return_value = MagicMock(info=MagicMock(run_id="fake_run_id"))
            mock_version = MagicMock()
            mock_version.version = "1"
            mock_mlflow_mod.register_model.return_value = mock_version
            
            _run_training(job_id=job_id)
    
    print(f"DEBUG LOGS: {_jobs[job_id]['logs']}")
    assert _jobs[job_id]["status"] == "completed", f"Job failed with error: {_jobs[job_id].get('error')}"
    assert any("Model registered" in log for log in _jobs[job_id]["logs"])




