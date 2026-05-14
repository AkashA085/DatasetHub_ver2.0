import pytest
from unittest.mock import patch, MagicMock
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np
from app.api.routes.training import _jobs

@pytest.fixture
def mock_run_training():
    with patch("app.api.routes.training._run_training"):
        yield

@pytest.fixture(autouse=True)
def clear_jobs():
    _jobs.clear()
    yield
    _jobs.clear()

def test_list_training_jobs_empty(client):
    response = client.get("/train/jobs")
    assert response.status_code == 200
    assert "jobs" in response.json()
    assert len(response.json()["jobs"]) == 0

def test_start_training_no_dataset(client):
    payload = {
        "dataset_id": str(uuid.uuid4()),
        "model": "yolov8n.pt",
        "epochs": 1,
        "batch_size": 1,
        "image_size": 640,
        "val_split": 0.2,
        "test_split": 0.1,
        "augmentation_enabled": False
    }
    response = client.post("/train/start", json=payload)
    assert response.status_code == 404

def test_start_training_success(client, sample_dataset, mock_run_training):
    payload = {
        "dataset_id": sample_dataset.id,
        "model": "yolov8n.pt",
        "epochs": 1,
        "batch_size": 1,
        "image_size": 640,
        "val_split": 0.2,
        "test_split": 0.1,
        "augmentation_enabled": False
    }
    response = client.post("/train/start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert data["dataset_id"] == sample_dataset.id
    
    # Verify job is in DB
    job_id = data["job_id"]
    response = client.get(f"/train/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id

def test_device_detection(client):
    response = client.get("/train/detect-devices")
    assert response.status_code == 200
    data = response.json()
    assert "cuda_available" in data
    assert "recommended_device" in data

def test_training_job_not_found(client):
    response = client.get("/train/jobs/non-existent-id")
    assert response.status_code == 404

def test_stop_training_not_found(client):
    response = client.post("/train/jobs/non-existent-id/stop")
    assert response.status_code == 404

def test_stop_training_success_v1(client, sample_dataset, mock_run_training):
    # Start a job first
    payload = {
        "dataset_id": sample_dataset.id,
        "model": "yolov8n.pt",
        "epochs": 1,
        "batch_size": 1,
        "image_size": 640,
        "val_split": 0.2,
        "test_split": 0.1,
        "augmentation_enabled": False
    }
    resp = client.post("/train/start", json=payload)
    job_id = resp.json()["job_id"]
    
    # Stop it
    response = client.post(f"/train/jobs/{job_id}/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"

def test_list_training_jobs(client, sample_dataset, mock_run_training):
    # Start a job
    payload = {
        "dataset_id": sample_dataset.id,
        "model": "yolov8n.pt",
        "epochs": 1,
        "batch_size": 1,
        "image_size": 640,
        "val_split": 0.2,
        "test_split": 0.1,
        "augmentation_enabled": False
    }
    client.post("/train/start", json=payload)
    
    # List all
    response = client.get("/train/jobs")
    assert response.status_code == 200
    assert len(response.json()["jobs"]) >= 1
    
    # List by dataset
    response = client.get(f"/train/jobs?dataset_id={sample_dataset.id}")
    assert response.status_code == 200
    assert all(j["dataset_id"] == sample_dataset.id for j in response.json()["jobs"])

def test_get_job_details_success(client, sample_dataset, mock_run_training):
    payload = {
        "dataset_id": sample_dataset.id,
        "model": "yolov8n.pt",
        "epochs": 1,
        "batch_size": 1,
        "image_size": 640,
        "val_split": 0.2,
        "test_split": 0.1,
        "augmentation_enabled": False
    }
    resp = client.post("/train/start", json=payload)
    job_id = resp.json()["job_id"]
    
    response = client.get(f"/train/jobs/{job_id}")
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id

def test_predict_image_no_data(client):
    response = client.post("/train/predict", data={"dataset_id": "some_id"})
    assert response.status_code == 400

def test_predict_image_both_data(client):
    response = client.post("/train/predict", data={
        "dataset_id": "some_id",
        "image_url": "http://example.com/img.jpg"
    }, files={"image_file": ("img.jpg", b"fake image content")})
    assert response.status_code == 400

def test_predict_image_url_success(client, sample_dataset):
    mock_model = MagicMock()
    mock_yolo = MagicMock(return_value=mock_model)
    mock_model.predict.return_value = [] # Mock results
    
    # We need a job with weights to find
    with patch("app.api.routes.training.find_latest_weights", return_value=Path("/tmp/best.pt")), \
         patch("app.api.routes.training.cv2.imread", return_value=np.zeros((100,100,3), dtype=np.uint8)), \
         patch("PIL.Image.open") as mock_pil, \
         patch("app.api.routes.training.requests.get") as mock_get, \
         patch.dict("sys.modules", {"ultralytics": MagicMock(), "ultralytics.YOLO": mock_yolo}):
        
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"fake image content"
        
        # Mock PIL image size
        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_pil.return_value = mock_img
        
        # Mock model names
        mock_model.names = {0: "person"}
        
        response = client.post("/train/predict", data={
            "dataset_id": sample_dataset.id,
            "image_url": "http://example.com/img.jpg"
        })
        assert response.status_code == 200



def test_find_latest_weights(tmp_path):
    from app.api.routes.training import find_latest_weights, TRAINING_JOBS_DIR
    import json
    
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path):
        # Empty
        assert find_latest_weights("ds1") is None
        
        # With job but no best.pt
        job1 = tmp_path / "job1"
        job1.mkdir()
        assert find_latest_weights("ds1") is None
        
        # With best.pt and metadata
        weights = job1 / "runs" / "train" / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").touch()
        (job1 / "job_meta.json").write_text(json.dumps({"dataset_id": "ds1"}))
        
        found = find_latest_weights("ds1")
        assert found == weights / "best.pt"

def test_get_job_logs_via_details(client, sample_dataset, mock_run_training):
    payload = {
        "dataset_id": sample_dataset.id,
        "model": "yolov8n.pt",
        "epochs": 1,
        "batch_size": 1,
        "image_size": 640,
        "val_split": 0.2,
        "test_split": 0.1,
        "augmentation_enabled": False
    }
    resp = client.post("/train/start", json=payload)
    job_id = resp.json()["job_id"]
    
    # Add a log manually to _jobs
    _jobs[job_id]["logs"].append("test log")
    
    # Force the API to use _jobs by ensuring it's not in DB or by mocking DB
    from app.core.database import TrainingJob
    with patch("app.api.routes.training.Session.query") as mock_query:
        mock_query.return_value.filter.return_value.first.return_value = None
        response = client.get(f"/train/jobs/{job_id}")
    
    assert response.status_code == 200
    assert any("test log" in log for log in response.json()["logs"])

def test_delete_training_job_success(client, sample_dataset, mock_run_training):
    payload = {
        "dataset_id": sample_dataset.id,
        "model": "yolov8n.pt",
        "epochs": 1,
        "batch_size": 1,
        "image_size": 640,
        "val_split": 0.2,
        "test_split": 0.1,
        "augmentation_enabled": False
    }
    resp = client.post("/train/start", json=payload)
    job_id = resp.json()["job_id"]
    
    response = client.delete(f"/train/jobs/{job_id}")
    assert response.status_code == 200
    # response is TrainingJobResponse, status will be 'queued' or 'cancelling'
    assert response.json()["job_id"] == job_id
    assert job_id not in _jobs

def test_stop_training_job_success(client, sample_dataset, mock_run_training):
    payload = {
        "dataset_id": sample_dataset.id,
        "model": "yolov8n.pt",
        "epochs": 1,
        "batch_size": 1,
        "image_size": 640,
        "val_split": 0.2,
        "test_split": 0.1,
        "augmentation_enabled": False
    }
    resp = client.post("/train/start", json=payload)
    job_id = resp.json()["job_id"]
    
    response = client.post(f"/train/jobs/{job_id}/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"
    assert _jobs[job_id]["stop_requested"] is True

def test_download_job_not_found(client):
    response = client.get("/train/jobs/non-existent/download")
    assert response.status_code == 404

def test_stop_job_not_found(client):
    response = client.post("/train/jobs/non-existent/stop")
    assert response.status_code == 404

def test_download_job_success(client, sample_dataset, tmp_path, db_session):
    # Setup job in DB with artifacts
    from app.core.database import TrainingJob
    job_id = f"job_dl_{uuid.uuid4()}"
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "best.pt").touch()
    
    job_rec = TrainingJob(
        id=job_id,
        dataset_id=sample_dataset.id,
        status="completed",
        artifacts={"run_dir": str(run_dir)}
    )
    db_session.add(job_rec)
    db_session.commit()
    
    with patch("app.utils.file_utils.create_zip_archive"), \
         patch("app.utils.file_utils.EXPORTS_DIR", tmp_path / "exports"):
        # Create dummy zip
        (tmp_path / "exports").mkdir()
        (tmp_path / "exports" / f"training_job_{job_id}.zip").touch()
        
        response = client.get(f"/train/jobs/{job_id}/download")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

def test_run_training_direct(sample_dataset, tmp_path):
    from app.api.routes.training import _run_training, _jobs, TrainingStartRequest
    from app.core.database import SessionLocal
    
    job_id = f"job_direct_{uuid.uuid4()}"
    params = TrainingStartRequest(
        dataset_id=sample_dataset.id,
        model="yolov8n.pt",
        epochs=1,
        batch_size=1,
        image_size=640,
        val_split=0.2,
        test_split=0.1,
        augmentation_enabled=False
    ).model_dump()
    
    _jobs[job_id] = {
        "job_id": job_id,
        "dataset_id": sample_dataset.id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "params": params,
        "logs": [],
        "stop_requested": False
    }
    
    # Mocking everywhere to avoid real training
    with patch("app.api.routes.training._prepare_yolo_dataset") as mock_prep, \
         patch("app.api.routes.training.YOLO") as mock_yolo, \
         patch("app.api.routes.training.mlflow"):
        
        mock_prep.return_value = {
            "job_dir": tmp_path / "job_dir",
            "yaml_path": tmp_path / "data.yaml",
            "n_train": 1,
            "n_val": 0,
            "n_test": 0
        }
        
        _run_training(job_id)
        
    status = _jobs[job_id]["status"]
    logs = "\n".join(_jobs[job_id]["logs"])
    assert status == "completed", f"Job failed with status {status}. Logs: {logs}"

def test_predict_image_missing_params(client):
    # dataset_id is required
    response = client.post("/train/predict", data={"dataset_id": ""})
    assert response.status_code == 422

def test_predict_image_no_source(client):
    response = client.post("/train/predict", data={"dataset_id": "ds1"})
    assert response.status_code == 400

def test_prepare_yolo_dataset_no_images(sample_dataset, tmp_path, db_session):
    from app.api.routes.training import _prepare_yolo_dataset
    job = {"logs": []}
    # sample_dataset has 0 real images on disk in this test env
    with pytest.raises(Exception) as exc:
        _prepare_yolo_dataset(sample_dataset.id, tmp_path / "job", 0.2, 0.1, db_session, job)
    assert "No labeled images found" in str(exc.value)





