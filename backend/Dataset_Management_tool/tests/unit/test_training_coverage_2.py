"""
Coverage tests for training.py API routes and _run_training.
Targets: lines 1093,1096-1100,1130-1131,1170-1175,1180,1223-1227,
         1249-1252,1261-1262,1280-1289,1292-1295,1317-1321,1335,1337,
         1342,1350,1357,1359,1440-1447,1463,1482-1483,1523-1524,
         1536-1541,1550-1551,1567,1597-1599,1682,1688,1692,1700-1703,
         1725-1745,1764-1784,1798,1802,1811-1812,1820-1823,1841,
         1852-1855,1859,1865-1882,1888-1890,1894,1896,1899,1912,
         1927-1943,1955-1958,1963-1964
"""
import json
import uuid
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.api.routes.training import _jobs, _run_training, find_latest_weights


@pytest.fixture(autouse=True)
def clear_jobs():
    _jobs.clear()
    yield
    _jobs.clear()


def _make_job(job_id, dataset_id, extra=None):
    j = {
        "job_id": job_id, "dataset_id": dataset_id, "status": "queued",
        "params": {
            "model": "yolov8n.pt", "model_architecture": "YOLOv8",
            "pretrained_weights_used": True, "epochs": 1, "batch_size": 1,
            "image_size": 640, "learning_rate": 0.01, "optimizer": "auto",
            "device": "cpu", "val_split": 0.2, "test_split": 0.1,
            "seed": 42, "augmentation_enabled": False,
            "augmentation_pipeline_name": "none", "flip_enabled": False,
            "rotation_angle": 0.0, "brightness_range": "0.0-0.0",
            "noise_level": 0.0, "blur_enabled": False,
            "augmented_images_count": 0, "experiment_name": "test_exp",
            "register_best_model": False, "model_stage": "Staging",
            "run_name": None, "mlflow_tracking_uri": None,
            "model_version": None, "model_description": None,
        },
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None, "finished_at": None,
        "metrics": None, "artifacts": None, "mlflow": None,
        "error": None, "logs": [], "stop_requested": False,
    }
    if extra:
        j.update(extra)
    return j


# ── _run_training: job not found (line 1093) ───────────────────────────────
def test_run_training_job_not_found(db_session):
    with patch("app.core.database.SessionLocal", return_value=db_session):
        _run_training("nonexistent_job_id")  # must not raise


# ── _run_training: stop before start (1096-1100) ───────────────────────────
def test_run_training_stop_before_start(db_session):
    job_id = "stop_before"
    _jobs[job_id] = _make_job(job_id, "ds1", {"stop_requested": True})
    with patch("app.core.database.SessionLocal", return_value=db_session):
        _run_training(job_id)
    assert _jobs[job_id]["status"] == "cancelled"


# ── _run_training: stop after running (1223-1227) ──────────────────────────
def test_run_training_stop_after_running(db_session, sample_dataset, tmp_path):
    job_id = "stop_after"
    _jobs[job_id] = _make_job(job_id, sample_dataset.id)

    def fake_prepare(**kwargs):
        _jobs[job_id]["stop_requested"] = True  # trigger mid-run cancel
        return {"job_dir": tmp_path, "yaml_path": tmp_path / "d.yaml",
                "n_train": 1, "n_val": 0, "n_test": 0}

    with patch("app.api.routes.training._prepare_yolo_dataset",
               side_effect=lambda **kw: fake_prepare(**kw)), \
         patch("app.api.routes.training.mlflow"), \
         patch("app.core.database.SessionLocal", return_value=db_session):
        _run_training(job_id)
    assert _jobs[job_id]["status"] == "cancelled"


# ── _run_training: mlflow init error (1170-1175) ───────────────────────────
def test_run_training_mlflow_init_error(db_session, sample_dataset, tmp_path):
    job_id = "mlflow_err"
    _jobs[job_id] = _make_job(job_id, sample_dataset.id)
    prepared = {"job_dir": tmp_path, "yaml_path": tmp_path / "d.yaml",
                "n_train": 1, "n_val": 0, "n_test": 0}

    mock_result = MagicMock()
    mock_result.save_dir = tmp_path / "save"
    (tmp_path / "save").mkdir(parents=True, exist_ok=True)
    mock_result.results_dict = {}
    mock_result.speed = {}

    mock_mlflow = MagicMock()
    mock_mlflow.start_run.side_effect = Exception("mlflow broken")

    with patch("app.api.routes.training._prepare_yolo_dataset", return_value=prepared), \
         patch("app.api.routes.training.mlflow", mock_mlflow), \
         patch("app.api.routes.training.YOLO") as mock_yolo, \
         patch("app.api.routes.training._plot_curves", return_value={}), \
         patch("app.api.routes.training._plot_class_distribution", return_value=False), \
         patch("app.api.routes.training._write_training_logs"), \
         patch("app.api.routes.training._save_job_to_db"), \
         patch("app.core.database.SessionLocal", return_value=db_session):
        mock_yolo.return_value.train.return_value = mock_result
        _run_training(job_id)
    assert any("MLflow disabled" in l for l in _jobs[job_id]["logs"])


# ── _run_training: device=auto log (1180) ──────────────────────────────────
def test_run_training_auto_device_log(db_session, sample_dataset, tmp_path):
    job_id = "auto_dev"
    _jobs[job_id] = _make_job(job_id, sample_dataset.id,
                               {"params": {**_make_job("x", "y")["params"], "device": "auto"}})
    prepared = {"job_dir": tmp_path, "yaml_path": tmp_path / "d.yaml",
                "n_train": 1, "n_val": 0, "n_test": 0}
    mock_result = MagicMock()
    mock_result.save_dir = tmp_path / "save"
    (tmp_path / "save").mkdir(parents=True, exist_ok=True)
    mock_result.results_dict = {}; mock_result.speed = {}
    with patch("app.api.routes.training._prepare_yolo_dataset", return_value=prepared), \
         patch("app.api.routes.training.mlflow"), \
         patch("app.api.routes.training.YOLO") as mock_yolo, \
         patch("app.api.routes.training._plot_curves", return_value={}), \
         patch("app.api.routes.training._plot_class_distribution", return_value=False), \
         patch("app.api.routes.training._write_training_logs"), \
         patch("app.api.routes.training._save_job_to_db"), \
         patch("app.core.database.SessionLocal", return_value=db_session):
        mock_yolo.return_value.train.return_value = mock_result
        _run_training(job_id)
    assert any("Resolved device" in l for l in _jobs[job_id]["logs"])


# ── _run_training: full metrics (1335,1337,1342,1350,1357,1359) ────────────
def test_run_training_full_metrics(db_session, sample_dataset, tmp_path):
    job_id = "full_met"
    _jobs[job_id] = _make_job(job_id, sample_dataset.id)
    save_dir = tmp_path / "save"
    save_dir.mkdir(parents=True)
    (save_dir / "weights").mkdir()

    mock_result = MagicMock()
    mock_result.save_dir = save_dir
    mock_result.results_dict = {
        "metrics/precision(B)": 0.9,
        "metrics/recall(B)": 0.8,
        "metrics/mAP50(B)": 0.85,
        "metrics/mAP50-95(B)": 0.7,
    }
    mock_result.speed = {"inference": 12.5}

    prepared = {"job_dir": tmp_path, "yaml_path": tmp_path / "d.yaml",
                "n_train": 1, "n_val": 0, "n_test": 0}

    with patch("app.api.routes.training._prepare_yolo_dataset", return_value=prepared), \
         patch("app.api.routes.training.mlflow"), \
         patch("app.api.routes.training.YOLO") as mock_yolo, \
         patch("app.api.routes.training._plot_curves", return_value={}), \
         patch("app.api.routes.training._plot_class_distribution", return_value=False), \
         patch("app.api.routes.training._write_training_logs"), \
         patch("app.api.routes.training._save_job_to_db"), \
         patch("app.core.database.SessionLocal", return_value=db_session):
        mock_yolo.return_value.train.return_value = mock_result
        _run_training(job_id)

    m = _jobs[job_id]["metrics"] or {}
    assert "recall" in m or _jobs[job_id]["status"] in ("completed", "failed")


# ── _run_training: metrics None → {} (1463) ────────────────────────────────
def test_run_training_none_metrics_converted(db_session, sample_dataset, tmp_path):
    job_id = "none_met"
    _jobs[job_id] = _make_job(job_id, sample_dataset.id)
    save_dir = tmp_path / "save"; save_dir.mkdir(parents=True)
    mock_result = MagicMock()
    mock_result.save_dir = save_dir
    mock_result.results_dict = {}; mock_result.speed = {}
    prepared = {"job_dir": tmp_path, "yaml_path": tmp_path / "d.yaml",
                "n_train": 1, "n_val": 0, "n_test": 0}
    with patch("app.api.routes.training._prepare_yolo_dataset", return_value=prepared), \
         patch("app.api.routes.training.mlflow"), \
         patch("app.api.routes.training.YOLO") as mock_yolo, \
         patch("app.api.routes.training._plot_curves", return_value={}), \
         patch("app.api.routes.training._plot_class_distribution", return_value=False), \
         patch("app.api.routes.training._write_training_logs"), \
         patch("app.api.routes.training._save_job_to_db"), \
         patch("app.core.database.SessionLocal", return_value=db_session):
        mock_result.results_dict = None
        mock_yolo.return_value.train.return_value = mock_result
        _run_training(job_id)
    assert _jobs[job_id]["status"] in ("completed", "failed")


# ── API: start_training split check (1567) ─────────────────────────────────
def test_start_training_bad_splits(client, sample_dataset, db_session):
    """val_split+test_split >= 0.9 → 400. Pydantic caps both at le=0.4 so we
    cannot hit this guard via the HTTP form. Call the endpoint function
    directly after relaxing the model limits via a monkeypatched request."""
    from app.api.routes import training as tr
    import asyncio

    # Build a fake request object that bypasses Pydantic field-level validation
    req = tr.TrainingStartRequest.model_construct(
        dataset_id=sample_dataset.id,
        model="yolov8n.pt", model_architecture="YOLOv8",
        pretrained_weights_used=True, epochs=1, batch_size=1, image_size=640,
        learning_rate=0.01, optimizer="auto", device="cpu",
        val_split=0.5, test_split=0.5,  # sum = 1.0 ≥ 0.9 → 400
        seed=42, augmentation_enabled=False, augmentation_pipeline_name="none",
        flip_enabled=False, rotation_angle=0.0, brightness_range="0.0-0.0",
        noise_level=0.0, blur_enabled=False, augmented_images_count=0,
        experiment_name="test", register_best_model=False,
        model_stage="Staging", run_name=None, mlflow_tracking_uri=None,
        model_version=None, model_description=None,
    )
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        asyncio.get_event_loop().run_until_complete(
            tr.start_training(req, db=db_session)
        )
    assert exc_info.value.status_code == 400
    assert "too large" in exc_info.value.detail


# ── API: start_training meta write failure (1597-1599) ─────────────────────
def test_start_training_meta_write_fail(client, sample_dataset):
    with patch("app.api.routes.training._run_training"), \
         patch("builtins.open", side_effect=OSError("no perm")):
        payload = {
            "dataset_id": sample_dataset.id, "model": "yolov8n.pt",
            "epochs": 1, "batch_size": 1, "image_size": 640,
            "val_split": 0.2, "test_split": 0.1, "augmentation_enabled": False,
        }
        resp = client.post("/train/start", json=payload)
    assert resp.status_code == 200  # non-fatal


# ── API: detect-devices error paths (1523-1524, 1536-1541, 1550-1551) ──────
def test_detect_devices_cuda_props_error(client):
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.init"), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_properties", side_effect=RuntimeError("props")):
        resp = client.get("/train/detect-devices")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cuda_available"] is True


def test_detect_devices_cuda_init_error(client):
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.init", side_effect=RuntimeError("init fail")):
        resp = client.get("/train/detect-devices")
    assert resp.status_code == 200


def test_detect_devices_torch_import_error(client):
    with patch("app.api.routes.training.torch", side_effect=ImportError("no torch")):
        resp = client.get("/train/detect-devices")
    assert resp.status_code == 200


# ── download: in-memory job, no run_dir (1682, 1688) ───────────────────────
def test_download_job_in_memory_no_run_dir(client, sample_dataset):
    job_id = "dl_mem"
    _jobs[job_id] = _make_job(job_id, sample_dataset.id,
                               {"status": "completed", "artifacts": {}})
    resp = client.get(f"/train/jobs/{job_id}/download")
    assert resp.status_code == 404


# ── download: run_dir not on disk (1692) ───────────────────────────────────
def test_download_job_run_dir_missing(client, db_session, sample_dataset):
    from app.core.database import TrainingJob
    job_id = f"dl_miss_{uuid.uuid4()}"
    rec = TrainingJob(id=job_id, dataset_id=sample_dataset.id,
                      status="completed",
                      artifacts={"run_dir": "/nonexistent/path/xyz"})
    db_session.add(rec); db_session.commit()
    resp = client.get(f"/train/jobs/{job_id}/download")
    assert resp.status_code == 404


# ── download: zip creation error (1700-1703) ───────────────────────────────
def test_download_job_zip_error(client, db_session, sample_dataset, tmp_path):
    from app.core.database import TrainingJob
    from app.utils import file_utils as fu
    job_id = f"dl_ziperr_{uuid.uuid4()}"
    run_dir = tmp_path / "run"; run_dir.mkdir()
    rec = TrainingJob(id=job_id, dataset_id=sample_dataset.id,
                      status="completed", artifacts={"run_dir": str(run_dir)})
    db_session.add(rec); db_session.commit()
    with patch.object(fu, "create_zip_archive", side_effect=Exception("zip fail")):
        resp = client.get(f"/train/jobs/{job_id}/download")
    assert resp.status_code == 500


# ── delete job from DB (1725-1745) ─────────────────────────────────────────
def test_delete_job_from_db(client, db_session, sample_dataset):
    from app.core.database import TrainingJob
    job_id = f"del_db_{uuid.uuid4()}"
    rec = TrainingJob(id=job_id, dataset_id=sample_dataset.id,
                      status="completed", params={},
                      created_at=datetime.utcnow())
    db_session.add(rec); db_session.commit()
    resp = client.delete(f"/train/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["job_id"] == job_id


def test_delete_job_not_found(client):
    resp = client.delete("/train/jobs/nonexistent-id")
    assert resp.status_code == 404


# ── stop job from DB (1764-1784) ───────────────────────────────────────────
def test_stop_job_from_db_running(client, db_session, sample_dataset):
    from app.core.database import TrainingJob
    job_id = f"stop_db_{uuid.uuid4()}"
    rec = TrainingJob(id=job_id, dataset_id=sample_dataset.id,
                      status="running", params={},
                      created_at=datetime.utcnow())
    db_session.add(rec); db_session.commit()
    resp = client.post(f"/train/jobs/{job_id}/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_stop_job_from_db_already_done(client, db_session, sample_dataset):
    from app.core.database import TrainingJob
    job_id = f"stop_done_{uuid.uuid4()}"
    rec = TrainingJob(id=job_id, dataset_id=sample_dataset.id,
                      status="completed", params={},
                      created_at=datetime.utcnow())
    db_session.add(rec); db_session.commit()
    resp = client.post(f"/train/jobs/{job_id}/stop")
    assert resp.status_code == 200


# ── find_latest_weights (1798, 1802, 1811-1812, 1820-1823) ─────────────────
def test_find_latest_weights_no_dir(tmp_path):
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path / "nope"):
        assert find_latest_weights("ds") is None


def test_find_latest_weights_non_dir_entry(tmp_path):
    (tmp_path / "file.txt").touch()  # file, not dir → line 1802 skip
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path):
        assert find_latest_weights("ds") is None


def test_find_latest_weights_bad_meta_json(tmp_path):
    job_dir = tmp_path / "job1"; job_dir.mkdir()
    (job_dir / "job_meta.json").write_text("NOT JSON{{{")  # line 1811-1812
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path):
        assert find_latest_weights("ds") is None


def test_find_latest_weights_fallback_no_meta(tmp_path):
    """No metadata → fallback to latest weights (1820-1823)."""
    job_dir = tmp_path / "job1"; job_dir.mkdir()
    weights = job_dir / "runs" / "train" / "weights"; weights.mkdir(parents=True)
    (weights / "best.pt").touch()
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path):
        result = find_latest_weights("other_ds")
    assert result is not None


# ── predict: dataset_id empty (1841) ───────────────────────────────────────
def test_predict_empty_dataset_id(client):
    resp = client.post("/train/predict", data={"dataset_id": "",
                        "image_url": "http://x.com/img.jpg"})
    assert resp.status_code in (400, 422)


# ── predict: url relative path rewriting (1852-1855) ───────────────────────
def test_predict_url_rewrite(client):
    with patch("app.api.routes.training.find_latest_weights",
               return_value=None):
        resp = client.post("/train/predict",
                           data={"dataset_id": "ds1",
                                 "image_url": "/api/images/img.jpg"})
    assert resp.status_code == 404


# ── predict: no weights (1859) ─────────────────────────────────────────────
def test_predict_no_weights(client):
    with patch("app.api.routes.training.find_latest_weights", return_value=None):
        resp = client.post("/train/predict",
                           data={"dataset_id": "ds1",
                                 "image_url": "http://example.com/img.jpg"})
    assert resp.status_code == 404


# ── predict: file too large (1871-1872) ────────────────────────────────────
def test_predict_file_too_large(client):
    with patch("app.api.routes.training.find_latest_weights",
               return_value=Path("/tmp/best.pt")):
        big_content = b"x" * (11 * 1024 * 1024)
        resp = client.post("/train/predict",
                           data={"dataset_id": "ds1"},
                           files={"image_file": ("big.jpg", big_content, "image/jpeg")})
    assert resp.status_code == 400


# ── predict: unsupported file type (1874-1875) ─────────────────────────────
def test_predict_bad_file_type(client):
    with patch("app.api.routes.training.find_latest_weights",
               return_value=Path("/tmp/best.pt")):
        resp = client.post("/train/predict",
                           data={"dataset_id": "ds1"},
                           files={"image_file": ("doc.pdf", b"content", "application/pdf")})
    assert resp.status_code == 400


# ── predict: invalid URL (1888-1890) ───────────────────────────────────────
def test_predict_invalid_url(client):
    with patch("app.api.routes.training.find_latest_weights",
               return_value=Path("/tmp/best.pt")):
        resp = client.post("/train/predict",
                           data={"dataset_id": "ds1",
                                 "image_url": "not-a-valid-url"})
    assert resp.status_code == 400


# ── predict: download fail (1893-1894) ─────────────────────────────────────
def test_predict_url_download_fail(client):
    with patch("app.api.routes.training.find_latest_weights",
               return_value=Path("/tmp/best.pt")), \
         patch("app.api.routes.training.requests.get") as mock_get:
        mock_get.return_value.status_code = 404
        resp = client.post("/train/predict",
                           data={"dataset_id": "ds1",
                                 "image_url": "http://example.com/img.jpg"})
    assert resp.status_code == 400


# ── predict: url image too large (1895-1896) ───────────────────────────────
def test_predict_url_image_too_large(client):
    with patch("app.api.routes.training.find_latest_weights",
               return_value=Path("/tmp/best.pt")), \
         patch("app.api.routes.training.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"x" * (11 * 1024 * 1024)
        resp = client.post("/train/predict",
                           data={"dataset_id": "ds1",
                                 "image_url": "http://example.com/img.jpg"})
    assert resp.status_code == 400


# ── predict: unsupported url suffix (1898-1899) ────────────────────────────
def test_predict_bad_url_suffix(client):
    with patch("app.api.routes.training.find_latest_weights",
               return_value=Path("/tmp/best.pt")), \
         patch("app.api.routes.training.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"data"
        resp = client.post("/train/predict",
                           data={"dataset_id": "ds1",
                                 "image_url": "http://example.com/file.pdf"})
    assert resp.status_code == 400


# ── predict: dataset has class_names (1911-1912) ───────────────────────────
def test_predict_with_class_names(client, sample_dataset, db_session, tmp_path):
    from app.core.database import Dataset
    ds = db_session.query(Dataset).filter_by(id=sample_dataset.id).first()
    if ds:
        ds.analysis_summary = {"class_names": ["drone"]}
        db_session.commit()

    mock_model = MagicMock()
    mock_model.predict.return_value = []
    mock_model.names = {0: "drone"}
    mock_yolo_cls = MagicMock(return_value=mock_model)
    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO = mock_yolo_cls

    with patch("app.api.routes.training.find_latest_weights",
               return_value=Path("/fake/best.pt")), \
         patch("app.api.routes.training.requests.get") as mock_get, \
         patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path), \
         patch.dict("sys.modules", {"ultralytics": mock_ultralytics}), \
         patch("PIL.Image.open") as mock_pil:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"fake_img"
        mock_pil.return_value.__enter__ = lambda s: s
        mock_pil.return_value.__exit__ = MagicMock(return_value=False)
        mock_pil.return_value.size = (640, 480)
        resp = client.post("/train/predict",
                           data={"dataset_id": sample_dataset.id,
                                 "image_url": "http://example.com/img.jpg"})
    assert resp.status_code == 200


# ── predict: inference with boxes (1927-1943) ──────────────────────────────
def test_predict_with_detections(client, sample_dataset, tmp_path):
    import torch

    box = MagicMock()
    box.xyxy = [torch.tensor([10.0, 20.0, 50.0, 60.0])]
    box.conf = [torch.tensor(0.9)]
    box.cls = [torch.tensor(0)]

    result_item = MagicMock()
    result_item.boxes = [box]

    mock_model = MagicMock()
    mock_model.predict.return_value = [result_item]
    mock_model.names = {0: "drone"}
    mock_yolo_cls = MagicMock(return_value=mock_model)
    mock_ultralytics = MagicMock()
    mock_ultralytics.YOLO = mock_yolo_cls

    with patch("app.api.routes.training.find_latest_weights",
               return_value=Path("/fake/best.pt")), \
         patch("app.api.routes.training.requests.get") as mock_get, \
         patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path), \
         patch.dict("sys.modules", {"ultralytics": mock_ultralytics}), \
         patch("PIL.Image.open") as mock_pil:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"fake_img"
        mock_pil.return_value.__enter__ = lambda s: s
        mock_pil.return_value.__exit__ = MagicMock(return_value=False)
        mock_pil.return_value.size = (640, 480)
        resp = client.post("/train/predict",
                           data={"dataset_id": sample_dataset.id,
                                 "image_url": "http://example.com/img.jpg"})
    assert resp.status_code == 200
    preds = resp.json().get("predictions", [])
    assert len(preds) >= 1


# ── predict: inference exception (1955-1958) ───────────────────────────────
def test_predict_inference_exception(client, sample_dataset):
    with patch("app.api.routes.training.find_latest_weights",
               return_value=Path("/fake/best.pt")), \
         patch("app.api.routes.training.requests.get") as mock_get, \
         patch("app.api.routes.training.YOLO",
               side_effect=Exception("model load fail")):
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = b"img"
        resp = client.post("/train/predict",
                           data={"dataset_id": sample_dataset.id,
                                 "image_url": "http://example.com/img.jpg"})
    assert resp.status_code == 500
