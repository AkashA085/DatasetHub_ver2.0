"""
Coverage tests for training.py helper functions.
Targets: lines 226,230,238,249-250,288,316,325,339,341,345,355,360-361,
         367-370,439,528,539,543-545,576-578,593-598,604,613,615-616,
         622-624,626,655-656,660-661,664,666,679-680,684-685,687,692,
         697-709,734-735,781,790,793-799,802,806,809-810,833-835,846,
         854-879,921-922,925-926,933-934,949-950,954-955,960-962,970,990
"""
import csv
import json
import os
import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from app.api.routes.training import (
    _append_log, _save_job_to_db, _count_job_dataset_images,
    _compute_job_metrics_fallback, _build_loss_metrics, _plot_curves,
    _plot_class_distribution, _resolve_device, _validate_and_setup_gpu,
    _get_optimal_batch_size, write_augmented_split, DroneMLflowCallback,
    _prepare_yolo_dataset,
)


# ── _append_log truncation (line 528) ──────────────────────────────────────
def test_append_log_truncates():
    job = {"logs": [f"msg{i}" for i in range(205)]}
    _append_log(job, "new")
    assert len(job["logs"]) <= 200


# ── _save_job_to_db branches (539, 543-545, 576-578) ──────────────────────
def test_save_job_to_db_creates_new(db_session):
    job = {
        "job_id": "new_j1", "dataset_id": "ds1", "status": "queued",
        "params": {}, "created_at": datetime.utcnow().isoformat(),
        "started_at": None, "finished_at": None,
        "metrics": None, "artifacts": None, "mlflow": None,
        "error": None, "logs": [],
    }
    _save_job_to_db(job, db_session)


def test_save_job_to_db_updates_existing(db_session):
    from app.core.database import TrainingJob
    rec = TrainingJob(id="upd_j1", dataset_id="ds1", status="queued", params={})
    db_session.add(rec); db_session.commit()
    job = {
        "job_id": "upd_j1", "dataset_id": "ds1", "status": "running",
        "params": {}, "created_at": datetime.utcnow().isoformat(),
        "started_at": datetime.utcnow().isoformat(), "finished_at": None,
        "metrics": None, "artifacts": None, "mlflow": None,
        "error": None, "logs": [],
    }
    _save_job_to_db(job, db_session)
    db_session.refresh(rec)
    assert rec.status == "running"


def test_save_job_to_db_exception_rollback(db_session):
    job = {
        "job_id": "bad_j1", "dataset_id": "ds1", "status": "queued",
        "params": {}, "created_at": "not-a-date",
        "started_at": None, "finished_at": None,
        "metrics": None, "artifacts": None, "mlflow": None,
        "error": None, "logs": [],
    }
    with patch.object(db_session, "commit", side_effect=Exception("DB fail")):
        _save_job_to_db(job, db_session)  # must not raise


def test_save_job_to_db_bad_iso_string(db_session):
    job = {
        "job_id": "bad_iso", "dataset_id": "ds1", "status": "queued",
        "params": {}, "created_at": datetime.utcnow().isoformat(),
        "started_at": "not-valid-iso", "finished_at": None,
        "metrics": None, "artifacts": None, "mlflow": None,
        "error": None, "logs": [],
    }
    _save_job_to_db(job, db_session)  # must not raise (bad iso → None)


# ── _count_job_dataset_images (593-598) ────────────────────────────────────
def test_count_job_dataset_images(tmp_path):
    job_id = "cnt_job"
    img_dir = tmp_path / job_id / "dataset" / "train" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "a.jpg").touch()
    (img_dir / "b.jpg").touch()
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path):
        from app.api.routes.training import _count_job_dataset_images
        assert _count_job_dataset_images(job_id) == 2


def test_count_job_dataset_images_no_dir(tmp_path):
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path):
        from app.api.routes.training import _count_job_dataset_images
        assert _count_job_dataset_images("no_job") == 0


# ── _compute_job_metrics_fallback (604, 613, 615-616, 622-624, 626) ────────
def test_compute_metrics_non_dict_metrics():
    job = {"job_id": "j", "metrics": "bad", "started_at": None}
    result = _compute_job_metrics_fallback(job)
    assert isinstance(result, dict)


def test_compute_metrics_no_finished_at():
    now = datetime.utcnow()
    job = {"job_id": "j", "metrics": None,
           "started_at": (now - timedelta(seconds=5)).isoformat(),
           "finished_at": None}
    result = _compute_job_metrics_fallback(job)
    assert result.get("total_training_time") is not None


def test_compute_metrics_exception_in_time():
    job = {"job_id": "j", "metrics": None,
           "started_at": "bad-date", "finished_at": None}
    result = _compute_job_metrics_fallback(job)
    assert isinstance(result, dict)


def test_compute_metrics_image_count_from_dir(tmp_path):
    job_id = "img_cnt"
    img_dir = tmp_path / job_id / "dataset" / "train" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "x.jpg").touch()
    job = {"job_id": job_id, "metrics": None, "started_at": None}
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path):
        result = _compute_job_metrics_fallback(job)
    assert result.get("images_trained") == 1


def test_compute_metrics_total_images_fallback():
    job = {"job_id": "j2", "metrics": {"total_images": 5}, "started_at": None}
    result = _compute_job_metrics_fallback(job)
    assert result.get("images_trained") == 5


# ── _build_loss_metrics (655-666) ──────────────────────────────────────────
def test_build_loss_metrics_full(tmp_path):
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "train/box_loss,train/cls_loss,train/dfl_loss,"
        "val/box_loss,val/cls_loss,val/dfl_loss\n"
        "0.1,0.2,0.3,0.05,0.06,0.07\n"
    )
    result = _build_loss_metrics(csv_path)
    assert "training_loss" in result
    assert "validation_loss" in result
    assert abs(result["training_loss"] - 0.6) < 0.01


def test_build_loss_metrics_empty_file(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")
    result = _build_loss_metrics(csv_path)
    assert result == {}


def test_build_loss_metrics_missing_file(tmp_path):
    result = _build_loss_metrics(tmp_path / "no.csv")
    assert result == {}


# ── _plot_curves error paths (679-680, 684-685, 687, 692, 697-709) ─────────
def test_plot_curves_no_csv(tmp_path):
    result = _plot_curves(tmp_path / "no.csv", tmp_path / "l.png", tmp_path / "a.png")
    assert result == {}


def test_plot_curves_matplotlib_missing(tmp_path):
    csv_path = tmp_path / "r.csv"
    csv_path.write_text("a,b\n1,2\n")
    with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None,
                                     "pandas": None}):
        result = _plot_curves(csv_path, tmp_path / "l.png", tmp_path / "a.png")
    assert result == {}


def test_plot_curves_empty_df(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("epoch,val\n")
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    result = _plot_curves(csv_path, tmp_path / "l.png", tmp_path / "a.png")
    assert isinstance(result, dict)


def test_plot_curves_no_epoch_col(tmp_path):
    csv_path = tmp_path / "r.csv"
    csv_path.write_text(
        "train/box_loss,metrics/mAP50(B)\n0.1,0.8\n0.09,0.85\n"
    )
    import matplotlib
    matplotlib.use("Agg")
    result = _plot_curves(csv_path, tmp_path / "l.png", tmp_path / "a.png")
    assert isinstance(result, dict)


def test_plot_curves_with_losses_and_map(tmp_path):
    csv_path = tmp_path / "r.csv"
    csv_path.write_text(
        "epoch,train/box_loss,val/box_loss,metrics/mAP50(B)\n"
        "0,0.5,0.4,0.7\n1,0.4,0.3,0.8\n"
    )
    import matplotlib
    matplotlib.use("Agg")
    result = _plot_curves(csv_path, tmp_path / "l.png", tmp_path / "a.png")
    assert "loss_curve.png" in result or "accuracy_curve.png" in result


# ── _plot_class_distribution error (734-735) ───────────────────────────────
def test_plot_class_dist_empty():
    result = _plot_class_distribution({}, Path("/tmp/out.png"))
    assert result is False


def test_plot_class_dist_matplotlib_missing(tmp_path):
    with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
        result = _plot_class_distribution({"a": 1}, tmp_path / "d.png")
    assert result is False


# ── _resolve_device branches (781, 790, 793-799, 802, 806, 809-810) ────────
def test_resolve_device_empty_no_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        assert _resolve_device("") == "cpu"
        assert _resolve_device("none") == "cpu"
        assert _resolve_device("auto") == "cpu"
        assert _resolve_device("cuda") == "cpu"
        assert _resolve_device("cuda:0") == "cpu"
        assert _resolve_device("0") == "cpu"


def test_resolve_device_cuda_idx_valid():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2):
        assert _resolve_device("cuda:0") == "0"
        assert _resolve_device("cuda:1") == "1"


def test_resolve_device_cuda_idx_out_of_range():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1):
        assert _resolve_device("cuda:5") == "cpu"


def test_resolve_device_digit_valid():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2):
        assert _resolve_device("0") == "0"


def test_resolve_device_digit_out_of_range():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1):
        assert _resolve_device("5") == "cpu"


def test_resolve_device_exception():
    with patch("torch.cuda.is_available", side_effect=RuntimeError("oops")):
        assert _resolve_device("0") == "cpu"


def test_resolve_device_cuda_bare_no_devices():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=0):
        assert _resolve_device("cuda") == "cpu"


# ── _validate_and_setup_gpu (833-835, 846) ─────────────────────────────────
def test_validate_gpu_cpu_device():
    job = {"logs": []}
    _validate_and_setup_gpu("cpu", job)
    assert any("WARNING" in l for l in job["logs"])


def test_validate_gpu_no_cuda():
    job = {"logs": []}
    with patch("torch.cuda.is_available", return_value=False):
        _validate_and_setup_gpu("0", job)
    assert any("not available" in l or "CUDA" in l for l in job["logs"])


def test_validate_gpu_exception():
    job = {"logs": []}
    with patch("torch.cuda.is_available", side_effect=RuntimeError("boom")):
        _validate_and_setup_gpu("0", job)
    assert any("warning" in l.lower() or "GPU" in l for l in job["logs"])


# ── _get_optimal_batch_size (854-879) ──────────────────────────────────────
def test_get_optimal_batch_size_all_branches():
    cases = [
        (1280, 24, 64), (1280, 16, 32), (1280, 8, 16), (1280, 4, 8),
        (640, 24, 128), (640, 16, 64), (640, 8, 32), (640, 4, 16),
        (320, 24, 256), (320, 16, 128), (320, 8, 64), (320, 4, 32),
    ]
    for img_size, vram_gb, expected in cases:
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties") as mp:
            mp.return_value.total_memory = vram_gb * (1024**3)
            result = _get_optimal_batch_size("cuda", img_size)
        assert result == expected, f"img={img_size} vram={vram_gb}GB → got {result}, want {expected}"


def test_get_optimal_batch_size_exception():
    with patch("torch.cuda.is_available", side_effect=RuntimeError):
        assert _get_optimal_batch_size("cuda", 640) == 16


# ── write_augmented_split branches (226, 230, 238, 249-250) ────────────────
@patch("app.api.routes.training.cv2")
def test_write_augmented_split_missing_label(mock_cv2, tmp_path):
    src_img = tmp_path / "imgs"; src_img.mkdir()
    src_lbl = tmp_path / "lbls"; src_lbl.mkdir()
    (src_img / "img1.jpg").touch()  # no label file
    mock_cv2.BORDER_CONSTANT = 0
    count = write_augmented_split(str(src_img), str(src_lbl),
                                  str(tmp_path / "di"), str(tmp_path / "dl"))
    assert count == 0


@patch("app.api.routes.training.cv2")
def test_write_augmented_split_imread_none(mock_cv2, tmp_path):
    src_img = tmp_path / "imgs"; src_img.mkdir()
    src_lbl = tmp_path / "lbls"; src_lbl.mkdir()
    (src_img / "img1.jpg").touch()
    (src_lbl / "img1.txt").write_text("0 0.5 0.5 0.1 0.1")
    mock_cv2.BORDER_CONSTANT = 0
    mock_cv2.imread.return_value = None  # line 230
    count = write_augmented_split(str(src_img), str(src_lbl),
                                  str(tmp_path / "di"), str(tmp_path / "dl"))
    assert count == 0


@patch("app.api.routes.training.cv2")
def test_write_augmented_split_short_line(mock_cv2, tmp_path):
    src_img = tmp_path / "imgs"; src_img.mkdir()
    src_lbl = tmp_path / "lbls"; src_lbl.mkdir()
    (src_img / "img1.jpg").touch()
    (src_lbl / "img1.txt").write_text("0 0.5")  # < 5 parts → line 238 continue
    mock_cv2.BORDER_CONSTANT = 0; mock_cv2.COLOR_BGR2RGB = 4
    mock_cv2.COLOR_RGB2BGR = 4; mock_cv2.IMWRITE_JPEG_QUALITY = 1
    mock_cv2.imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_cv2.cvtColor.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    count = write_augmented_split(str(src_img), str(src_lbl),
                                  str(tmp_path / "di"), str(tmp_path / "dl"),
                                  multiplier=0)
    assert count == 1  # original only (0 bbox)


@patch("app.api.routes.training.cv2")
def test_write_augmented_split_aug_exception(mock_cv2, tmp_path):
    src_img = tmp_path / "imgs"; src_img.mkdir()
    src_lbl = tmp_path / "lbls"; src_lbl.mkdir()
    (src_img / "img1.jpg").touch()
    (src_lbl / "img1.txt").write_text("0 0.5 0.5 0.1 0.1")
    mock_cv2.BORDER_CONSTANT = 0; mock_cv2.COLOR_BGR2RGB = 4
    mock_cv2.COLOR_RGB2BGR = 4; mock_cv2.IMWRITE_JPEG_QUALITY = 1
    mock_cv2.imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_cv2.cvtColor.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    with patch("app.api.routes.training.build_train_transform") as mbt:
        mbt.return_value.side_effect = Exception("aug fail")  # lines 249-250
        count = write_augmented_split(str(src_img), str(src_lbl),
                                      str(tmp_path / "di"), str(tmp_path / "dl"),
                                      multiplier=1)
    assert count >= 1  # original always written


# ── DroneMLflowCallback exception paths (316, 325, 339, 341, 345) ──────────
def test_drone_callback_exception_paths():
    from app.api.routes.training import DroneMLflowCallback
    cb = DroneMLflowCallback(MagicMock(), {"best_map50": 0.0})
    trainer = MagicMock()
    trainer.epoch = 1
    # Values that make float() inside try/except raise (bare excepts at 316, 325)
    trainer.label_loss_items.return_value = {"box": object()}
    trainer.metrics = {"mAP50": object()}
    # lr must succeed (line 328 not in bare except) so return a real float
    trainer.scheduler.get_last_lr.return_value = [0.001]
    trainer.save_dir = "/nonexistent/path"
    # Patch only log_metric for loss/val (not lr call on line 328)
    with patch("app.api.routes.training.mlflow") as mm:
        # make log_metric silently succeed for lr call, fail for others
        mm.log_metric.return_value = None
        cb.on_train_epoch_end(trainer)  # bare excepts at 316, 325 swallow errors
        # on_train_end: no files exist so log_artifact not called; bare except at 345
        mm.log_metric.side_effect = Exception("fail")
        cb.on_train_end(trainer)  # line 345 bare except swallows error


# ── _prepare_yolo_dataset error branches ───────────────────────────────────
def test_prepare_yolo_dataset_invalid_split(db_session, tmp_path):
    from app.core.database import Dataset, Image, User, Project
    user = User(id="u99", email="u99@x.com"); db_session.add(user)
    proj = Project(id="p99", name="p99", user_id="u99"); db_session.add(proj)
    ds = Dataset(id="ds99", format_type="yolo", project_id="p99"); db_session.add(ds)
    img_dir = tmp_path / "images"; img_dir.mkdir()
    lbl_dir = tmp_path / "labels"; lbl_dir.mkdir()
    for i in range(3):
        ip = img_dir / f"i{i}.jpg"; ip.touch()
        lp = lbl_dir / f"i{i}.txt"; lp.write_text("0 0.5 0.5 0.1 0.1")
        db_session.add(Image(id=f"i99_{i}", dataset_id="ds99",
                             file_path=str(ip), file_name=ip.name, has_label=True))
    db_session.commit()
    job = {"job_id": "j99", "logs": []}
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path / "jobs"), \
         pytest.raises(ValueError, match="split"):
        _prepare_yolo_dataset("ds99", 42, 0.8, 0.8, db_session, job)  # line 990


def test_prepare_yolo_dataset_image_not_file(db_session, tmp_path):
    from app.core.database import Dataset, Image, User, Project
    user = User(id="u88", email="u88@x.com"); db_session.add(user)
    proj = Project(id="p88", name="p88", user_id="u88"); db_session.add(proj)
    ds = Dataset(id="ds88", format_type="yolo", project_id="p88"); db_session.add(ds)
    img_dir = tmp_path / "images"; img_dir.mkdir()
    # Create a directory where a file is expected (is_file() → False, line 925-926)
    fake_img = img_dir / "img.jpg"; fake_img.mkdir()
    db_session.add(Image(id="i88_0", dataset_id="ds88",
                         file_path=str(fake_img), file_name="img.jpg", has_label=True))
    db_session.commit()
    job = {"job_id": "j88", "logs": []}
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path / "jobs"), \
         pytest.raises((ValueError, Exception)):
        _prepare_yolo_dataset("ds88", 42, 0.2, 0.1, db_session, job)


def test_prepare_yolo_dataset_no_images_dir(db_session, tmp_path):
    """Image path not under 'images' subdir → line 933-934."""
    from app.core.database import Dataset, Image, User, Project
    user = User(id="u77", email="u77@x.com"); db_session.add(user)
    proj = Project(id="p77", name="p77", user_id="u77"); db_session.add(proj)
    ds = Dataset(id="ds77", format_type="yolo", project_id="p77"); db_session.add(ds)
    flat = tmp_path / "flat"; flat.mkdir()
    ip = flat / "img.jpg"; ip.touch()
    db_session.add(Image(id="i77_0", dataset_id="ds77",
                         file_path=str(ip), file_name="img.jpg", has_label=True))
    db_session.commit()
    job = {"job_id": "j77", "logs": []}
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path / "jobs"), \
         pytest.raises((ValueError, Exception)):
        _prepare_yolo_dataset("ds77", 42, 0.2, 0.1, db_session, job)


def test_prepare_yolo_dataset_empty_label(db_session, tmp_path):
    """Empty label file → line 954-955."""
    from app.core.database import Dataset, Image, User, Project
    user = User(id="u66", email="u66@x.com"); db_session.add(user)
    proj = Project(id="p66", name="p66", user_id="u66"); db_session.add(proj)
    ds = Dataset(id="ds66", format_type="yolo", project_id="p66"); db_session.add(ds)
    img_dir = tmp_path / "images"; img_dir.mkdir()
    lbl_dir = tmp_path / "labels"; lbl_dir.mkdir()
    ip = img_dir / "img.jpg"; ip.touch()
    lp = lbl_dir / "img.txt"; lp.write_text("")  # empty label
    db_session.add(Image(id="i66_0", dataset_id="ds66",
                         file_path=str(ip), file_name="img.jpg", has_label=True))
    db_session.commit()
    job = {"job_id": "j66", "logs": []}
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path / "jobs"), \
         pytest.raises((ValueError, Exception)):
        _prepare_yolo_dataset("ds66", 42, 0.2, 0.1, db_session, job)


def test_prepare_yolo_dataset_more_than_5_invalid(db_session, tmp_path):
    """More than 5 invalid images → line 970."""
    from app.core.database import Dataset, Image, User, Project
    user = User(id="u55", email="u55@x.com"); db_session.add(user)
    proj = Project(id="p55", name="p55", user_id="u55"); db_session.add(proj)
    ds = Dataset(id="ds55", format_type="yolo", project_id="p55"); db_session.add(ds)
    for i in range(8):
        db_session.add(Image(id=f"i55_{i}", dataset_id="ds55",
                             file_path=f"/nonexistent/img{i}.jpg",
                             file_name=f"img{i}.jpg", has_label=True))
    db_session.commit()
    job = {"job_id": "j55", "logs": []}
    with patch("app.api.routes.training.TRAINING_JOBS_DIR", tmp_path / "jobs"), \
         pytest.raises((ValueError, Exception)):
        _prepare_yolo_dataset("ds55", 42, 0.2, 0.1, db_session, job)
    assert any("more" in l for l in job["logs"])
