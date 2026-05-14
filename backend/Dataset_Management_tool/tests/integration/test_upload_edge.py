import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
import asyncio
import io
import zipfile
from pathlib import Path

def test_upload_dataset_timeout(client):
    # Mock _upload_dataset_impl to simulate timeout
    with patch("app.api.routes.upload._upload_dataset_impl", side_effect=asyncio.TimeoutError()):
        files = {
            "images_zip": ("images.zip", b"fake", "application/zip"),
            "labels_zip": ("labels.zip", b"fake", "application/zip")
        }
        data = {"format_type": "yolo"}
        response = client.post("/upload-dataset", files=files, data=data)
        assert response.status_code == 408
        assert "Upload timed out" in response.json()["detail"]

def test_upload_dataset_generic_exception(client):
    with patch("app.api.routes.upload._upload_dataset_impl", side_effect=Exception("Unexpected Error")):
        files = {
            "images_zip": ("images.zip", b"fake", "application/zip"),
            "labels_zip": ("labels.zip", b"fake", "application/zip")
        }
        data = {"format_type": "yolo"}
        response = client.post("/upload-dataset", files=files, data=data)
        assert response.status_code == 500
        assert "Upload failed" in response.json()["detail"]

def test_upload_dataset_empty_zip(client):
    # Mock _save_upload_file to return 0 size
    with patch("app.api.routes.upload._save_upload_file", return_value=0):
        files = {
            "images_zip": ("images.zip", b"fake", "application/zip"),
            "labels_zip": ("labels.zip", b"fake", "application/zip")
        }
        data = {"format_type": "yolo"}
        response = client.post("/upload-dataset", files=files, data=data)
        assert response.status_code == 400
        assert "ZIP file is empty" in response.json()["detail"]

def test_upload_dataset_invalid_zip(client, tmp_path):
    # Real empty zip but we want to trigger extract_zip failure
    with patch("app.api.routes.upload.extract_zip", side_effect=Exception("Corrupted")):
        files = {
            "images_zip": ("images.zip", b"fake", "application/zip"),
            "labels_zip": ("labels.zip", b"fake", "application/zip")
        }
        data = {"format_type": "yolo"}
        response = client.post("/upload-dataset", files=files, data=data)
        assert response.status_code == 400
        assert "Invalid or corrupted ZIP file" in response.json()["detail"]

def test_upload_dataset_validation_failed(client):
    # Mock validator to return empty annotations
    mock_report = {
        "total_images": 0, "total_labels": 0, "missing_labels": 0, "orphan_labels": 0,
        "empty_labels": 0, "corrupted_images": 0, "class_ids_found": [],
        "missing_label_images": [], "orphan_label_files": [], "empty_label_files": [],
        "corrupted_image_files": []
    }
    mock_validator = MagicMock()
    mock_validator.validate.return_value = (mock_report, [], {}, {}, {})
    with patch("app.api.routes.upload.DatasetValidator", return_value=mock_validator), \
         patch("app.api.routes.upload._save_upload_file", return_value=100), \
         patch("app.api.routes.upload.extract_zip"):
        files = {
            "images_zip": ("images.zip", b"fake", "application/zip"),
            "labels_zip": ("labels.zip", b"fake", "application/zip")
        }
        data = {"format_type": "yolo"}
        response = client.post("/upload-dataset", files=files, data=data)
        assert response.status_code == 400
        assert "No matched image/label pairs found" in response.json()["detail"]

def test_upload_dataset_save_internal_failed(client):
    with patch("app.api.routes.upload.DatasetValidator"), \
         patch("app.api.routes.upload._save_upload_file", return_value=100), \
         patch("app.api.routes.upload.extract_zip"), \
         patch("builtins.open", side_effect=Exception("Permission Denied")):
        
        # We need to mock validator to return non-empty annotations to reach save_internal
        mock_val = MagicMock()
        mock_val.validate.return_value = (MagicMock(), [MagicMock()], {}, {}, {})
        with patch("app.api.routes.upload.DatasetValidator", return_value=mock_val):
            files = {
                "images_zip": ("images.zip", b"fake", "application/zip"),
                "labels_zip": ("labels.zip", b"fake", "application/zip")
            }
            data = {"format_type": "yolo"}
            response = client.post("/upload-dataset", files=files, data=data)
            # Line 148: Failed to save internal format
            assert response.status_code == 500
            assert "Failed to save internal format" in response.json()["detail"]

def test_upload_dataset_db_persistence_failed(client):
    # Mock everything up to DB persistence and make it fail
    mock_report = {
        "total_images": 1, "total_labels": 1, "missing_labels": 0, "orphan_labels": 0,
        "empty_labels": 0, "corrupted_images": 0, "class_ids_found": ["0"],
        "missing_label_images": [], "orphan_label_files": [], "empty_label_files": [],
        "corrupted_image_files": []
    }
    mock_summary = {
        "total_images": 1, "total_labels": 1, "total_classes": 1, "total_objects": 1,
        "avg_objects_per_image": 1.0, "missing_label_count": 0, "corrupted_image_count": 0,
        "class_distribution": {"0": 1}
    }
    
    with patch("app.api.routes.upload.DatasetValidator") as mock_v_cls, \
         patch("app.api.routes.upload._save_upload_file", return_value=100), \
         patch("app.api.routes.upload.extract_zip"), \
         patch("app.api.routes.upload.DatasetAnalyzer") as mock_a_cls, \
         patch("app.api.routes.upload.ExportService.export_dataset", return_value=(Path("fake.zip"), 1)), \
         patch("app.api.routes.upload.shutil.copy2"), \
         patch("app.api.routes.upload.get_db"), \
         patch("app.api.routes.upload.json.dump"), \
         patch("builtins.open", MagicMock()):
        
        mock_v_cls.return_value.validate.return_value = (MagicMock(**mock_report), [MagicMock(image_name="fake", objects=[])], {"fake": Path("fake.jpg")}, {}, {"0": "drone"})
        mock_analyzer = mock_a_cls.return_value
        mock_analyzer.analyze.return_value = MagicMock(**mock_summary)
        mock_analyzer.analyze.return_value.model_dump.return_value = mock_summary
        
        # Mock DB failure
        with patch("app.api.routes.upload.Session.commit", side_effect=Exception("DB Persistence Error")):
            files = {
                "images_zip": ("images.zip", b"fake", "application/zip"),
                "labels_zip": ("labels.zip", b"fake", "application/zip")
            }
            data = {"format_type": "yolo"}
            response = client.post("/upload-dataset", files=files, data=data)
            # Should still return 200 because DB is non-critical
            assert response.status_code == 200
