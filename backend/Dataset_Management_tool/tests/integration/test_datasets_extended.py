import pytest
from app.core.database import Dataset, Project, User, DatasetValidation, ClassDistribution, Image as DBImage, Label as DBLabel
import uuid
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_get_dataset_statistics(client, sample_dataset, db_session):
    # Setup some stats
    val = DatasetValidation(dataset_id=sample_dataset.id, total_images=10, total_labels=10)
    db_session.add(val)
    cd = ClassDistribution(dataset_id=sample_dataset.id, class_id="0", object_count=5)
    db_session.add(cd)
    db_session.commit()
    
    response = client.get(f"/datasets/{sample_dataset.id}/statistics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_images"] == 10
    assert len(data["class_distribution"]) >= 1

def test_list_dataset_image_issues(client, sample_dataset, db_session, tmp_path):
    # Setup images with issues
    img = DBImage(id=str(uuid.uuid4()), dataset_id=sample_dataset.id, file_name="blurry.jpg", file_path=str(tmp_path / "blurry.jpg"))
    db_session.add(img)
    
    # Create the blurry image on disk
    from PIL import Image
    import numpy as np
    Image.new("L", (100, 100), color=255).save(tmp_path / "blurry.jpg")
    
    db_session.commit()
    
    response = client.get(f"/datasets/{sample_dataset.id}/images/issues", params={"blur_threshold": 1000.0})
    assert response.status_code == 200
    data = response.json()
    assert len(data["flagged_images"]) >= 1
    assert "blurry_image" in data["flagged_images"][0]["issues"]

def test_list_all_images_global(client, sample_dataset, db_session):
    # Add another image
    img = DBImage(id=str(uuid.uuid4()), dataset_id=sample_dataset.id, file_name="global_img.jpg", has_label=True)
    db_session.add(img)
    db_session.commit()
    
    response = client.get("/images", params={"dataset_id": sample_dataset.id, "has_label": True})
    assert response.status_code == 200
    data = response.json()
    assert len(data["images"]) >= 1

def test_update_image_labels(client, sample_dataset, db_session, tmp_path):
    # Setup image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    
    img_path = img_dir / "test_update.jpg"
    img_path.write_text("fake image")
    
    img = DBImage(id=str(uuid.uuid4()), dataset_id=sample_dataset.id, file_name="test_update.jpg", file_path=str(img_path))
    db_session.add(img)
    db_session.commit()
    
    labels = [
        {"class_id": "1", "bbox": {"yolo": [0.5, 0.5, 0.2, 0.2]}}
    ]
    
    response = client.put(f"/datasets/{sample_dataset.id}/images/{img.id}/labels", json=labels)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Verify label file was created
    label_file = label_dir / "test_update.txt"
    assert label_file.exists()
    assert "1 0.5 0.5 0.2 0.2" in label_file.read_text()
    
    # Verify DB update
    db_session.refresh(img)
    assert img.has_label == True
    assert db_session.query(DBLabel).filter(DBLabel.image_id == img.id).count() == 1

def test_list_dataset_images_fallbacks(client, sample_dataset, db_session, tmp_path):
    # Setup image with label on disk
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir = tmp_path / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = img_dir / "fallback.jpg"
    img_path.write_text("fake")
    label_path = label_dir / "fallback.txt"
    label_path.write_text("0 0.1 0.1 0.2 0.2")
    
    img = DBImage(id=str(uuid.uuid4()), dataset_id=sample_dataset.id, file_name="fallback.jpg", file_path=str(img_path))
    db_session.add(img)
    db_session.commit()
    
    response = client.get(f"/datasets/{sample_dataset.id}/images")
    assert response.status_code == 200
    data = response.json()
    assert len(data["images"][0]["labels"]) == 1
    assert data["images"][0]["labels"][0]["class_id"] == "0"

def test_delete_dataset_with_files(client, sample_dataset, db_session, tmp_path):
    # Add a mock zip path
    dataset_id = sample_dataset.id
    zip_path = tmp_path / f"{dataset_id}.zip"
    zip_path.write_text("fake zip")
    
    sample_dataset.zip_file_path = str(zip_path)
    db_session.commit()
    
    # Mock STORAGE_ROOT and cleanup_session
    with patch("app.api.routes.datasets.STORAGE_ROOT", tmp_path), \
         patch("app.api.routes.datasets.cleanup_session") as mock_cleanup:
        
        # Create a folder matching dataset_id in a subfolder to test rglob
        subfolder = tmp_path / "some_user" / "some_project" / dataset_id
        subfolder.mkdir(parents=True)
        
        response = client.delete(f"/datasets/{dataset_id}")
        assert response.status_code == 200
        assert not zip_path.exists()
        assert not subfolder.exists()
        mock_cleanup.assert_called_with(dataset_id)

def test_delete_dataset_exception(client, sample_dataset):
    with patch("app.api.routes.datasets.cleanup_session", side_effect=Exception("Simulated Failure")):
        response = client.delete(f"/datasets/{sample_dataset.id}")
        assert response.status_code == 500
        assert "Failed to delete dataset" in response.json()["detail"]

def test_list_dataset_images_not_found(client):
    response = client.get(f"/datasets/{uuid.uuid4()}/images")
    assert response.status_code == 404

def test_list_dataset_image_issues_not_found(client):
    response = client.get(f"/datasets/{uuid.uuid4()}/images/issues")
    assert response.status_code == 404

def test_list_dataset_images_filter_false(client, sample_dataset, db_session):
    img = DBImage(id=str(uuid.uuid4()), dataset_id=sample_dataset.id, file_name="no_label.jpg", has_label=False)
    db_session.add(img)
    db_session.commit()
    response = client.get(f"/datasets/{sample_dataset.id}/images", params={"has_label": False})
    assert response.status_code == 200
    data = response.json()
    assert any(im["file_name"] == "no_label.jpg" for im in data["images"])

def test_list_dataset_image_issues_all_types(client, sample_dataset, db_session, tmp_path):
    # Setup image with multiple issues
    img = DBImage(id=str(uuid.uuid4()), dataset_id=sample_dataset.id, file_name="issues.jpg", file_path=str(tmp_path / "issues.jpg"))
    db_session.add(img)
    # Add an invalid label (out of bounds)
    lbl = DBLabel(id=str(uuid.uuid4()), image_id=img.id, class_id="0", bbox_data={"yolo": [2.0, 2.0, 0.1, 0.1]})
    db_session.add(lbl)
    # Add a small object label
    lbl2 = DBLabel(id=str(uuid.uuid4()), image_id=img.id, class_id="1", bbox_data={"yolo": [0.5, 0.5, 0.01, 0.01]})
    db_session.add(lbl2)
    db_session.commit()
    
    response = client.get(f"/datasets/{sample_dataset.id}/images/issues")
    assert response.status_code == 200
    data = response.json()
    issues = data["flagged_images"][0]["issues"]
    assert "invalid_bbox" in issues
    assert "object_too_small" in issues
    assert data["flagged_images"][0]["invalid_label_count"] == 1

def test_get_dataset_statistics_not_found(client):
    response = client.get(f"/datasets/{uuid.uuid4()}/statistics")
    assert response.status_code == 404

def test_update_image_labels_not_found(client, sample_dataset):
    response = client.put(f"/datasets/{sample_dataset.id}/images/{uuid.uuid4()}/labels", json=[])
    assert response.status_code == 404
