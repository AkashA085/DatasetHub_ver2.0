import pytest
import json
from pathlib import Path
from app.utils.file_utils import PROCESSED_DIR

def test_update_label_success(client, tmp_path):
    # Mock processed dir and annotations.json
    session_id = "test_session"
    processed_dir = PROCESSED_DIR / session_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    ann_path = processed_dir / "annotations.json"
    initial_ann = [
        {"image_name": "img1", "width": 100, "height": 100, "objects": []}
    ]
    ann_path.write_text(json.dumps(initial_ann))
    
    payload = {
        "dataset_id": session_id,
        "image_name": "img1",
        "objects": [
            {"class_id": "0", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}
        ]
    }
    
    response = client.post("/label", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Verify file content
    updated_ann = json.loads(ann_path.read_text())
    assert len(updated_ann[0]["objects"]) == 1
    assert updated_ann[0]["objects"][0]["class_id"] == 0

def test_update_label_dataset_not_found(client):
    payload = {
        "dataset_id": "nonexistent",
        "image_name": "img1",
        "objects": []
    }
    response = client.post("/label", json=payload)
    assert response.status_code == 404
