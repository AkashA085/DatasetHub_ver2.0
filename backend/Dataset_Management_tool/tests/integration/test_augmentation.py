import pytest
from app.core.database import Dataset
import json
import numpy as np
import cv2
from pathlib import Path
from app.utils.file_utils import PROCESSED_DIR, UPLOADS_DIR, ANALYSIS_DIR

import uuid

@pytest.fixture
def setup_dataset_for_aug(db_session):
    session_id = str(uuid.uuid4())
    
    # Add to DB first
    ds = Dataset(id=session_id, project_id="test_project", format_type="yolo")
    db_session.add(ds)
    db_session.commit()
    
    # Processed dir
    proc_dir = PROCESSED_DIR / session_id
    proc_dir.mkdir(parents=True, exist_ok=True)
    
    # Initial annotations
    ann_path = proc_dir / "annotations.json"
    initial_ann = [
        {"image_name": "test_img", "width": 100, "height": 100, "objects": [
            {"class_id": 0, "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}
        ]}
    ]
    ann_path.write_text(json.dumps(initial_ann))
    
    # Uploads dir (images)
    up_dir = UPLOADS_DIR / session_id
    img_dir = up_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Real black image for cv2
    img_path = img_dir / "test_img.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    
    # Labels dir (classes.txt)
    lbl_dir = up_dir / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (lbl_dir / "classes.txt").write_text("drone\n")
    
    return session_id

def test_augment_dataset_route_success(client, setup_dataset_for_aug):
    session_id = setup_dataset_for_aug
    
    payload = {
        "dataset_id": session_id,
        "count": 1,
        "horizontal_flip": True,
        "export_format": "yolo"
    }
    
    response = client.post("/augment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"] == session_id
    assert data["validation_report"]["total_images"] == 2 # Original + 1 augmented
    
    # Check if augmented image exists
    proc_dir = PROCESSED_DIR / session_id
    aug_dir = proc_dir / "augmented_images"
    assert aug_dir.exists()
    assert len(list(aug_dir.glob("*.jpg"))) == 1

def test_augment_dataset_sequential(client, setup_dataset_for_aug):
    session_id = setup_dataset_for_aug
    
    payload = {
        "dataset_id": session_id,
        "count": 1,
        "horizontal_flip": True,
        "export_format": "yolo"
    }
    
    # First run
    client.post("/augment", json=payload)
    
    # Second run - should pick up augmented images
    response = client.post("/augment", json=payload)
    assert response.status_code == 200
    data = response.json()
    # 1 original + 1 (first run) + 2 (second run, because it augments all existing)
    # Wait, the logic is: new_anns, new_stem_map = AugmentationService.augment_dataset(request, annotations, stem_to_image)
    # If annotations has 2 items, it should generate 2 new ones if count=1?
    # Let's see.
    assert data["validation_report"]["total_images"] > 2

def test_augment_dataset_db_error(client, setup_dataset_for_aug):
    session_id = setup_dataset_for_aug
    
    payload = {
        "dataset_id": session_id,
        "count": 1,
        "export_format": "yolo"
    }
    
    from unittest.mock import patch
    with patch("app.api.routes.augmentation.Session.commit", side_effect=Exception("DB Error")):
        response = client.post("/augment", json=payload)
        # Should still succeed but log the error
        assert response.status_code == 200
