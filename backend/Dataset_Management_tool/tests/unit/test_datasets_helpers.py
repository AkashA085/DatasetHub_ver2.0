import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import json
from unittest.mock import patch, MagicMock
from app.api.routes.datasets import _to_storage_url, _parse_yolo_bbox, _is_invalid_bbox, _compute_blur_score

def test_to_storage_url():
    assert _to_storage_url("/home/user/storage/data/img.jpg") == "/storage/data/img.jpg"
    assert _to_storage_url("/absolute/path/no/match.jpg") == ""
    assert _to_storage_url("") == ""
    # "storage_no_slash" contains "storage", so it returns "/storage_no_slash"
    assert _to_storage_url("storage_no_slash") == "/storage_no_slash"
    # Case for line 146: If split results in only 1 part (defensive check)
    mock_str = MagicMock(spec=str)
    mock_str.__contains__.return_value = True
    mock_str.split.return_value = ["only_one"]
    mock_str.replace.return_value = mock_str # needed for some cases
    assert _to_storage_url(mock_str) == ""

def test_parse_yolo_bbox():
    assert _parse_yolo_bbox({"yolo": [0.5, 0.5, 0.2, 0.2]}) == [0.5, 0.5, 0.2, 0.2]
    assert _parse_yolo_bbox({"yolo": ["0.5", "0.5", "0.2", "0.2"]}) == [0.5, 0.5, 0.2, 0.2]
    assert _parse_yolo_bbox({"yolo": [0.5, 0.5]}) is None
    assert _parse_yolo_bbox("not a dict") is None
    assert _parse_yolo_bbox({"yolo": "not a list"}) is None
    assert _parse_yolo_bbox({"yolo": [0.5, 0.5, "invalid", 0.2]}) is None

def test_is_invalid_bbox():
    # Valid
    assert _is_invalid_bbox([0.5, 0.5, 0.2, 0.2]) is False
    # Negative width/height
    assert _is_invalid_bbox([0.5, 0.5, -0.1, 0.2]) is True
    # Out of bounds
    assert _is_invalid_bbox([1.1, 0.5, 0.2, 0.2]) is True
    # Bbox edges out of bounds
    assert _is_invalid_bbox([0.05, 0.5, 0.2, 0.2]) is True # x_min = 0.05 - 0.1 = -0.05

def test_compute_blur_score(tmp_path):
    # Test with a real image
    img_path = tmp_path / "test.jpg"
    # Create a 100x100 white image
    img = Image.new("L", (100, 100), color=255)
    img.save(img_path)
    # Variance of white image is 0
    assert _compute_blur_score(str(img_path)) == 0.0
    
    # Create a pattern image
    img_data = np.zeros((100, 100), dtype=np.uint8)
    img_data[50:, :] = 255
    img = Image.fromarray(img_data)
    img.save(img_path)
    score = _compute_blur_score(str(img_path))
    assert score > 0
    
    # Test exception
    assert _compute_blur_score("non_existent.jpg") is None
    
    # Case for line 185: Empty image
    with patch("PIL.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_img.__enter__.return_value = mock_img
        mock_img.convert.return_value = mock_img
        # np.asarray of mock will have size 0 if it's empty
        with patch("numpy.asarray", return_value=np.array([])):
            assert _compute_blur_score("fake.jpg") is None

def test_load_annotations_yolo_map(tmp_path):
    from app.api.routes.datasets import _load_annotations_yolo_map
    
    dataset_id = "test_ds"
    processed_dir = tmp_path / "processed"
    ds_dir = processed_dir / dataset_id
    ds_dir.mkdir(parents=True)
    ann_path = ds_dir / "annotations.json"
    
    # Mock PROCESSED_DIR
    with patch("app.api.routes.datasets.PROCESSED_DIR", processed_dir):
        # Case: file does not exist
        assert _load_annotations_yolo_map(dataset_id) == {}
        
        # Case: valid file
        annotations = [
            {
                "image_name": "img1",
                "width": 100,
                "height": 100,
                "objects": [
                    {"xmin": 10, "ymin": 10, "xmax": 20, "ymax": 20, "class_id": "0"}
                ]
            }
        ]
        with open(ann_path, "w") as f:
            json.dump(annotations, f)
            
        result = _load_annotations_yolo_map(dataset_id)
        assert "img1" in result
        assert len(result["img1"]) == 1
        assert result["img1"][0].class_id == "0"
        
        # Case: invalid file (triggers exception in line 236)
        with open(ann_path, "w") as f:
            f.write("invalid json")
        assert _load_annotations_yolo_map(dataset_id) == {}
