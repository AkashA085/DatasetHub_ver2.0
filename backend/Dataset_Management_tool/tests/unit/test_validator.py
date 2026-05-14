import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from app.services.validator import DatasetValidator

@pytest.fixture
def validator(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    return DatasetValidator(images_dir, labels_dir)

def test_normalize_stem(validator):
    assert validator._normalize_stem(Path("image_abc123456789.jpg")) == "image"
    assert validator._normalize_stem(Path("normal_image.png")) == "normal_image"

def test_alt_match_key(validator):
    assert validator._alt_match_key("fwd_drone_A_01") == "a_1"
    assert validator._alt_match_key("simple") == "simple"

@patch("PIL.Image.open")
def test_validate_matched_pair(mock_pil_open, validator, tmp_path):
    # Setup files
    img_path = validator.images_dir / "test.jpg"
    img_path.touch()
    
    lbl_path = validator.labels_dir / "test.txt"
    # YOLO: class x_center y_center width height
    lbl_path.write_text("0 0.5 0.5 0.2 0.2")
    
    # Mock PIL image size
    mock_img = MagicMock()
    mock_img.size = (1000, 1000)
    mock_pil_open.return_value.__enter__.return_value = mock_img
    
    report, anns, img_map, lbl_map, classes = validator.validate()
    
    assert report.total_images == 1
    assert report.total_labels == 1
    assert len(anns) == 1
    assert anns[0].image_name == "test"
    assert len(anns[0].objects) == 1
    assert anns[0].objects[0].class_id == 0
    # check conversion to absolute pixels
    # xmin = (0.5 - 0.2/2) * 1000 = 0.4 * 1000 = 400
    assert anns[0].objects[0].xmin == 400

def test_validate_missing_label(validator):
    img_path = validator.images_dir / "missing.jpg"
    img_path.touch()
    
    report, _, _, _, _ = validator.validate()
    assert report.total_images == 1
    assert report.missing_labels == 1
    assert "missing.jpg" in report.missing_label_images

def test_validate_orphan_label(validator):
    lbl_path = validator.labels_dir / "orphan.txt"
    lbl_path.touch()
    
    report, _, _, _, _ = validator.validate()
    assert report.total_labels == 1
    assert report.orphan_labels == 1
    assert "orphan.txt" in report.orphan_label_files

def test_validate_alt_matching(validator):
    # A_01.jpg vs labels_01.txt should match via "a_1" or "1" logic
    # Wait, the logic is: _alt_match_key("A_01") -> "a_1", _alt_match_key("labels_01") -> "1"
    # Actually, let's use something that matches exactly.
    img_path = validator.images_dir / "fwd_drone_A_01.jpg"
    img_path.touch()
    lbl_path = validator.labels_dir / "labels_A_01.txt"
    lbl_path.write_text("0 0.5 0.5 0.1 0.1")
    
    # Mock PIL
    with patch("PIL.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_open.return_value.__enter__.return_value = mock_img
        report, anns, _, _, _ = validator.validate()
    
    # Both stems give alt_key "a_1"
    assert report.total_images == 1
    assert report.total_labels == 1
    assert report.missing_labels == 0
    assert len(anns) == 1

def test_validate_uppercase_extension(validator):
    img_path = validator.images_dir / "TEST.JPG"
    img_path.touch()
    report, _, _, _, _ = validator.validate()
    assert report.total_images == 1

def test_validate_empty_label(validator):
    img_path = validator.images_dir / "empty.jpg"
    img_path.touch()
    lbl_path = validator.labels_dir / "empty.txt"
    lbl_path.touch() # Empty file
    
    with patch("PIL.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_open.return_value.__enter__.return_value = mock_img
        report, _, _, _, _ = validator.validate()
    
    assert report.total_images == 1
    assert report.empty_labels == 1
    assert "empty.txt" in report.empty_label_files

def test_validate_corrupted_image(validator):
    img_path = validator.images_dir / "corrupted.jpg"
    img_path.touch()
    
    with patch("PIL.Image.open", side_effect=Exception("Corrupted")):
        report, _, _, _, _ = validator.validate()
    
    assert report.total_images == 1
    assert report.corrupted_images == 1
    assert "corrupted.jpg" in report.corrupted_image_files



