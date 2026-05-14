import pytest
import json
from pathlib import Path
from app.services.export_service import ExportService
from app.models.schemas import ImageAnnotation, BoundingBox, ValidationReport

@pytest.fixture
def sample_annotations():
    return [
        ImageAnnotation(image_name="test1", width=640, height=480, objects=[
            BoundingBox(class_id=0, xmin=100, ymin=100, xmax=200, ymax=200)
        ]),
        ImageAnnotation(image_name="test2", width=640, height=480, objects=[
            BoundingBox(class_id=0, xmin=100, ymin=100, xmax=200, ymax=200)
        ]),
        ImageAnnotation(image_name="test3", width=640, height=480, objects=[
            BoundingBox(class_id=0, xmin=100, ymin=100, xmax=200, ymax=200)
        ])
    ]

@pytest.fixture
def sample_report():
    return ValidationReport(
        total_images=3, total_labels=3, missing_labels=0, 
        orphan_labels=0, empty_labels=0, corrupted_images=0, 
        class_ids_found=[0]
    )

def test_export_yolo(tmp_path, sample_annotations, sample_report):
    session_id = "test_yolo_unique"
    class_names = {0: "drone"}
    stem_to_image = {
        "test1": tmp_path / "test1.jpg",
        "test2": tmp_path / "test2.jpg",
        "test3": tmp_path / "test3.jpg"
    }
    for p in stem_to_image.values(): p.touch()
    
    zip_path, count = ExportService.export_dataset(
        session_id, sample_annotations, sample_report, class_names, stem_to_image, "yolo"
    )
    assert zip_path.exists()
    assert count > 0

def test_export_coco(tmp_path, sample_annotations, sample_report):
    session_id = "test_coco_unique"
    class_names = {0: "drone"}
    stem_to_image = {
        "test1": tmp_path / "test1.jpg",
        "test2": tmp_path / "test2.jpg",
        "test3": tmp_path / "test3.jpg"
    }
    for p in stem_to_image.values(): p.touch()
    
    zip_path, count = ExportService.export_dataset(
        session_id, sample_annotations, sample_report, class_names, stem_to_image, "coco"
    )
    assert zip_path.exists()
    assert count > 0

def test_export_pascal_voc(tmp_path, sample_annotations, sample_report):
    session_id = "test_pascal_unique"
    class_names = {0: "drone"}
    stem_to_image = {
        "test1": tmp_path / "test1.jpg",
        "test2": tmp_path / "test2.jpg",
        "test3": tmp_path / "test3.jpg"
    }
    for p in stem_to_image.values(): p.touch()
    
    zip_path, count = ExportService.export_dataset(
        session_id, sample_annotations, sample_report, class_names, stem_to_image, "pascal_voc"
    )
    assert zip_path.exists()
    assert count > 0
