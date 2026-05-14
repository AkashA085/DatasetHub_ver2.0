import pytest
from app.models.schemas import ImageAnnotation, BoundingBox
from app.services.splitter import DatasetSplitter

def test_split_basic():
    # 10 labeled images
    anns = [
        ImageAnnotation(
            image_name=f"img_{i}", width=100, height=100, 
            objects=[BoundingBox(class_id="0", xmin=0, ymin=0, xmax=10, ymax=10)]
        ) for i in range(10)
    ]
    
    train, val, test = DatasetSplitter.split(anns, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    
    assert len(train) == 7
    assert len(val) == 2
    assert len(test) == 1
    # Check deterministic seed
    train2, _, _ = DatasetSplitter.split(anns, seed=42)
    assert train[0].image_name == train2[0].image_name

def test_split_filters_empty_labels():
    anns = [
        ImageAnnotation(image_name="labeled_1", width=100, height=100, objects=[BoundingBox(class_id="0", xmin=0, ymin=0, xmax=10, ymax=10)]),
        ImageAnnotation(image_name="labeled_2", width=100, height=100, objects=[BoundingBox(class_id="0", xmin=0, ymin=0, xmax=10, ymax=10)]),
        ImageAnnotation(image_name="labeled_3", width=100, height=100, objects=[BoundingBox(class_id="0", xmin=0, ymin=0, xmax=10, ymax=10)]),
        ImageAnnotation(image_name="labeled_4", width=100, height=100, objects=[BoundingBox(class_id="0", xmin=0, ymin=0, xmax=10, ymax=10)]),
        ImageAnnotation(image_name="empty", width=100, height=100, objects=[])
    ]
    
    train, val, test = DatasetSplitter.split(anns, train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)
    # Only "labeled" should be in the split (4 items total)
    assert len(train) + len(val) + len(test) == 4
    assert len(train) == 2
    assert len(val) == 2
