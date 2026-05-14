import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
from app.services.augmentation import AugmentationService
from app.models.schemas import AugmentationRequest, ImageAnnotation, BoundingBox

@pytest.fixture
def sample_ann():
    return ImageAnnotation(
        image_name="test_img",
        width=100,
        height=100,
        objects=[BoundingBox(class_id=0, xmin=10, ymin=10, xmax=50, ymax=50)]
    )

@pytest.fixture
def mock_img(tmp_path):
    import cv2
    img_path = tmp_path / "test_img.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path

def test_augment_dataset_basic(sample_ann, mock_img, tmp_path):
    request = AugmentationRequest(
        dataset_id="test_ds",
        horizontal_flip=True,
        count=1
    )
    stem_to_image = {"test_img": mock_img}
    
    with patch("app.services.augmentation.PROCESSED_DIR", tmp_path):
        aug_anns, aug_stems = AugmentationService.augment_dataset(request, [sample_ann], stem_to_image)
    
    assert len(aug_anns) == 1
    assert len(aug_stems) == 1
    assert "test_img_aug_" in aug_anns[0].image_name

def test_augment_dataset_all_transforms(sample_ann, mock_img, tmp_path):
    request = AugmentationRequest(
        dataset_id="test_ds",
        horizontal_flip=True,
        vertical_flip=True,
        rotation=90,
        blur=5,
        brightness=0.2,
        contrast=0.2,
        noise=0.1,
        count=1
    )
    stem_to_image = {"test_img": mock_img}
    
    with patch("app.services.augmentation.PROCESSED_DIR", tmp_path):
        aug_anns, aug_stems = AugmentationService.augment_dataset(request, [sample_ann], stem_to_image)
    
    assert len(aug_anns) == 1

def test_augment_dataset_missing_image(sample_ann, tmp_path):
    request = AugmentationRequest(dataset_id="test_ds", count=1)
    stem_to_image = {"test_img": Path("/non/existent.jpg")}
    
    aug_anns, aug_stems = AugmentationService.augment_dataset(request, [sample_ann], stem_to_image)
    assert len(aug_anns) == 0

def test_augment_dataset_invalid_image(sample_ann, mock_img, tmp_path):
    request = AugmentationRequest(dataset_id="test_ds", count=1)
    stem_to_image = {"test_img": mock_img}
    
    with patch("cv2.imread", return_value=None):
        aug_anns, aug_stems = AugmentationService.augment_dataset(request, [sample_ann], stem_to_image)
    assert len(aug_anns) == 0

def test_augment_dataset_invalid_bbox(sample_ann, mock_img, tmp_path):
    # Bbox with xmax <= xmin
    sample_ann.objects[0].xmax = 5
    request = AugmentationRequest(dataset_id="test_ds", count=1)
    stem_to_image = {"test_img": mock_img}
    
    with patch("app.services.augmentation.PROCESSED_DIR", tmp_path):
        aug_anns, aug_stems = AugmentationService.augment_dataset(request, [sample_ann], stem_to_image)
    
    # Bbox should be filtered out, so aug_anns[0].objects should be empty
    assert len(aug_anns) == 1
    assert len(aug_anns[0].objects) == 0

def test_augment_dataset_exception(sample_ann, mock_img, tmp_path):
    request = AugmentationRequest(dataset_id="test_ds", count=1)
    stem_to_image = {"test_img": mock_img}
    
    # Patch compose call instead of constructor
    with patch("app.services.augmentation.A.Compose") as mock_compose:
        mock_compose.return_value.side_effect = Exception("Aug error")
        with patch("app.services.augmentation.PROCESSED_DIR", tmp_path):
            aug_anns, aug_stems = AugmentationService.augment_dataset(request, [sample_ann], stem_to_image)
    
    assert len(aug_anns) == 0

def test_augment_dataset_invalid_labels(sample_ann, mock_img, tmp_path):
    request = AugmentationRequest(dataset_id="test_ds", count=1)
    stem_to_image = {"test_img": mock_img}
    
    with patch("app.services.augmentation.A.Compose") as mock_compose:
        # Return different number of bboxes and labels
        mock_compose.return_value.return_value = {
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "bboxes": [[10, 10, 50, 50]],
            "class_labels": [] # Mismatch
        }
        with patch("app.services.augmentation.PROCESSED_DIR", tmp_path):
            aug_anns, aug_stems = AugmentationService.augment_dataset(request, [sample_ann], stem_to_image)
    
    assert len(aug_anns) == 0

