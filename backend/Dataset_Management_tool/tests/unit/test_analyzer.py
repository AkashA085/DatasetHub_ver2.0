import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from app.models.schemas import ImageAnnotation, BoundingBox, ValidationReport
from app.services.analyzer import DatasetAnalyzer

@pytest.fixture
def analyzer(tmp_path):
    analysis_dir = tmp_path / "analysis"
    return DatasetAnalyzer("test_session", analysis_dir)

@pytest.fixture
def sample_data():
    anns = [
        ImageAnnotation(
            image_name="img1", width=100, height=100,
            objects=[
                BoundingBox(class_id="0", xmin=10, ymin=10, xmax=50, ymax=50),
                BoundingBox(class_id="1", xmin=20, ymin=20, xmax=60, ymax=60)
            ]
        ),
        ImageAnnotation(
            image_name="img2", width=200, height=200,
            objects=[
                BoundingBox(class_id="0", xmin=5, ymin=5, xmax=15, ymax=15)
            ]
        )
    ]
    report = ValidationReport(
        total_images=2, total_labels=2, missing_labels=0, orphan_labels=0,
        empty_labels=0, corrupted_images=0, class_ids_found=["0", "1"],
        missing_label_images=[], orphan_label_files=[], empty_label_files=[],
        corrupted_image_files=[]
    )
    return anns, report

@patch("matplotlib.pyplot.savefig")
def test_analyze(mock_savefig, analyzer, sample_data):
    anns, report = sample_data
    summary = analyzer.analyze(anns, report)
    
    assert summary.total_images == 2
    assert summary.total_objects == 3
    assert summary.total_classes == 2
    assert summary.class_distribution == {"0": 2, "1": 1}
    assert summary.avg_objects_per_image == 1.5
    
    # Check that files were created
    assert (analyzer.analysis_dir / "dataset_statistics.csv").exists()
    assert mock_savefig.called

def test_analyze_empty(analyzer):
    report = ValidationReport(
        total_images=0, total_labels=0, missing_labels=0, orphan_labels=0,
        empty_labels=0, corrupted_images=0, class_ids_found=[],
        missing_label_images=[], orphan_label_files=[], empty_label_files=[],
        corrupted_image_files=[]
    )
    summary = analyzer.analyze([], report)
    assert summary.total_images == 0
    assert summary.total_objects == 0
