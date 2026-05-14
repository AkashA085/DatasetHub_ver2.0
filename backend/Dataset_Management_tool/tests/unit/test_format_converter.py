import pytest
from pathlib import Path
from app.services.format_converter import FormatConverter
from app.models.schemas import ImageAnnotation, BoundingBox

def test_to_coco(tmp_path):
    anns = [
        ImageAnnotation(image_name="img1", width=100, height=100, objects=[
            BoundingBox(class_id=1, xmin=10, ymin=20, xmax=50, ymax=60)
        ])
    ]
    class_ids = [1]
    coco = FormatConverter.to_coco(anns, tmp_path, class_ids)
    
    assert len(coco["images"]) == 1
    assert coco["images"][0]["file_name"] == "img1"
    assert len(coco["annotations"]) == 1
    # COCO bbox: [x, y, width, height]
    assert coco["annotations"][0]["bbox"] == [10.0, 20.0, 40.0, 40.0]
    assert coco["categories"][0]["id"] == 1

def test_to_pascal_voc(tmp_path):
    anns = [
        ImageAnnotation(image_name="img1", width=100, height=100, objects=[
            BoundingBox(class_id=0, xmin=10, ymin=20, xmax=50, ymax=60)
        ])
    ]
    FormatConverter.to_pascal_voc(anns, tmp_path)
    
    xml_path = tmp_path / "Annotations" / "img1.xml"
    assert xml_path.exists()
    xml_content = xml_path.read_text()
    assert "<filename>img1</filename>" in xml_content
    assert "<xmin>10</xmin>" in xml_content
    assert "<xmax>50</xmax>" in xml_content
    assert "<name>class_0</name>" in xml_content
