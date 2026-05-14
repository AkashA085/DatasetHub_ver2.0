import pytest
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

def create_zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    buf.seek(0)
    return buf

def get_fake_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

def test_upload_dataset_success(client):
    # Prepare dummy images zip
    fake_img = get_fake_image()
    images = {
        "img1.jpg": fake_img,
        "img2.jpg": fake_img
    }
    img_zip = create_zip(images)
    
    # Prepare dummy labels zip (YOLO format)
    labels = {
        "img1.txt": "0 0.5 0.5 0.2 0.2",
        "img2.txt": "1 0.3 0.3 0.1 0.1",
        "classes.txt": "drone\nperson"
    }
    lbl_zip = create_zip(labels)
    
    files = {
        "images_zip": ("images.zip", img_zip, "application/zip"),
        "labels_zip": ("labels.zip", lbl_zip, "application/zip")
    }
    data = {
        "format_type": "yolo"
    }
    
    response = client.post("/upload-dataset", data=data, files=files)
    assert response.status_code == 200
    res_data = response.json()
    assert "dataset_id" in res_data
    assert res_data["validation_report"]["total_images"] == 2
    assert res_data["analysis_summary"]["total_objects"] == 2

def test_upload_dataset_invalid_format(client):
    img_zip = create_zip({"a.jpg": get_fake_image()})
    lbl_zip = create_zip({"a.txt": b"0 0.5 0.5 0.1 0.1"})
    
    files = {
        "images_zip": ("images.zip", img_zip, "application/zip"),
        "labels_zip": ("labels.zip", lbl_zip, "application/zip")
    }
    response = client.post("/upload-dataset", data={"format_type": "invalid"}, files=files)
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]

def test_upload_dataset_no_matches(client):
    img_zip = create_zip({"img1.jpg": get_fake_image()})
    lbl_zip = create_zip({"diff.txt": b"0 0.5 0.5 0.1 0.1"})
    
    files = {
        "images_zip": ("images.zip", img_zip, "application/zip"),
        "labels_zip": ("labels.zip", lbl_zip, "application/zip")
    }
    response = client.post("/upload-dataset", data={"format_type": "yolo"}, files=files)
    assert response.status_code == 400
    assert "No matched image/label pairs found" in response.json()["detail"]
def test_upload_dataset_empty_zip(client):
    img_zip = create_zip({})
    lbl_zip = create_zip({})
    
    files = {
        "images_zip": ("images.zip", img_zip, "application/zip"),
        "labels_zip": ("labels.zip", lbl_zip, "application/zip")
    }
    response = client.post("/upload-dataset", data={"format_type": "yolo"}, files=files)
    assert response.status_code == 400
    # The message is "No matched image/label pairs..." if zips are empty but exist
    assert "no matched" in response.json()["detail"].lower()

def test_upload_dataset_corrupted_zip(client):
    files = {
        "images_zip": ("images.zip", b"not a zip", "application/zip"),
        "labels_zip": ("labels.zip", b"not a zip", "application/zip")
    }
    response = client.post("/upload-dataset", data={"format_type": "yolo"}, files=files)
    assert response.status_code == 400
    assert "invalid or corrupted" in response.json()["detail"].lower()

def test_upload_dataset_pascal_voc(client):
    # Prepare dummy images zip
    fake_img = get_fake_image()
    img_zip = create_zip({"img1.jpg": fake_img})
    
    # Prepare dummy labels zip (Pascal VOC format)
    # Important: stem must match img1
    xml_content = """<annotation>
        <filename>img1.jpg</filename>
        <size><width>100</width><height>100</height></size>
        <object><name>drone</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
    </annotation>"""
    lbl_zip = create_zip({"img1.xml": xml_content})
    
    files = {
        "images_zip": ("images.zip", img_zip, "application/zip"),
        "labels_zip": ("labels.zip", lbl_zip, "application/zip")
    }
    response = client.post("/upload-dataset", data={"format_type": "pascal_voc"}, files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["validation_report"]["total_images"] == 1
    assert data["analysis_summary"]["total_objects"] == 1

def test_upload_dataset_coco(client):
    img_zip = create_zip({"img1.jpg": get_fake_image()})
    coco_json = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "drone"}]
    }
    lbl_zip = create_zip({"annotations.json": json.dumps(coco_json)})
    
    files = {
        "images_zip": ("images.zip", img_zip, "application/zip"),
        "labels_zip": ("labels.zip", lbl_zip, "application/zip")
    }
    response = client.post("/upload-dataset", data={"format_type": "coco"}, files=files)
    assert response.status_code == 200
    assert response.json()["validation_report"]["total_images"] == 1

def test_upload_dataset_pascal_alias(client):
    # Test "pascal" alias normalization
    img_zip = create_zip({"img1.jpg": get_fake_image()})
    xml_content = """<annotation>
        <filename>img1.jpg</filename>
        <size><width>100</width><height>100</height></size>
        <object><name>drone</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
    </annotation>"""
    lbl_zip = create_zip({"img1.xml": xml_content})
    
    files = {
        "images_zip": ("images.zip", img_zip, "application/zip"),
        "labels_zip": ("labels.zip", lbl_zip, "application/zip")
    }
    response = client.post("/upload-dataset", data={"format_type": "pascal"}, files=files)
    assert response.status_code == 200

def test_upload_dataset_roboflow_success(client):
    img_zip = create_zip({"img1.jpg": get_fake_image()})
    lbl_zip = create_zip({"img1.txt": "0 0.5 0.5 0.1 0.1", "classes.txt": "drone"})
    
    files = {
        "images_zip": ("images.zip", img_zip, "application/zip"),
        "labels_zip": ("labels.zip", lbl_zip, "application/zip")
    }
    response = client.post("/upload-dataset", data={"format_type": "roboflow"}, files=files)
    assert response.status_code == 200


