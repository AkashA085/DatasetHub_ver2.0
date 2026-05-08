import json
import yaml
import os
from pathlib import Path
from typing import List, Dict
import xml.etree.ElementTree as ET
from app.models.schemas import ImageAnnotation

class FormatConverter:
    @staticmethod
    def to_yolo(annotations: List[ImageAnnotation], output_dir: Path, class_ids: List[int]):
        """Convert internal format to YOLO txt files."""
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        for ann in annotations:
            label_file = labels_dir / f"{ann.image_name}.txt"
            with open(label_file, 'w') as f:
                for obj in ann.objects:
                    x_center = (obj.xmin + obj.xmax) / 2 / ann.width
                    y_center = (obj.ymin + obj.ymax) / 2 / ann.height
                    w = (obj.xmax - obj.xmin) / ann.width
                    h = (obj.ymax - obj.ymin) / ann.height
                    f.write(f"{obj.class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    @staticmethod
    def to_coco(annotations: List[ImageAnnotation], output_dir: Path, class_ids: List[int]) -> Dict:
        """Convert internal format to COCO JSON structure."""
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": int(i), "name": f"class_{i}"} for i in class_ids]
        }

        ann_id = 1
        for i, ann in enumerate(annotations):
            img_id = i + 1
            coco["images"].append({
                "id": img_id,
                "file_name": ann.image_name, 
                "width": ann.width,
                "height": ann.height
            })

            for obj in ann.objects:
                w = obj.xmax - obj.xmin
                h = obj.ymax - obj.ymin
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(obj.class_id),
                    "bbox": [float(obj.xmin), float(obj.ymin), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0
                })
                ann_id += 1
        
        return coco

    @staticmethod
    def to_pascal_voc(annotations: List[ImageAnnotation], output_dir: Path):
        """
        Convert internal format to Pascal VOC XML files.
        Saves all XMLs into output_dir / 'Annotations'
        """
        ann_dir = output_dir / "Annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)

        for ann in annotations:
            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = "JPEGImages"
            ET.SubElement(root, "filename").text = ann.image_name
            
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(ann.width)
            ET.SubElement(size, "height").text = str(ann.height)
            ET.SubElement(size, "depth").text = "3"

            # Even if objects list is empty, we produce the XML root and size
            for obj in ann.objects:
                obj_elem = ET.SubElement(root, "object")
                ET.SubElement(obj_elem, "name").text = f"class_{obj.class_id}"
                ET.SubElement(obj_elem, "pose").text = "Unspecified"
                ET.SubElement(obj_elem, "truncated").text = "0"
                ET.SubElement(obj_elem, "difficult").text = "0"
                
                bndbox = ET.SubElement(obj_elem, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(obj.xmin))
                ET.SubElement(bndbox, "ymin").text = str(int(obj.ymin))
                ET.SubElement(bndbox, "xmax").text = str(int(obj.xmax))
                ET.SubElement(bndbox, "ymax").text = str(int(obj.ymax))

            # Proper XML declaration as requested
            tree = ET.ElementTree(root)
            xml_path = ann_dir / f"{ann.image_name}.xml"
            
            # Using custom writing to ensure UTF-8 and header
            with open(xml_path, "wb") as f:
                f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
                tree.write(f, encoding='utf-8', xml_declaration=False)

    @staticmethod
    def generate_voc_imagesets(output_dir: Path, splits: Dict[str, List[ImageAnnotation]]):
        """Generate ImageSets/Main/*.txt files for Pascal VOC."""
        imagesets_dir = output_dir / "ImageSets" / "Main"
        imagesets_dir.mkdir(parents=True, exist_ok=True)

        for split_name, annotations in splits.items():
            # filename stem without extension as requested
            stems = [ann.image_name for ann in annotations]
            txt_path = imagesets_dir / f"{split_name}.txt"
            with open(txt_path, 'w') as f:
                f.write("\n".join(stems))

    @staticmethod
    def generate_data_yaml(output_path: Path, class_ids: List[int], train_path: str, val_path: str, test_path: str = None, custom_names: List[str] = None):
        """Generate YOLO data.yaml file with specific formatting."""
        nc = len(class_ids)
        if custom_names and len(custom_names) > 0:
            names = custom_names
        else:
            names = [f"class_{i}" for i in class_ids]

        with open(output_path, 'w') as f:
            f.write(f"train: {train_path}\n")
            f.write(f"val: {val_path}\n")
            if test_path:
                f.write(f"test: {test_path}\n")
            f.write(f"nc: {nc}\n")
            names_str = ", ".join([f"'{name}'" for name in names])
            f.write(f"names: [{names_str}]\n")

    @staticmethod
    def generate_roboflow_metadata(output_dir: Path, dataset_name: str):
        """Generate Roboflow-style metadata files."""
        with open(output_dir / "README.roboflow.txt", 'w') as f:
            f.write(f"Dataset Name: {dataset_name}\n")
            f.write("Exported via Dataset Management Backend.\n")
        
        metadata = {"name": dataset_name, "version": 1, "format": "yolov8"}
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
