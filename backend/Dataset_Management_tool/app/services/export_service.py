from pathlib import Path
import shutil
import json
from app.utils.file_utils import EXPORTS_DIR, create_zip_archive
from app.services.format_converter import FormatConverter
from app.services.splitter import DatasetSplitter
from app.models.schemas import ValidationReport, ImageAnnotation

class ExportService:
    @staticmethod
    def export_dataset(session_id: str, annotations: list[ImageAnnotation], report: ValidationReport, class_names: dict, stem_to_image: dict, format_type: str):
        """
        Exports the dataset to the specified format and saves it as a ZIP file.
        """
        session_export_dir = EXPORTS_DIR / session_id
        dataset_folder_name = f"dataset_{session_id}"
        export_output_dir = session_export_dir / dataset_folder_name
        
        # Clean up existing export if any (for re-exporting after augmentation)
        if export_output_dir.exists():
            shutil.rmtree(export_output_dir)
        export_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Perform split
        train_anns, val_anns, test_anns = DatasetSplitter.split(annotations)
        
        splits = {
            "train": train_anns,
            "valid": val_anns,
            "test": test_anns
        }

        total_copied_images = 0

        if format_type in ["yolo", "roboflow"]:
            for split_name, split_anns in splits.items():
                if not split_anns: continue
                
                split_dir = export_output_dir / split_name
                images_out = split_dir / "images"
                labels_out = split_dir / "labels"
                images_out.mkdir(parents=True, exist_ok=True)
                labels_out.mkdir(parents=True, exist_ok=True)

                # 1. Convert labels
                FormatConverter.to_yolo(split_anns, split_dir, report.class_ids_found)
                
                # 2. Physically copy images
                for ann in split_anns:
                    stem = ann.image_name
                    src_img = stem_to_image.get(stem)
                    # If src_img is a Path object, check existence. 
                    # Note: stem_to_image might contain original paths or augmented paths
                    if src_img and Path(src_img).exists():
                        shutil.copy2(src_img, images_out / Path(src_img).name)
                        total_copied_images += 1
                    else:
                        print(f"Warning: Image not found for {stem}: {src_img}")

            FormatConverter.generate_data_yaml(
                export_output_dir / "data.yaml", 
                report.class_ids_found, 
                "train/images", "valid/images", "test/images",
                custom_names=class_names
            )
            if format_type == "roboflow":
                FormatConverter.generate_roboflow_metadata(export_output_dir, "Custom Dataset")
                
        elif format_type == "coco":
            for split_name, split_anns in splits.items():
                if not split_anns: continue
                
                split_dir = export_output_dir / split_name
                images_out = split_dir / "images"
                images_out.mkdir(parents=True, exist_ok=True)
                
                # 1. Generate COCO JSON
                coco_data = FormatConverter.to_coco(split_anns, split_dir, report.class_ids_found)
                with open(split_dir / "annotations.json", "w") as f:
                    json.dump(coco_data, f, indent=4)
                
                # 2. Physically copy images
                for ann in split_anns:
                    stem = ann.image_name
                    src_img = stem_to_image.get(stem)
                    if src_img and Path(src_img).exists():
                        shutil.copy2(src_img, images_out / Path(src_img).name)
                        total_copied_images += 1

        elif format_type == "pascal_voc":
            # Structure: JPEGImages/, Annotations/, ImageSets/Main/
            images_out = export_output_dir / "JPEGImages"
            images_out.mkdir(parents=True, exist_ok=True)
            
            # 1. Convert all matched pairs to XMLs
            FormatConverter.to_pascal_voc(annotations, export_output_dir)
            
            # 2. Generate ImageSets/Main
            FormatConverter.generate_voc_imagesets(export_output_dir, splits)
            
            # 3. Physically copy all matched images to JPEGImages/
            for ann in annotations:
                stem = ann.image_name
                src_img = stem_to_image.get(stem)
                if src_img and Path(src_img).exists():
                    shutil.copy2(src_img, images_out / Path(src_img).name)
                    total_copied_images += 1

        # Zip the root folder
        zip_path = session_export_dir / f"{session_id}.zip"
        # Ensure older zip is removed before creating new one
        if zip_path.exists():
            zip_path.unlink()
            
        create_zip_archive(session_export_dir, dataset_folder_name, zip_path)
        
        return zip_path, total_copied_images
