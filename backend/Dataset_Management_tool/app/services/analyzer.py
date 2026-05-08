import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from app.models.schemas import ImageAnnotation, AnalysisSummary, ValidationReport

class DatasetAnalyzer:
    def __init__(self, session_id: str, analysis_dir: Path):
        self.session_id = session_id
        self.analysis_dir = analysis_dir
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, annotations: List[ImageAnnotation], validation_report: ValidationReport) -> AnalysisSummary:
        """Analyze annotations and generate graphs and CSV."""
        
        # total_images must reflect the count of images used in analysis (matched pairs)
        dataset_size = len(annotations)
        
        if dataset_size == 0:
            return self._empty_summary(validation_report)

        obj_data = []
        img_data = []
        class_counts = {}

        for ann in annotations:
            num_objs = len(ann.objects)
            img_data.append({
                "image_name": ann.image_name,
                "width": ann.width,
                "height": ann.height,
                "num_objects": num_objs
            })
            for obj in ann.objects:
                area = (obj.xmax - obj.xmin) * (obj.ymax - obj.ymin)
                obj_data.append({
                    "class_id": obj.class_id,
                    "area": area
                })
                cls_key = str(obj.class_id)
                class_counts[cls_key] = class_counts.get(cls_key, 0) + 1

        df_img = pd.DataFrame(img_data)
        df_obj = pd.DataFrame(obj_data)

        # Generate Graphs
        self._generate_graphs(df_img, df_obj)

        total_objects = len(df_obj)
        avg_objs = total_objects / dataset_size

        summary = AnalysisSummary(
            total_images=dataset_size,
            total_labels=dataset_size, # Number of label files processed (matched)
            total_classes=len(class_counts),
            total_objects=total_objects,
            avg_objects_per_image=float(round(avg_objs, 2)),
            class_distribution=class_counts,
            missing_label_count=validation_report.missing_labels,
            corrupted_image_count=validation_report.corrupted_images
        )

        # Save CSV
        self._save_csv(summary)

        return summary

    def _generate_graphs(self, df_img, df_obj):
        try:
            # Class distribution graph
            plt.figure(figsize=(10, 6))
            df_obj['class_id'].value_counts().sort_index().plot(kind='bar')
            plt.title('Class Distribution')
            plt.xlabel('Class ID')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "class_distribution.png")
            plt.close()

            # Objects per image histogram
            plt.figure(figsize=(10, 6))
            df_img['num_objects'].hist(bins=range(0, int(df_img['num_objects'].max() + 2)))
            plt.title('Objects per Image')
            plt.xlabel('Number of Objects')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "objects_per_image.png")
            plt.close()

            # Bounding box area distribution
            plt.figure(figsize=(10, 6))
            df_obj['area'].hist(bins=50)
            plt.title('Bounding Box Area Distribution')
            plt.xlabel('Area (pixels)')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "bbox_area_distribution.png")
            plt.close()

            # Image resolution distribution
            plt.figure(figsize=(10, 6))
            plt.scatter(df_img['width'], df_img['height'], alpha=0.5)
            plt.title('Image Resolutions')
            plt.xlabel('Width')
            plt.ylabel('Height')
            plt.tight_layout()
            plt.savefig(self.analysis_dir / "resolution_distribution.png")
            plt.close()
        except Exception:
            # Skip graphing if error (e.g. empty data)
            plt.close('all')

    def _save_csv(self, summary: AnalysisSummary):
        stats = {
            "total_images": summary.total_images,
            "total_labels": summary.total_labels,
            "total_classes": summary.total_classes,
            "total_objects": summary.total_objects,
            "avg_objects_per_image": summary.avg_objects_per_image,
            "missing_label_count": summary.missing_label_count,
            "corrupted_image_count": summary.corrupted_image_count,
            "class_distribution": str(summary.class_distribution)
        }
        df = pd.DataFrame([stats])
        df.to_csv(self.analysis_dir / "dataset_statistics.csv", index=False)

    def _empty_summary(self, validation_report: ValidationReport) -> AnalysisSummary:
        return AnalysisSummary(
            total_images=0,
            total_labels=0,
            total_classes=0,
            total_objects=0,
            avg_objects_per_image=0.0,
            class_distribution={},
            missing_label_count=validation_report.missing_labels,
            corrupted_image_count=validation_report.corrupted_images
        )
