from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class BoundingBox(BaseModel):
    class_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class ImageAnnotation(BaseModel):
    image_name: str
    width: int
    height: int
    objects: List[BoundingBox]

class DatasetInternalFormat(BaseModel):
    annotations: List[ImageAnnotation]

class LabelUpdateRequest(BaseModel):
    dataset_id: str
    image_name: str
    objects: List[BoundingBox]

class ValidationReport(BaseModel):
    total_images: int
    total_labels: int
    missing_labels: int
    orphan_labels: int
    empty_labels: int
    corrupted_images: int
    class_ids_found: List[int]
    missing_label_images: List[str] = Field(default_factory=list)
    orphan_label_files: List[str] = Field(default_factory=list)
    empty_label_files: List[str] = Field(default_factory=list)
    corrupted_image_files: List[str] = Field(default_factory=list)

class AnalysisSummary(BaseModel):
    total_images: int
    total_labels: int
    total_classes: int
    total_objects: int
    avg_objects_per_image: float
    class_distribution: Dict[str, int]
    missing_label_count: int
    corrupted_image_count: int

class UploadResponse(BaseModel):
    dataset_id: str
    validation_report: ValidationReport
    analysis_summary: AnalysisSummary
    csv_file_path: str
    download_url: str

class DownloadURLResponse(BaseModel):
    download_url: str

class AugmentationRequest(BaseModel):
    dataset_id: str
    export_format: str = "yolo" 
    count: int = 1  # Number of augmented versions per image
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotation: int = 0
    brightness: float = 0.0
    contrast: float = 0.0
    blur: int = 0
    noise: float = 0.0
