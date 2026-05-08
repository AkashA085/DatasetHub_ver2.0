from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from app.models.schemas import AugmentationRequest, UploadResponse, ImageAnnotation, ValidationReport
from app.services.augmentation import AugmentationService
from app.services.export_service import ExportService
from app.services.validator import DatasetValidator
from app.services.analyzer import DatasetAnalyzer
from app.utils.file_utils import PROCESSED_DIR, UPLOADS_DIR, ANALYSIS_DIR, EXPORTS_DIR
import json
import shutil
from pathlib import Path

router = APIRouter()

@router.post("/augment", response_model=UploadResponse)
async def augment_dataset(request: AugmentationRequest, req: Request):
    """
    Apply augmentations to an existing dataset.
    - Loads existing internal annotations.
    - Applies transformations using Albumentations.
    - Saves new images and updates annotations.
    - Re-exports the dataset.
    """
    session_id = request.dataset_id
    session_processed_dir = PROCESSED_DIR / session_id
    internal_format_path = session_processed_dir / "annotations.json"
    
    if not internal_format_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload first.")
        
    # Load existing annotations
    with open(internal_format_path, "r") as f:
        annotations_raw = json.load(f)
        
    # Convert to objects (reconstruct from dicts)
    annotations = [ImageAnnotation(**ann) for ann in annotations_raw]
    
    # Reconstruct stem_to_image mapping
    session_upload_dir = UPLOADS_DIR / session_id
    extract_images_dir = session_upload_dir / "images"
    aug_images_dir = session_processed_dir / "augmented_images"
    
    stem_to_image = {}
    
    # Re-scan for original images
    if extract_images_dir.exists():
        # Use rglob to find images recursively in subdirectories (e.g., images/images)
        for file_path in extract_images_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Store lowercase stem for case-insensitive matching
                stem_to_image[file_path.stem.lower()] = file_path
                
    # Scan for already augmented images (if applying augmentations sequentially)
    if aug_images_dir.exists():
        for file_path in aug_images_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                stem_to_image[file_path.stem.lower()] = file_path

    # Apply Augmentations
    new_anns, new_stem_map = AugmentationService.augment_dataset(request, annotations, stem_to_image)
    
    # Merge annotations
    all_annotations = annotations + new_anns
    stem_to_image.update(new_stem_map)
    
    # Save updated annotations
    with open(internal_format_path, "w") as f:
        json.dump([ann.dict() for ann in all_annotations], f, indent=4)
        
    # Re-Analyze
    session_analysis_dir = ANALYSIS_DIR / session_id
    analyzer = DatasetAnalyzer(session_id, session_analysis_dir)
    
    # Create a simplified report for analysis (re-use class finding logic or simple set)
    class_counts = set()
    for ann in all_annotations:
        for obj in ann.objects:
            class_counts.add(obj.class_id)
            
    report = ValidationReport(
        total_images=len(all_annotations),
        total_labels=len(all_annotations),
        missing_labels=0,
        orphan_labels=0,
        empty_labels=0,
        corrupted_images=0,
        class_ids_found=list(class_counts),
        missing_label_images=[],
        orphan_label_files=[],
        empty_label_files=[],
        corrupted_image_files=[]
    )
    
    summary = analyzer.analyze(all_annotations, report)
    
    # Re-Export
    # Use the requested format or default to yolo
    target_format = request.export_format
    
    # Attempt to recover class names from original classes.txt
    class_names = {}
    classes_file = session_upload_dir / "labels" / "classes.txt"
    if classes_file.exists():
        with open(classes_file, "r") as f:
            lines = f.readlines()
            for idx, name in enumerate(lines):
                class_names[idx] = name.strip()
    
    zip_path, total_copied_images = ExportService.export_dataset(
        session_id, all_annotations, report, class_names, stem_to_image, target_format
    )
    
    base_url = str(req.base_url).rstrip("/")
    download_url = f"{base_url}/download/{session_id}"

    return UploadResponse(
        dataset_id=session_id,
        validation_report=report,
        analysis_summary=summary,
        csv_file_path=str(session_analysis_dir / "dataset_statistics.csv"),
        download_url=download_url
    )
