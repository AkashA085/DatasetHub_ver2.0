from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Depends
from app.models.schemas import AugmentationRequest, UploadResponse, ImageAnnotation, ValidationReport
from app.services.augmentation import AugmentationService
from app.services.export_service import ExportService
from app.services.validator import DatasetValidator
from app.services.analyzer import DatasetAnalyzer
from app.utils.file_utils import PROCESSED_DIR, UPLOADS_DIR, ANALYSIS_DIR, EXPORTS_DIR
import json
import shutil
from pathlib import Path
from sqlalchemy.orm import Session
import uuid
from app.core.database import get_db, Dataset, DatasetValidation, ClassDistribution, Image, Label

router = APIRouter()

@router.post("/augment", response_model=UploadResponse)
async def augment_dataset(request: AugmentationRequest, req: Request, db: Session = Depends(get_db)):
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

    # If this dataset exists in the DB, update its aggregated stats and validation/class distribution
    try:
        db_dataset = db.query(Dataset).filter(Dataset.id == session_id).first()
        if db_dataset:
            # Use analyzer `summary` and validator `report` so DB mirrors the API response exactly
            db_dataset.total_images = summary.total_images
            db_dataset.total_labels = summary.total_labels
            db_dataset.total_classes = summary.total_classes
            db_dataset.total_objects = summary.total_objects
            db_dataset.avg_objects_per_image = summary.avg_objects_per_image
            db_dataset.csv_file_path = str(session_analysis_dir / "dataset_statistics.csv")
            db_dataset.zip_file_path = str(zip_path) if zip_path else db_dataset.zip_file_path
            db_dataset.missing_label_count = summary.missing_label_count
            db_dataset.corrupted_image_count = summary.corrupted_image_count
            # persist full analysis summary JSON
            try:
                db_dataset.analysis_summary = summary.__dict__ if hasattr(summary, '__dict__') else None
            except Exception:
                db_dataset.analysis_summary = None

            # Update or create dataset validation row using the `report` returned in the response
            val = db.query(DatasetValidation).filter(DatasetValidation.dataset_id == session_id).first()
            if not val:
                val = DatasetValidation(dataset_id=session_id)
                db.add(val)

            val.total_images = report.total_images
            val.total_labels = report.total_labels
            val.missing_labels = report.missing_labels
            val.orphan_labels = report.orphan_labels if hasattr(report, 'orphan_labels') else 0
            val.empty_labels = report.empty_labels if hasattr(report, 'empty_labels') else 0
            val.corrupted_images = report.corrupted_images            # Persist ValidationReport lists and class ids
            try:
                val.class_ids_found = report.class_ids_found if hasattr(report, 'class_ids_found') else None
            except Exception:
                val.class_ids_found = None
            val.missing_label_images = report.missing_label_images if hasattr(report, 'missing_label_images') else None
            val.orphan_label_files = report.orphan_label_files if hasattr(report, 'orphan_label_files') else None
            val.empty_label_files = report.empty_label_files if hasattr(report, 'empty_label_files') else None
            val.corrupted_image_files = report.corrupted_image_files if hasattr(report, 'corrupted_image_files') else None
            # Rebuild class distributions from `summary.class_distribution`
            db.query(ClassDistribution).filter(ClassDistribution.dataset_id == session_id).delete()
            for cls_id, cnt in summary.class_distribution.items():
                db.add(ClassDistribution(dataset_id=session_id, class_id=str(cls_id), object_count=int(cnt)))

            # --- Persist image & label rows for any newly-added (augmented) images ---
            # Build existing-image stem map to avoid duplicates
            existing_images = { (img.file_name.rsplit('.', 1)[0]).lower(): img for img in db.query(Image).filter(Image.dataset_id == session_id).all() }

            # Insert missing Image rows (look up file path from stem_to_image or processed aug folder)
            for ann in all_annotations:
                stem = ann.image_name.lower()
                if stem in existing_images:
                    continue

                img_path = stem_to_image.get(stem)
                if not img_path:
                    candidate = (session_processed_dir / "augmented_images" / (ann.image_name + ".jpg"))
                    if candidate.exists():
                        img_path = candidate

                if not img_path:
                    # if we still can't find a file, skip DB image insertion for this ann
                    continue

                img_row = Image(
                    id=str(uuid.uuid4()),
                    dataset_id=session_id,
                    file_name=img_path.name,
                    file_path=str(img_path),
                    has_label=(len(ann.objects) > 0),
                )
                db.add(img_row)
                existing_images[stem] = img_row
            db.commit()

            # Insert Label rows for images that do not already have labels in DB
            for ann in all_annotations:
                stem = ann.image_name.lower()
                img_row = existing_images.get(stem)
                if not img_row:
                    continue

                # If labels already exist for this image, skip (prevents duplicates)
                existing_label_count = db.query(Label).filter(Label.image_id == img_row.id).count()
                if existing_label_count > 0:
                    continue

                for obj in ann.objects:
                    bbox_json = {"pascal_voc": [obj.xmin, obj.ymin, obj.xmax, obj.ymax]}
                    db.add(Label(id=str(uuid.uuid4()), image_id=img_row.id, class_id=str(obj.class_id), bbox_data=bbox_json))
            db.commit()
    except Exception as e:
        # don't block response on DB errors; log and continue
        try:
            print(f"Failed to update DB after augmentation for {session_id}: {e}")
        except Exception:
            pass

    return UploadResponse(
        dataset_id=session_id,
        validation_report=report,
        analysis_summary=summary,
        csv_file_path=str(session_analysis_dir / "dataset_statistics.csv"),
        download_url=download_url
    )
