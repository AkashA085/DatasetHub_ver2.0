import os
import shutil
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends
from app.utils.file_utils import generate_session_id, UPLOADS_DIR, PROCESSED_DIR, ANALYSIS_DIR, EXPORTS_DIR, extract_zip, STORAGE_ROOT
from app.services.validator import DatasetValidator
from app.services.analyzer import DatasetAnalyzer
from app.services.export_service import ExportService
from app.models.schemas import UploadResponse, DatasetInternalFormat, DownloadURLResponse

# DB / persistence
from app.db.session import get_db
from app.db import models as db_models
from sqlalchemy.orm import Session
import uuid

router = APIRouter()

@router.post("/upload-dataset", response_model=UploadResponse)
async def upload_dataset(
    request: Request,
    images_zip: UploadFile = File(...),
    labels_zip: UploadFile = File(...),
    format_type: str = Form(...),
    db: Session = Depends(get_db)
):
    session_id = generate_session_id()
    
    # Paths for this session
    session_upload_dir = UPLOADS_DIR / session_id
    session_processed_dir = PROCESSED_DIR / session_id
    session_analysis_dir = ANALYSIS_DIR / session_id
    session_export_dir = EXPORTS_DIR / session_id
    
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    session_processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded zips
    images_zip_path = session_upload_dir / "images.zip"
    labels_zip_path = session_upload_dir / "labels.zip"
    
    with open(images_zip_path, "wb") as buffer:
        shutil.copyfileobj(images_zip.file, buffer)
    with open(labels_zip_path, "wb") as buffer:
        shutil.copyfileobj(labels_zip.file, buffer)
        
    # Extract
    extract_images_dir = session_upload_dir / "images"
    extract_labels_dir = session_upload_dir / "labels"
    extract_images_dir.mkdir(exist_ok=True)
    extract_labels_dir.mkdir(exist_ok=True)
    
    try:
        extract_zip(images_zip_path, extract_images_dir)
        extract_zip(labels_zip_path, extract_labels_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid zip file: {str(e)}")

    # Normalize format_type
    format_type = format_type.strip().lower().replace(" ", "_").replace("-", "_")
    if format_type == "pascal" or "pascal_voc" in format_type or format_type == "voc":
        format_type = "pascal_voc"
    
    supported_formats = ["yolo", "coco", "pascal_voc", "roboflow", "voc"]
    if format_type not in supported_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format_type}. Supported: {supported_formats}")

    # Validate
    validator = DatasetValidator(extract_images_dir, extract_labels_dir)
    report, annotations, stem_to_image, stem_to_label, class_names = validator.validate()
    
    print(f"DEBUG: Session {session_id} - Matched: {len(annotations)}, Orphans: {report.orphan_labels}")

    # Save internal format
    internal_format_path = session_processed_dir / "annotations.json"
    with open(internal_format_path, "w") as f:
        json.dump([ann.dict() for ann in annotations], f, indent=4)
        
    # Analyze
    analyzer = DatasetAnalyzer(session_id, session_analysis_dir)
    summary = analyzer.analyze(annotations, report)
    
    # Export (Initial) using ExportService
    zip_path, total_copied_images = ExportService.export_dataset(
        session_id, annotations, report, class_names, stem_to_image, format_type
    )

    # Verify count consistency
    # Note: validation matches annotations to images, so len(annotations) should be exported
    if total_copied_images != len(annotations):
        # This check is now less critical as ExportService handles it, but good to keep
         print(f"Warning: Expected {len(annotations)} images, processed {total_copied_images}")
         # raise HTTPException(status_code=500, detail=f"Export Failed: Expected {len(annotations)} images, but only {total_copied_images} were processed.")

    # -----------------------
    # Persist dataset metadata + files to DB/storage
    # -----------------------
    try:
        # Static user/project for now (supports future auth)
        user_id = "00000000-0000-0000-0000-000000000001"
        project_name = "default-project"

        # Create storage path: STORAGE_ROOT/{user_id}/{project_name}/{dataset_id}/
        storage_base = STORAGE_ROOT / user_id / project_name / session_id
        images_out = storage_base / "images"
        labels_out = storage_base / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # Copy extracted uploads into structured storage (do not change original export behavior)
        # extracted files live in `extract_images_dir` and `extract_labels_dir`
        for p in extract_images_dir.rglob("*"):
            if p.is_file():
                shutil.copy2(p, images_out / p.name)
        for p in extract_labels_dir.rglob("*.txt"):
            if p.is_file():
                shutil.copy2(p, labels_out / p.name)

        # Ensure user exists
        user = db.query(db_models.User).filter_by(id=user_id).first()
        if not user:
            user = db_models.User(id=user_id, email="user@example.com")
            db.add(user)
            db.commit()

        # Ensure project exists
        project = db.query(db_models.Project).filter_by(name=project_name, user_id=user_id).first()
        if not project:
            project = db_models.Project(id=str(uuid.uuid4()), name=project_name, user_id=user_id)
            db.add(project)
            db.commit()

        # Insert Dataset row (use session_id as dataset id)
        dataset = db_models.Dataset(
            id=session_id,
            project_id=project.id,
            format_type=format_type,
            total_images=summary.total_images,
            total_labels=summary.total_labels,
            total_classes=summary.total_classes,
            total_objects=summary.total_objects,
            avg_objects_per_image=summary.avg_objects_per_image,
            missing_label_count=summary.missing_label_count,
            corrupted_image_count=summary.corrupted_image_count,
            csv_file_path=str(session_analysis_dir / "dataset_statistics.csv"),
            zip_file_path=str(zip_path),
        )
        db.add(dataset)
        db.commit()

        # Insert Image rows (from storage)
        image_files = [p for p in images_out.iterdir() if p.is_file()]
        for img in sorted(image_files):
            img_row = db_models.Image(
                id=str(uuid.uuid4()),
                dataset_id=dataset.id,
                file_name=img.name,
                file_path=str(img),
                has_label=(labels_out / (img.stem + ".txt")).exists(),
            )
            db.add(img_row)
        db.commit()

        # Insert Label rows (one DB row per object line in each .txt)
        for lbl in sorted(labels_out.glob("*.txt")):
            # try to find associated image by stem
            stem = lbl.stem
            image_row = db.query(db_models.Image).filter(db_models.Image.file_name.like(f"{stem}%")).first()
            with open(lbl, "r", encoding="utf-8", errors="ignore") as fh:
                for ln in fh:
                    parts = ln.strip().split()
                    if not parts:
                        continue
                    class_id = parts[0]
                    bbox = parts[1:]
                    label_row = db_models.Label(
                        id=str(uuid.uuid4()),
                        image_id=(image_row.id if image_row else None),
                        class_id=str(class_id),
                        bbox_data={"yolo": bbox},
                    )
                    db.add(label_row)
        db.commit()

        # Insert validation summary
        val = db_models.DatasetValidation(
            dataset_id=dataset.id,
            total_images=report.total_images,
            total_labels=report.total_labels,
            missing_labels=report.missing_labels,
            orphan_labels=report.orphan_labels,
            empty_labels=report.empty_labels,
            corrupted_images=report.corrupted_images,
        )
        db.add(val)
        db.commit()

        # Insert class distribution
        for cls_id, cnt in summary.class_distribution.items():
            cd = db_models.ClassDistribution(dataset_id=dataset.id, class_id=str(cls_id), object_count=int(cnt))
            db.add(cd)
        db.commit()

    except Exception as _e:
        # Do not break endpoint behavior if DB/storage persistence fails. Log for debugging.
        print("DB persistence failed:", str(_e))


    base_url = str(request.base_url).rstrip("/")
    download_url = f"{base_url}/download/{session_id}"
    
    return UploadResponse(
        dataset_id=session_id,
        validation_report=report,
        analysis_summary=summary,
        csv_file_path=str(session_analysis_dir / "dataset_statistics.csv"),
        download_url=download_url
    )
