import os
import shutil
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import uuid

from app.utils.file_utils import (
    generate_session_id,
    UPLOADS_DIR,
    PROCESSED_DIR,
    ANALYSIS_DIR,
    EXPORTS_DIR,
    extract_zip,
    STORAGE_ROOT,
    cleanup_session,
)
from app.services.validator import DatasetValidator
from app.services.analyzer import DatasetAnalyzer
from app.services.export_service import ExportService
from app.models.schemas import UploadResponse
from datetime import datetime
from sqlalchemy.orm import Session
from app.core import User, get_db, Project, Dataset, Image, Label, DatasetValidation, ClassDistribution



# Placeholder user utilities

def get_placeholder_user(db: Session):
    """Return a placeholder user, creating it if missing."""
    placeholder_id = "00000000-0000-0000-0000-000000000001"
    user = db.query(User).filter_by(id=placeholder_id).first()
    if not user:
        user = User(id=placeholder_id, email="placeholder@example.com", created_at=datetime.utcnow())
        db.add(user)
        db.commit()
    return user



router = APIRouter()

# If the main FastAPI app does not already include CORS, you can add it here.
# The following middleware can be mounted in the main app module.
# from fastapi import FastAPI
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

async def _save_upload_file(src: UploadFile, dst: Path, chunk_size: int = 4 * 1024 * 1024) -> int:
    """Stream uploaded file to disk in chunks to handle large payloads safely."""
    total_written = 0
    with open(dst, "wb") as out:
        while True:
            chunk = await src.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            total_written += len(chunk)
    await src.close()
    return total_written

@router.post("/upload-dataset", response_model=UploadResponse)
async def upload_dataset(
    request: Request,
    images_zip: UploadFile = File(...),
    labels_zip: UploadFile = File(...),
    format_type: str = Form(...),
    storage_path: str = Form(None),
    db: Session = Depends(get_db),
):
    """Handle dataset upload, validation, analysis, export and persistence.

    The flow is:
    1. Save and extract zip files.
    2. Validate image/label pairs.
    3. Generate internal JSON format.
    4. Run analysis.
    5. Export dataset (copy files to a permanent storage location).
    6. Persist all metadata in the database.
    """
    print(
        f"Received upload request: images_zip={images_zip.filename}, labels_zip={labels_zip.filename}, format_type={format_type}, storage_path={storage_path}"
    )
    # ---------------------------------------------------------------------
    # 1. Save uploaded zip files
    # ---------------------------------------------------------------------
    session_id = generate_session_id()
    session_upload_dir = UPLOADS_DIR / session_id
    session_processed_dir = PROCESSED_DIR / session_id
    session_analysis_dir = ANALYSIS_DIR / session_id
    session_export_dir = EXPORTS_DIR / session_id
    for p in [session_upload_dir, session_processed_dir, session_analysis_dir, session_export_dir]:
        p.mkdir(parents=True, exist_ok=True)

    images_zip_path = session_upload_dir / "images.zip"
    labels_zip_path = session_upload_dir / "labels.zip"

    try:
        await _save_upload_file(images_zip, images_zip_path)
        await _save_upload_file(labels_zip, labels_zip_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save uploaded files: {e}")

    # ---------------------------------------------------------------------
    # 2. Extract archives
    # ---------------------------------------------------------------------
    extract_images_dir = session_upload_dir / "images"
    extract_labels_dir = session_upload_dir / "labels"
    extract_images_dir.mkdir(exist_ok=True)
    extract_labels_dir.mkdir(exist_ok=True)
    try:
        extract_zip(images_zip_path, extract_images_dir)
        extract_zip(labels_zip_path, extract_labels_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted ZIP file: {e}")

    # ---------------------------------------------------------------------
    # 3. Validation
    # ---------------------------------------------------------------------
    try:
        validator = DatasetValidator(extract_images_dir, extract_labels_dir)
        report, annotations, stem_to_image, stem_to_label, class_names = validator.validate()
        if not annotations:
            # No matched image/label pairs found – proceed with empty annotations
            print('Warning: No matched image/label pairs; proceeding with empty dataset')
            # Continue without raising an exception
            annotations = []
            # Optionally you could still create a minimal dataset entry
            # but downstream steps will handle empty data gracefully.

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dataset validation failed: {e}")

    # ---------------------------------------------------------------------
    # 4. Save internal JSON representation
    # ---------------------------------------------------------------------
    internal_path = session_processed_dir / "annotations.json"
    try:
        with open(internal_path, "w") as f:
            json.dump([a.dict() for a in annotations], f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write internal format: {e}")

    # ---------------------------------------------------------------------
    # 5. Analysis
    # ---------------------------------------------------------------------
    try:
        analyzer = DatasetAnalyzer(session_id, session_analysis_dir)
        summary = analyzer.analyze(annotations, report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset analysis failed: {e}")

    # ---------------------------------------------------------------------
    # 6. Export (copy files to permanent storage and create zip)
    # ---------------------------------------------------------------------
    try:
        zip_path, total_copied_images = ExportService.export_dataset(
            session_id, annotations, report, class_names, stem_to_image, format_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset export failed: {e}")

    # ---------------------------------------------------------------------
    # 7. Prepare payload for DB persistence
    # ---------------------------------------------------------------------
    analysis_payload = summary.model_dump() if hasattr(summary, "model_dump") else summary
    if class_names:
        analysis_payload["class_names"] = class_names

    # ---------------------------------------------------------------------
    # 8. Retrieve or create Project entry (using placeholder user)
    # ---------------------------------------------------------------------
    placeholder_user = get_placeholder_user(db)
    user_id = placeholder_user.id
    project_name = "default-project"
    project = db.query(Project).filter_by(name=project_name, user_id=user_id).first()
    if not project:
        project = Project(id=str(uuid.uuid4()), name=project_name, user_id=user_id)
        db.add(project)
        db.commit()


    # ---------------------------------------------------------------------
    # 9. Create storage directory for this dataset under STORAGE_ROOT
    # ---------------------------------------------------------------------
    storage_base = STORAGE_ROOT / user_id / project_name / session_id
    images_out = storage_base / "images"
    labels_out = storage_base / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Copy matched pairs first (this mirrors ExportService but ensures we have a local copy for DB rows)
    for stem in sorted(stem_to_image.keys()):
        src_img = stem_to_image[stem]
        src_lbl = stem_to_label.get(stem)
        if not src_img.exists():
            continue
        dst_img = images_out / f"{stem}{src_img.suffix.lower()}"
        try:
            os.symlink(src_img, dst_img)
        except OSError:
            shutil.copy2(src_img, dst_img)
        if src_lbl and src_lbl.exists():
            dst_lbl = labels_out / f"{stem}.txt"
            try:
                os.symlink(src_lbl, dst_lbl)
            except OSError:
                shutil.copy2(src_lbl, dst_lbl)

    # Handle images with missing or empty labels (negative/empty cases)
    for stem in report.missing_label_images + report.empty_label_images:
        candidates = list(Path(extract_images_dir).glob(f"{stem}.*"))
        if not candidates:
            continue
        src_img = candidates[0]
        dst_img = images_out / f"{stem}{src_img.suffix.lower()}"
        try:
            os.symlink(src_img, dst_img)
        except OSError:
            shutil.copy2(src_img, dst_img)
        # No label file is created for these images

    # ---------------------------------------------------------------------
    # 10. Persist Dataset entry
    # ---------------------------------------------------------------------
    # Copy CSV and ZIP to permanent storage before cleanup deletes temp dirs
    perm_csv_dir = storage_base / "stats"
    perm_csv_dir.mkdir(parents=True, exist_ok=True)
    perm_csv_path = perm_csv_dir / "dataset_statistics.csv"
    csv_src = session_analysis_dir / "dataset_statistics.csv"
    if csv_src.exists():
        shutil.copy2(str(csv_src), str(perm_csv_path))

    perm_zip_dir = storage_base / "export"
    perm_zip_dir.mkdir(parents=True, exist_ok=True)
    perm_zip_path = perm_zip_dir / f"{session_id}.zip"
    if zip_path.exists():
        shutil.copy2(str(zip_path), str(perm_zip_path))

    dataset = Dataset(
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
        csv_file_path=str(perm_csv_path),
        zip_file_path=str(perm_zip_path),
        analysis_summary=analysis_payload,
    )
    db.add(dataset)
    db.commit()

    # ---------------------------------------------------------------------
    # 11. Insert Image rows
    # ---------------------------------------------------------------------
    stem_to_image_row = {}
    for img_path in sorted(images_out.iterdir()):
        if not img_path.is_file():
            continue
        has_label = (labels_out / f"{img_path.stem}.txt").exists() and (
            (labels_out / f"{img_path.stem}.txt").stat().st_size > 0
        )
        img_row = Image(
            id=str(uuid.uuid4()),
            dataset_id=dataset.id,
            file_name=img_path.name,
            file_path=str(img_path),
            has_label=has_label,
        )
        db.add(img_row)
        stem_to_image_row[img_path.stem] = img_row
    db.commit()

    # ---------------------------------------------------------------------
    # 12. Insert Label rows (objects per image)
    # ---------------------------------------------------------------------
    try:
        for lbl_path in sorted(labels_out.glob("*.txt")):
            stem = lbl_path.stem
            img_row = stem_to_image_row.get(stem)
            with open(lbl_path, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = parts[0]
                    try:
                        bbox = [float(v) for v in parts[1:]]
                    except ValueError:
                        bbox = parts[1:]
                    label_row = Label(
                        id=str(uuid.uuid4()),
                        image_id=img_row.id if img_row else None,
                        class_id=str(class_id),
                        bbox_data={"yolo": bbox},
                    )
                    db.add(label_row)
        db.commit()
    except Exception as lbl_e:
        print(f"Failed to insert labels: {lbl_e}")
        db.rollback()

    # ---------------------------------------------------------------------
    # 13. Insert validation summary
    # ---------------------------------------------------------------------
    val = DatasetValidation(
        dataset_id=dataset.id,
        total_images=report.total_images,
        total_labels=report.total_labels,
        missing_labels=report.missing_labels,
        orphan_labels=report.orphan_labels,
        empty_labels=report.empty_labels,
        corrupted_images=report.corrupted_images,
        class_ids_found=getattr(report, "class_ids_found", None),
        missing_label_images=getattr(report, "missing_label_images", None),
        orphan_label_files=getattr(report, "orphan_label_files", None),
        empty_label_files=getattr(report, "empty_label_files", None),
        corrupted_image_files=getattr(report, "corrupted_image_files", None),
    )
    db.add(val)
    db.commit()

    # ---------------------------------------------------------------------
    # 14. Insert class distribution
    # ---------------------------------------------------------------------
    for cls_id, cnt in summary.class_distribution.items():
        cd = ClassDistribution(dataset_id=dataset.id, class_id=str(cls_id), object_count=int(cnt))
        db.add(cd)
    db.commit()

    # ---------------------------------------------------------------------
    # 15. Cleanup temporary session files
    # ---------------------------------------------------------------------
    try:
        cleanup_session(session_id)
    except Exception as e:
        print(f"Failed to cleanup session {session_id}: {e}")

    # ---------------------------------------------------------------------
    # 16. Return response
    # ---------------------------------------------------------------------
    base_url = str(request.base_url).rstrip("/")
    download_url = f"{base_url}/download/{session_id}"
    print(f"Upload completed successfully for session {session_id}")
    return UploadResponse(
        dataset_id=session_id,
        validation_report=report,
        analysis_summary=summary,
        csv_file_path=str(perm_csv_path),
        download_url=download_url,
    )
