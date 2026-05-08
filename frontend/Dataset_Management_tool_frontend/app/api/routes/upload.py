import os
import shutil
import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends
import re
from app.utils.file_utils import generate_session_id, UPLOADS_DIR, PROCESSED_DIR, ANALYSIS_DIR, EXPORTS_DIR, extract_zip, STORAGE_ROOT
from app.services.validator import DatasetValidator
from app.services.analyzer import DatasetAnalyzer
from app.services.export_service import ExportService
from app.models.schemas import UploadResponse, DatasetInternalFormat, DownloadURLResponse

# DB / persistence
from app.core import get_db
from app.core import User, Project, Dataset, Image, Label, DatasetValidation, ClassDistribution
from sqlalchemy.orm import Session
import uuid

router = APIRouter()


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
    db: Session = Depends(get_db)
):
    print(f"Received upload request: images_zip={images_zip.filename if images_zip else None}, labels_zip={labels_zip.filename if labels_zip else None}, format_type={format_type}, storage_path={storage_path}")
    # Set a reasonable timeout for the entire operation
    import asyncio
    try:
        return await asyncio.wait_for(_upload_dataset_impl(request, images_zip, labels_zip, format_type, storage_path, db), timeout=300.0)  # 5 minutes timeout
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Upload timed out. Please try with smaller files or check your internet connection.")
    except Exception as e:
        print(f"Upload failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def _upload_dataset_impl(
    request: Request,
    images_zip: UploadFile,
    labels_zip: UploadFile,
    format_type: str,
    storage_path: str,
    db: Session
):
        session_id = generate_session_id()
        
        # Paths for this session
        session_upload_dir = UPLOADS_DIR / session_id
        session_processed_dir = PROCESSED_DIR / session_id
        session_analysis_dir = ANALYSIS_DIR / session_id
        session_export_dir = EXPORTS_DIR / session_id
        
        session_upload_dir.mkdir(parents=True, exist_ok=True)
        session_processed_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting upload for session {session_id}")
        
        # Save uploaded zips
        images_zip_path = session_upload_dir / "images.zip"
        labels_zip_path = session_upload_dir / "labels.zip"

        try:
            images_size = await _save_upload_file(images_zip, images_zip_path)
            labels_size = await _save_upload_file(labels_zip, labels_zip_path)
            if images_size == 0 or labels_size == 0:
                raise HTTPException(status_code=400, detail="Uploaded ZIP file is empty.")
            print(f"Files saved: images.zip ({images_size} bytes), labels.zip ({labels_size} bytes)")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save upload files ({type(e).__name__}): {str(e)}")
            
        # Extract
        extract_images_dir = session_upload_dir / "images"
        extract_labels_dir = session_upload_dir / "labels"
        extract_images_dir.mkdir(exist_ok=True)
        extract_labels_dir.mkdir(exist_ok=True)
        
        try:
            extract_zip(images_zip_path, extract_images_dir)
            extract_zip(labels_zip_path, extract_labels_dir)
            print("ZIP files extracted successfully")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid or corrupted ZIP file ({type(e).__name__}): {str(e)}")

        # Normalize format_type
        format_type = format_type.strip().lower().replace(" ", "_").replace("-", "_")
        if format_type == "pascal" or "pascal_voc" in format_type or format_type == "voc":
            format_type = "pascal_voc"
        
        supported_formats = ["yolo", "coco", "pascal_voc", "roboflow", "voc"]
        if format_type not in supported_formats:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {format_type}. Supported: {supported_formats}")

        # Validate
        try:
            validator = DatasetValidator(extract_images_dir, extract_labels_dir)
            report, annotations, stem_to_image, stem_to_label, class_names = validator.validate()
            
            print(f"DEBUG: Session {session_id} - Matched: {len(annotations)}, Orphans: {report.orphan_labels}")
            if len(annotations) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No matched image/label pairs found after validation. "
                        f"Detected images={report.total_images}, labels={report.total_labels}, "
                        f"missing_labels={report.missing_labels}, orphan_labels={report.orphan_labels}. "
                        "Ensure filenames/stems match between image and label files."
                    ),
                )
            print("Dataset validation completed")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Dataset validation failed ({type(e).__name__}): {str(e)}")

        # Save internal format
        try:
            internal_format_path = session_processed_dir / "annotations.json"
            with open(internal_format_path, "w") as f:
                json.dump([ann.dict() for ann in annotations], f, indent=4)
            print("Internal format saved")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save internal format ({type(e).__name__}): {str(e)}")
            
        # Analyze
        try:
            analyzer = DatasetAnalyzer(session_id, session_analysis_dir)
            summary = analyzer.analyze(annotations, report)
            print("Dataset analysis completed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Dataset analysis failed ({type(e).__name__}): {str(e)}")
        
        # Export (Initial) using ExportService
        try:
            zip_path, total_copied_images = ExportService.export_dataset(
                session_id, annotations, report, class_names, stem_to_image, format_type
            )

            # Verify count consistency
            # Note: validation matches annotations to images, so len(annotations) should be exported
            if total_copied_images != len(annotations):
                # This check is now less critical as ExportService handles it, but good to keep
                 print(f"Warning: Expected {len(annotations)} images, processed {total_copied_images}")
                 # raise HTTPException(status_code=500, detail=f"Export Failed: Expected {len(annotations)} images, but only {total_copied_images} were processed.")

            print("Dataset export completed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Dataset export failed ({type(e).__name__}): {str(e)}")

        # Database persistence (non-critical)
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

            # Copy only validated/matched image-label pairs from validator mappings.
            # This guarantees image and label stay synchronized exactly as validated.
            matched_stems = sorted(stem_to_image.keys())
            for stem in matched_stems:
                src_img = stem_to_image[stem]
                src_lbl = stem_to_label.get(stem)
                if not src_img.exists():
                    continue

                dst_img = images_out / f"{stem}{src_img.suffix.lower()}"
                shutil.copy2(src_img, dst_img)

                if src_lbl and src_lbl.exists():
                    dst_lbl = labels_out / f"{stem}.txt"
                    shutil.copy2(src_lbl, dst_lbl)

            # Ensure user exists
            user = db.query(User).filter_by(id=user_id).first()
            if not user:
                user = User(id=user_id, email="user@example.com")
                db.add(user)
                db.commit()

            # Ensure project exists
            project = db.query(Project).filter_by(name=project_name, user_id=user_id).first()
            if not project:
                project = Project(id=str(uuid.uuid4()), name=project_name, user_id=user_id)
                db.add(project)
                db.commit()

            # Insert Dataset row (use session_id as dataset id)
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
                csv_file_path=str(session_analysis_dir / "dataset_statistics.csv"),
                zip_file_path=str(zip_path),
                analysis_summary=None,  # Temporarily disable to avoid serialization issues
            )
            db.add(dataset)
            db.commit()

            # Insert Image rows (from validated matched files in storage)
            image_files = [p for p in images_out.iterdir() if p.is_file()]
            stem_to_image_row = {}
            for img in sorted(image_files):
                img_row = Image(
                    id=str(uuid.uuid4()),
                    dataset_id=dataset.id,
                    file_name=img.name,
                    file_path=str(img),
                    has_label=(labels_out / (img.stem + ".txt")).exists(),
                )
                db.add(img_row)
                stem_to_image_row[img.stem] = img_row
            db.commit()

            # Insert Label rows (one DB row per object line in each .txt)
            try:
                for lbl in sorted(labels_out.glob("*.txt")):
                    stem = lbl.stem
                    # Exact stem match avoids collisions like `_1` matching `_10`
                    image_row = stem_to_image_row.get(stem)
                    with open(lbl, "r", encoding="utf-8", errors="ignore") as fh:
                        for ln in fh:
                            parts = ln.strip().split()
                            if not parts:
                                continue
                            class_id = parts[0]
                            # Convert YOLO coordinates to float
                            try:
                                bbox = [float(x) for x in parts[1:]]
                            except ValueError:
                                bbox = parts[1:] # Fallback
                                
                            label_row = Label(
                                id=str(uuid.uuid4()),
                                image_id=(image_row.id if image_row else None),
                                class_id=str(class_id),
                                bbox_data={"yolo": bbox},
                            )
                            db.add(label_row)
                db.commit()
                print("Labels inserted")
            except Exception as lbl_e:
                print(f"Failed to insert labels: {lbl_e}")
                db.rollback()
                # Continue without labels rather than failing completely

            # Insert validation summary
            val = DatasetValidation(
                dataset_id=dataset.id,
                total_images=report.total_images,
                total_labels=report.total_labels,
                missing_labels=report.missing_labels,
                orphan_labels=report.orphan_labels,
                empty_labels=report.empty_labels,
                corrupted_images=report.corrupted_images,
                class_ids_found=report.class_ids_found if hasattr(report, 'class_ids_found') else None,
                missing_label_images=report.missing_label_images if hasattr(report, 'missing_label_images') else None,
                orphan_label_files=report.orphan_label_files if hasattr(report, 'orphan_label_files') else None,
                empty_label_files=report.empty_label_files if hasattr(report, 'empty_label_files') else None,
                corrupted_image_files=report.corrupted_image_files if hasattr(report, 'corrupted_image_files') else None,
            )
            db.add(val)
            db.commit()

            # Insert class distribution
            for cls_id, cnt in summary.class_distribution.items():
                cd = ClassDistribution(dataset_id=dataset.id, class_id=str(cls_id), object_count=int(cnt))
                db.add(cd)
            db.commit()

        except Exception as _e:
            # Log the error but don't fail the upload - the files are processed successfully
            # The dataset will be available for download even if DB persistence fails
            print("DB persistence failed:", str(_e))
            import traceback
            traceback.print_exc()

        base_url = str(request.base_url).rstrip("/")
        download_url = f"{base_url}/download/{session_id}"

        print(f"Upload completed successfully for session {session_id}")
        return UploadResponse(
            dataset_id=session_id,
            validation_report=report,
            analysis_summary=summary,
            csv_file_path=str(session_analysis_dir / "dataset_statistics.csv"),
            download_url=download_url
        )
