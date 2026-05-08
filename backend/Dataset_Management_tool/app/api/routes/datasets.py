"""
API routes for dataset management - listing, details, images, statistics.
These endpoints provide read-only access to the database for frontend display.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc, asc
from typing import Optional, List
from app.core.database import get_db, Dataset, Image, Label, DatasetValidation, ClassDistribution, Project
import os
import uuid
import shutil
from pathlib import Path
import json
from pydantic import BaseModel
from app.utils.file_utils import cleanup_session, STORAGE_ROOT, PROCESSED_DIR
from datetime import datetime
from PIL import Image as PILImage
import numpy as np

router = APIRouter()

# Response Models
class DatasetSummary(BaseModel):
    id: str
    format_type: str
    total_images: int
    total_labels: int
    total_classes: int
    total_objects: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class PaginatedDatasetsResponse(BaseModel):
    datasets: List[DatasetSummary]
    total: int
    page: int
    limit: int
    total_pages: int

class BboxData(BaseModel):
    yolo: List[float]

class LabelInfo(BaseModel):
    class_id: str
    bbox: BboxData

class ImageInfo(BaseModel):
    id: str
    file_name: str
    url: str
    has_label: bool
    labels: List[LabelInfo] = []
    
    class Config:
        from_attributes = True

class PaginatedImagesResponse(BaseModel):
    images: List[ImageInfo]
    total: int
    page: int
    limit: int
    total_pages: int

class GlobalImageInfo(BaseModel):
    id: str
    dataset_id: str
    file_name: str
    url: str
    has_label: bool

    class Config:
        from_attributes = True

class PaginatedGlobalImagesResponse(BaseModel):
    images: List[GlobalImageInfo]
    total: int
    page: int
    limit: int
    total_pages: int

class LabelUpdateResponse(BaseModel):
    status: str
    message: str

class ClassDistributionItem(BaseModel):
    class_id: str
    object_count: int
    
    class Config:
        from_attributes = True

class DatasetDetailResponse(BaseModel):
    id: str
    format_type: str
    total_images: int
    total_labels: int
    total_classes: int
    total_objects: int
    avg_objects_per_image: float
    missing_label_count: int
    corrupted_image_count: int
    csv_file_path: Optional[str]
    zip_file_path: Optional[str]
    created_at: datetime
    validation: Optional[dict]
    class_distribution: List[ClassDistributionItem]
    analysis_summary: Optional[dict]
    
    class Config:
        from_attributes = True

class StatisticsResponse(BaseModel):
    total_images: int
    total_labels: int
    total_classes: int
    total_objects: int
    avg_objects_per_image: float
    missing_label_count: int
    corrupted_image_count: int
    class_distribution: List[ClassDistributionItem]
    validation_metrics: Optional[dict]

class ImageIssueInfo(BaseModel):
    id: str
    file_name: str
    url: str
    issues: List[str]
    blur_score: Optional[float]
    invalid_label_count: int

class ImageIssuesResponse(BaseModel):
    flagged_images: List[ImageIssueInfo]
    total_flagged: int


def _to_storage_url(file_path: str) -> str:
    """Convert an absolute file path to mounted `/storage/...` URL."""
    if not file_path or "storage" not in file_path:
        return ""
    parts = file_path.split("storage")
    if len(parts) <= 1:
        return ""
    return "/storage" + parts[1].replace("\\", "/")


def _parse_yolo_bbox(raw_bbox) -> Optional[List[float]]:
    if not isinstance(raw_bbox, dict):
        return None
    yolo = raw_bbox.get("yolo", [])
    if not isinstance(yolo, list) or len(yolo) < 4:
        return None
    try:
        return [float(yolo[0]), float(yolo[1]), float(yolo[2]), float(yolo[3])]
    except (TypeError, ValueError):
        return None


def _is_invalid_bbox(yolo_bbox: List[float]) -> bool:
    cx, cy, w, h = yolo_bbox
    if w <= 0 or h <= 0:
        return True
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
        return True

    x_min = cx - (w / 2)
    x_max = cx + (w / 2)
    y_min = cy - (h / 2)
    y_max = cy + (h / 2)
    return x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1


def _compute_blur_score(file_path: str) -> Optional[float]:
    """
    Lightweight blur score based on gradient variance.
    Lower scores indicate blurrier images.
    """
    try:
        with PILImage.open(file_path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float32)
            if gray.size == 0:
                return None
            gx = np.diff(gray, axis=1)
            gy = np.diff(gray, axis=0)
            return float(np.var(gx) + np.var(gy))
    except Exception:
        return None


def _load_annotations_yolo_map(dataset_id: str) -> dict:
    """
    Load validator output saved at upload time and expose labels as YOLO normalized coords.
    Keyed by image stem lowercase.
    """
    result = {}
    ann_path = PROCESSED_DIR / dataset_id / "annotations.json"
    if not ann_path.exists():
        return result

    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        for ann in annotations:
            stem = str(ann.get("image_name", "")).strip().lower()
            width = float(ann.get("width", 0) or 0)
            height = float(ann.get("height", 0) or 0)
            if not stem or width <= 0 or height <= 0:
                continue

            labels = []
            for obj in ann.get("objects", []):
                try:
                    xmin = float(obj.get("xmin"))
                    ymin = float(obj.get("ymin"))
                    xmax = float(obj.get("xmax"))
                    ymax = float(obj.get("ymax"))
                    class_id = str(obj.get("class_id"))
                except Exception:
                    continue

                cx = ((xmin + xmax) / 2.0) / width
                cy = ((ymin + ymax) / 2.0) / height
                bw = max(0.0, (xmax - xmin) / width)
                bh = max(0.0, (ymax - ymin) / height)

                labels.append(LabelInfo(
                    class_id=class_id,
                    bbox=BboxData(yolo=[cx, cy, bw, bh])
                ))

            result[stem] = labels
    except Exception:
        return {}

    return result


@router.get("/datasets", response_model=PaginatedDatasetsResponse)
async def list_datasets(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    format_type: Optional[str] = None,
    sort_by: str = Query("created_at", regex="^(created_at|total_images|total_classes)$"),
    order: str = Query("desc", regex="^(asc|desc)$"),
    db: Session = Depends(get_db)
):
    """
    List all datasets with pagination, filtering, and sorting.
    """
    query = db.query(Dataset)
    
    # Filter by format type if provided
    if format_type:
        query = query.filter(Dataset.format_type == format_type.lower())
    
    # Sorting
    sort_column = getattr(Dataset, sort_by)
    if order == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(asc(sort_column))
    
    # Get total count
    total = query.count()
    
    # Pagination
    offset = (page - 1) * limit
    datasets = query.offset(offset).limit(limit).all()
    
    # Calculate total pages
    total_pages = (total + limit - 1) // limit
    
    return PaginatedDatasetsResponse(
        datasets=[DatasetSummary.from_orm(d) for d in datasets],
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages
    )


@router.get("/datasets/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset_details(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific dataset.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get validation info
    validation = db.query(DatasetValidation).filter(DatasetValidation.dataset_id == dataset_id).first()
    validation_dict = None
    if validation:
        validation_dict = {
            "total_images": validation.total_images,
            "total_labels": validation.total_labels,
            "missing_labels": validation.missing_labels,
            "orphan_labels": validation.orphan_labels,
            "empty_labels": validation.empty_labels,
            "corrupted_images": validation.corrupted_images,
            "class_ids_found": validation.class_ids_found,
            "missing_label_images": validation.missing_label_images,
            "orphan_label_files": validation.orphan_label_files,
            "empty_label_files": validation.empty_label_files,
            "corrupted_image_files": validation.corrupted_image_files,
        }
    
    # Get class distribution
    class_dist = db.query(ClassDistribution).filter(ClassDistribution.dataset_id == dataset_id).all()
    
    return DatasetDetailResponse(
        id=dataset.id,
        format_type=dataset.format_type,
        total_images=dataset.total_images,
        total_labels=dataset.total_labels,
        total_classes=dataset.total_classes,
        total_objects=dataset.total_objects,
        avg_objects_per_image=dataset.avg_objects_per_image,
        missing_label_count=dataset.missing_label_count,
        corrupted_image_count=dataset.corrupted_image_count,
        csv_file_path=dataset.csv_file_path,
        zip_file_path=dataset.zip_file_path,
        created_at=dataset.created_at,
        validation=validation_dict,
        class_distribution=[ClassDistributionItem.from_orm(cd) for cd in class_dist],
        analysis_summary=dataset.analysis_summary
    )


@router.get("/datasets/{dataset_id}/images", response_model=PaginatedImagesResponse)
async def list_dataset_images(
    dataset_id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=100),
    has_label: Optional[bool] = None,
    force_file_labels: bool = Query(True),
    db: Session = Depends(get_db)
):
    """
    List images in a specific dataset with pagination.
    """
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    query = db.query(Image).filter(Image.dataset_id == dataset_id).options(joinedload(Image.labels))
    
    # Filter by label status if provided
    if has_label is not None:
        query = query.filter(Image.has_label == has_label)
    
    # Get total count
    total = query.count()
    
    # Pagination
    offset = (page - 1) * limit
    images = query.offset(offset).limit(limit).all()
    
    # Calculate total pages
    total_pages = (total + limit - 1) // limit
    
    annotations_map = _load_annotations_yolo_map(dataset_id)

    # Process images to include URLs and labels
    processed_images = []
    for img in images:
        # Convert absolute path to relative URL
        rel_path = ""
        if "storage" in img.file_path:
            parts = img.file_path.split("storage")
            if len(parts) > 1:
                rel_path = "/storage" + parts[1].replace("\\", "/")
        
        labels_info = []
        image_stem = Path(img.file_name).stem.lower() if img.file_name else ""

        # 0) Prefer current label files from disk (source of truth after edits/augmentation).
        if not labels_info:
            try:
                img_path = Path(img.file_path)
                img_path_parts = list(img_path.parts)
                if "images" in img_path_parts:
                    idx = img_path_parts.index("images")
                    label_parts = list(img_path_parts)
                    label_parts[idx] = "labels"
                    label_file_path = Path(*label_parts).with_suffix(".txt")

                    if label_file_path.exists():
                        with open(label_file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                line_parts = line.strip().split()
                                if len(line_parts) >= 5:
                                    try:
                                        labels_info.append(LabelInfo(
                                            class_id=line_parts[0],
                                            bbox=BboxData(yolo=[float(x) for x in line_parts[1:5]])
                                        ))
                                    except ValueError:
                                        continue
            except Exception as e:
                print(f"Error reading label file: {e}")

        # 1) Fallback to upload-time validator annotations.
        if not labels_info and image_stem and image_stem in annotations_map:
            labels_info = annotations_map.get(image_stem, [])

        # 2) Optional fallback: DB labels if file labels were not found and caller asks.
        if not labels_info and not force_file_labels and img.labels:
            for lbl in img.labels:
                labels_info.append(LabelInfo(
                    class_id=lbl.class_id,
                    bbox=BboxData(yolo=lbl.bbox_data.get("yolo", []))
                ))
            
        processed_images.append(ImageInfo(
            id=img.id,
            file_name=img.file_name,
            url=rel_path,
            has_label=img.has_label or len(labels_info) > 0,
            labels=labels_info
        ))
    
    return PaginatedImagesResponse(
        images=processed_images,
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages
    )


@router.get("/datasets/{dataset_id}/images/issues", response_model=ImageIssuesResponse)
async def list_dataset_image_issues(
    dataset_id: str,
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=200),
    blur_threshold: float = Query(20.0, gt=0),
    db: Session = Depends(get_db)
):
    """
    Flag images needing review:
    - missing_label: no labels present
    - invalid_bbox: bbox outside valid YOLO normalized range / frame
    - blurry_image: low gradient variance score
    - object_too_small: bbox area very small (likely hard-to-see target)
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    query = db.query(Image).filter(Image.dataset_id == dataset_id).options(joinedload(Image.labels))
    offset = (page - 1) * limit
    images = query.offset(offset).limit(limit).all()

    flagged_images: List[ImageIssueInfo] = []
    for img in images:
        issues: List[str] = []
        invalid_label_count = 0

        parsed_boxes: List[List[float]] = []
        for lbl in img.labels:
            yolo_bbox = _parse_yolo_bbox(lbl.bbox_data)
            if not yolo_bbox or _is_invalid_bbox(yolo_bbox):
                invalid_label_count += 1
                continue
            parsed_boxes.append(yolo_bbox)

        if len(img.labels) == 0:
            issues.append("missing_label")
        if invalid_label_count > 0:
            issues.append("invalid_bbox")
        if any((w * h) < 0.002 for _, _, w, h in parsed_boxes):
            issues.append("object_too_small")

        blur_score = _compute_blur_score(img.file_path)
        if blur_score is not None and blur_score < blur_threshold:
            issues.append("blurry_image")

        if issues:
            flagged_images.append(
                ImageIssueInfo(
                    id=img.id,
                    file_name=img.file_name,
                    url=_to_storage_url(img.file_path),
                    issues=issues,
                    blur_score=blur_score,
                    invalid_label_count=invalid_label_count,
                )
            )

    return ImageIssuesResponse(
        flagged_images=flagged_images,
        total_flagged=len(flagged_images),
    )


@router.get("/datasets/{dataset_id}/statistics", response_model=StatisticsResponse)
async def get_dataset_statistics(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed statistics for charts and analysis.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get validation metrics
    validation = db.query(DatasetValidation).filter(DatasetValidation.dataset_id == dataset_id).first()
    validation_metrics = None
    if validation:
        validation_metrics = {
            "total_images": validation.total_images,
            "total_labels": validation.total_labels,
            "missing_labels": validation.missing_labels,
            "orphan_labels": validation.orphan_labels,
            "empty_labels": validation.empty_labels,
            "corrupted_images": validation.corrupted_images,
        }
    
    # Get class distribution
    class_dist = db.query(ClassDistribution).filter(ClassDistribution.dataset_id == dataset_id).all()
    
    return StatisticsResponse(
        total_images=dataset.total_images,
        total_labels=dataset.total_labels,
        total_classes=dataset.total_classes,
        total_objects=dataset.total_objects,
        avg_objects_per_image=dataset.avg_objects_per_image,
        missing_label_count=dataset.missing_label_count,
        corrupted_image_count=dataset.corrupted_image_count,
        class_distribution=[ClassDistributionItem.from_orm(cd) for cd in class_dist],
        validation_metrics=validation_metrics
    )


@router.get("/images", response_model=PaginatedGlobalImagesResponse)
async def list_all_images(
    page: int = Query(1, ge=1),
    limit: int = Query(48, ge=1, le=200),
    has_label: Optional[bool] = None,
    dataset_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all images across datasets (global gallery) with pagination.
    """
    query = db.query(Image)

    if dataset_id:
        query = query.filter(Image.dataset_id == dataset_id)

    if has_label is not None:
        query = query.filter(Image.has_label == has_label)

    total = query.count()
    offset = (page - 1) * limit
    images = query.order_by(Image.file_name.asc()).offset(offset).limit(limit).all()
    total_pages = (total + limit - 1) // limit

    mapped = [
        GlobalImageInfo(
            id=img.id,
            dataset_id=img.dataset_id,
            file_name=img.file_name,
            url=_to_storage_url(img.file_path),
            has_label=img.has_label,
        )
        for img in images
    ]

    return PaginatedGlobalImagesResponse(
        images=mapped,
        total=total,
        page=page,
        limit=limit,
        total_pages=total_pages
    )

@router.put("/datasets/{dataset_id}/images/{image_id}/labels", response_model=LabelUpdateResponse)
async def update_image_labels(
    dataset_id: str,
    image_id: str,
    labels: List[LabelInfo],
    db: Session = Depends(get_db)
):
    # Verify image exists and belongs to dataset
    img = db.query(Image).filter(Image.id == image_id, Image.dataset_id == dataset_id).first()
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
        
    # 1. Update Database
    # Remove old labels
    db.query(Label).filter(Label.image_id == image_id).delete()
    
    # Add new labels
    for lbl_info in labels:
        new_label = Label(
            id=str(uuid.uuid4()),
            image_id=image_id,
            class_id=lbl_info.class_id,
            bbox_data={"yolo": lbl_info.bbox.yolo}
        )
        db.add(new_label)
    
    # Update has_label flag
    img.has_label = len(labels) > 0
    db.commit()
    
    # 2. Update File System (YOLO format)
    # Find .txt file path robustly
    img_path = Path(img.file_path)
    img_path_parts = list(img_path.parts)
    
    label_file_path = None
    if "images" in img_path_parts:
        idx = img_path_parts.index("images")
        label_parts = list(img_path_parts)
        label_parts[idx] = "labels"
        label_file_path = Path(*label_parts).with_suffix(".txt")
    else:
        # Fallback to simple structure
        labels_dir = img_path.parent.parent / "labels"
        label_file_path = labels_dir / (img_path.stem + ".txt")
    
    try:
        if label_file_path:
            label_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_file_path, "w", encoding="utf-8") as f:
                for lbl_info in labels:
                    yolo = lbl_info.bbox.yolo
                    f.write(f"{lbl_info.class_id} {' '.join(map(str, yolo))}\n")
    except Exception as e:
        print(f"Failed to update label file: {str(e)}")
        # We don't raise here to keep DB in sync, but maybe we should?
        # For now just log it.
        
    return LabelUpdateResponse(status="success", message="Labels updated successfully")

@router.delete("/datasets/{dataset_id}", response_model=LabelUpdateResponse)
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Permanently delete a dataset and ALL associated files from every storage location.
    """
    # 1. Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # 2. Filesystem Cleanup (100% thorough)
        
        # A. Cleanup via session utility (Uploads, Processed, Analysis, Exports)
        # This covers: uploads/{id}, processed/{id}, analysis/{id}, exports/{id}
        cleanup_session(dataset_id)
        
        # B. Cleanup Structured Storage (STORAGE_ROOT/user_id/project_id/dataset_id)
        # We also scan for any folder matching the dataset_id in the entire storage tree
        # to handle accidental nested folders or path mistakes.
        search_roots = [STORAGE_ROOT, Path("dataset_backend/app/storage")]
        for root in search_roots:
            if not root.exists():
                continue
            for p in root.rglob(dataset_id):
                if p.is_dir():
                    try:
                        shutil.rmtree(p)
                    except Exception as e:
                        print(f"Warning: Could not delete path {p}: {e}")

        # C. Cleanup specific export ZIP from DB record if still exists
        if dataset.zip_file_path:
            zip_p = Path(dataset.zip_file_path)
            if zip_p.exists():
                try:
                    zip_p.unlink()
                except Exception:
                    pass

        # 3. Cascading Database Deletion
        db.query(Label).filter(Label.image_id.in_(
            db.query(Image.id).filter(Image.dataset_id == dataset_id)
        )).delete(synchronize_session=False)
        
        db.query(Image).filter(Image.dataset_id == dataset_id).delete(synchronize_session=False)
        db.query(DatasetValidation).filter(DatasetValidation.dataset_id == dataset_id).delete(synchronize_session=False)
        db.query(ClassDistribution).filter(ClassDistribution.dataset_id == dataset_id).delete(synchronize_session=False)
        db.query(Dataset).filter(Dataset.id == dataset_id).delete(synchronize_session=False)
        
        db.commit()
        return LabelUpdateResponse(status="success", message="Dataset and all associated files deleted successfully")
        
    except Exception as e:
        db.rollback()
        print(f"Failed to delete dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")
