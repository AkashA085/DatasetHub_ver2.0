from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from app.utils.file_utils import EXPORTS_DIR, STORAGE_ROOT
from app.models.schemas import DownloadURLResponse
from pathlib import Path

router = APIRouter()

@router.get(
    "/download/{dataset_id}",
    response_model=DownloadURLResponse,
    summary="Get Dataset Download URL",
    description="Returns a URL to download the processed dataset ZIP file.",
    responses={
        404: {"description": "Dataset ZIP not found"}
    },
    tags=["Export"]
)
async def download_dataset(dataset_id: str, request: Request):
    # Check multiple possible locations for the ZIP file
    candidates = [
        EXPORTS_DIR / dataset_id / f"{dataset_id}.zip",
        STORAGE_ROOT / "exports" / dataset_id / f"{dataset_id}.zip",
        STORAGE_ROOT / "uploads" / dataset_id / f"{dataset_id}.zip",
        STORAGE_ROOT / "exports" / f"{dataset_id}.zip",
    ]
    
    zip_path = None
    for candidate in candidates:
        if candidate.exists():
            zip_path = candidate
            break
    
    if not zip_path:
        raise HTTPException(status_code=404, detail="Dataset ZIP not found. Please upload or process first.")
        
    base_url = str(request.base_url).rstrip("/")
    download_url = f"{base_url}/files/{dataset_id}/{dataset_id}.zip"
    
    return DownloadURLResponse(download_url=download_url)
