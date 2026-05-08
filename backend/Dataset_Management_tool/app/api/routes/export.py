from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from app.utils.file_utils import EXPORTS_DIR
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
    session_export_dir = EXPORTS_DIR / dataset_id
    zip_path = session_export_dir / f"{dataset_id}.zip"
    
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Dataset ZIP not found. Please upload or process first.")
        
    # Construct the download URL using the request base URL to ensure it works in any environment
    # Point to the static file mount: /files/{dataset_id}/{dataset_id}.zip
    base_url = str(request.base_url).rstrip("/")
    download_url = f"{base_url}/files/{dataset_id}/{dataset_id}.zip"
    
    return DownloadURLResponse(download_url=download_url)
