import pytest
from pathlib import Path
from app.utils.file_utils import EXPORTS_DIR

def test_download_dataset_success(client):
    session_id = "test_export"
    session_export_dir = EXPORTS_DIR / session_id
    session_export_dir.mkdir(parents=True, exist_ok=True)
    zip_path = session_export_dir / f"{session_id}.zip"
    zip_path.write_text("fake zip content")
    
    response = client.get(f"/download/{session_id}")
    assert response.status_code == 200
    assert f"/files/{session_id}/{session_id}.zip" in response.json()["download_url"]

def test_download_dataset_not_found(client):
    response = client.get("/download/nonexistent")
    assert response.status_code == 404
