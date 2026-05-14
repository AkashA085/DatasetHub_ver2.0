import pytest
from pathlib import Path
import os

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Dataset Management Backend API"}

def test_serve_storage_file(client, tmp_path):
    # Setup a dummy file in storage
    from app.utils.file_utils import STORAGE_ROOT
    
    # We need to ensure STORAGE_ROOT exists for this test if it doesn't
    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    
    test_file = STORAGE_ROOT / "test_image.jpg"
    test_file.write_text("dummy image content")
    
    # The URL should be /storage/test_image.jpg
    response = client.get("/storage/test_image.jpg")
    assert response.status_code == 200
    assert response.text == "dummy image content"

def test_serve_nonexistent_file(client):
    response = client.get("/storage/nonexistent.jpg")
    assert response.status_code == 404

def test_limit_request_size(client):
    # Test middleware for large requests
    headers = {"Content-Length": str(200 * 1024 * 1024)} # 200MB
    response = client.post("/some-endpoint", headers=headers)
    assert response.status_code == 413
    assert "Request too large" in response.json()["detail"]

def test_serve_storage_file_traversal(client):
    # Test directory traversal protection
    response = client.get("/storage/../secrets.txt")
    assert response.status_code == 404

def test_limit_request_size_upload_bypass(client):
    # Test that /upload-dataset bypasses the 100MB limit
    headers = {"Content-Length": str(200 * 1024 * 1024)} # 200MB
    # Note: we don't need a real payload, the middleware checks headers FIRST
    # We might get a 422 or 404 later, but NOT a 413
    response = client.post("/upload-dataset", headers=headers)
    assert response.status_code != 413
