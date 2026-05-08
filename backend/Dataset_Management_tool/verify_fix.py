from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)

def test_download_dataset():
    # Use the known dataset ID from the user request
    dataset_id = "a3bc16c0-2d73-4481-bba2-26f8dffdd67e"
    
    print(f"Testing download for dataset_id: {dataset_id}")
    
    response = client.get(f"/download/{dataset_id}")
    
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Content-Disposition: {response.headers.get('content-disposition')}")
    
    assert response.status_code == 200
    assert response.headers.get("content-type") == "application/zip"
    assert f"dataset_{dataset_id}.zip" in response.headers.get("content-disposition")
    
    # Check if content is actually delivered (not empty)
    assert len(response.content) > 0
    print(f"Downloaded size: {len(response.content)} bytes")
    print("Download test PASSED")

if __name__ == "__main__":
    try:
        test_download_dataset()
    except Exception as e:
        print(f"Test FAILED: {str(e)}")
        exit(1)
