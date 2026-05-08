from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_static_download_flow():
    # Use the known dataset ID
    dataset_id = "a3bc16c0-2d73-4481-bba2-26f8dffdd67e"
    
    print(f"Testing static download flow for dataset_id: {dataset_id}")
    
    # 1. Test /download/{dataset_id}
    print("1. Calling /download endpoint...")
    response = client.get(f"/download/{dataset_id}")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response JSON: {response.json()}")
    
    assert response.status_code == 200
    data = response.json()
    assert "download_url" in data
    download_url = data["download_url"]
    
    # Verify the URL structure matches what we expect from static files
    # Should end with /files/{dataset_id}/{dataset_id}.zip
    expected_suffix = f"/files/{dataset_id}/{dataset_id}.zip"
    print(f"Received Download URL: {download_url}")
    print(f"Expected suffix: {expected_suffix}")
    
    assert download_url.endswith(expected_suffix)
    
    # 2. Test the static URL
    # Extract the relative path
    relative_path = "/files/" + download_url.split("/files/")[-1]
    
    print(f"2. Calling static file path: {relative_path}")
    file_response = client.get(relative_path)
    
    print(f"File Status Code: {file_response.status_code}")
    # StaticFiles might assume different content types, but zip is standard
    print(f"File Content-Type: {file_response.headers.get('content-type')}")
    print(f"File Size: {len(file_response.content)} bytes")
    
    assert file_response.status_code == 200
    # Provide leniency on content-type if system MIME types vary, but typically application/zip or application/x-zip-compressed
    # app/zip is standard.
    assert "zip" in file_response.headers.get("content-type", "").lower()
    assert len(file_response.content) > 0
    
    print("Static download flow test PASSED")

if __name__ == "__main__":
    try:
        test_static_download_flow()
    except Exception as e:
        print(f"Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
