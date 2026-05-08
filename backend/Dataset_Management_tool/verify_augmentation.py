import os
import shutil
import time
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Use existing dataset for testing
# Use existing dataset for testing
DATASET_ID = "a3bc16c0-2d73-4481-bba2-26f8dffdd67e"

def reset_dataset():
    print(f"Resetting dataset {DATASET_ID} to original state...")
    from app.utils.file_utils import PROCESSED_DIR
    import json
    
    session_processed_dir = PROCESSED_DIR / DATASET_ID
    internal_format_path = session_processed_dir / "annotations.json"
    aug_images_dir = session_processed_dir / "augmented_images"
    
    if internal_format_path.exists():
        with open(internal_format_path, 'r') as f:
            anns = json.load(f)
        
        # Filter out augmented images
        original_anns = [ann for ann in anns if "_aug_" not in ann["image_name"]]
        
        print(f"Restoring annotations from {len(anns)} to {len(original_anns)}")
        
        with open(internal_format_path, 'w') as f:
            json.dump(original_anns, f, indent=4)
            
    if aug_images_dir.exists():
        print("Removing augmented images directory...")
        shutil.rmtree(aug_images_dir)
        aug_images_dir.mkdir()
        
    print("Reset complete.")

def test_augmentation_flow():
    print(f"Testing Augmentation Flow for {DATASET_ID}")
    
    # 1. Augmentation Request
    # Apply a simple flip with count=1 (meaning 1 extra version per image)
    # Total images should double.
    # Current images: 642. Expected new: 1284.
    
    print("1. Sending Augmentation Request...", flush=True)
    payload = {
        "dataset_id": DATASET_ID,
        "export_format": "yolo",
        "count": 1,
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation": 0,
        "brightness": 0.0,
        "contrast": 0.0,
        "blur": 0,
        "noise": 0.0
    }
    
    start_time = time.time()
    try:
        response = client.post("/augment", json=payload)
    except Exception as e:
        print(f"CRITICAL ERROR during request: {e}", flush=True)
        return

    end_time = time.time()
    
    print(f"Augmentation took {end_time - start_time:.2f} seconds", flush=True)
    
    if response.status_code != 200:
        print(f"FAILED: {response.text}", flush=True)
        return
        
    data = response.json()
    print(f"Status Code: {response.status_code}")
    
    # Check analysis summary to verify count increase
    summary = data.get("analysis_summary", {})
    total_images = summary.get("total_images", 0)
    print(f"New Total Images: {total_images}")
    
    # We expect roughly double, but depends on if augmentation succeeded for all
    # Previous count was 642. 642 * 2 = 1284.
    if total_images > 642:
        print("SUCCESS: Image count increased.")
    else:
        print("WARNING: Image count did not increase as expected.")

    # 2. Check Download Link
    download_url = data.get("download_url")
    print(f"Download URL: {download_url}")
    
    # 3. Test Download
    # The download URL is like http://testserver/download/{id}
    # TestClient needs a relative path -> /download/{id}
    
    if "http" in download_url:
        # Extract path component
        from urllib.parse import urlparse
        parsed = urlparse(download_url)
        relative_path = parsed.path
    else:
        relative_path = download_url
        
    print(f"Downloading from: {relative_path}", flush=True)
    
    # The /download/{id} endpoint RETURNS A JSON with "download_url".
    # We first need to GET this JSON, then extract the URL, then GET the file.
    
    json_response = client.get(relative_path)
    if json_response.status_code != 200:
        print(f"Failed to get download URL: {json_response.status_code}", flush=True)
        return
        
    download_info = json_response.json()
    actual_file_url = download_info.get("download_url")
    print(f"Actual File URL: {actual_file_url}", flush=True)
    
    # Now download the actual file
    if "http" in actual_file_url:
        from urllib.parse import urlparse
        parsed = urlparse(actual_file_url)
        file_path = parsed.path
    else:
        file_path = actual_file_url
        
    print(f"Fetching file from: {file_path}", flush=True)
    dl_response = client.get(file_path)
    if dl_response.status_code == 200:
        print(f"Download Successful. Size: {len(dl_response.content)} bytes", flush=True)
    else:
        print(f"Download FAILED: {dl_response.status_code}")

if __name__ == "__main__":
    try:
        reset_dataset()
        test_augmentation_flow()
    except Exception as e:
        print(f"Test Error: {e}")
