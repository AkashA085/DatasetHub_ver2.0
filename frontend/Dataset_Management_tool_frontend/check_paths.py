import os
from pathlib import Path

# Mocking the constants from app.utils.file_utils
STORAGE_ROOT = Path("dataset_backend/app/storage")
EXPORTS_DIR = STORAGE_ROOT / "exports"

def check_download(dataset_id: str):
    session_export_dir = EXPORTS_DIR / dataset_id
    zip_path = session_export_dir / f"{dataset_id}.zip"
    
    print(f"Checking path: {zip_path}")
    print(f"Absolute path: {zip_path.absolute()}")
    print(f"Exists: {zip_path.exists()}")

if __name__ == "__main__":
    dataset_id = "a3bc16c0-2d73-4481-bba2-26f8dffdd67e"
    check_download(dataset_id)
