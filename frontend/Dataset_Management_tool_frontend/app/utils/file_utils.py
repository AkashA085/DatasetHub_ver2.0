import os
import shutil
import zipfile
import uuid
import hashlib
from pathlib import Path

# Storage root can be configured via env var.
# Default is outside backend app tree to avoid dev-server autoreload loops during large uploads.
_DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parents[4] / "datasethub_storage"
STORAGE_ROOT = Path(os.getenv("DATASET_STORAGE_ROOT", str(_DEFAULT_STORAGE_ROOT))).resolve()

UPLOADS_DIR = STORAGE_ROOT / "uploads"
PROCESSED_DIR = STORAGE_ROOT / "processed"
ANALYSIS_DIR = STORAGE_ROOT / "analysis"
EXPORTS_DIR = STORAGE_ROOT / "exports"

def ensure_dirs():
    """Ensure all required storage directories exist."""
    for directory in [UPLOADS_DIR, PROCESSED_DIR, ANALYSIS_DIR, EXPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

def generate_session_id() -> str:
    """Generate a unique session UUID."""
    return str(uuid.uuid4())

def extract_zip(zip_path: Path, extract_to: Path):
    """Extract a zip file to a specific directory with safety checks.
    Handles long nested paths on Windows by flattening to hashed filenames when needed.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for info in zip_ref.infolist():
            # Skip directories and invalid names
            name = info.filename
            if not name or name.endswith("/"):
                continue

            # Normalize separators and prevent zip-slip paths
            safe_name = name.replace("\\", "/").lstrip("/")
            if ".." in Path(safe_name).parts:
                continue

            target = extract_to / Path(safe_name)
            target_str = str(target)

            # Windows long-path fallback: flatten filename using a hash based on path WITHOUT suffix.
            # This keeps image/label stem alignment intact (e.g., a.jpg and a.txt remain same stem).
            if os.name == "nt" and len(target_str) > 240:
                original = Path(safe_name)
                suffix = original.suffix.lower()
                stem = original.stem[:80]
                stem_key = str(original.with_suffix(""))
                digest = hashlib.sha1(stem_key.encode("utf-8", errors="ignore")).hexdigest()[:12]
                target = extract_to / f"{stem}_{digest}{suffix}"

            target.parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(info, "r") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)

def cleanup_session(session_id: str):
    """Remove all files associated with a session ID."""
    for directory in [UPLOADS_DIR, PROCESSED_DIR, ANALYSIS_DIR, EXPORTS_DIR]:
        session_path = directory / session_id
        if session_path.exists():
            shutil.rmtree(session_path)

def create_zip_archive(parent_dir: Path, folder_to_zip: str, output_zip_path: Path):
    """
    Create a zip archive where folder_to_zip is the root folder inside the ZIP.
    - parent_dir: The directory containing the folder to zip.
    - folder_to_zip: The name of the folder within parent_dir to be zipped.
    - output_zip_path: The full path where the .zip file should be saved.
    """
    base_name = str(output_zip_path).replace(".zip", "")
    shutil.make_archive(
        base_name=base_name,
        format='zip',
        root_dir=parent_dir,
        base_dir=folder_to_zip
    )
    return output_zip_path
