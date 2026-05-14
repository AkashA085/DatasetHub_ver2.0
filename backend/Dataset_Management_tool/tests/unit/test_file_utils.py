import os
import zipfile
from pathlib import Path
import pytest
from app.utils.file_utils import (
    generate_session_id,
    extract_zip,
    create_zip_archive,
    STORAGE_ROOT
)

def test_generate_session_id():
    session_id = generate_session_id()
    assert isinstance(session_id, str)
    assert len(session_id) == 36  # UUID length

def test_create_and_extract_zip(tmp_path):
    # Setup: Create a dummy folder with files
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "file1.txt").write_text("hello")
    (source_dir / "file2.txt").write_text("world")
    
    zip_out = tmp_path / "test.zip"
    
    # Test create_zip_archive
    create_zip_archive(tmp_path, "source", zip_out)
    assert zip_out.exists()
    
    # Test extract_zip
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    extract_zip(zip_out, extract_dir)
    
    assert (extract_dir / "source" / "file1.txt").exists()
    assert (extract_dir / "source" / "file1.txt").read_text() == "hello"
    assert (extract_dir / "source" / "file2.txt").read_text() == "world"

def test_extract_zip_safety(tmp_path):
    # Test protection against zip-slip (path traversal)
    zip_path = tmp_path / "malicious.zip"
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr("../evil.txt", "content")
        
    extract_to = tmp_path / "safe_dir"
    extract_to.mkdir()
    
    # Should not raise error but should skip the malicious file
    extract_zip(zip_path, extract_to)
    assert not (tmp_path / "evil.txt").exists()
