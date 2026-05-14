import pytest
from pathlib import Path
from app.main import _resolve_storage_file
from app.utils.file_utils import STORAGE_ROOT

def test_resolve_storage_file_valid(tmp_path):
    # Create a test file in the real STORAGE_ROOT if possible, or mock STORAGE_ROOTS
    # Since STORAGE_ROOTS is a global in app.main, we might need to patch it
    with patch("app.main.STORAGE_ROOTS", [tmp_path]):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        
        resolved = _resolve_storage_file("test.txt")
        assert resolved == test_file.resolve()

from unittest.mock import patch

def test_resolve_storage_file_traversal():
    assert _resolve_storage_file("../secret.txt") is None
    assert _resolve_storage_file("dir/../../secret.txt") is None

def test_resolve_storage_file_absolute():
    assert _resolve_storage_file("/etc/passwd") is None

def test_resolve_storage_file_outside_root(tmp_path):
    root1 = tmp_path / "root1"
    root1.mkdir()
    with patch("app.main.STORAGE_ROOTS", [root1]):
        # Mock resolve to return something outside root1
        with patch.object(Path, "resolve", return_value=Path("/tmp/outside")):
            assert _resolve_storage_file("test.txt") is None
