import os
import pytest
from unittest.mock import patch, MagicMock
from importlib import reload
import sqlalchemy
from pathlib import Path

# Import the module once
import app.core.database as database

def test_get_db_core():
    # Test the get_db generator
    gen = database.get_db()
    db = next(gen)
    assert db is not None
    try:
        next(gen)
    except StopIteration:
        pass

def test_database_url_sqlite_core():
    # Test sqlite engine creation
    with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///./test_core.db"}):
        with patch("sqlalchemy.create_engine") as mock_create_engine:
            reload(database)
            mock_create_engine.assert_called()
            args, kwargs = mock_create_engine.call_args
            assert "sqlite" in args[0]
            assert kwargs.get("connect_args", {}).get("check_same_thread") is False

def test_database_url_postgres_core():
    # Test postgres engine creation
    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost/db"}):
        with patch("sqlalchemy.create_engine") as mock_create_engine:
            reload(database)
            mock_create_engine.assert_called()
            # Find the postgres call
            found = False
            for call in mock_create_engine.call_args_list:
                if call.args[0].startswith("postgresql"):
                    found = True
                    break
            assert found

def test_database_url_none_core():
    # Test fallback when no DATABASE_URL is provided
    with patch.dict(os.environ, {}, clear=True):
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        with patch("sqlalchemy.create_engine") as mock_create_engine:
            reload(database)
            mock_create_engine.assert_called()
            found_sqlite = False
            for call in mock_create_engine.call_args_list:
                if "sqlite" in str(call.args[0]):
                    found_sqlite = True
                    break
            assert found_sqlite

def test_ensure_additional_columns():
    mock_engine = MagicMock()
    mock_inspector = MagicMock()
    
    with patch("sqlalchemy.inspect", return_value=mock_inspector):
        # Case 1: Table doesn't exist
        mock_inspector.get_table_names.return_value = []
        database.ensure_additional_columns(mock_engine)
        
        # Case 2: Table exists, columns missing
        mock_inspector.get_table_names.return_value = ['datasets', 'dataset_validations']
        mock_inspector.get_columns.return_value = [] # No columns found
        
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        database.ensure_additional_columns(mock_engine)
        assert mock_conn.execute.called
        assert mock_conn.commit.called

def test_ensure_additional_columns_error_handling():
    mock_engine = MagicMock()
    mock_inspector = MagicMock()
    
    with patch("sqlalchemy.inspect", return_value=mock_inspector):
        mock_inspector.get_table_names.return_value = ['datasets']
        mock_inspector.get_columns.return_value = []
        
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.side_effect = Exception("DB Error")
        
        # Should not raise exception
        database.ensure_additional_columns(mock_engine)

def test_metadata_create_all_error():
    # Test error handling in module-level create_all
    with patch("sqlalchemy.sql.schema.MetaData.create_all", side_effect=Exception("Metadata Error")):
        # We need to reload to trigger the module-level try-except
        with patch("sqlalchemy.create_engine"):
            reload(database)
            # If we reached here, the exception was caught as expected
