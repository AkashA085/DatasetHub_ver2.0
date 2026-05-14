import os
import pytest
from unittest.mock import patch, MagicMock
from importlib import reload
import sqlalchemy

# We import the module once to have the reference for reload
import app.db.session as session

def test_get_db():
    # Test the get_db generator
    # We don't need reload here, just use the imported version
    gen = session.get_db()
    db = next(gen)
    assert db is not None
    # Simulate closing
    try:
        next(gen)
    except StopIteration:
        pass

def test_database_url_postgresql_valid():
    # Test postgres engine creation
    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost/db"}):
        # Patch create_engine in sqlalchemy because session.py imports it
        with patch("sqlalchemy.create_engine") as mock_create_engine:
            reload(session)
            assert session.DATABASE_URL == "postgresql://user:pass@localhost/db"
            mock_create_engine.assert_called()
            # Find the call that uses the postgres URL
            found = False
            for call in mock_create_engine.call_args_list:
                if call.args[0] == "postgresql://user:pass@localhost/db":
                    found = True
                    break
            assert found, f"create_engine not called with expected URL. Calls: {mock_create_engine.call_args_list}"

def test_database_url_postgresql_invalid_host():
    # Test fallback to sqlite when host is "host"
    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@host/db"}):
        with patch("sqlalchemy.create_engine") as mock_create_engine:
            reload(session)
            assert session.DATABASE_URL is None
            # It should fall back to sqlite
            mock_create_engine.assert_called()
            # Check that at least one call was for sqlite
            found_sqlite = False
            for call in mock_create_engine.call_args_list:
                if "sqlite" in str(call.args[0]):
                    found_sqlite = True
                    break
            assert found_sqlite

def test_database_url_none():
    # Test fallback when no DATABASE_URL is provided
    with patch.dict(os.environ, {}, clear=True):
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        with patch("sqlalchemy.create_engine") as mock_create_engine:
            reload(session)
            assert session.DATABASE_URL is None
            mock_create_engine.assert_called()
            found_sqlite = False
            for call in mock_create_engine.call_args_list:
                if "sqlite" in str(call.args[0]):
                    found_sqlite = True
                    break
            assert found_sqlite
