import pytest
from app.db.models import User, Project
from app.db.session import engine, SessionLocal

def test_db_models_instantiation():
    u = User(id="1", email="test@test.com")
    assert u.id == "1"
    p = Project(id="2", name="test", user_id="1")
    assert p.name == "test"

def test_db_session_creation():
    session = SessionLocal()
    assert session is not None
    session.close()
