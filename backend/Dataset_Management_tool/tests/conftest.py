import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.core.database import get_db, Base

# Use a separate SQLite database for testing to avoid touching production data
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session(db_engine):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def client(db_session):
    # Override get_db to use the test session
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
@pytest.fixture
def sample_dataset(db_session):
    # Setup: Create a User, Project, and Dataset
    import uuid
    from app.core.database import Dataset, Project, User
    user = User(id=str(uuid.uuid4()), email="test@example.com")
    db_session.add(user)
    db_session.commit()
    
    project = Project(id=str(uuid.uuid4()), name="Test Project", user_id=user.id)
    db_session.add(project)
    db_session.commit()
    
    dataset = Dataset(
        id=str(uuid.uuid4()),
        project_id=project.id,
        format_type="yolo",
        total_images=10,
        total_labels=5
    )
    db_session.add(dataset)
    db_session.commit()
    return dataset
