"""Database configuration and ORM models."""

import os
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, ForeignKey, Boolean, Float, JSON
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # For development: use PostgreSQL if available, else raise error for production awareness
    raise ValueError(
        "DATABASE_URL environment variable required. "
        "Set it to: postgresql+psycopg2://user:password@host:5432/Database_management"
    )

engine = create_engine(DATABASE_URL, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db():
    """Dependency for FastAPI to provide database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_additional_columns(engine):
    """Run lightweight, idempotent ALTER TABLE statements to add any newly-introduced JSON columns.
    This is NOT a full migration system but helps create missing columns in development.
    """
    from sqlalchemy import inspect, text
    inspector = inspect(engine)

    # datasets table: ensure `analysis_summary` exists
    cols = {c['name'] for c in inspector.get_columns('datasets')} if 'datasets' in inspector.get_table_names() else set()
    if 'analysis_summary' not in cols:
        try:
            with engine.connect() as conn:
                conn.execute(text('ALTER TABLE datasets ADD COLUMN analysis_summary JSON'))
                conn.commit()
        except Exception:
            pass

    # dataset_validations table: add JSON list columns
    cols = {c['name'] for c in inspector.get_columns('dataset_validations')} if 'dataset_validations' in inspector.get_table_names() else set()
    missing = []
    for c in ['class_ids_found', 'missing_label_images', 'orphan_label_files', 'empty_label_files', 'corrupted_image_files']:
        if c not in cols:
            missing.append(c)
    if missing:
        try:
            with engine.connect() as conn:
                for c in missing:
                    conn.execute(text(f'ALTER TABLE dataset_validations ADD COLUMN {c} JSON'))
                conn.commit()
        except Exception:
            pass


# ============================================================================
# ORM Models (SQLAlchemy)
# ============================================================================

class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True)
    email = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    projects = relationship("Project", back_populates="user")


class Project(Base):
    __tablename__ = "projects"
    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="projects")
    datasets = relationship("Dataset", back_populates="project")


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String(36), primary_key=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    format_type = Column(String(50))
    total_images = Column(Integer, default=0)
    total_labels = Column(Integer, default=0)
    total_classes = Column(Integer, default=0)
    total_objects = Column(Integer, default=0)
    avg_objects_per_image = Column(Float, default=0.0)
    missing_label_count = Column(Integer, default=0)
    corrupted_image_count = Column(Integer, default=0)
    csv_file_path = Column(String(1024))
    zip_file_path = Column(String(1024))
    # Store full analysis summary JSON (mirror API `analysis_summary`)
    analysis_summary = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="datasets")
    images = relationship("Image", back_populates="dataset")
    validation = relationship("DatasetValidation", uselist=False, back_populates="dataset")
    class_distributions = relationship("ClassDistribution", back_populates="dataset")


class Image(Base):
    __tablename__ = "images"
    id = Column(String(36), primary_key=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    file_name = Column(String(1024))
    file_path = Column(String(4096))
    has_label = Column(Boolean, default=False)

    dataset = relationship("Dataset", back_populates="images")
    labels = relationship("Label", back_populates="image")


class Label(Base):
    __tablename__ = "labels"
    id = Column(String(36), primary_key=True)
    image_id = Column(String(36), ForeignKey("images.id"), nullable=True)
    class_id = Column(String(50))
    bbox_data = Column(JSON)

    image = relationship("Image", back_populates="labels")


class DatasetValidation(Base):
    __tablename__ = "dataset_validations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, unique=True)
    total_images = Column(Integer)
    total_labels = Column(Integer)
    missing_labels = Column(Integer)
    orphan_labels = Column(Integer)
    empty_labels = Column(Integer)
    corrupted_images = Column(Integer)
    # Additional fields to mirror ValidationReport
    class_ids_found = Column(JSON, nullable=True)
    missing_label_images = Column(JSON, nullable=True)
    orphan_label_files = Column(JSON, nullable=True)
    empty_label_files = Column(JSON, nullable=True)
    corrupted_image_files = Column(JSON, nullable=True)

    dataset = relationship("Dataset", back_populates="validation")


class ClassDistribution(Base):
    __tablename__ = "class_distributions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    class_id = Column(String(50))
    object_count = Column(Integer)

    dataset = relationship("Dataset", back_populates="class_distributions")


class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id = Column(String(36), primary_key=True)  # job_id
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    status = Column(String(50), default="queued")  # queued, preparing, running, completed, failed
    params = Column(JSON, nullable=True)  # training parameters
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    metrics = Column(JSON, nullable=True)  # training metrics
    artifacts = Column(JSON, nullable=True)  # model weights paths, etc.
    mlflow = Column(JSON, nullable=True)  # mlflow tracking info
    error = Column(String(2048), nullable=True)  # error message if failed
    logs = Column(JSON, nullable=True)  # list of log messages

    dataset = relationship("Dataset")
