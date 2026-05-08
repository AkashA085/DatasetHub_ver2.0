import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean, Float, JSON
from sqlalchemy.orm import relationship
from .session import Base


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
    # store full analysis summary JSON
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
    # Mirror ValidationReport lists
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
