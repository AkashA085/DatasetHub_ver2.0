import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Prefer DATABASE_URL env var (Postgres). If not provided, fall back to a local sqlite file for development.
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgresql"):
    engine = create_engine(DATABASE_URL, future=True)
else:
    # sqlite fallback (development) — safe default so the app still runs without a PG server
    SQLITE_URL = os.getenv("DEV_DATABASE_URL", "sqlite:///./dataset_management.db")
    engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False}, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
