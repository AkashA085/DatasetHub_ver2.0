# FastAPI Dataset Management Backend - Restructured Architecture

## Project Structure

```
app/
├── main.py                    # FastAPI application entry point
├── core/
│   ├── __init__.py
│   └── database.py            # SQLAlchemy engine, models, and session management
├── models/
│   └── schemas.py             # Pydantic request/response schemas
├── services/
│   ├── validator.py           # Dataset validation logic
│   ├── analyzer.py            # Dataset analysis and statistics
│   ├── export_service.py      # Dataset export functionality
│   ├── format_converter.py    # Format conversion utilities
│   ├── augmentation.py        # Data augmentation service
│   ├── splitter.py            # Train/val/test split logic
│   └── zipper.py              # Compression utilities
├── api/
│   ├── __init__.py
│   └── routes/
│       ├── __init__.py
│       ├── upload.py          # Dataset upload endpoint
│       ├── export.py          # Dataset download endpoint
│       ├── labeling.py        # Label update endpoint
│       └── augmentation.py    # Augmentation endpoint
├── utils/
│   └── file_utils.py          # File and directory utilities
└── storage/                   # Dataset files on disk

migrations/
.env
.env.example
requirements.txt
```

## Getting Started

### Prerequisites
- Python 3.9+
- PostgreSQL 12+ (required - no SQLite fallback)
- pip

### Installation

1. **Clone the repo** and navigate to the project:
```bash
cd dataset_backend
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env and set your DATABASE_URL
# Example: postgresql+psycopg2://postgres:password@localhost:5432/Dataset_management
```

4. **Run the application**:
```bash
export DATABASE_URL="postgresql+psycopg2://postgres:password@localhost:5432/Dataset_management"
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Upload & Analysis
- **POST** `/upload-dataset` - Upload and analyze a new dataset
- **POST** `/augment` - Augment an existing dataset

### Import & Export
- **GET** `/download/{dataset_id}` - Download processed dataset
- **POST** `/label` - Update dataset labels

## Database

### Models
The application uses SQLAlchemy ORM with the following tables:

- `users` - User accounts
- `projects` - User projects
- `datasets` - Dataset metadata
- `images` - Image records
- `labels` - Object labels in images
- `dataset_validations` - Validation reports
- `class_distributions` - Class frequency statistics

Tables are automatically created on application startup.

## Environment Variables

**Required:**
- `DATABASE_URL` - PostgreSQL connection string (format: `postgresql+psycopg2://user:password@host:5432/dbname`)

**Optional:**
- `ENVIRONMENT` - Set to `development` or `production` (default: `development`)

## Storage

Dataset files are stored on disk at:
```
storage/{user_id}/{project_id}/{dataset_id}/
├── images/
└── labels/
```

No image binaries are stored in the database - only metadata and file paths.

## Architecture

### Core (`app/core/`)
Database configuration and ORM models using SQLAlchemy.

### Models (`app/models/`)
Pydantic schemas for FastAPI request/response validation.

### Services (`app/services/`)
Business logic including:
- Validation and analysis
- Format conversion
- Data augmentation
- File compression

### API Routes (`app/api/routes/`)
FastAPI endpoint handlers for:
- Dataset upload
- Export/download
- Labeling
- Augmentation

### Utils (`app/utils/`)
Helper functions for file operations.

## Notes

- All endpoint paths remain unchanged from the original implementation
- Response formats are identical to the original API
- All business logic and validation functions work exactly as before
- The restructuring only reorganizes files into a production-ready layout
- Database persistence is appended without breaking any existing functionality
