# Quick Start Guide

## Prerequisites
- Python 3.9+
- PostgreSQL 12+ 
- pip

## 1. Clone and Setup

```bash
cd dataset_backend
pip install -r requirements.txt
```

## 2. Configure Database

### Option A: Using Environment Variable
```bash
export DATABASE_URL="postgresql+psycopg2://postgres:password@localhost:5432/Dataset_management"
```

### Option B: Using .env File
```bash
cp .env.example .env
# Edit .env and set your DATABASE_URL
```

## 3. Run the Application

```bash
uvicorn app.main:app --reload --port 8000
```

## 4. Access the API

- **API Base URL**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 5. Test an Endpoint

```bash
curl -X GET http://localhost:8000/
```

Should return:
```json
{"message": "Welcome to the Dataset Management Backend API"}
```

## Common Issues

### Error: DATABASE_URL not provided
```
ValueError: DATABASE_URL environment variable required
```
**Solution**: Set the DATABASE_URL environment variable
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/Database_management"
```

### Error: psycopg2 connection failed
```
psycopg2.OperationalError: connection to server... failed
```
**Solution**: Make sure PostgreSQL is running and your connection string is correct

### Error: ModuleNotFoundError
```
ModuleNotFoundError: No module named 'app'
```
**Solution**: Make sure you're running from the `dataset_backend` directory:
```bash
cd dataset_backend
uvicorn app.main:app --reload
```

## Project Structure

```
dataset_backend/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── core/                # Database & ORM
│   ├── api/routes/          # API endpoints
│   ├── services/            # Business logic
│   ├── models/              # Pydantic schemas
│   ├── utils/               # Utilities
│   └── storage/             # Dataset files
├── requirements.txt
├── .env.example
└── ARCHITECTURE.md
```

## API Endpoints

### Upload Dataset
```bash
curl -X POST http://localhost:8000/upload-dataset \
  -F "images_zip=@images.zip" \
  -F "labels_zip=@labels.zip" \
  -F "format_type=yolo"
```

### Download Dataset  
```bash
curl -X GET http://localhost:8000/download/{dataset_id} -o dataset.zip
```

### Augment Dataset
```bash
curl -X POST http://localhost:8000/augment \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "...",
    "augmentation_type": "rotation",
    "augmentation_config": {...}
  }'
```

### Update Labels
```bash
curl -X POST http://localhost:8000/label \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "...",
    "image_name": "image.jpg",
    "labels": [...]
  }'
```

## Database Tables

The following tables are auto-created:
- `users` - User accounts
- `projects` - User projects
- `datasets` - Dataset metadata
- `images` - Image records
- `labels` - Object labels
- `dataset_validations` - Validation reports
- `class_distributions` - Class statistics

## Environment Variables

**Required:**
- `DATABASE_URL` - PostgreSQL connection string

**Optional:**
- `ENVIRONMENT` - `development` or `production` (default: development)

## Next Steps

1. **Read the documentation**
   - [ARCHITECTURE.md](ARCHITECTURE.md) - Complete architecture guide
   - [BEFORE_AFTER.md](BEFORE_AFTER.md) - Structural changes made

2. **Test the API**
   - Go to http://localhost:8000/docs for interactive API docs

3. **Check the code**
   - Review the source files in `app/`
   - Read comments in `app/core/database.py` for model definitions

4. **Deploy to production**
   - Set `ENVIRONMENT=production`
   - Use a production database server
   - Set up proper environment variables
   - Use a production ASGI server like Gunicorn

## Support

For questions about the structure:
- See [ARCHITECTURE.md](ARCHITECTURE.md)
- See [BEFORE_AFTER.md](BEFORE_AFTER.md)
- See [MIGRATION_CHECKLIST.md](MIGRATION_CHECKLIST.md)

## Production Deployment

For production, use:
```bash
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

**Happy coding!** 
