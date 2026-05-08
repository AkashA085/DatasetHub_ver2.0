from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
load_dotenv()

# NOW import everything else that needs DATABASE_URL
from app.api.routes import upload, labeling, export, augmentation, datasets, training
from app.utils.file_utils import ensure_dirs, EXPORTS_DIR, STORAGE_ROOT

STORAGE_ROOTS = [STORAGE_ROOT]

app = FastAPI(title="Dataset Management Backend", max_request_size=100*1024*1024*1024)  # 100GB limit

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ensure storage directories exist on startup
ensure_dirs()

# Initialize & create DB tables (will use DATABASE_URL env var if provided)
from app.core import Base, engine, ensure_additional_columns
Base.metadata.create_all(bind=engine)
# Ensure any newly-added JSON columns exist (safe / idempotent helper)
try:
    ensure_additional_columns(engine)
except Exception:
    pass

from fastapi import Request
from fastapi.responses import JSONResponse

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Request: {request.method} {request.url} Content-Length: {request.headers.get('content-length', 'unknown')}")
    response = await call_next(request)
    print(f"Response: {response.status_code}")
    return response

@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    # Allow large requests for upload-dataset
    if request.url.path == "/upload-dataset":
        max_size = 100 * 1024 * 1024 * 1024  # 100GB
    else:
        max_size = 100 * 1024 * 1024  # 100MB for other requests
    
    content_length = request.headers.get("content-length")
    if content_length:
        size = int(content_length)
        if size > max_size:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request too large. Maximum size is {max_size} bytes."}
            )
    
    response = await call_next(request)
    return response


def _resolve_storage_file(file_path: str) -> Path | None:
    # URL path is always relative to one of our storage roots.
    raw = unquote(file_path.replace("\\", "/")).lstrip("/")
    rel = Path(raw)
    if rel.is_absolute() or ".." in rel.parts:
        return None

    for root in STORAGE_ROOTS:
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


@app.get("/storage/{file_path:path}")
async def serve_storage_file(file_path: str):
    file_on_disk = _resolve_storage_file(file_path)
    if not file_on_disk:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_on_disk)

# Mount exports directory for static file serving
app.mount("/files", StaticFiles(directory=EXPORTS_DIR), name="files")

# Include routers
app.include_router(datasets.router, tags=["Datasets"])
app.include_router(upload.router, tags=["Upload & Analysis"])
app.include_router(augmentation.router, tags=["Augmentation"])
app.include_router(labeling.router, tags=["Labeling"])
app.include_router(export.router, tags=["Export"])
app.include_router(training.router, tags=["Training"])


@app.get("/")
async def root():
    return {"message": "Welcome to the Dataset Management Backend API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
