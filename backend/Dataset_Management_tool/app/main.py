from pathlib import Path
from urllib.parse import unquote
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio, os, shutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
dotenv_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path)

# NOW import everything else that needs DATABASE_URL
from .api.routes import upload, labeling, export, augmentation, datasets, training
from .utils.file_utils import ensure_dirs, EXPORTS_DIR, STORAGE_ROOT
STORAGE_ROOTS = [STORAGE_ROOT]

app = FastAPI(title="Dataset Management Backend")
# Configure CORS for frontend access
allowed_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ensure storage directories exist on startup
ensure_dirs()

# Initialize & create DB tables (will use DATABASE_URL env var if provided)
from .core import Base, engine, ensure_additional_columns
# Ensure any newly-added JSON columns exist (safe / idempotent helper)
try:
    ensure_additional_columns(engine)
except Exception:
    pass

from fastapi import Request
from fastapi.responses import JSONResponse

class SuppressBrokenPipeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            response = await call_next(request)
            return response

        except asyncio.CancelledError:
            print("Client disconnected")
            raise

        except Exception as e:
            print(f"Unhandled error: {e}")
            raise

app.add_middleware(SuppressBrokenPipeMiddleware)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    is_polling = "/train/jobs" in request.url.path

    try:
        if not is_polling:
            print(
                f"Request: {request.method} {request.url} "
                f"Content-Length: {request.headers.get('content-length', 'unknown')}"
            )

        response = await call_next(request)

        if not is_polling:
            print(f"Response: {response.status_code}")

        return response

    except Exception as e:
        print(f"Middleware error: {e}")
        raise e

@app.middleware("http")
async def limit_request_size(request: Request, call_next):

    if request.url.path.endswith("/upload-dataset"):
        max_size = 10 * 1024 * 1024 * 1024  # 10GB
    else:
        max_size = 100 * 1024 * 1024  # 100MB

    try:
        content_length = request.headers.get("content-length")

        if content_length:
            size = int(content_length)

            if size > max_size:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"Request too large. Maximum size is {max_size} bytes."
                    },
                )

        response = await call_next(request)
        return response

    except Exception as e:
        print(f"Request middleware failed: {e}")
        raise e


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
app.include_router(datasets.router, prefix="/api/v1", tags=["Datasets"])
app.include_router(upload.router, prefix="/api/v1", tags=["Upload & Analysis"])
app.include_router(augmentation.router, prefix="/api/v1", tags=["Augmentation"])
app.include_router(labeling.router, prefix="/api/v1", tags=["Labeling"])
from .train_endpoint import router as train_router
app.include_router(train_router, prefix="/api/v1", tags=["Training"])
app.include_router(training.router, prefix="/api/v1", tags=["Training"])


@app.get("/health")
async def health():
    status = {"status": "ok"}
    try:
        from .core.database import engine
        with engine.connect() as conn:
            conn.execute(__import__('sqlalchemy').text("SELECT 1"))
        status["database"] = "connected"
    except Exception:
        status["database"] = "disconnected"
        status["status"] = "degraded"
    try:
        import torch
        status["gpu"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
    except Exception:
        status["gpu"] = False
    status["disk_free_gb"] = round(shutil.disk_usage(str(STORAGE_ROOT)).free / (1024**3), 1)
    return status


@app.get("/")
async def root():
    return {"message": "Welcome to the Dataset Management Backend API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
