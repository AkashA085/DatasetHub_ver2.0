# DatasetHub

This repository contains a full dataset management application with:

- **Backend**: FastAPI + SQLAlchemy + YOLOv8 training
- **Frontend**: React + Vite
- **Docker Compose**: PostgreSQL, backend, and frontend services

## Project Structure

- `backend/Dataset_Management_tool` – Backend application
- `frontend/Dataset_Management_tool_frontend` – Frontend application
- `docker-compose.yml` – Compose stack for local development

---

## Backend Setup

### 1. Create and activate the Python virtual environment
```powershell
cd d:\datasethub\backend\Dataset_Management_tool
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### 3. Install backend dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Install PyTorch separately

For CPU-only usage:
```powershell
python -m pip install torch torchvision torchaudio
```

For NVIDIA GPU support:
```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Verify the backend
```powershell
python -c "import app.main; print('IMPORT_OK')"
```

### 6. Start the backend server
```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Frontend Setup

### 1. Install Node.js and npm
Make sure Node.js 16+ is installed.

### 2. Install frontend dependencies
```powershell
cd d:\datasethub\frontend\Dataset_Management_tool_frontend
npm install
```

### 3. Start the frontend
```powershell
npm run dev
```

The frontend will normally run on `http://localhost:5173` and should connect to the backend at `http://localhost:8000`.

---

## Docker Compose Setup

If you prefer a containerized setup, use Docker Compose from the repository root:

```powershell
docker compose up --build
```

This will start:

- `db` on PostgreSQL
- `backend` on `http://localhost:8000`
- `frontend` on `http://localhost:5173`

---

## Notes

- Backend install is pinned to compatible versions to avoid dependency conflicts.
- If dependencies fail, try:
  ```powershell
  python -m pip install --prefer-binary -r requirements.txt
  ```
- GPU training support requires installing the correct PyTorch CUDA wheel.
- The file `backend/Dataset_Management_tool/INSTALL.md` includes step-by-step backend installation notes.

---

## Useful commands

```powershell
# backend
cd backend/Dataset_Management_tool
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# frontend
cd frontend/Dataset_Management_tool_frontend
npm install
npm run dev

# docker
cd d:\datasethub
docker compose up --build
```
