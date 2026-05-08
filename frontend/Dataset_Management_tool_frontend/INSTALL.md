# Backend Installation Guide

## Prerequisites
- Windows
- Python 3.13 installed
- Git and command line access
- Optional: NVIDIA GPU drivers and CUDA for GPU training

## Recommended setup
1. Open PowerShell in `d:\datasethub\backend\Dataset_Management_tool`
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Upgrade pip:
   ```powershell
   python -m pip install --upgrade pip
   ```

## Install backend dependencies
1. Install common dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
2. Install PyTorch separately:
   - CPU only:
     ```powershell
     python -m pip install torch torchvision torchaudio
     ```
   - NVIDIA GPU (CUDA 12.1):
     ```powershell
     python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - NVIDIA GPU (CUDA 11.8):
     ```powershell
     python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

## Verify install
1. Check package installation:
   ```powershell
   .\.venv\Scripts\python.exe -m pip list | findstr /R "fastapi uvicorn pydantic numpy pandas mlflow ultralytics albumentations"
   ```
2. Test backend startup:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
3. Optional GPU verification:
   ```powershell
   .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
   ```

## Notes
- `pydantic` is pinned to `2.9.2` to satisfy both FastAPI and Albumentations.
- `pandas` is pinned to `2.2.3` to satisfy MLflow.
- If a dependency install fails, use `python -m pip install --prefer-binary -r requirements.txt`.

## Starting the backend
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Common issues
- If NumPy tries to build from source, make sure `numpy==2.4.4` is installed first:
  ```powershell
  python -m pip install numpy==2.4.4
  ```
- For GPU training, install the PyTorch CUDA wheel from the official PyTorch index.
