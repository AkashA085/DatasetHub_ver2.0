import os
import subprocess
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.file_utils import STORAGE_ROOT

router = APIRouter()

PROJECT_ROOT = STORAGE_ROOT.parent  # Project root


class TrainRequest(BaseModel):
    """
    `dataset_yaml` – path **relative to the project root**, e.g.
    "datasets/my_dataset.yaml".
    """
    dataset_yaml: str
    use_gpu: bool = True


@router.post("/train")
def start_training(req: TrainRequest):
    dataset_path = os.path.abspath(os.path.join(str(PROJECT_ROOT), req.dataset_yaml))

    if not os.path.isfile(dataset_path):
        raise HTTPException(status_code=400, detail="Dataset yaml not found")

    training_cmd = f"""
    cd {PROJECT_ROOT}/backend/Dataset_Management_tool && \
    python3 -c "
import sys; sys.path.insert(0, '.')
from app.utils.file_utils import STORAGE_ROOT
os.environ.setdefault('DATASET_STORAGE_ROOT', str(STORAGE_ROOT))
os.environ.setdefault('DATABASE_URL', 'postgresql+psycopg2://postgres:postgres@localhost:5432/dataset_management')
exec(open('retrain_model.py').read().replace('sys.argv[1]', '\"{dataset_path}\"'))
"
"""
    logs_dir = str(PROJECT_ROOT / "training_logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"{int(__import__('time').time())}.log")

    try:
        subprocess.Popen(
            ["bash", "-c", training_cmd],
            stdout=open(log_file, "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {e}")

    return {"status": "training_started", "log_file": log_file}
