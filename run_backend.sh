#!/bin/bash
# Run the backend locally (no Docker)
cd "$(dirname "$0")/backend/Dataset_Management_tool"

echo "Starting backend on http://0.0.0.0:8001"
exec python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --timeout-keep-alive 120
