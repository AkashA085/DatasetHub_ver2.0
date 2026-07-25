#!/bin/bash
# Run the frontend locally (no Docker)
cd "$(dirname "$0")/frontend/Dataset_Management_tool_frontend"

echo "Starting frontend on http://0.0.0.0:5173"
exec npm run dev -- --host 0.0.0.0
