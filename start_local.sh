#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Cleanup on exit ────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down services..."
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    pkill -f "vite.*--host" 2>/dev/null || true
    echo "All services stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ── PostgreSQL ──────────────────────────────────────────────
docker start local-postgres 2>/dev/null || docker run -d \
  --name local-postgres \
  -e POSTGRES_DB=dataset_management \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -v model_trainig_agents_postgres_data:/var/lib/postgresql/data \
  --restart unless-stopped \
  postgres:16-alpine

echo "Waiting for PostgreSQL..."
until pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1; do
    sleep 1
done
echo "PostgreSQL is ready."

# ── Kill any existing backend/frontend ──────────────────────
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite.*--host" 2>/dev/null || true
sleep 1

# ── Backend ─────────────────────────────────────────────────
echo "Starting backend on port 8001..."
setsid bash -c "cd '$DIR/backend/Dataset_Management_tool' && exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 120" > /tmp/backend.log 2>&1 &
echo "  Backend started"

# ── Frontend ────────────────────────────────────────────────
echo "Starting frontend on port 5173..."
setsid bash -c "cd '$DIR/frontend/Dataset_Management_tool_frontend' && exec npm run dev -- --host 0.0.0.0" > /tmp/frontend.log 2>&1 &
echo "  Frontend started"

# ── Wait for services to be ready ───────────────────────────
echo ""
echo "Waiting for backend..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8001/health 2>/dev/null | grep -q '"status"'; then
        echo "Backend is ready!"
        break
    fi
    sleep 1
done

echo "Waiting for frontend..."
for i in $(seq 1 15); do
    if curl -s -o /dev/null http://localhost:5173/ 2>/dev/null; then
        echo "Frontend is ready!"
        break
    fi
    sleep 1
done

echo ""
echo "========================================="
echo "  All services started!"
echo "  Backend:  http://localhost:8001"
echo "  Frontend: http://localhost:5173"
echo "  Database: localhost:5432 (Docker)"
echo "  API Docs: http://localhost:8001/docs"
echo "========================================="
echo ""
echo "Logs:"
echo "  tail -f /tmp/backend.log"
echo "  tail -f /tmp/frontend.log"
echo ""
echo "Press Ctrl+C to stop all services."

wait
