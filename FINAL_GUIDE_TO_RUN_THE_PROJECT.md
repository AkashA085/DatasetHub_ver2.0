# Project Execution & GPU-Accelerated Training Guide

This guide provides step-by-step instructions to run, rebuild, and maintain the GPU-accelerated YOLO training platform.

---

## 🛠️ Prerequisites
Before running the project, ensure your host machine has:
1. **NVIDIA GPU Drivers** installed (verify with `nvidia-smi` on the host).
2. **Docker** and **Docker Compose** installed.
3. **NVIDIA Container Toolkit** installed and registered as a Docker runtime.

---

## 🚀 Scenario A: Starting from a Clean Slate (Docker fully vanished)

If the Docker daemon is fresh or the containers/images/networks have been completely cleared, follow these steps to set up the workspace:

### Step 1: Configure the NVIDIA Container Runtime on the Host
The host needs to register the `nvidia` container runtime with Docker. Use the helper scripts located in the `./scripts` directory:

1. Allow executing the scripts:
   ```bash
   chmod +x scripts/setup_nvidia_runtime.sh
   chmod +x scripts/setup_and_run_backend.sh
   ```
2. Run the registration and setup script:
   ```bash
   ./scripts/setup_and_run_backend.sh
   ```
   *This script installs the NVIDIA Docker runtime packages (if missing), adds `"nvidia"` to `/etc/docker/daemon.json`, restarts the Docker daemon, and builds the backend.*

### Step 2: Build and Start the Entire Stack
To build and start the database, backend, and frontend containers in the background:
```bash
docker compose up -d --build
```

---

## ⚡ Scenario B: Running the Existing Build (Fast Start / Local Development)

If the containers have already been built and the Docker daemon is configured, you don't need to rebuild or restart the containers when making changes to the source code.

### Start the Services
```bash
docker compose up -d
```
All services will boot up in seconds:
* **Frontend Application**: Accessible at [http://localhost:5174](http://localhost:5174)
* **Backend API server**: Running at [http://localhost:8001/api/v1](http://localhost:8001/api/v1)
* **PostgreSQL Database**: Listening on port `5432` internally

### Stop the Services
```bash
docker compose down
```

---

## 🔥 Hot-Reloading & Live Development

The Docker container stack is configured for **production-level developer experience** with automatic, instant updates:

### Backend Auto-Reload
* **How it works**: The host's `./backend/Dataset_Management_tool` folder is mounted directly to `/app` inside the container.
* **Result**: Any changes to Python files in the backend trigger Uvicorn's `StatReload` automatically. The server restarts instantly inside the container.

### Frontend Hot-Module-Replacement (HMR)
* **How it works**: The host's `./frontend/Dataset_Management_tool_frontend` folder is mounted to `/app` inside the container.
* **Result**: Vite dev server runs inside the container and watches for frontend changes. When you update any JSX, CSS, or JS component, the browser updates instantly without a page refresh.

---

## 📋 Common Management & Debugging Commands

### View Logs
* **All Services**:
  ```bash
  docker compose logs -f
  ```
* **Backend Only**:
  ```bash
  docker compose logs backend -f
  ```
* **Frontend Only**:
  ```bash
  docker compose logs frontend -f
  ```

### Verify GPU Detection Inside Backend
To verify that the backend container can see the host's GPU:
```bash
docker compose exec backend nvidia-smi
```

### Access Postgres Database CLI
```bash
docker compose exec db psql -U postgres -d dataset_management
```

---

## 📦 File Paths and Weights Downloads
* **Zip Archives & Results**: When a training job completes, clicking **Download Full Training Folder (ZIP)** will retrieve the compressed run direct from the backend API using the `API_BASE_URL` dynamically.
* **Weights files (`best.pt`, `last.pt`)**: Served directly via the backend static serving path `/storage/training/jobs/...`.
