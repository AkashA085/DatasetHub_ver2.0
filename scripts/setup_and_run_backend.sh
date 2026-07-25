#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# 1️⃣ Install NVIDIA Docker runtime (if not already installed)
# -------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}") && pwd)"
# Run the existing setup script which adds the required repos and installs nvidia-docker2
if ! command -v nvidia-docker >/dev/null 2>&1 && ! command -v nvidia-container-runtime >/dev/null 2>&1; then
  echo "Installing NVIDIA Docker runtime..."
  sudo bash "${SCRIPT_DIR}/setup_nvidia_runtime.sh"
else
  echo "NVIDIA Docker runtime already installed. Skipping setup script."
fi

# -------------------------------------------------------------------
# 2️⃣ Ensure Docker daemon knows about the "nvidia" runtime
# -------------------------------------------------------------------
DAEMON_JSON="/etc/docker/daemon.json"
NEED_RESTART=false
if sudo test -f "${DAEMON_JSON}"; then
  if ! sudo grep -q '"nvidia"' "${DAEMON_JSON}"; then
    echo "Adding nvidia runtime to existing ${DAEMON_JSON}"
    sudo jq '. + {"runtimes": {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}}}' "${DAEMON_JSON}" | sudo tee "${DAEMON_JSON}" > /dev/null
    NEED_RESTART=true
  fi
else
  echo "Creating ${DAEMON_JSON} with nvidia runtime configuration"
  sudo mkdir -p "$(dirname "${DAEMON_JSON}")"
  sudo tee "${DAEMON_JSON}" > /dev/null <<'EOF'
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
  NEED_RESTART=true
fi

# -------------------------------------------------------------------
# 3️⃣ Restart Docker daemon if configuration changed
# -------------------------------------------------------------------
if [ "$NEED_RESTART" = true ]; then
  echo "Restarting Docker daemon..."
  sudo systemctl restart docker
fi

# -------------------------------------------------------------------
# 4️⃣ Verify the runtime is registered with Docker
# -------------------------------------------------------------------
if ! docker info | grep -iq "nvidia"; then
  echo "ERROR: NVIDIA runtime not registered after configuration. Check Docker daemon logs."
  exit 1
fi

echo "✅ NVIDIA runtime successfully registered."

# -------------------------------------------------------------------
# 5️⃣ Build and launch the backend service using Docker Compose
# -------------------------------------------------------------------
cd "${SCRIPT_DIR}/.."  # Move to repository root (MODEL_TRAINIG_AGENTS)

echo "Building and starting backend..."
# The compose file already specifies `runtime: nvidia`; with the daemon configured this will work.

docker compose up -d --build backend

echo "🚀 Backend container should now be running. Use \`docker compose ps\` to verify."
