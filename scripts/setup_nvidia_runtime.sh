#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# 1️⃣ Clean old (incorrect) NVIDIA repository files (if they exist)
# -------------------------------------------------------------------
sudo rm -f /etc/apt/sources.list.d/nvidia-docker.list \
           /etc/apt/sources.list.d/nvidia-container-runtime.list \
           /etc/apt/sources.list.d/libnvidia-container.list

# -------------------------------------------------------------------
# 2️⃣ Determine distribution string (should be "ubuntu22.04" for Jammy)
# -------------------------------------------------------------------
export DISTRIBUTION=$(. /etc/os-release && echo "$ID$VERSION_ID")
if [[ "$DISTRIBUTION" != "ubuntu22.04" ]]; then
  echo "Warning: detected distribution $DISTRIBUTION – script expects ubuntu22.04"
fi

echo "Using distribution: $DISTRIBUTION"

# -------------------------------------------------------------------
# 3️⃣ Add the libnvidia-container repository
# -------------------------------------------------------------------
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/libnvidia-container-archive-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/${DISTRIBUTION}/amd64/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/libnvidia-container.list

# -------------------------------------------------------------------
# 4️⃣ Add the nvidia-container-runtime repository
# -------------------------------------------------------------------
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-runtime-archive-keyring.gpg
curl -s -L https://nvidia.github.io/nvidia-container-runtime/${DISTRIBUTION}/amd64/nvidia-container-runtime.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

# -------------------------------------------------------------------
# 5️⃣ Add the nvidia-docker repository
# -------------------------------------------------------------------
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker-archive-keyring.gpg
curl -s -L https://nvidia.github.io/nvidia-docker/${DISTRIBUTION}/amd64/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# -------------------------------------------------------------------
# 6️⃣ Install the NVIDIA Docker runtime package
# -------------------------------------------------------------------
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# -------------------------------------------------------------------
# 7️⃣ Restart Docker daemon so it picks up the new runtime
# -------------------------------------------------------------------
sudo systemctl restart docker

# -------------------------------------------------------------------
# 8️⃣ Verify the runtime is registered (should list "nvidia")
# -------------------------------------------------------------------
docker info | grep -i runtime

echo "NVIDIA Docker runtime installation complete."
