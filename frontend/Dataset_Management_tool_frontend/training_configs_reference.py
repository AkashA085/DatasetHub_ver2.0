#!/usr/bin/env python3
"""
Quick Reference: Training Configurations
Pre-configured settings for different scenarios
"""

# === RECOMMENDED TRAINING CONFIGURATIONS ===

# 1. FAST TRAINING (Quick prototyping)
FAST_TRAINING = {
    "epochs": 20,
    "batch_size": 32,
    "image_size": 416,
    "device": "0",  # GPU
    "learning_rate": 0.001,
    "val_split": 0.1,
    "test_split": 0.05,
    "augmentation_enabled": False,  # Skip to save time
}
# Expected time: 20-30 minutes

# 2. BALANCED TRAINING (Good accuracy & speed)
BALANCED_TRAINING = {
    "epochs": 100,
    "batch_size": 64,
    "image_size": 640,
    "device": "0",  # GPU
    "learning_rate": 0.001,
    "val_split": 0.2,
    "test_split": 0.1,
    "augmentation_enabled": True,
}
# Expected time: 60-90 minutes

# 3. HIGH QUALITY TRAINING (Best accuracy, slower)
HIGH_QUALITY_TRAINING = {
    "epochs": 300,
    "batch_size": 128,
    "image_size": 1280,
    "device": "0",  # GPU
    "learning_rate": 0.0001,
    "val_split": 0.2,
    "test_split": 0.1,
    "augmentation_enabled": True,
    "register_best_model": True,
}
# Expected time: 3-4 hours
# Requires: 24GB VRAM GPU

# 4. RESOURCE-LIMITED TRAINING (Limited GPU memory)
LIMITED_GPU_TRAINING = {
    "epochs": 100,
    "batch_size": 16,  # Small batch for 6-8GB GPU
    "image_size": 512,
    "device": "0",
    "learning_rate": 0.001,
    "val_split": 0.2,
    "test_split": 0.1,
    "augmentation_enabled": True,
}
# Expected time: 2-3 hours
# For: 6-8GB VRAM GPUs

# 5. CPU TRAINING (No GPU)
CPU_TRAINING = {
    "epochs": 30,
    "batch_size": 8,
    "image_size": 416,
    "device": "cpu",
    "learning_rate": 0.001,
    "val_split": 0.2,
    "test_split": 0.1,
    "augmentation_enabled": False,
}
# Expected time: 8-12 hours
# Note: Very slow, not recommended for production

# === GPU MEMORY REQUIREMENTS ===

BATCH_SIZE_BY_GPU = {
    # GPU Memory: [image_size_416, image_size_640, image_size_1280]
    "4GB": [8, 4, 2],       # Entry-level GPU
    "6GB": [16, 8, 4],      # Mid-range
    "8GB": [32, 16, 8],     # Common
    "12GB": [64, 32, 16],   # Good
    "16GB": [64, 32, 16],   # Very Good
    "24GB": [128, 64, 32],  # Excellent
    "48GB": [256, 128, 64], # Enterprise
}

# === LEARNING RATE RECOMMENDATIONS ===

LEARNING_RATE_BY_SIZE = {
    "small": 0.001,      # yolov8n - Fast convergence
    "medium": 0.0005,    # yolov8m - Balanced
    "large": 0.0001,     # yolov8l - Slower convergence
    "xlarge": 0.00005,   # yolov8x - Very slow
}

# === API CALL EXAMPLES ===

"""
Example 1: Fast prototyping on RTX 3060 (12GB)
POST /train/start
{
    "dataset_id": "my-dataset",
    "device": "0",
    "epochs": 30,
    "batch_size": 64,
    "image_size": 640,
    "learning_rate": 0.001,
    "model": "yolov8n.pt"
}
Expected: ~45 minutes

Example 2: Production training on RTX 3090 (24GB)
POST /train/start
{
    "dataset_id": "my-dataset",
    "device": "0",
    "epochs": 300,
    "batch_size": 128,
    "image_size": 1280,
    "learning_rate": 0.0001,
    "model": "yolov8l.pt",
    "register_best_model": true,
    "model_description": "Production detector"
}
Expected: ~3 hours

Example 3: Budget GPU (RTX 2060, 6GB)
POST /train/start
{
    "dataset_id": "my-dataset",
    "device": "0",
    "epochs": 100,
    "batch_size": 16,
    "image_size": 512,
    "learning_rate": 0.001,
    "model": "yolov8n.pt"
}
Expected: ~2-3 hours

Example 4: CPU only (no GPU)
POST /train/start
{
    "dataset_id": "my-dataset",
    "device": "cpu",
    "epochs": 30,
    "batch_size": 8,
    "image_size": 416,
    "learning_rate": 0.001,
    "model": "yolov8n.pt"
}
Expected: ~8-12 hours (not recommended)
"""

# === OPTIMIZATION CHECKLIST ===

OPTIMIZATION_CHECKLIST = [
    "✓ GPU drivers installed: nvidia-smi works",
    "✓ CUDA installed: matches PyTorch version",
    "✓ PyTorch with CUDA: pip install torch...cu121",
    "✓ Enough VRAM: 8GB+ recommended",
    "✓ Dataset has labels: required for training",
    "✓ Device set to GPU: device='0'",
    "✓ Monitoring GPU: nvidia-smi -l 1",
    "✓ Batch size appropriate: 8-16 per GB VRAM",
    "✓ Image size reasonable: 416-1280",
    "✓ Val/test splits sum <0.5: leaves 50%+ for training",
]

# === PERFORMANCE BASELINE ===

PERFORMANCE_BASELINE = """
Training Performance (100 epochs, 640px, batch=64):

Device          | Time/Epoch | Total Time | Cost/Hour
----------------|------------|------------|----------
CPU (Modern)    | 3-5 mins   | 5-8 hours  | N/A
RTX 2060 (6GB)  | 30 secs    | 50 minutes | Low
RTX 3060 (12GB) | 20 secs    | 33 minutes | Low
RTX 3090 (24GB) | 10 secs    | 17 minutes | High
A100 (40GB)     | 5 secs     | 8 minutes  | Very High

With GPU optimizations, expect:
✓ 3-5x speedup over basic settings
✓ 85%+ GPU utilization
✓ Stable memory usage
✓ Early stopping after ~60 epochs
"""

# === COMMON ISSUES & FIXES ===

TROUBLESHOOTING = """
Issue: "CUDA out of memory"
Fix: Reduce batch_size or image_size by 50%
     batch_size: 64 → 32 or image_size: 640 → 512

Issue: "GPU utilization low (<50%)"
Fix: Increase workers (already at 8), increase batch_size
     Or increase image_size (more computation)

Issue: "GPU not found"
Fix: Check nvidia-smi, reinstall PyTorch with CUDA
     pip install torch...cu121 --force-reinstall

Issue: "Training very slow on GPU"
Fix: Ensure device="0" in API call, check nvidia-smi output
     GPU should show >80% utilization after epoch 1

Issue: "CUDA version mismatch"
Fix: Check CUDA: nvcc --version
     Reinstall PyTorch with matching version
"""

# === MONITORING COMMANDS ===

MONITORING_COMMANDS = """
# Real-time GPU stats (refresh every second)
nvidia-smi -l 1

# Detailed GPU info
nvidia-smi -q

# Log GPU stats to file every 5 seconds
nvidia-smi -lms 5000 > gpu_log.txt &

# Monitor specific GPU (GPU 0)
nvtop  # (requires: pip install gpustat)

# Watch GPU memory
watch -n 1 'nvidia-smi | grep Memory'

# Python: Check GPU inside code
import torch
print(torch.cuda.get_device_name(0))  # Device name
print(torch.cuda.get_device_properties(0))  # Full info
"""

if __name__ == "__main__":
    print("GPU Training Quick Reference")
    print("=" * 60)
    print("\nFast Configuration:")
    print(FAST_TRAINING)
    print("\nBalanced Configuration:")
    print(BALANCED_TRAINING)
    print("\nBatch Size by GPU:")
    for gpu, sizes in BATCH_SIZE_BY_GPU.items():
        print(f"  {gpu:8s}: 416px={sizes[0]:3d}, 640px={sizes[1]:3d}, 1280px={sizes[2]:3d}")
    print("\nPerformance Baseline:")
    print(PERFORMANCE_BASELINE)
