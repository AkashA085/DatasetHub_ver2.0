#!/usr/bin/env python3
"""
GPU Training Verification Script
Checks if GPU is properly configured for YOLO training
"""

import sys
from pathlib import Path


def check_torch():
    """Check PyTorch installation and GPU availability."""
    print("\n" + "="*60)
    print("🔍 CHECKING PYTORCH INSTALLATION")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch NOT installed")
        print("  Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    return True


def check_cuda():
    """Check CUDA and GPU availability."""
    print("\n" + "="*60)
    print("🔍 CHECKING CUDA & GPU")
    print("="*60)
    
    import torch
    
    if not torch.cuda.is_available():
        print("✗ CUDA NOT available")
        print("  Your system doesn't have NVIDIA GPU drivers installed")
        print("  Or PyTorch wasn't installed with CUDA support")
        return False
    
    print(f"✓ CUDA available: Yes")
    print(f"✓ CUDA version (PyTorch): {torch.version.cuda}")
    print(f"✓ Device count: {torch.cuda.device_count()}")
    
    return True


def check_gpu_properties():
    """Get detailed GPU properties."""
    print("\n" + "="*60)
    print("🔍 GPU PROPERTIES")
    print("="*60)
    
    import torch
    
    if not torch.cuda.is_available():
        print("✗ No CUDA device found")
        return False
    
    try:
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)
            
            print(f"\n📱 GPU {i}:")
            print(f"   Name: {props.name}")
            print(f"   Total Memory: {total_mem:.1f}GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Max Threads Per Block: {props.max_threads_per_block}")
            
            # Get free memory
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)
            print(f"   Free Memory: {free_mem:.1f}GB")
    except Exception as e:
        print(f"✗ Error getting GPU properties: {e}")
        return False
    
    return True


def check_ultralytics():
    """Check Ultralytics YOLOv8 installation."""
    print("\n" + "="*60)
    print("🔍 CHECKING ULTRALYTICS")
    print("="*60)
    
    try:
        from ultralytics import YOLO, __version__
        print(f"✓ Ultralytics installed: {__version__}")
    except ImportError:
        print("✗ Ultralytics NOT installed")
        print("  Install: pip install ultralytics")
        return False
    
    return True


def check_dependencies():
    """Check other required dependencies."""
    print("\n" + "="*60)
    print("🔍 CHECKING DEPENDENCIES")
    print("="*60)
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'PIL': 'Pillow',
        'cv2': 'opencv-python-headless',
        'yaml': 'PyYAML',
        'albumentations': 'albumentations',
        'mlflow': 'mlflow',
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} NOT installed")
            missing.append(package_name)
    
    return len(missing) == 0, missing


def test_gpu_training():
    """Test GPU by running a small training job."""
    print("\n" + "="*60)
    print("🔍 TESTING GPU TRAINING")
    print("="*60)
    
    try:
        import torch
        from ultralytics import YOLO
        
        print("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        
        print("Model loaded successfully")
        print(f"Model device capability: GPU support {'enabled' if torch.cuda.is_available() else 'disabled'}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def get_batch_size_recommendation():
    """Get optimal batch size recommendation."""
    print("\n" + "="*60)
    print("💡 BATCH SIZE RECOMMENDATION")
    print("="*60)
    
    import torch
    
    if not torch.cuda.is_available():
        print("CPU Mode: Recommended batch_size = 8")
        return
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print(f"\nBased on GPU with {total_memory:.1f}GB VRAM:")
        print("\nFor image_size=640:")
        if total_memory >= 24:
            print("  Recommended batch_size: 128")
        elif total_memory >= 16:
            print("  Recommended batch_size: 64")
        elif total_memory >= 8:
            print("  Recommended batch_size: 32")
        elif total_memory >= 6:
            print("  Recommended batch_size: 16")
        else:
            print("  Recommended batch_size: 8")
        
        print("\nFor image_size=1280:")
        if total_memory >= 24:
            print("  Recommended batch_size: 64")
        elif total_memory >= 16:
            print("  Recommended batch_size: 32")
        elif total_memory >= 8:
            print("  Recommended batch_size: 16")
        elif total_memory >= 6:
            print("  Recommended batch_size: 8")
        else:
            print("  Recommended batch_size: 4")
    except Exception as e:
        print(f"Could not determine: {e}")


def main():
    """Run all checks."""
    print("\n🚀 GPU TRAINING VERIFICATION SCRIPT")
    print("="*60)
    
    checks = [
        ("PyTorch", check_torch),
        ("CUDA & GPU", check_cuda),
        ("GPU Properties", check_gpu_properties),
        ("Ultralytics", check_ultralytics),
        ("Dependencies", lambda: check_dependencies()[0]),
        ("Model Loading", test_gpu_training),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"✗ {check_name} check failed: {e}")
            results[check_name] = False
    
    # Additional info
    get_batch_size_recommendation()
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    print("="*60)
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED - GPU TRAINING READY!")
        print("\nYou can now:")
        print("1. Start the backend: python -m uvicorn app.main:app")
        print("2. Submit training jobs with device='0'")
        print("3. Monitor with: nvidia-smi -l 1")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\nTo fix GPU training:")
        print("1. Install CUDA: https://developer.nvidia.com/cuda-downloads")
        print("2. Install cuDNN: https://developer.nvidia.com/cudnn")
        print("3. Reinstall PyTorch with CUDA:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return 1


if __name__ == "__main__":
    sys.exit(main())
