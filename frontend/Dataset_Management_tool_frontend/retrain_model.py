"""
Retrain YOLOv8 Model on Drone Dataset
This script trains a fresh model that will actually detect drones
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml


def retrain_model(dataset_yaml_path, epochs=100, batch_size=16, img_size=640, device="cpu"):
    """
    Retrain YOLOv8 model on your dataset
    
    Args:
        dataset_yaml_path: Path to data.yaml
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size
        device: 'cpu' or 'cuda' or GPU number
    """
    
    print("\n" + "="*60)
    print("🚀 YOLO DRONE DETECTOR - TRAINING")
    print("="*60)
    
    # Validate data.yaml
    dataset_yaml = Path(dataset_yaml_path)
    if not dataset_yaml.exists():
        print(f"❌ data.yaml not found: {dataset_yaml_path}")
        return False
    
    print(f"✅ Found data.yaml: {dataset_yaml_path}")
    
    # Load and display config
    with open(dataset_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n📊 Dataset Configuration:")
    print(f"   Training path: {config.get('train')}")
    print(f"   Validation path: {config.get('val')}")
    print(f"   Classes: {config.get('names')}")
    print(f"   Number of classes: {config.get('nc')}")
    
    # Load pretrained model
    print(f"\n📦 Loading YOLOv8 pretrained model...")
    model = YOLO('yolov8n.pt')  # nano model for speed, use yolov8s.pt for better accuracy
    
    print(f"✅ Model loaded: YOLOv8 Nano")
    
    # Train the model
    print(f"\n🏋️  Starting training with these parameters:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {img_size}x{img_size}")
    print(f"   Device: {device}")
    print(f"\n💡 This will take several minutes...\n")
    
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            patience=20,  # early stopping
            save=True,
            exist_ok=True,
            project='runs/detect',
            name='train',
            verbose=True,
            seed=42,
        )
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Find the trained model
        best_model_path = Path('runs/detect/train/weights/best.pt')
        if best_model_path.exists():
            print(f"\n🎯 Best model saved to: {best_model_path.absolute()}")
            print(f"\n📝 Next steps:")
            print(f"   1. Test the model:")
            print(f"      python test_model_inference.py --model {best_model_path} --image test.jpg")
            print(f"   2. Test on video:")
            print(f"      python test_model_inference.py --model {best_model_path} --video drone_video.mp4")
            
            return True
        else:
            print("❌ Error: best.pt not created")
            return False
            
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Retrain YOLOv8 model on drone dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python retrain_model.py --yaml data.yaml
  
  # With custom epochs
  python retrain_model.py --yaml data.yaml --epochs 200 --batch 32
  
  # GPU training (if available)
  python retrain_model.py --yaml data.yaml --device 0
  
  # Faster training (more epochs)
  python retrain_model.py --yaml data.yaml --epochs 300 --batch 64
        """
    )
    
    parser.add_argument(
        "--yaml",
        type=str,
        required=True,
        help="Path to data.yaml file"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16, use 32 or 64 for faster training if you have RAM)"
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size (default: 640)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu', 'cuda', or GPU number like '0' (default: cpu)"
    )
    
    args = parser.parse_args()
    
    success = retrain_model(
        args.yaml,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device
    )
    
    sys.exit(0 if success else 1)
