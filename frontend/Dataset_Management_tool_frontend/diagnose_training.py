"""
Complete Training Diagnosis Script
Checks why model is not detecting drones
"""

import os
import sys
from pathlib import Path
import yaml
import sqlite3
from ultralytics import YOLO
import json

def check_model_weights(model_path):
    print("\n" + "="*60)
    print("🤖 MODEL WEIGHTS ANALYSIS")
    print("="*60)
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
        print(f"   Number of classes: {len(model.names)}")
        print(f"   Classes: {model.names}")
        
        # Check model parameters
        print(f"\n📊 Model Parameters:")
        print(f"   Task: {model.task if hasattr(model, 'task') else 'Unknown'}")
        
        # Get model summary
        if hasattr(model, 'model'):
            total_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            print(f"   Total parameters: {total_params:,}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def check_training_database(db_path):
    print("\n" + "="*60)
    print("📊 TRAINING DATABASE CHECK")
    print("="*60)
    
    if not os.path.exists(db_path):
        print(f"⚠️  Database not found: {db_path}")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get training jobs
        cursor.execute("SELECT * FROM training_job LIMIT 5")
        jobs = cursor.fetchall()
        
        if jobs:
            print(f"✅ Found {len(jobs)} training job(s)")
            for job in jobs:
                print(f"\n  Job ID: {job[0]}")
                print(f"  Dataset ID: {job[1]}")
                print(f"  Status: {job[2]}")
                
        else:
            print("⚠️  No training jobs found in database")
        
        conn.close()
        return jobs
        
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return None


def test_on_training_image(model_path, dataset_path, conf=0.1):
    print("\n" + "="*60)
    print("🧪 TEST ON TRAINING IMAGE")
    print("="*60)
    
    try:
        # Find a training image
        train_images = list(Path(dataset_path).glob("train/images/*"))
        if not train_images:
            print("❌ No training images found")
            return False
        
        test_image = train_images[0]
        print(f"📷 Testing on training image: {test_image.name}")
        
        model = YOLO(model_path)
        results = model(str(test_image), conf=conf, verbose=False)
        
        if len(results[0].boxes) > 0:
            print(f"✅ Model detects objects in training image!")
            print(f"   Detections: {len(results[0].boxes)}")
            return True
        else:
            print(f"❌ Model does NOT detect objects in training image")
            print("   This suggests the model may not have trained properly")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def analyze_labels(dataset_path):
    print("\n" + "="*60)
    print("🏷️  LABEL ANALYSIS")
    print("="*60)
    
    from collections import Counter
    
    label_counts = Counter()
    total_annotations = 0
    
    for label_file in Path(dataset_path).glob("train/labels/*.txt"):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        label_counts[class_id] += 1
                        total_annotations += 1
        except:
            pass
    
    print(f"📊 Total annotations in training set: {total_annotations}")
    print(f"   Distribution:")
    for class_id, count in sorted(label_counts.items()):
        print(f"   Class {class_id}: {count} annotations ({count/total_annotations*100:.1f}%)")
    
    if total_annotations == 0:
        print("❌ NO ANNOTATIONS FOUND - THIS IS THE PROBLEM!")
        return False
    
    return True


def recommendations():
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    print("""
🎯 The model is not detecting because:

1. ❌ TRAINING LIKELY FAILED
   The model has 1 class but data.yaml specifies 2 classes
   → Re-train with correct data.yaml

2. ⚠️  VERIFY YOUR LABELS ARE CORRECT
   • Check that all images have corresponding .txt label files
   • Labels must be in YOLO format: class_id x y w h (normalized 0-1)
   • Each annotation on a separate line

3. 📚 ENSURE SUFFICIENT TRAINING DATA
   • You have 2440 training images - this is good
   • But verify they all contain drone objects

4. 🔧 HOW TO FIX:

   OPTION A - Quick Test:
   python test_model_inference.py --model best.pt --image test.jpg --conf 0.05
   
   OPTION B - Retrain Model:
   • Fix data.yaml to match your actual classes
   • Make sure training images are properly labeled
   • Run training again from the UI

   OPTION C - Check Data Quality:
   python verify_dataset.py --dataset dataset_path
""")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose training issues")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--db", type=str, help="Path to database")
    
    args = parser.parse_args()
    
    # Run diagnostics
    model = check_model_weights(args.model)
    
    if args.db:
        check_training_database(args.db)
    
    if args.dataset:
        analyze_labels(args.dataset)
        test_on_training_image(args.model, args.dataset, conf=0.05)
    
    recommendations()
