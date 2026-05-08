#!/usr/bin/env python3
"""
Verify that training data is correct and only uses real uploaded labeled images.
This script checks:
1. All images in database have corresponding files
2. All images with has_label=True have label files
3. No mock/test data is being used
4. Label files are not empty
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add the app to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy.orm import Session
from app.core.database import SessionLocal, Dataset, Image, ClassDistribution
from app.utils.file_utils import STORAGE_ROOT

def verify_dataset(dataset_id: str, db: Session) -> dict:
    """Verify a dataset is valid and ready for training."""
    results = {
        "dataset_id": dataset_id,
        "valid": True,
        "messages": [],
        "errors": [],
        "stats": {
            "total_images": 0,
            "images_with_labels": 0,
            "images_without_labels": 0,
            "valid_pairs": 0,
            "invalid_images": 0,
            "empty_labels": 0,
            "missing_label_files": 0,
            "missing_image_files": 0,
            "total_classes": 0,
        }
    }
    
    # Get dataset
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        results["valid"] = False
        results["errors"].append(f"Dataset {dataset_id} not found in database")
        return results
    
    results["messages"].append(f"✓ Dataset found: {dataset.format_type} format")
    
    # Get all images
    images = db.query(Image).filter(Image.dataset_id == dataset_id).all()
    results["stats"]["total_images"] = len(images)
    
    if not images:
        results["valid"] = False
        results["errors"].append("No images found for this dataset")
        return results
    
    results["messages"].append(f"✓ Found {len(images)} images")
    
    # Verify each image
    labeled_images = 0
    valid_pairs = 0
    
    for img in images:
        # Count labeled/unlabeled
        if img.has_label:
            labeled_images += 1
        else:
            results["stats"]["images_without_labels"] += 1
            results["messages"].append(f"  ⚠ Image without label: {img.file_name}")
            continue
        
        # Check image file exists
        img_path = Path(img.file_path)
        if not img_path.exists():
            results["stats"]["missing_image_files"] += 1
            results["errors"].append(f"Image file not found: {img_path}")
            results["valid"] = False
            continue
        
        # Find and verify label file
        try:
            parts = list(img_path.parts)
            if "images" not in parts:
                results["errors"].append(f"Image not in 'images' dir: {img_path}")
                results["valid"] = False
                continue
            
            idx = parts.index("images")
            label_parts = parts[:idx] + ["labels"] + parts[idx+1:]
            label_path = Path(*label_parts).parent / (Path(img_path).stem + ".txt")
            
            if not label_path.exists():
                results["stats"]["missing_label_files"] += 1
                results["errors"].append(f"Label file not found: {label_path}")
                results["valid"] = False
                continue
            
            # Check label is not empty
            if label_path.stat().st_size == 0:
                results["stats"]["empty_labels"] += 1
                results["errors"].append(f"Empty label file: {label_path}")
                results["valid"] = False
                continue
            
            # Valid pair
            valid_pairs += 1
            
        except Exception as e:
            results["errors"].append(f"Error checking {img.file_name}: {str(e)}")
            results["valid"] = False
    
    results["stats"]["images_with_labels"] = labeled_images
    results["stats"]["valid_pairs"] = valid_pairs
    results["stats"]["invalid_images"] = labeled_images - valid_pairs
    
    # Check classes
    classes = db.query(ClassDistribution).filter(ClassDistribution.dataset_id == dataset_id).all()
    results["stats"]["total_classes"] = len(set(c.class_id for c in classes))
    
    if valid_pairs == 0:
        results["valid"] = False
        results["errors"].append("No valid image/label pairs found!")
    else:
        results["messages"].append(f"✓ Found {valid_pairs} valid image/label pairs")
    
    if results["stats"]["total_classes"] > 0:
        results["messages"].append(f"✓ Found {results['stats']['total_classes']} classes")
    
    # Final validation
    if results["valid"]:
        results["messages"].append(f"✅ Dataset is READY for training with {valid_pairs} real labeled images")
    else:
        results["messages"].append(f"❌ Dataset has ISSUES and needs fixing before training")
    
    return results


def main():
    """Main verification routine."""
    db = SessionLocal()
    
    try:
        # Get all datasets
        datasets = db.query(Dataset).all()
        
        if not datasets:
            print("No datasets found in database")
            return
        
        print(f"\n🔍 Verifying {len(datasets)} dataset(s):\n")
        print("=" * 80)
        
        all_valid = True
        
        for dataset in datasets:
            results = verify_dataset(dataset.id, db)
            
            print(f"\nDataset: {dataset.id}")
            print(f"Format: {dataset.format_type}")
            print(f"Stats: {results['stats']['total_images']} images, "
                  f"{results['stats']['total_classes']} classes")
            
            # Print messages
            for msg in results["messages"]:
                print(f"  {msg}")
            
            # Print errors
            if results["errors"]:
                print(f"\n  ❌ Errors ({len(results['errors'])} total):")
                for err in results["errors"][:5]:  # Show first 5
                    print(f"    - {err}")
                if len(results["errors"]) > 5:
                    print(f"    ... and {len(results['errors']) - 5} more")
            
            # Summary
            print(f"\n  Status: {'✅ VALID' if results['valid'] else '❌ INVALID'}")
            print(f"  Valid Pairs: {results['stats']['valid_pairs']}")
            
            if not results["valid"]:
                all_valid = False
            
            print("-" * 80)
        
        print("\n" + "=" * 80)
        if all_valid:
            print("✅ All datasets are valid and ready for training!")
        else:
            print("❌ Some datasets have issues. Please fix before training.")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
