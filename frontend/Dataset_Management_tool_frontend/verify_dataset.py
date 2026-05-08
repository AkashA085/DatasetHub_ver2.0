"""
Dataset and Training Verification Script
Checks your dataset structure, labels, and training setup for common issues
"""

import os
import sys
from pathlib import Path
import yaml
import json
from collections import Counter


def verify_dataset_structure(dataset_path: str):
    """
    Verify the dataset structure and check for common issues
    """
    print("\n" + "="*60)
    print("🔍 DATASET STRUCTURE VERIFICATION")
    print("="*60)

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False

    # Check for required directories
    required_dirs = ['images', 'labels']
    splits = ['train', 'val', 'test']

    issues = []

    for split in splits:
        split_path = dataset_path / split
        if not split_path.exists():
            issues.append(f"Missing split directory: {split}")
            continue

        for subdir in required_dirs:
            subdir_path = split_path / subdir
            if not subdir_path.exists():
                issues.append(f"Missing {subdir} directory in {split}")
            else:
                files = list(subdir_path.glob('*'))
                print(f"📁 {split}/{subdir}: {len(files)} files")

                if subdir == 'images':
                    img_exts = ['.jpg', '.jpeg', '.png', '.bmp']
                    img_files = [f for f in files if f.suffix.lower() in img_exts]
                    print(f"   🖼️  Image files: {len(img_files)}")
                    if len(img_files) == 0:
                        issues.append(f"No image files found in {split}/images")

                elif subdir == 'labels':
                    txt_files = [f for f in files if f.suffix.lower() == '.txt']
                    print(f"   📝 Label files: {len(txt_files)}")
                    if len(txt_files) == 0:
                        issues.append(f"No label files found in {split}/labels")

    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Dataset structure looks good!")
        return True


def verify_labels(dataset_path: str):
    """
    Verify label files and check for common labeling issues
    """
    print("\n" + "="*60)
    print("🏷️  LABEL VERIFICATION")
    print("="*60)

    dataset_path = Path(dataset_path)
    splits = ['train', 'val', 'test']
    all_labels = []
    issues = []

    for split in splits:
        labels_dir = dataset_path / split / 'labels'
        if not labels_dir.exists():
            continue

        txt_files = list(labels_dir.glob('*.txt'))

        for txt_file in txt_files:
            try:
                with open(txt_file, 'r') as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        issues.append(f"{txt_file.name}: Line {line_num} - Invalid format (need at least 5 values)")
                        continue

                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]

                        if len(coords) != 4:
                            issues.append(f"{txt_file.name}: Line {line_num} - Expected 4 coordinates, got {len(coords)}")
                            continue

                        x_center, y_center, width, height = coords

                        # Check coordinate ranges (YOLO format: normalized 0-1)
                        for coord_name, coord_val in [('x_center', x_center), ('y_center', y_center),
                                                    ('width', width), ('height', height)]:
                            if not (0 <= coord_val <= 1):
                                issues.append(f"{txt_file.name}: Line {line_num} - {coord_name}={coord_val} out of range [0,1]")

                        all_labels.append(class_id)

                    except ValueError as e:
                        issues.append(f"{txt_file.name}: Line {line_num} - Invalid number format: {e}")

            except Exception as e:
                issues.append(f"Error reading {txt_file.name}: {e}")

    if all_labels:
        class_counts = Counter(all_labels)
        print("📊 Class distribution:")
        for class_id, count in sorted(class_counts.items()):
            print(f"  Class {class_id}: {count} instances")

        print(f"\n📈 Total annotations: {len(all_labels)}")
        print(f"🏷️  Unique classes: {len(class_counts)}")

    if issues:
        print("\n❌ LABEL ISSUES FOUND:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
        return False
    else:
        print("✅ Labels look good!")
        return True


def verify_data_yaml(yaml_path: str):
    """
    Verify the data.yaml configuration file
    """
    print("\n" + "="*60)
    print("⚙️  DATA.YAML VERIFICATION")
    print("="*60)

    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        print(f"❌ data.yaml not found: {yaml_path}")
        return False

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        print("📄 data.yaml contents:")
        for key, value in data.items():
            print(f"  {key}: {value}")

        # Check required fields
        required_fields = ['path', 'train', 'val', 'names', 'nc']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            print(f"\n❌ Missing required fields: {missing_fields}")
            return False

        # Verify paths exist
        base_path = Path(data['path'])

        for split in ['train', 'val']:
            split_path = base_path / data[split]
            if not split_path.exists():
                print(f"❌ Split path does not exist: {split_path}")
                return False

        # Check class names
        if not isinstance(data['names'], list):
            print("❌ 'names' should be a list")
            return False

        if data['nc'] != len(data['names']):
            print(f"❌ nc ({data['nc']}) doesn't match names length ({len(data['names'])})")
            return False

        print("✅ data.yaml looks good!")
        return True

    except Exception as e:
        print(f"❌ Error reading data.yaml: {e}")
        return False


def check_training_artifacts(run_path: str):
    """
    Check training artifacts and results
    """
    print("\n" + "="*60)
    print("🏃 TRAINING ARTIFACTS CHECK")
    print("="*60)

    run_path = Path(run_path)

    if not run_path.exists():
        print(f"❌ Training run path does not exist: {run_path}")
        return False

    # Check for weights
    weights_dir = run_path / 'weights'
    if weights_dir.exists():
        best_pt = weights_dir / 'best.pt'
        last_pt = weights_dir / 'last.pt'

        if best_pt.exists():
            print(f"✅ Best model found: {best_pt}")
        else:
            print("❌ best.pt not found")

        if last_pt.exists():
            print(f"✅ Last checkpoint found: {last_pt}")
        else:
            print("⚠️  last.pt not found (training may have failed)")
    else:
        print("❌ weights directory not found")
        return False

    # Check for results
    results_csv = run_path / 'results.csv'
    if results_csv.exists():
        print(f"✅ Training results found: {results_csv}")
    else:
        print("⚠️  results.csv not found")

    # Check for args.yaml (training config)
    args_yaml = run_path / 'args.yaml'
    if args_yaml.exists():
        print(f"✅ Training config found: {args_yaml}")
    else:
        print("⚠️  args.yaml not found")

    return True


def diagnose_common_issues():
    """
    Provide diagnosis and solutions for common issues
    """
    print("\n" + "="*60)
    print("🔧 COMMON ISSUE DIAGNOSIS & SOLUTIONS")
    print("="*60)

    print("""
🚨 IF YOUR MODEL DETECTS NOTHING:

1. 📊 LOW CONFIDENCE THRESHOLD
   • Try confidence threshold of 0.1 instead of 0.25
   • Command: python test_model_inference.py --model best.pt --image test.jpg --conf 0.1

2. 🏷️  LABEL FORMAT ISSUES
   • YOLO labels must be normalized (0-1 range)
   • Format: class_id x_center y_center width height
   • Check with: python verify_dataset.py --dataset /path/to/dataset

3. 📚 INSUFFICIENT TRAINING DATA
   • Need at least 100-200 images per class
   • Ensure good variety in lighting, angles, backgrounds

4. 🔧 TRAINING ISSUES
   • Check if training completed successfully
   • Look for best.pt vs last.pt
   • Verify data.yaml is correct

5. 🎯 CLASS IMBALANCE
   • Some classes may have very few examples
   • Model may not learn rare classes well

6. 📏 IMAGE SIZE MISMATCH
   • Training and inference image sizes should match
   • Default is 640x640, but check your training settings

🔍 DEBUGGING STEPS:
1. Run: python verify_dataset.py --dataset your_dataset_path
2. Run: python test_model_inference.py --model best.pt --diagnose
3. Test with a single image from your training set
4. Try different confidence thresholds

📞 IF STILL NOT WORKING:
• Share the output of verification scripts
• Check training logs for errors
• Ensure your test images contain the objects you're trying to detect
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify dataset structure, labels, and training setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify dataset structure
  python verify_dataset.py --dataset runs/detect/train/dataset
  
  # Check training run
  python verify_dataset.py --run runs/detect/train
  
  # Verify data.yaml
  python verify_dataset.py --yaml data.yaml
  
  # Full diagnosis
  python verify_dataset.py --dataset dataset_path --run run_path --yaml data.yaml
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset directory (with train/val/test splits)"
    )

    parser.add_argument(
        "--run",
        type=str,
        help="Path to training run directory"
    )

    parser.add_argument(
        "--yaml",
        type=str,
        help="Path to data.yaml file"
    )

    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Show common issue diagnosis and solutions"
    )

    args = parser.parse_args()

    success = True

    if args.dataset:
        success &= verify_dataset_structure(args.dataset)
        success &= verify_labels(args.dataset)

    if args.yaml:
        success &= verify_data_yaml(args.yaml)

    if args.run:
        success &= check_training_artifacts(args.run)

    if args.diagnose or not any([args.dataset, args.run, args.yaml]):
        diagnose_common_issues()

    sys.exit(0 if success else 1)