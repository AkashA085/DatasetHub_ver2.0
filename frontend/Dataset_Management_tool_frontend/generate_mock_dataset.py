import os
import zipfile
from pathlib import Path
from PIL import Image
import random

def create_mock_dataset(output_dir: Path):
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create 10 mock images and labels
    for i in range(10):
        img_name = f"image_{i}.jpg"
        img_path = images_dir / img_name
        
        # Create a simple colored image
        img = Image.new('RGB', (640, 480), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        img.save(img_path)
        
        # Create YOLO label
        label_path = labels_dir / f"image_{i}.txt"
        with open(label_path, 'w') as f:
            # 1-3 random objects
            for _ in range(random.randint(1, 3)):
                class_id = random.randint(0, 2)
                x_center = random.uniform(0.1, 0.9)
                y_center = random.uniform(0.1, 0.9)
                w = random.uniform(0.1, 0.3)
                h = random.uniform(0.1, 0.3)
                f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

    # Zip them up
    with zipfile.ZipFile(output_dir / "images.zip", "w") as z:
        for f in images_dir.iterdir():
            z.write(f, f.name)
            
    with zipfile.ZipFile(output_dir / "labels.zip", "w") as z:
        for f in labels_dir.iterdir():
            z.write(f, f.name)

if __name__ == "__main__":
    create_mock_dataset(Path("mock_dataset"))
    print("Mock dataset created in mock_dataset/")
