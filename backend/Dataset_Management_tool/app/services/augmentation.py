import cv2
import albumentations as A
import numpy as np
from pathlib import Path
from app.models.schemas import AugmentationRequest, ImageAnnotation, BoundingBox
from app.utils.file_utils import PROCESSED_DIR, UPLOADS_DIR
import uuid
import copy

class AugmentationService:
    @staticmethod
    def augment_dataset(request: AugmentationRequest, annotations: list[ImageAnnotation], stem_to_image: dict):
        """
        Apply augmentations to the dataset.
        Returns multiple lists/dicts:
        - new_annotations: Combined list of original + augmented annotations
        - new_stem_to_image: Combined dict of stem -> image path
        """
        
        # Define augmentation pipeline
        transforms = []
        if request.horizontal_flip:
            transforms.append(A.HorizontalFlip(p=0.5))
        if request.vertical_flip:
            transforms.append(A.VerticalFlip(p=0.5))
        if request.rotation > 0:
            transforms.append(A.Rotate(limit=request.rotation, p=0.5))
        if request.blur > 0:
            transforms.append(A.Blur(blur_limit=request.blur, p=0.5))
        if request.brightness != 0 or request.contrast != 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=request.brightness, 
                contrast_limit=request.contrast, 
                p=0.5
            ))
        if request.noise > 0:
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.5)) # Simplified noise
            
        # If no transforms selected but count > 1, user might just want copies or re-sampling (though albumentations usually needs transforms)
        # If count > 1 and no transforms, we might effectively just get duplicates if we force it.
        # But let's assume at least one transform or we add a NoOp.
        if not transforms:
            transforms.append(A.NoOp(p=1.0))
            
        compose = A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
        
        augmented_annotations = []
        augmented_stem_to_image = {}
        
        session_processed_dir = PROCESSED_DIR / request.dataset_id
        aug_images_dir = session_processed_dir / "augmented_images"
        aug_images_dir.mkdir(parents=True, exist_ok=True)

        for ann in annotations:
            stem = ann.image_name
            # Case-insensitive lookup
            src_img_path = stem_to_image.get(stem.lower())
            
            if not src_img_path or not src_img_path.exists():
                continue
                
            image = cv2.imread(str(src_img_path))
            if image is None:
                continue
                
            h, w = image.shape[:2]
            
            # Prepare bboxes for albumentations
            # Pascal VOC format: [x_min, y_min, x_max, y_max]
            bboxes = []
            class_labels = []
            for obj in ann.objects:
                # Clip coordinates to be within image bounds to prevent errors
                xmin = max(0, min(obj.xmin, w))
                ymin = max(0, min(obj.ymin, h))
                xmax = max(0, min(obj.xmax, w))
                ymax = max(0, min(obj.ymax, h))
                
                # Verify valid box
                if xmax <= xmin or ymax <= ymin:
                    continue
                    
                bboxes.append([xmin, ymin, xmax, ymax])
                class_labels.append(obj.class_id)
            
            # Generate augmentations
            # request.count determines how many *additional* versions we want, or total?
            # User said "add this augmented images to main dataesy".
            # Usually users specify "Generate 3x versions".
            # Let's assume request.count is the number of NEW augmented images to generate per original image.
            
            for i in range(request.count):
                try:
                    transformed = compose(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_labels = transformed['class_labels'] # Should match aug_bboxes
                    
                    if len(aug_bboxes) != len(aug_labels):
                        # Should not happen with albumentations
                        continue

                    # Save augmented image
                    aug_filename = f"{stem}_aug_{uuid.uuid4().hex[:8]}.jpg"
                    aug_path = aug_images_dir / aug_filename
                    cv2.imwrite(str(aug_path), aug_image)
                    
                    # Create annotation object
                    new_objects = []
                    for bbox, cls_id in zip(aug_bboxes, aug_labels):
                        new_objects.append(BoundingBox(
                            class_id=cls_id,
                            xmin=bbox[0],
                            ymin=bbox[1],
                            xmax=bbox[2],
                            ymax=bbox[3]
                        ))
                    
                    new_ann = ImageAnnotation(
                        image_name=aug_path.stem, # Stem used for matching
                        width=w,
                        height=h,
                        objects=new_objects
                    )
                    
                    augmented_annotations.append(new_ann)
                    augmented_stem_to_image[aug_path.stem] = aug_path
                    
                except Exception as e:
                    print(f"Augmentation failed for {stem}: {e}")
                    continue

        return augmented_annotations, augmented_stem_to_image
