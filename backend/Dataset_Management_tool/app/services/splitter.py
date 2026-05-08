import random
from typing import List, Tuple
from app.models.schemas import ImageAnnotation

class DatasetSplitter:
    @staticmethod
    def split(annotations: List[ImageAnnotation], train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[ImageAnnotation], List[ImageAnnotation], List[ImageAnnotation]]:
        """Perform a deterministic split of the dataset."""
        # Filter out images without labels if necessary (though requirement says "Do not split images without labels" - which might mean if an image has NO labels, we might want to exclude it or just be careful. The prompt says "Do not split images without labels" which I'll interpret as: only include images that have at least one object in the split).
        
        valid_annotations = [ann for ann in annotations if len(ann.objects) > 0]
        
        random.seed(seed)
        random.shuffle(valid_annotations)

        total = len(valid_annotations)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_anns = valid_annotations[:train_end]
        val_anns = valid_annotations[train_end:val_end]
        test_anns = valid_annotations[val_end:]

        return train_anns, val_anns, test_anns
