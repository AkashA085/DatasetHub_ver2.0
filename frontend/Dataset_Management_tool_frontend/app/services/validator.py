import os
import re
from pathlib import Path
from PIL import Image
from typing import Dict, List, Set, Tuple
from app.models.schemas import ValidationReport, BoundingBox, ImageAnnotation

class DatasetValidator:
    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self._hash_suffix_pattern = re.compile(r"_[0-9a-f]{12}$")

    def _normalize_stem(self, p: Path) -> str:
        """Normalize stem for matching image/label pairs after long-path fallback renames."""
        stem = p.stem.lower()
        return self._hash_suffix_pattern.sub("", stem)

    def _alt_match_key(self, stem: str) -> str:
        """Build a fallback key using trailing id pattern, e.g. A_01 -> a_1."""
        tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", stem.lower()) if t]
        if not tokens:
            return stem
        num_idx = None
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i].isdigit():
                num_idx = i
                break
        if num_idx is None:
            return tokens[-1]
        num = str(int(tokens[num_idx]))
        prefix = ""
        if num_idx - 1 >= 0 and tokens[num_idx - 1].isalpha() and len(tokens[num_idx - 1]) <= 3:
            prefix = tokens[num_idx - 1]
        return f"{prefix}_{num}" if prefix else num

    def validate(self) -> Tuple[ValidationReport, List[ImageAnnotation], Dict[str, Path], Dict[str, Path], List[str]]:
        """Validate the dataset and return a report, internal annotations, path mappings, and class names."""
        
        # 1. Recursive search for all files
        all_image_paths = []
        for ext in self.image_extensions:
            all_image_paths.extend(list(self.images_dir.rglob(f"*{ext}")))
            all_image_paths.extend(list(self.images_dir.rglob(f"*{ext.upper()}")))
        
        # Deduplicate paths (resolving to absolute to be sure)
        all_image_paths = {p.resolve(): p for p in all_image_paths}
        all_label_paths = {p.resolve(): p for p in self.labels_dir.rglob("*.txt")}

        # Search for classes.txt specifically for metadata
        class_names = []
        for p in all_label_paths.values():
            if p.name.lower() == "classes.txt":
                try:
                    with open(p, 'r') as f:
                        class_names = [line.strip() for line in f if line.strip()]
                except Exception:
                    pass
                break # We found it, but we LEAVE it in all_label_paths for orphan detection

        # 2. Map stem (lower) to file path
        # EVERY .txt file is a label candidate
        image_map: Dict[str, Path] = {}
        label_map: Dict[str, Path] = {}
        for p in all_image_paths.values():
            key = self._normalize_stem(p)
            if key not in image_map:
                image_map[key] = p
        for p in all_label_paths.values():
            key = self._normalize_stem(p)
            if key not in label_map:
                label_map[key] = p

        image_stems = set(image_map.keys())
        label_stems = set(label_map.keys())

        # 3. Set Logic
        matched_stems = image_stems & label_stems
        matched_pairs: Dict[str, str] = {s: s for s in matched_stems}
        missing_label_stems = set(image_stems - label_stems)
        orphan_label_stems = set(label_stems - image_stems)

        # Fallback pairing for datasets whose image/label stems differ by prefixes
        # (example: fwd_drone_A_01.jpg vs fwd_labels_A_01.txt).
        if not matched_stems and missing_label_stems and orphan_label_stems:
            alt_image_map: Dict[str, str] = {}
            alt_label_map: Dict[str, str] = {}
            for s in missing_label_stems:
                k = self._alt_match_key(s)
                if k not in alt_image_map:
                    alt_image_map[k] = s
            for s in orphan_label_stems:
                k = self._alt_match_key(s)
                if k not in alt_label_map:
                    alt_label_map[k] = s

            alt_keys = set(alt_image_map.keys()) & set(alt_label_map.keys())
            for k in alt_keys:
                img_stem = alt_image_map[k]
                lbl_stem = alt_label_map[k]
                matched_stems.add(img_stem)
                matched_pairs[img_stem] = lbl_stem
                if img_stem in missing_label_stems:
                    missing_label_stems.remove(img_stem)
                if lbl_stem in orphan_label_stems:
                    orphan_label_stems.remove(lbl_stem)

        corrupted_images = 0
        internal_annotations = []
        class_ids_found = set()

        missing_label_images_list = []
        orphan_label_files_list = []
        empty_label_files_list = []
        corrupted_image_files_list = []
        
        # Mappings for export logic
        stem_to_image_path = {}
        stem_to_label_path = {}

        # 4. Process Matched Pairs
        for stem in matched_stems:
            img_path = image_map[stem]
            label_path = label_map[matched_pairs.get(stem, stem)]

            # Check image corruption & get size
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception:
                corrupted_image_files_list.append(img_path.name)
                continue

            # Store absolute paths for export
            stem_to_image_path[stem] = img_path
            stem_to_label_path[stem] = label_path

            # Parse YOLO label
            objects = self._parse_yolo(label_path, width, height)
            
            if not objects:
                empty_label_files_list.append(label_path.name)
            
            for obj in objects:
                class_ids_found.add(obj.class_id)

            internal_annotations.append(ImageAnnotation(
                image_name=stem, 
                width=width,
                height=height,
                objects=objects
            ))

        # 5. Handle missing labels
        for stem in missing_label_stems:
            img_path = image_map[stem]
            missing_label_images_list.append(img_path.name)
            try:
                with Image.open(img_path) as img:
                    pass
            except Exception:
                corrupted_image_files_list.append(img_path.name)

        # 6. Handle orphan labels
        for stem in orphan_label_stems:
            orphan_label_files_list.append(label_map[stem].name)

        # 7. Generate Report
        report = ValidationReport(
            total_images=len(image_stems),
            total_labels=len(label_stems),
            missing_labels=len(missing_label_stems),
            orphan_labels=len(orphan_label_stems),
            empty_labels=len(empty_label_files_list),
            corrupted_images=len(corrupted_image_files_list),
            class_ids_found=sorted(list(class_ids_found)),
            missing_label_images=sorted(missing_label_images_list),
            orphan_label_files=sorted(orphan_label_files_list),
            empty_label_files=sorted(empty_label_files_list),
            corrupted_image_files=sorted(corrupted_image_files_list)
        )
        
        return report, internal_annotations, stem_to_image_path, stem_to_label_path, class_names

    def _parse_yolo(self, label_path: Path, width: int, height: int) -> List[BoundingBox]:
        boxes = []
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5: # Basic YOLO format check
                        try:
                            cls_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])

                            # Convert to absolute pixels
                            xmin = (x_center - w / 2) * width
                            ymin = (y_center - h / 2) * height
                            xmax = (x_center + w / 2) * width
                            ymax = (y_center + h / 2) * height

                            boxes.append(BoundingBox(
                                class_id=cls_id,
                                xmin=max(0, xmin),
                                ymin=max(0, ymin),
                                xmax=min(width, xmax),
                                ymax=min(height, ymax)
                            ))
                        except (ValueError, IndexError):
                            continue
        except Exception:
            pass
        return boxes
