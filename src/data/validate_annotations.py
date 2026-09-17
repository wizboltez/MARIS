"""
Inspect and validate the raw TrashCan COCO annotations.

Checks performed (all computed from the actual files, nothing fabricated):
  - image file exists / is a readable, non-corrupt JPEG
  - image dimensions match what the COCO json claims
  - bbox is within image bounds and has positive width/height
  - segmentation polygon has at least 3 vertices (a valid polygon) and a
    positive area
  - near-duplicate images, via perceptual hash (imagehash.phash)
  - class distribution, image-size distribution, objects-per-image

Run standalone:
    python -m src.data.validate_annotations
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
from PIL import Image

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (  # noqa: E402
    CLASS_NAMES,
    RAW_TRAIN_IMAGES,
    RAW_TRAIN_JSON,
    RAW_VAL_IMAGES,
    RAW_VAL_JSON,
)

VIDEO_ID_RE = re.compile(r"vid_(\d+)_frame(\d+)\.jpg$")


@dataclass
class ValidationReport:
    n_images_listed: int = 0
    n_images_missing: int = 0
    n_images_corrupt: int = 0
    n_images_dim_mismatch: int = 0
    n_annotations: int = 0
    n_annotations_invalid_bbox: int = 0
    n_annotations_invalid_polygon: int = 0
    n_annotations_zero_area: int = 0
    class_counts: Counter = field(default_factory=Counter)
    image_sizes: Counter = field(default_factory=Counter)
    objects_per_image: Counter = field(default_factory=Counter)
    duplicate_groups: list = field(default_factory=list)
    missing_files: list = field(default_factory=list)
    corrupt_files: list = field(default_factory=list)
    invalid_annotation_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["class_counts"] = dict(self.class_counts)
        d["image_sizes"] = {f"{w}x{h}": c for (w, h), c in self.image_sizes.items()}
        d["objects_per_image"] = dict(self.objects_per_image)
        d["n_duplicate_groups"] = len(self.duplicate_groups)
        return d


def load_coco(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def extract_video_id(file_name: str) -> str | None:
    m = VIDEO_ID_RE.match(file_name)
    return m.group(1) if m else None


def validate_split(coco: dict, images_dir: Path, report: ValidationReport,
                    compute_hashes: bool = True) -> dict[int, dict]:
    """Validate one COCO split in-place on `report`. Returns image_id -> image record
    (augmented with a resolved `path` and `video_id`)."""
    images_by_id: dict[int, dict] = {}
    hashes: dict[str, list[str]] = defaultdict(list)

    for img in coco["images"]:
        report.n_images_listed += 1
        path = images_dir / img["file_name"]
        img = dict(img)
        img["path"] = path
        img["video_id"] = extract_video_id(img["file_name"])

        if not path.exists():
            report.n_images_missing += 1
            report.missing_files.append(str(path))
            images_by_id[img["id"]] = img
            continue

        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                w, h = im.size
                if compute_hashes:
                    hashes[str(imagehash.phash(im))].append(img["file_name"])
        except Exception:
            report.n_images_corrupt += 1
            report.corrupt_files.append(str(path))
            images_by_id[img["id"]] = img
            continue

        if (w, h) != (img["width"], img["height"]):
            report.n_images_dim_mismatch += 1
        report.image_sizes[(w, h)] += 1
        images_by_id[img["id"]] = img

    if compute_hashes:
        for _, names in hashes.items():
            if len(names) > 1:
                report.duplicate_groups.append(names)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    obj_count = Counter()
    for ann in coco["annotations"]:
        report.n_annotations += 1
        img = images_by_id.get(ann["image_id"])
        w = img["width"] if img else None
        h = img["height"] if img else None

        x, y, bw, bh = ann["bbox"]
        valid_bbox = bw > 0 and bh > 0
        if valid_bbox and w is not None:
            valid_bbox = 0 <= x and 0 <= y and (x + bw) <= w + 1 and (y + bh) <= h + 1
        if not valid_bbox:
            report.n_annotations_invalid_bbox += 1
            report.invalid_annotation_ids.append(ann["id"])

        seg = ann.get("segmentation")
        valid_poly = isinstance(seg, list) and len(seg) > 0 and all(
            isinstance(poly, list) and len(poly) >= 6 for poly in seg
        )
        if not valid_poly:
            report.n_annotations_invalid_polygon += 1

        if ann.get("area", 0) <= 0:
            report.n_annotations_zero_area += 1

        report.class_counts[cat_id_to_name.get(ann["category_id"], "unknown")] += 1
        obj_count[ann["image_id"]] += 1

    for img_id in images_by_id:
        report.objects_per_image[obj_count.get(img_id, 0)] += 1

    return images_by_id


def validate_full_dataset(compute_hashes: bool = True) -> tuple[ValidationReport, dict, dict]:
    """Validate the combined (train+val) material_version dataset.

    Returns (report, train_coco, val_coco) so callers (prepare_dataset.py)
    don't have to re-parse the JSON files.
    """
    report = ValidationReport()
    train_coco = load_coco(RAW_TRAIN_JSON)
    val_coco = load_coco(RAW_VAL_JSON)

    assert [c["name"] for c in train_coco["categories"]] == CLASS_NAMES, (
        "category order in instances_train_trashcan.json no longer matches "
        "src/config.CLASS_NAMES -- update CLASS_NAMES if the dataset changed."
    )

    validate_split(train_coco, RAW_TRAIN_IMAGES, report, compute_hashes)
    validate_split(val_coco, RAW_VAL_IMAGES, report, compute_hashes)
    return report, train_coco, val_coco


def _print_report(report: ValidationReport) -> None:
    print(f"Images listed in annotations : {report.n_images_listed}")
    print(f"Images missing on disk       : {report.n_images_missing}")
    print(f"Images corrupt/unreadable    : {report.n_images_corrupt}")
    print(f"Images with dim mismatch     : {report.n_images_dim_mismatch}")
    print(f"Annotations total            : {report.n_annotations}")
    print(f"Annotations invalid bbox     : {report.n_annotations_invalid_bbox}")
    print(f"Annotations invalid polygon  : {report.n_annotations_invalid_polygon}")
    print(f"Annotations zero area        : {report.n_annotations_zero_area}")
    print(f"Near-duplicate image groups  : {len(report.duplicate_groups)}")
    print(f"Distinct image sizes         : {dict(report.image_sizes)}")
    print("Class distribution:")
    for name, count in report.class_counts.most_common():
        print(f"  {name:<22} {count}")


if __name__ == "__main__":
    report, _, _ = validate_full_dataset()
    _print_report(report)
