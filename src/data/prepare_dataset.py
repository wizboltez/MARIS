"""
End-to-end preprocessing: validate -> leak-free split -> COCO->YOLO-seg conversion.

Pipeline:
  1. Validate both raw COCO splits (src/data/validate_annotations.py), save an
     EDA report + figures.
  2. Pool all valid images from official train+val together.
  3. Build "leak groups" = source video id, UNIONED across any two videos
     that share a near-duplicate image (so a duplicate frame can't land in
     both train and test even if it was captured on two different dives).
  4. GroupShuffleSplit those leak groups into train/val/test per
     SPLIT_FRACTIONS (seeded, reproducible).
  5. Convert each split's images+annotations to YOLO segmentation format and
     copy into data/processed/{split}/{images,labels}/.
  6. Write data/processed/data.yaml (Ultralytics dataset config) and
     data/processed/split_manifest.csv (full audit trail).

Run:
    python -m src.data.prepare_dataset
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (  # noqa: E402
    CLASS_NAMES,
    DATA_YAML_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    SPLIT_FRACTIONS,
    SPLIT_MANIFEST_PATH,
)
from src.data.validate_annotations import validate_full_dataset  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    plot_class_distribution,
    plot_image_size_distribution,
    plot_objects_per_image,
)


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_leak_groups(images_by_id: dict, duplicate_groups: list) -> dict:
    """Map image_id -> leak-group id, where two images are in the same group
    if they share a video_id OR are near-duplicates of each other."""
    video_ids = {img["video_id"] for img in images_by_id.values() if img.get("video_id")}
    uf = UnionFind(video_ids)

    name_to_video = {img["file_name"]: img.get("video_id") for img in images_by_id.values()}
    cross_video_dupe_groups = 0
    for group in duplicate_groups:
        vids = {name_to_video[n] for n in group if name_to_video.get(n)}
        if len(vids) > 1:
            cross_video_dupe_groups += 1
            vids = list(vids)
            for v in vids[1:]:
                uf.union(vids[0], v)
    if cross_video_dupe_groups:
        print(f"Note: {cross_video_dupe_groups} duplicate-image group(s) span multiple "
              f"videos; those videos were merged into one leak-group so they stay together.")

    return {
        img["id"]: uf.find(img["video_id"]) if img.get("video_id") else f"novideo_{img['id']}"
        for img in images_by_id.values()
    }


def group_split(image_ids: list, leak_group_of: dict, seed: int) -> dict:
    groups = np.array([leak_group_of[i] for i in image_ids])
    ids = np.array(image_ids)

    val_test_frac = SPLIT_FRACTIONS["val"] + SPLIT_FRACTIONS["test"]
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_test_frac, random_state=seed)
    train_idx, rest_idx = next(gss1.split(ids, groups=groups))

    rest_ids, rest_groups = ids[rest_idx], groups[rest_idx]
    test_frac_of_rest = SPLIT_FRACTIONS["test"] / val_test_frac
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_frac_of_rest, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest_ids, groups=rest_groups))

    assignment = {i: "train" for i in ids[train_idx]}
    assignment.update({i: "val" for i in rest_ids[val_idx]})
    assignment.update({i: "test" for i in rest_ids[test_idx]})

    g_train = set(groups[train_idx])
    g_val = set(rest_groups[val_idx])
    g_test = set(rest_groups[test_idx])
    assert not (g_train & g_val) and not (g_train & g_test) and not (g_val & g_test), \
        "leak-group overlap detected between splits -- this should be impossible"
    return assignment


def coco_poly_to_yolo_lines(anns: list, cat_id_to_yolo: dict, img_w: int, img_h: int) -> list:
    """Convert this image's valid COCO annotations to YOLO-seg label lines.

    For an instance whose segmentation has multiple polygon parts (occlusion
    splits it into disjoint regions), we keep only the largest-area part.
    This is a deliberate simplification for this prototype: it slightly
    undercounts a heavily-occluded instance's pixel area, but keeps every
    label line a single valid closed polygon, which is what a YOLO
    segmentation label line requires.
    """
    lines = []
    for ann in anns:
        seg = ann.get("segmentation")
        if not isinstance(seg, list) or not seg:
            continue
        parts = [p for p in seg if isinstance(p, list) and len(p) >= 6]
        if not parts:
            continue

        def poly_area(p):
            xs, ys = p[0::2], p[1::2]
            return abs(sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
                           for i in range(len(xs)))) / 2

        poly = max(parts, key=poly_area)
        cls = cat_id_to_yolo[ann["category_id"]]
        norm = []
        for i in range(0, len(poly), 2):
            x = min(max(poly[i] / img_w, 0.0), 1.0)
            y = min(max(poly[i + 1] / img_h, 0.0), 1.0)
            norm.extend([f"{x:.6f}", f"{y:.6f}"])
        lines.append(f"{cls} " + " ".join(norm))
    return lines


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Validating raw annotations (this also computes EDA stats)...")
    from src.data.validate_annotations import validate_split, load_coco
    from src.config import RAW_TRAIN_IMAGES, RAW_VAL_IMAGES, RAW_TRAIN_JSON, RAW_VAL_JSON

    report, train_coco, val_coco = validate_full_dataset(compute_hashes=True)

    with open(METRICS_DIR / "dataset_validation_report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    plot_class_distribution(dict(report.class_counts), FIGURES_DIR / "class_distribution.png")
    plot_image_size_distribution(
        {f"{w}x{h}": c for (w, h), c in report.image_sizes.items()},
        FIGURES_DIR / "image_size_distribution.png",
    )
    plot_objects_per_image(dict(report.objects_per_image), FIGURES_DIR / "objects_per_image.png")

    # Re-run validate_split to get the augmented image records (path, video_id)
    # and per-image annotation lists we need for conversion.
    images_by_id: dict = {}
    anns_by_image: dict = {}
    cat_id_to_yolo = {c["id"]: i for i, c in enumerate(train_coco["categories"])}
    assert [c["name"] for c in train_coco["categories"]] == CLASS_NAMES

    for coco, img_dir in [(train_coco, RAW_TRAIN_IMAGES), (val_coco, RAW_VAL_IMAGES)]:
        sub_report = report.__class__()
        imgs = validate_split(coco, img_dir, sub_report, compute_hashes=False)
        for img in imgs.values():
            images_by_id[img["id"]] = img
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    valid_image_ids = [
        i for i, img in images_by_id.items()
        if img["path"].exists()
    ]
    n_dropped = len(images_by_id) - len(valid_image_ids)
    if n_dropped:
        print(f"Dropping {n_dropped} image(s) missing/corrupt on disk from the pool.")

    leak_group_of = build_leak_groups(images_by_id, report.duplicate_groups)
    assignment = group_split(valid_image_ids, leak_group_of, RANDOM_SEED)

    manifest_rows = []
    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    for img_id in tqdm(valid_image_ids, desc="Converting + copying"):
        img = images_by_id[img_id]
        split = assignment[img_id]
        valid_anns = [
            a for a in anns_by_image.get(img_id, [])
            if a.get("area", 0) > 0 and isinstance(a.get("segmentation"), list) and a["segmentation"]
        ]
        lines = coco_poly_to_yolo_lines(valid_anns, cat_id_to_yolo, img["width"], img["height"])

        dst_img = PROCESSED_DIR / split / "images" / img["file_name"]
        shutil.copyfile(img["path"], dst_img)
        dst_lbl = PROCESSED_DIR / split / "labels" / (Path(img["file_name"]).stem + ".txt")
        dst_lbl.write_text("\n".join(lines))

        manifest_rows.append({
            "image_id": img_id,
            "file_name": img["file_name"],
            "video_id": img.get("video_id"),
            "split": split,
            "width": img["width"],
            "height": img["height"],
            "n_objects": len(lines),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)

    print("\nSplit sizes (images):")
    print(manifest["split"].value_counts())
    print("\nLeak-group video overlap check:")
    for a in ["train", "val", "test"]:
        for b in ["train", "val", "test"]:
            if a >= b:
                continue
            ga = set(manifest.loc[manifest.split == a, "video_id"].dropna())
            gb = set(manifest.loc[manifest.split == b, "video_id"].dropna())
            print(f"  {a} vs {b} shared video ids: {len(ga & gb)}")

    data_yaml = {
        "path": str(PROCESSED_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    with open(DATA_YAML_PATH, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"\nWrote {DATA_YAML_PATH}")
    print(f"Wrote {SPLIT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
