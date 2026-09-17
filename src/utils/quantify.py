"""Plastic quantification from a single Ultralytics Results object.

Shared by evaluate.py, predict.py and the Streamlit demo so the numbers
reported everywhere are computed the same way.
"""
from __future__ import annotations

import numpy as np

from src.config import CLASS_NAMES, PLASTIC_CLASSES, DEBRIS_CLASSES

PLASTIC_IDS = {CLASS_NAMES.index(c) for c in PLASTIC_CLASSES}
DEBRIS_IDS = {CLASS_NAMES.index(c) for c in DEBRIS_CLASSES}


def quantify(result) -> dict:
    """result: a single ultralytics.engine.results.Results (segmentation model).

    Coverage is computed from real pixel masks when the model has segmentation
    output; if only boxes are available (e.g. a detection-only model), we
    fall back to bbox-area coverage and name the field accordingly so it is
    never mistaken for true pixel coverage.
    """
    h, w = result.orig_shape
    total_px = h * w

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return _empty_result(has_masks=result.masks is not None)

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    is_plastic = np.array([c in PLASTIC_IDS for c in cls_ids])
    is_debris = np.array([c in DEBRIS_IDS for c in cls_ids])

    out = {
        "plastic_count": int(is_plastic.sum()),
        "debris_count": int(is_debris.sum()),
        "average_confidence": float(confs[is_plastic].mean()) if is_plastic.any() else 0.0,
        "max_confidence": float(confs[is_plastic].max()) if is_plastic.any() else 0.0,
    }

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()  # (N, mh, mw) at model resolution, resized below
        plastic_px = 0
        for i, m in enumerate(masks):
            if is_plastic[i]:
                plastic_px += int((m > 0.5).sum() * (total_px / m.size))
        out["plastic_area"] = plastic_px
        out["plastic_coverage_percentage"] = round(100.0 * plastic_px / total_px, 4)
        out["area_is_estimated_from_bbox"] = False
    else:
        xyxy = boxes.xyxy.cpu().numpy()
        plastic_area = sum(
            max(0, x2 - x1) * max(0, y2 - y1)
            for (x1, y1, x2, y2), keep in zip(xyxy, is_plastic) if keep
        )
        out["plastic_area"] = int(plastic_area)
        out["plastic_coverage_percentage"] = round(100.0 * plastic_area / total_px, 4)
        out["area_is_estimated_from_bbox"] = True

    class_wise = {}
    for cid in np.unique(cls_ids):
        class_wise[CLASS_NAMES[cid]] = int((cls_ids == cid).sum())
    out["class_wise_counts"] = class_wise

    return out


def _empty_result(has_masks: bool) -> dict:
    return {
        "plastic_count": 0,
        "debris_count": 0,
        "plastic_area": 0,
        "plastic_coverage_percentage": 0.0,
        "average_confidence": 0.0,
        "max_confidence": 0.0,
        "area_is_estimated_from_bbox": not has_masks,
        "class_wise_counts": {},
    }
