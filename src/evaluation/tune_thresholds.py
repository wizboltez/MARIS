"""Tune per-class confidence thresholds on VAL split and evaluate once on TEST split.
Handles both 480x270 and 480x360 images dynamically.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import BEST_MODEL_PATH, CLASS_NAMES, PROCESSED_DIR, PROJECT_ROOT

VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"
OUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLASTIC_ID = CLASS_NAMES.index("trash_plastic")


def box_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    if len(box1) == 0 or len(box2) == 0:
        return np.zeros((len(box1), len(box2)), dtype=np.float32)
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def load_ground_truths_with_sizes(split_dir: Path) -> dict[str, dict]:
    """Parse YOLO txt annotations into absolute xyxy coordinates using actual image sizes."""
    data = {}
    label_dir = split_dir / "labels"
    img_dir = split_dir / "images"

    for img_path in sorted(img_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = label_dir / f"{img_path.stem}.txt"
        items = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if len(parts) == 5:
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    x2 = (cx + bw / 2.0) * w
                    y2 = (cy + bh / 2.0) * h
                else:
                    coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
                    x1 = coords[:, 0].min() * w
                    y1 = coords[:, 1].min() * h
                    x2 = coords[:, 0].max() * w
                    y2 = coords[:, 1].max() * h
                items.append({"cls": cls_id, "box": np.array([x1, y1, x2, y2], dtype=np.float32)})

        data[img_path.name] = {"h": h, "w": w, "gts": items}
    return data


def run_split_inference(model: YOLO, split_dir: Path, conf_floor: float = 0.02, device=None) -> dict[str, list[dict]]:
    """Collect predictions per image using streaming inference to fit within 4GB VRAM."""
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    img_dir = split_dir / "images"
    results = model.predict(source=str(img_dir), conf=conf_floor, imgsz=640, device=device, stream=True, verbose=False)

    preds_by_img = {}
    for res in results:
        img_name = Path(res.path).name
        preds = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            for b, c, cl in zip(boxes, confs, clss):
                preds.append({"box": b, "conf": float(c), "cls": int(cl)})
        preds_by_img[img_name] = preds
    return preds_by_img


def sweep_metrics_for_class(gt_data: dict[str, dict], preds_by_img: dict[str, list[dict]], target_cls: int, thresholds: np.ndarray, iou_thresh: float = 0.5):
    n_gt = sum(sum(1 for item in d["gts"] if item["cls"] == target_cls) for d in gt_data.values())
    if n_gt == 0:
        return np.zeros_like(thresholds), np.zeros_like(thresholds), np.zeros_like(thresholds)

    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        tp, fp = 0, 0
        for img_name, d in gt_data.items():
            gt_boxes = np.array([item["box"] for item in d["gts"] if item["cls"] == target_cls], dtype=np.float32)
            preds = [p for p in preds_by_img.get(img_name, []) if p["cls"] == target_cls and p["conf"] >= t]
            if len(preds) == 0:
                continue
            pred_boxes = np.array([p["box"] for p in preds], dtype=np.float32)
            if len(gt_boxes) == 0:
                fp += len(pred_boxes)
                continue

            ious = box_iou_xyxy(pred_boxes, gt_boxes)
            matched_gt = set()
            pred_order = np.argsort([-p["conf"] for p in preds])
            for p_idx in pred_order:
                best_gt = -1
                best_iou = iou_thresh
                for g_idx in range(len(gt_boxes)):
                    if g_idx in matched_gt:
                        continue
                    if ious[p_idx, g_idx] >= best_iou:
                        best_iou = ious[p_idx, g_idx]
                        best_gt = g_idx
                if best_gt >= 0:
                    matched_gt.add(best_gt)
                    tp += 1
                else:
                    fp += 1

        fn = n_gt - tp
        p_val = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = (2 * p_val * r_val) / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        precisions.append(p_val)
        recalls.append(r_val)
        f1s.append(f1_val)

    return np.array(precisions), np.array(recalls), np.array(f1s)


def main():
    print(f"Loading model from {BEST_MODEL_PATH}...")
    model = YOLO(str(BEST_MODEL_PATH))

    print(f"Loading VAL ground truth (986 images, checking actual dimensions)...")
    val_gt = load_ground_truths_with_sizes(VAL_DIR)
    print("Running VAL inference (conf_floor=0.02)...")
    val_preds = run_split_inference(model, VAL_DIR, conf_floor=0.02)

    thresholds = np.linspace(0.02, 0.80, 79)
    optimal_thresholds = {}

    print("\n--- Tuning thresholds on VAL ---")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        prec, rec, f1 = sweep_metrics_for_class(val_gt, val_preds, cls_id, thresholds)
        if len(prec) == 0 or f1.max() == 0:
            optimal_thresholds[cls_name] = {"threshold": 0.25, "val_p": 0.0, "val_r": 0.0, "val_f1": 0.0}
            continue

        if cls_id == PLASTIC_ID:
            # Operational goal: Maximize F1 or recall subject to precision >= 0.50
            valid_p50 = np.where(prec >= 0.50)[0]
            if len(valid_p50) > 0:
                best_idx = valid_p50[np.argmax(rec[valid_p50])]
            else:
                best_idx = int(np.argmax(f1))
        else:
            best_idx = int(np.argmax(f1))

        chosen_t = float(thresholds[best_idx])
        optimal_thresholds[cls_name] = {
            "threshold": chosen_t,
            "val_p": float(prec[best_idx]),
            "val_r": float(rec[best_idx]),
            "val_f1": float(f1[best_idx]),
        }
        if cls_id in (PLASTIC_ID, CLASS_NAMES.index("animal_fish"), CLASS_NAMES.index("rov")):
            print(f"VAL: {cls_name:18s} -> Chosen thresh: {chosen_t:.2f} | P: {prec[best_idx]:.3f} | R: {rec[best_idx]:.3f} | F1: {f1[best_idx]:.3f}")

    val_thresh_file = OUT_DIR / "optimal_thresholds_val.json"
    with open(val_thresh_file, "w") as f:
        json.dump(optimal_thresholds, f, indent=2)
    print(f"Saved tuned thresholds to {val_thresh_file}")

    print("\n--- Evaluating TEST split strictly once with frozen thresholds ---")
    print("Loading TEST ground truth (1313 images)...")
    test_gt = load_ground_truths_with_sizes(TEST_DIR)
    print("Running TEST inference (conf_floor=0.02)...")
    test_preds = run_split_inference(model, TEST_DIR, conf_floor=0.02)

    test_metrics = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        chosen_t = optimal_thresholds[cls_name]["threshold"]
        prec, rec, f1 = sweep_metrics_for_class(test_gt, test_preds, cls_id, np.array([chosen_t]))
        test_metrics[cls_name] = {
            "threshold": chosen_t,
            "test_p": float(prec[0]),
            "test_r": float(rec[0]),
            "test_f1": float(f1[0]),
        }
        if cls_id in (PLASTIC_ID, CLASS_NAMES.index("animal_fish"), CLASS_NAMES.index("rov")):
            print(f"TEST: {cls_name:18s} @ thresh {chosen_t:.2f} -> P: {prec[0]:.3f} | R: {rec[0]:.3f} | F1: {f1[0]:.3f}")

    # Also evaluate plastic at the old static 0.18 threshold for direct comparison on the same script
    p_018, r_018, f_018 = sweep_metrics_for_class(test_gt, test_preds, PLASTIC_ID, np.array([0.18]))
    test_metrics["trash_plastic_at_018_baseline"] = {
        "threshold": 0.18,
        "test_p": float(p_018[0]),
        "test_r": float(r_018[0]),
        "test_f1": float(f_018[0]),
    }
    print(f"\nTEST COMPARISON for trash_plastic:")
    print(f"  At static 0.18: P = {p_018[0]:.3f}, R = {r_018[0]:.3f}, F1 = {f_018[0]:.3f}")
    print(f"  At tuned {optimal_thresholds['trash_plastic']['threshold']:.2f}: P = {test_metrics['trash_plastic']['test_p']:.3f}, R = {test_metrics['trash_plastic']['test_r']:.3f}, F1 = {test_metrics['trash_plastic']['test_f1']:.3f}")

    test_results_file = OUT_DIR / "held_out_test_metrics_tuned.json"
    with open(test_results_file, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved held-out test results to {test_results_file}")


if __name__ == "__main__":
    main()
