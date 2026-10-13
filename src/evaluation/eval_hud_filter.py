"""Evaluate On-Screen Text (HUD) Filtering on held-out TEST split.
Measures both false alarms removed AND true ground-truth detections lost.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import BEST_MODEL_PATH, CLASS_NAMES, PROCESSED_DIR, PROJECT_ROOT
from src.evaluation.tune_thresholds import box_iou_xyxy, load_ground_truths_with_sizes, run_split_inference

TEST_DIR = PROCESSED_DIR / "test"
OUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
PLASTIC_ID = CLASS_NAMES.index("trash_plastic")


def evaluate_hud_filter(test_gt: dict[str, dict], test_preds: dict[str, list[dict]], top_px: int, bot_px: int, conf_thresh: float = 0.18):
    """Filter detections whose center lies in top_px or bottom bot_px of image."""
    n_gt = sum(sum(1 for item in d["gts"] if item["cls"] == PLASTIC_ID) for d in test_gt.values())
    gt_lost = 0

    # Check how many ground truth boxes fall inside the filter zone
    for img_name, d in test_gt.items():
        h = d["h"]
        for item in d["gts"]:
            if item["cls"] == PLASTIC_ID:
                box = item["box"]
                cy = (box[1] + box[3]) / 2.0
                if cy < top_px or cy > (h - bot_px):
                    gt_lost += 1

    tp_raw, fp_raw = 0, 0
    tp_filt, fp_filt = 0, 0
    hud_fp_removed = 0

    for img_name, d in test_gt.items():
        h = d["h"]
        gt_boxes = np.array([item["box"] for item in d["gts"] if item["cls"] == PLASTIC_ID], dtype=np.float32)
        preds = [p for p in test_preds.get(img_name, []) if p["cls"] == PLASTIC_ID and p["conf"] >= conf_thresh]
        if len(preds) == 0:
            continue

        # 1. Raw evaluation
        raw_pred_boxes = np.array([p["box"] for p in preds], dtype=np.float32)
        if len(gt_boxes) == 0:
            fp_raw += len(raw_pred_boxes)
        else:
            ious = box_iou_xyxy(raw_pred_boxes, gt_boxes)
            matched_gt = set()
            for p_idx in np.argsort([-p["conf"] for p in preds]):
                best_gt = -1
                best_iou = 0.5
                for g_idx in range(len(gt_boxes)):
                    if g_idx in matched_gt:
                        continue
                    if ious[p_idx, g_idx] >= best_iou:
                        best_iou = ious[p_idx, g_idx]
                        best_gt = g_idx
                if best_gt >= 0:
                    matched_gt.add(best_gt)
                    tp_raw += 1
                else:
                    fp_raw += 1

        # 2. Filtered evaluation
        filt_preds = []
        for p in preds:
            box = p["box"]
            cy = (box[1] + box[3]) / 2.0
            if cy < top_px or cy > (h - bot_px):
                hud_fp_removed += 1
            else:
                filt_preds.append(p)

        if len(filt_preds) > 0:
            filt_pred_boxes = np.array([p["box"] for p in filt_preds], dtype=np.float32)
            if len(gt_boxes) == 0:
                fp_filt += len(filt_pred_boxes)
            else:
                ious = box_iou_xyxy(filt_pred_boxes, gt_boxes)
                matched_gt = set()
                for p_idx in np.argsort([-p["conf"] for p in filt_preds]):
                    best_gt = -1
                    best_iou = 0.5
                    for g_idx in range(len(gt_boxes)):
                        if g_idx in matched_gt:
                            continue
                        if ious[p_idx, g_idx] >= best_iou:
                            best_iou = ious[p_idx, g_idx]
                            best_gt = g_idx
                if best_gt >= 0:
                    matched_gt.add(best_gt)
                    tp_filt += 1
                else:
                    fp_filt += 1

    fn_raw = n_gt - tp_raw
    p_raw = tp_raw / (tp_raw + fp_raw) if (tp_raw + fp_raw) > 0 else 1.0
    r_raw = tp_raw / (tp_raw + fn_raw) if (tp_raw + fn_raw) > 0 else 0.0
    f1_raw = (2 * p_raw * r_raw) / (p_raw + r_raw) if (p_raw + r_raw) > 0 else 0.0

    fn_filt = n_gt - tp_filt
    p_filt = tp_filt / (tp_filt + fp_filt) if (tp_filt + fp_filt) > 0 else 1.0
    r_filt = tp_filt / (tp_filt + fn_filt) if (tp_filt + fn_filt) > 0 else 0.0
    f1_filt = (2 * p_filt * r_filt) / (p_filt + r_filt) if (p_filt + r_filt) > 0 else 0.0

    return {
        "top_px": top_px,
        "bot_px": bot_px,
        "conf_thresh": conf_thresh,
        "n_gt": n_gt,
        "gt_lost_in_zone": gt_lost,
        "tp_raw": tp_raw,
        "fp_raw": fp_raw,
        "p_raw": round(p_raw, 4),
        "r_raw": round(r_raw, 4),
        "f1_raw": round(f1_raw, 4),
        "tp_filt": tp_filt,
        "fp_filt": fp_filt,
        "p_filt": round(p_filt, 4),
        "r_filt": round(r_filt, 4),
        "f1_filt": round(f1_filt, 4),
        "hud_detections_removed": hud_fp_removed,
    }


def main():
    print("Loading test ground truth with exact dimensions...")
    test_gt = load_ground_truths_with_sizes(TEST_DIR)
    model = YOLO(str(BEST_MODEL_PATH))
    print("Running test predictions...")
    test_preds = run_split_inference(model, TEST_DIR, conf_floor=0.02)

    results = []
    # Test different zone margins: top and bottom banners
    margins = [(0, 0), (25, 25), (30, 30), (35, 35), (40, 40)]
    print("\n--- Evaluating HUD margins on TEST split ---")
    for top_m, bot_m in margins:
        res = evaluate_hud_filter(test_gt, test_preds, top_px=top_m, bot_px=bot_m, conf_thresh=0.18)
        results.append(res)
        print(f"Zone [{top_m:2d}px top, {bot_m:2d}px bot]: GT lost = {res['gt_lost_in_zone']:2d}/{res['n_gt']} | "
              f"Raw P/R/F1 = {res['p_raw']:.3f}/{res['r_raw']:.3f}/{res['f1_raw']:.3f} -> "
              f"Filt P/R/F1 = {res['p_filt']:.3f}/{res['r_filt']:.3f}/{res['f1_filt']:.3f} | "
              f"Detections removed = {res['hud_detections_removed']}")

    out_file = OUT_DIR / "hud_filter_test_ablation.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved HUD filter ablation to {out_file}")


if __name__ == "__main__":
    main()
