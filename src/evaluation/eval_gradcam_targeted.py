"""Evaluate Targeted Per-Detection Grad-CAM vs Global Grad-CAM on held-out TEST frames.
Measures heat percentage inside plastic bounding boxes and Deletion/Insertion AUC.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import BEST_MODEL_PATH, CLASS_NAMES, PROCESSED_DIR, PROJECT_ROOT
from src.modules.gradcam import YOLOGradCAM
from src.modules.gradcam_targeted import TargetedYOLOGradCAM, evaluate_faithfulness

TEST_DIR = PROCESSED_DIR / "test"
OUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
PLASTIC_ID = CLASS_NAMES.index("trash_plastic")


def main():
    print(f"Loading YOLO model from {BEST_MODEL_PATH}...")
    model = YOLO(str(BEST_MODEL_PATH))

    global_cam_engine = YOLOGradCAM(model, imgsz=640)
    targeted_cam_engine = TargetedYOLOGradCAM(model, imgsz=640)

    test_imgs = sorted((TEST_DIR / "images").glob("*.jpg"))
    print(f"Searching for test images with plastic detections at conf >= 0.20...")

    results = []
    # Test up to 25 plastic detections across the test split
    for p in test_imgs:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        res = model.predict(img_bgr, conf=0.20, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        plastic_indices = [i for i, c in enumerate(clss) if c == PLASTIC_ID]
        if not plastic_indices:
            continue

        # Evaluate the highest confidence plastic detection in this frame
        best_i = plastic_indices[int(np.argmax(confs[plastic_indices]))]
        target_box = boxes[best_i]
        conf_val = float(confs[best_i])

        # 1. Global Grad-CAM heat inside box
        global_res = global_cam_engine.compute(img_bgr, class_id=PLASTIC_ID, conf=0.18)
        global_cam = global_res["cam"]
        bx1, by1, bx2, by2 = map(int, target_box)
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(img_bgr.shape[1], bx2), min(img_bgr.shape[0], by2)

        tot_global = float(global_cam.sum())
        box_global = float(global_cam[by1:by2, bx1:bx2].sum())
        pct_global = (box_global / tot_global * 100.0) if tot_global > 0 else 0.0

        # 2. Targeted Grad-CAM + Faithfulness test
        faith_res = evaluate_faithfulness(model, targeted_cam_engine, img_bgr, target_box, steps=8)

        record = {
            "image": p.name,
            "plastic_conf": round(conf_val, 3),
            "box": [round(float(c), 1) for c in target_box],
            "global_cam_heat_in_box_pct": round(pct_global, 2),
            "targeted_cam_heat_in_box_pct": faith_res["heat_inside_box_pct"],
            "deletion_auc": faith_res["deletion_auc"],
            "insertion_auc": faith_res["insertion_auc"],
        }
        results.append(record)
        print(f"[{len(results):2d}] {p.name}: Global Heat in Box = {pct_global:5.1f}% -> Targeted = {faith_res['heat_inside_box_pct']:5.1f}% | "
              f"Del AUC = {faith_res['deletion_auc']:.3f}, Ins AUC = {faith_res['insertion_auc']:.3f}")

        if len(results) >= 20:
            break

    if not results:
        print("No test images with plastic detections found.")
        return

    # Compute macro-averages
    avg_global_heat = float(np.mean([r["global_cam_heat_in_box_pct"] for r in results]))
    avg_targeted_heat = float(np.mean([r["targeted_cam_heat_in_box_pct"] for r in results]))
    avg_del_auc = float(np.mean([r["deletion_auc"] for r in results]))
    avg_ins_auc = float(np.mean([r["insertion_auc"] for r in results]))

    summary = {
        "n_samples": len(results),
        "mean_global_cam_heat_in_box_pct": round(avg_global_heat, 2),
        "mean_targeted_cam_heat_in_box_pct": round(avg_targeted_heat, 2),
        "mean_deletion_auc": round(avg_del_auc, 3),
        "mean_insertion_auc": round(avg_ins_auc, 3),
        "samples": results,
    }

    out_file = OUT_DIR / "gradcam_targeted_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Grad-CAM Evaluation Summary ---")
    print(f"Mean Global Grad-CAM heat inside box:   {avg_global_heat:.2f}%")
    print(f"Mean Targeted Grad-CAM heat inside box: {avg_targeted_heat:.2f}%")
    print(f"Mean Deletion AUC (lower is better):    {avg_del_auc:.3f}")
    print(f"Mean Insertion AUC (higher is better):   {avg_ins_auc:.3f}")
    print(f"Saved evaluation to {out_file}")


if __name__ == "__main__":
    main()
