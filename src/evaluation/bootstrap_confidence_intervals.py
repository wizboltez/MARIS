"""Vectorized Cluster Bootstrap 95% Confidence Intervals over 53 Test Videos.
Precomputes per-image TP, FP, GT counts so B=1000 resampling runs in seconds.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.config import BEST_MODEL_PATH, CLASS_NAMES, PROCESSED_DIR, PROJECT_ROOT, SPLIT_MANIFEST_PATH
from src.evaluation.tune_thresholds import box_iou_xyxy, load_ground_truths_with_sizes, run_split_inference

TEST_DIR = PROCESSED_DIR / "test"
OUT_DIR = PROJECT_ROOT / "outputs" / "metrics"
PLASTIC_ID = CLASS_NAMES.index("trash_plastic")
FISH_ID = CLASS_NAMES.index("animal_fish")
ROV_ID = CLASS_NAMES.index("rov")


def compute_image_stats(gt_data: dict[str, dict], preds_by_img: dict[str, list[dict]], target_cls: int, conf_thresh: float):
    """Precompute (tp, fp, n_gt) per image."""
    stats = {}
    for img_name, d in gt_data.items():
        gt_boxes = np.array([item["box"] for item in d["gts"] if item["cls"] == target_cls], dtype=np.float32)
        n_gt = len(gt_boxes)
        preds = [p for p in preds_by_img.get(img_name, []) if p["cls"] == target_cls and p["conf"] >= conf_thresh]

        if len(preds) == 0:
            stats[img_name] = (0, 0, n_gt)
            continue

        pred_boxes = np.array([p["box"] for p in preds], dtype=np.float32)
        if len(gt_boxes) == 0:
            stats[img_name] = (0, len(pred_boxes), n_gt)
            continue

        ious = box_iou_xyxy(pred_boxes, gt_boxes)
        matched_gt = set()
        tp, fp = 0, 0
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
                tp += 1
            else:
                fp += 1
        stats[img_name] = (tp, fp, n_gt)
    return stats


def main(n_boot: int = 1000):
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH)
    test_manifest = manifest[manifest["split"] == "test"]
    unique_videos = np.array(sorted(test_manifest["video_id"].unique()))
    n_videos = len(unique_videos)
    print(f"Loaded {n_videos} test video clusters ({len(test_manifest)} images).")

    test_gt = load_ground_truths_with_sizes(TEST_DIR)
    model = YOLO(str(BEST_MODEL_PATH))
    test_preds = run_split_inference(model, TEST_DIR, conf_floor=0.02)

    thresh_file = OUT_DIR / "optimal_thresholds_val.json"
    optimal_thresholds = json.loads(thresh_file.read_text()) if thresh_file.exists() else {}
    t_plastic = optimal_thresholds.get("trash_plastic", {}).get("threshold", 0.47)
    t_fish = optimal_thresholds.get("animal_fish", {}).get("threshold", 0.17)
    t_rov = optimal_thresholds.get("rov", {}).get("threshold", 0.59)

    tasks = {
        "plastic_tuned": (PLASTIC_ID, t_plastic),
        "plastic_018": (PLASTIC_ID, 0.18),
        "fish": (FISH_ID, t_fish),
        "rov": (ROV_ID, t_rov),
    }

    # Aggregate stats per video: vid_stats[task][vid_idx] = np.array([tp, fp, n_gt])
    vid_arrays = {}
    for task_name, (cls_id, t_val) in tasks.items():
        img_stats = compute_image_stats(test_gt, test_preds, cls_id, t_val)
        arr = np.zeros((n_videos, 3), dtype=np.int32)
        for i, vid in enumerate(unique_videos):
            vid_imgs = test_manifest[test_manifest["video_id"] == vid]["file_name"]
            tp_sum = sum(img_stats[m][0] for m in vid_imgs)
            fp_sum = sum(img_stats[m][1] for m in vid_imgs)
            gt_sum = sum(img_stats[m][2] for m in vid_imgs)
            arr[i] = [tp_sum, fp_sum, gt_sum]
        vid_arrays[task_name] = arr

    rng = np.random.default_rng(42)
    boot_indices = rng.choice(n_videos, size=(n_boot, n_videos), replace=True)

    summary = {"n_videos": int(n_videos), "n_boot": n_boot, "metrics": {}}

    print("\n--- 95% Cluster Bootstrap Over 53 Test Videos (B = 1000) ---")
    for task_name, arr in vid_arrays.items():
        # Point estimate
        tot_tp, tot_fp, tot_gt = arr.sum(axis=0)
        p_pt = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
        r_pt = tot_tp / tot_gt if tot_gt > 0 else 0.0
        f1_pt = (2 * p_pt * r_pt) / (p_pt + r_pt) if (p_pt + r_pt) > 0 else 0.0

        # Resample
        boot_sums = arr[boot_indices].sum(axis=1)  # shape (n_boot, 3)
        b_tp = boot_sums[:, 0]
        b_fp = boot_sums[:, 1]
        b_gt = boot_sums[:, 2]

        b_p = np.where(b_tp + b_fp > 0, b_tp / (b_tp + b_fp), 0.0)
        b_r = np.where(b_gt > 0, b_tp / b_gt, 0.0)
        b_f1 = np.where(b_p + b_r > 0, (2 * b_p * b_r) / (b_p + b_r), 0.0)

        p_ci = [round(float(np.percentile(b_p, 2.5)), 3), round(float(np.percentile(b_p, 97.5)), 3)]
        r_ci = [round(float(np.percentile(b_r, 2.5)), 3), round(float(np.percentile(b_r, 97.5)), 3)]
        f1_ci = [round(float(np.percentile(b_f1, 2.5)), 3), round(float(np.percentile(b_f1, 97.5)), 3)]

        summary["metrics"][task_name] = {
            "threshold": tasks[task_name][1],
            "precision": f"{p_pt:.3f} [{p_ci[0]:.3f}, {p_ci[1]:.3f}]",
            "recall": f"{r_pt:.3f} [{r_ci[0]:.3f}, {r_ci[1]:.3f}]",
            "f1": f"{f1_pt:.3f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]",
        }
        print(f"{task_name:15s} | P: {summary['metrics'][task_name]['precision']} | R: {summary['metrics'][task_name]['recall']} | F1: {summary['metrics'][task_name]['f1']}")

    out_file = OUT_DIR / "test_bootstrap_confidence_intervals.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved bootstrap metrics to {out_file}")


if __name__ == "__main__":
    main()
