"""Train a single-class fish detector (FathomNet + TrashCan) and compare it with the 16-class model.

Model selection uses TrashCan VAL only (Ultralytics best.pt by val fitness). TEST is used once at the
end: both detectors are scored on the same TrashCan test fish at conf 0.18 (IoU >= 0.5), with a
95% cluster bootstrap over the 53 test videos.

Training runs on Colab (notebooks/colab_train_fish.ipynb). Locally, only the CPU test evaluation:
  python -m src.training.train_fish_detector --eval-only   (needs models/fish_detector/best_fish.pt)
"""
from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from src.config import BEST_MODEL_PATH, CLASS_NAMES, METRICS_DIR, PROCESSED_DIR, PROJECT_ROOT, SPLIT_MANIFEST_PATH
from src.evaluation.bootstrap_confidence_intervals import compute_image_stats
from src.evaluation.tune_thresholds import load_ground_truths_with_sizes, run_split_inference

FISH_DATA = PROJECT_ROOT / "data" / "processed_fish" / "data.yaml"
FISH_MODEL_DIR = PROJECT_ROOT / "models" / "fish_detector"
FISH_ID = CLASS_NAMES.index("animal_fish")
CONF = 0.18


def bootstrap(stats: dict, manifest: pd.DataFrame, n_boot: int = 1000) -> dict:
    vids = np.array(sorted(manifest["video_id"].unique()))
    arr = np.array([[sum(stats[f][k] for f in manifest[manifest["video_id"] == v]["file_name"]) for k in range(3)]
                    for v in vids])
    idx = np.random.default_rng(42).choice(len(vids), size=(n_boot, len(vids)), replace=True)
    b = arr[idx].sum(axis=1)
    bp = np.where(b[:, 0] + b[:, 1] > 0, b[:, 0] / np.maximum(1, b[:, 0] + b[:, 1]), 0.0)
    br = np.where(b[:, 2] > 0, b[:, 0] / np.maximum(1, b[:, 2]), 0.0)
    tp, fp, gt = arr.sum(axis=0)
    p, r = tp / max(1, tp + fp), tp / max(1, gt)

    def ci(x):
        return [round(float(np.percentile(x, 2.5)), 3), round(float(np.percentile(x, 97.5)), 3)]
    return {"precision": round(float(p), 3), "precision_ci": ci(bp), "recall": round(float(r), 3), "recall_ci": ci(br),
            "f1": round(float(2 * p * r / max(1e-9, p + r)), 3), "tp": int(tp), "fp": int(fp), "n_fish": int(gt)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--eval-only", action="store_true", help="skip training; evaluate best_fish.pt on CPU")
    args = ap.parse_args()
    if not args.eval_only:
        train(args)
    evaluate()


def train(args) -> None:

    device = 0 if torch.cuda.is_available() else "cpu"
    run = YOLO(args.model).train(
        data=str(FISH_DATA), epochs=args.epochs, patience=12, imgsz=640, batch=args.batch, device=device,
        seed=42, cos_lr=True, project=str(PROJECT_ROOT / "runs" / "fish"), name="yolo11s_fish", exist_ok=True,
        workers=2, plots=True,
    )
    best = PROJECT_ROOT / "runs" / "fish" / "yolo11s_fish" / "weights" / "best.pt"
    FISH_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, FISH_MODEL_DIR / "best_fish.pt")


def evaluate() -> None:
    # ---- single TEST evaluation: same TrashCan test fish, same matching, same threshold
    test_dir = PROCESSED_DIR / "test"
    gt = load_ground_truths_with_sizes(test_dir)
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH)
    manifest = manifest[manifest["split"] == "test"]

    old_preds = run_split_inference(YOLO(str(BEST_MODEL_PATH)), test_dir, conf_floor=0.02, device="cpu")
    old = bootstrap(compute_image_stats(gt, old_preds, FISH_ID, CONF), manifest)

    new_raw = run_split_inference(YOLO(str(FISH_MODEL_DIR / "best_fish.pt")), test_dir, conf_floor=0.02, device="cpu")
    new_preds = {k: [dict(p, cls=FISH_ID) for p in v] for k, v in new_raw.items()}  # class 0 'fish' -> TrashCan id
    new = bootstrap(compute_image_stats(gt, new_preds, FISH_ID, CONF), manifest)

    result = {
        "test_split": "TrashCan test (53 videos, never used for training or model selection)",
        "conf": CONF, "iou": 0.5,
        "old_16_class_model": old,
        "new_fish_detector": new,
        "new_model_better_f1": new["f1"] > old["f1"],
        "train_data": "FathomNet deep-sea fish + TrashCan train fish + 1000 TrashCan train negatives",
    }
    (METRICS_DIR / "fish_detector_comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
