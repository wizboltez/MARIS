"""
Evaluate a trained model on the held-out test split and produce the figures
+ error-analysis examples the report needs.

Run:
    python -m src.evaluation.evaluate --weights models/plastic_detector/best_model.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (  # noqa: E402
    BEST_MODEL_PATH, CLASS_NAMES, DATA_YAML_PATH, FIGURES_DIR, METRICS_DIR,
    PROCESSED_DIR, PLASTIC_CLASSES,
)
from src.utils.common import get_device  # noqa: E402
from src.utils.quantify import quantify  # noqa: E402
from src.utils.visualization import draw_detections, plot_sample_grid  # noqa: E402

PLASTIC_ID = CLASS_NAMES.index(PLASTIC_CLASSES[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=str, default=str(BEST_MODEL_PATH))
    p.add_argument("--data", type=str, default=str(DATA_YAML_PATH))
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n-error-examples", type=int, default=12)
    return p.parse_args()


def run_official_metrics(model, args) -> dict:
    """Ultralytics' own val() gives P/R/mAP50/mAP50-95 per class + overall,
    plus a confusion matrix and PR-curve PNGs saved directly to its run dir."""
    metrics = model.val(
        data=args.data, split=args.split, imgsz=args.imgsz, device=get_device(args.device),
        plots=True, project="runs/segment", name="eval",
    )
    save_dir = Path(metrics.save_dir)
    for fname in ["confusion_matrix.png", "confusion_matrix_normalized.png",
                  "BoxPR_curve.png", "MaskPR_curve.png", "BoxF1_curve.png", "MaskF1_curve.png"]:
        src = save_dir / fname
        if src.exists():
            (FIGURES_DIR / fname).write_bytes(src.read_bytes())

    box = metrics.box
    seg = metrics.seg if hasattr(metrics, "seg") else None
    summary = {
        "split": args.split,
        "box_precision": float(box.mp),
        "box_recall": float(box.mr),
        "box_map50": float(box.map50),
        "box_map50_95": float(box.map),
        "box_f1": float(2 * box.mp * box.mr / (box.mp + box.mr)) if (box.mp + box.mr) > 0 else 0.0,
    }
    if seg is not None:
        summary.update({
            "mask_precision": float(seg.mp),
            "mask_recall": float(seg.mr),
            "mask_map50": float(seg.map50),
            "mask_map50_95": float(seg.map),
            "mask_f1": float(2 * seg.mp * seg.mr / (seg.mp + seg.mr)) if (seg.mp + seg.mr) > 0 else 0.0,
        })
    return summary


def error_analysis(model, args) -> None:
    """Run inference over the test split ourselves (separately from model.val())
    so we can pick out qualitatively interesting cases: false positives,
    missed ground truth, low-confidence hits, and small objects."""
    img_dir = PROCESSED_DIR / args.split / "images"
    lbl_dir = PROCESSED_DIR / args.split / "labels"
    image_paths = sorted(img_dir.glob("*.jpg"))

    low_conf, small_objs, no_detections, many_detections = [], [], [], []

    for path in image_paths:
        result = model.predict(str(path), imgsz=args.imgsz, conf=0.1,
                                device=get_device(args.device), verbose=False)[0]
        gt_path = lbl_dir / (path.stem + ".txt")
        gt_lines = [l for l in gt_path.read_text().splitlines() if l.strip()] if gt_path.exists() else []

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            if gt_lines:
                no_detections.append((path, result))
            continue

        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        img_area = result.orig_shape[0] * result.orig_shape[1]

        if confs.min() < 0.35:
            low_conf.append((path, result, float(confs.min())))
        if (areas / img_area).min() < 0.01:
            small_objs.append((path, result))
        if len(boxes) >= 6:
            many_detections.append((path, result))

    def save_grid(items, name, caption_fn, k):
        items = items[:k]
        if not items:
            print(f"  (no examples found for {name})")
            return
        samples = []
        for entry in items:
            path, result = entry[0], entry[1]
            img = cv2.imread(str(path))
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            xyxy = result.boxes.xyxy.cpu().numpy()
            masks = None
            if result.masks is not None:
                mh, mw = img.shape[:2]
                masks = np.stack([
                    cv2.resize(m, (mw, mh)) > 0.5 for m in result.masks.data.cpu().numpy()
                ])
            annotated = draw_detections(img, xyxy, cls_ids, confs, CLASS_NAMES, masks)
            samples.append((annotated, caption_fn(entry)))
        plot_sample_grid(samples, FIGURES_DIR / f"error_analysis_{name}.png", title=name)
        print(f"  saved error_analysis_{name}.png ({len(samples)} examples)")

    print("Error analysis:")
    save_grid(low_conf, "low_confidence", lambda e: f"{e[0].name}\nmin_conf={e[2]:.2f}", args.n_error_examples)
    save_grid(small_objs, "small_objects", lambda e: e[0].name, args.n_error_examples)
    save_grid(no_detections, "false_negatives_no_detection", lambda e: e[0].name, args.n_error_examples)
    save_grid(many_detections, "cluttered_scenes", lambda e: e[0].name, args.n_error_examples)


def sample_predictions(model, args, n: int = 8) -> None:
    img_dir = PROCESSED_DIR / args.split / "images"
    paths = sorted(img_dir.glob("*.jpg"))[:n]
    samples = []
    for path in paths:
        result = model.predict(str(path), imgsz=args.imgsz, conf=args.conf,
                                device=get_device(args.device), verbose=False)[0]
        img = cv2.imread(str(path))
        stats = quantify(result)
        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            xyxy = result.boxes.xyxy.cpu().numpy()
            masks = None
            if result.masks is not None:
                mh, mw = img.shape[:2]
                masks = np.stack([cv2.resize(m, (mw, mh)) > 0.5 for m in result.masks.data.cpu().numpy()])
            img = draw_detections(img, xyxy, cls_ids, confs, CLASS_NAMES, masks)
        caption = f"{path.name}\nplastic={stats['plastic_count']} cov={stats['plastic_coverage_percentage']:.1f}%"
        samples.append((img, caption))
    plot_sample_grid(samples, FIGURES_DIR / "sample_predictions.png", title="Sample predictions")
    print("  saved sample_predictions.png")


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)

    print(f"Running official Ultralytics validation on '{args.split}' split...")
    summary = run_official_metrics(model, args)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / f"eval_metrics_{args.split}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    print("\nGenerating sample prediction visualizations...")
    sample_predictions(model, args)

    print("\nRunning qualitative error analysis...")
    error_analysis(model, args)

    print(f"\nAll figures saved to {FIGURES_DIR}")
    print(f"Metrics saved to {METRICS_DIR / f'eval_metrics_{args.split}.json'}")


if __name__ == "__main__":
    main()
