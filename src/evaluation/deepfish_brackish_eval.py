"""Brackish YOLOv8s fish detector (dronefreak/brackish-yolov8s, pretrained, no fine-tuning) on DeepFish.

Test data: the DeepFish test set with fish bounding boxes released by the YOLO-Fish authors
(github.com/tamim662/YOLO-Fish; 909 frames, YOLO format, single class 'fish'). DeepFish (Saleh et al. 2020,
James Cook University) is tropical Australian fish-habitat video; the model was trained only on Brackish
(Danish harbour), so this is a test on unseen water.

Protocol (same rules as the project's other fish evaluations):
- frames grouped by video clip (file-name prefix); 30% of clips = val, used only to pick the confidence
  threshold (max F1); the other 70% = test, reported once
- greedy matching, highest confidence first, IoU >= 0.5; AP50 on test; 95% bootstrap CIs over test clips
- the model's 'fish' and 'small_fish' classes both count as fish

Run:  python -m src.evaluation.deepfish_brackish_eval
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.config import METRICS_DIR, PROJECT_ROOT

DATA = PROJECT_ROOT / "data" / "external" / "deepfish_yolofish" / "extracted" / "test"
WEIGHTS = PROJECT_ROOT / "models" / "fish_brackish" / "best.pt"
FISH_CLASSES = {"fish", "small_fish"}


def clip(n: str) -> str:
    return re.sub(r"_f\d+$", "", n)


def split(n: str) -> str:
    return "val" if int(hashlib.md5(clip(n).encode()).hexdigest(), 16) % 10 < 3 else "test"


def load_gt(img: Path) -> np.ndarray:
    w, h = Image.open(img).size
    rows = [list(map(float, l.split()[1:5])) for l in img.with_suffix(".txt").read_text().splitlines() if l.strip()]
    b = np.array(rows, np.float32).reshape(-1, 4)
    return np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                     (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1)


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match_flags(gt, boxes, confs):
    order = np.argsort(-confs)
    ious = iou(boxes, gt) if len(boxes) and len(gt) else np.zeros((len(boxes), len(gt)))
    used, out = set(), []
    for i in order:
        cand = [(ious[i, g], g) for g in range(len(gt)) if g not in used and ious[i, g] >= 0.5]
        if cand:
            used.add(max(cand)[1])
        out.append((float(confs[i]), bool(cand)))
    return out


def counts(gt, boxes, confs, conf):
    k = confs >= conf
    f = match_flags(gt, boxes[k], confs[k])
    tp = sum(x for _, x in f)
    return tp, len(f) - tp, len(gt) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def ap50(gt: dict, preds: dict, ns: list[str]) -> float:
    flags = sorted((f for n in ns for f in match_flags(gt[n], *preds[n])), key=lambda t: -t[0])
    n_gt = sum(len(gt[n]) for n in ns)
    tps = np.cumsum([f for _, f in flags]); fps = np.cumsum([not f for _, f in flags])
    mrec = np.concatenate([[0], tps / n_gt, [1]])
    mpre = np.concatenate([[1], tps / np.maximum(tps + fps, 1), [0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def evaluate(gt: dict, preds: dict, splits: dict) -> dict:
    tot = lambda ns, c: tuple(sum(v) for v in zip(*(counts(gt[n], *preds[n], c) for n in ns)))
    table = [(prf(*tot(splits["val"], c))[2], float(c)) for c in np.round(np.arange(0.05, 0.91, 0.05), 2)]
    vf1, conf = max(table)
    ns = splits["test"]
    per = {n: counts(gt[n], *preds[n], conf) for n in ns}
    tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
    p, r, f1 = prf(tp, fp, fn)
    groups = collections.defaultdict(list)
    for n in ns:
        groups[clip(n)].append(n)
    rng = np.random.default_rng(0)
    keys = list(groups)
    boot = []
    for _ in range(1000):
        s = [n for k in rng.choice(keys, len(keys)) for n in groups[k]]
        boot.append(prf(*(sum(per[n][k] for n in s) for k in range(3))))
    ci = lambda j: [round(float(np.percentile([b[j] for b in boot], q)), 3) for q in (2.5, 97.5)]
    # image-level: does the model say "fish present" when there is at least one fish?
    has_fish = {n: len(gt[n]) > 0 for n in ns}
    said = {n: bool((preds[n][1] >= conf).any()) for n in ns}
    img_acc = float(np.mean([has_fish[n] == said[n] for n in ns]))
    return {"conf_chosen_on_val": conf, "val_f1": round(vf1, 3), "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(f1, 3), "ap50": round(ap50(gt, preds, ns), 3), "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision_ci": ci(0), "recall_ci": ci(1), "f1_ci": ci(2),
            "image_level_fish_present_accuracy": round(img_acc, 3),
            "images_with_fish": int(sum(has_fish.values())), "images_total": len(ns)}


def main() -> None:
    imgs = sorted(DATA.glob("*.jpg"))
    gt = {p.stem: load_gt(p) for p in imgs}
    splits = {s: sorted(n for n in gt if split(n) == s) for s in ("val", "test")}
    for s, ns in splits.items():
        print(s, len(ns), "frames", len({clip(n) for n in ns}), "clips", sum(len(gt[n]) for n in ns), "fish")

    model = YOLO(str(WEIGHTS))
    fish_ids = [i for i, c in model.names.items() if c in FISH_CLASSES]
    preds = {}
    for i in range(0, len(imgs), 16):
        batch = imgs[i:i + 16]
        for p, r in zip(batch, model.predict([str(x) for x in batch], conf=0.02, imgsz=640, verbose=False)):
            k = np.isin(r.boxes.cls.cpu().numpy().astype(int), fish_ids)
            preds[p.stem] = (r.boxes.xyxy.cpu().numpy()[k].astype(np.float32), r.boxes.conf.cpu().numpy()[k].astype(np.float32))
    print("predicted", len(preds), "frames")

    res = evaluate(gt, preds, splits)
    out = {
        "source": "src/evaluation/deepfish_brackish_eval.py, run locally 2026-10-15",
        "model": "dronefreak/brackish-yolov8s (Hugging Face), pretrained on Brackish only, no fine-tuning; "
                 "classes fish + small_fish counted as fish",
        "model_reported_score": "Brackish test mAP50 0.993 (model card, its own dataset)",
        "dataset": "DeepFish test set with boxes from the YOLO-Fish authors (909 frames); split by video clip "
                   "30% val (threshold only) / 70% test",
        "test_frames": len(splits["test"]), "test_clips": len({clip(n) for n in splits["test"]}),
        "test_fish": int(sum(len(gt[n]) for n in splits["test"])), "matching": "greedy IoU>=0.5",
        "ap50_all_909_frames": round(ap50(gt, preds, splits["val"] + splits["test"]), 3),
        "test": res,
    }
    path = METRICS_DIR / "deepfish_brackish_yolov8s_test.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
