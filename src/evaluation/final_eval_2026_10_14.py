"""Final evaluation package (2026-10-14): fish threshold table, per-class analysis, per-video ranking,
demo frames. All outputs go to outputs/experiments/2026-10-14/; baseline files are never modified.

Selection (thresholds, best/worst videos) uses VAL only. TEST numbers are read from the frozen
evaluations already on disk or computed once at frozen settings. Best-case examples are labelled as such.

Run:  python -m src.evaluation.final_eval_2026_10_14   (CPU; caches predictions in the output folder)
"""
from __future__ import annotations

import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.config import BEST_MODEL_PATH, CLASS_NAMES, METRICS_DIR, PROCESSED_DIR, PROJECT_ROOT
from src.evaluation.tune_thresholds import (box_iou_xyxy, load_ground_truths_with_sizes, run_split_inference,
                                            sweep_metrics_for_class)

OUT = PROJECT_ROOT / "outputs" / "experiments" / "2026-10-14"
FRAMES = OUT / "demo_frames"
FISH_MODEL = PROJECT_ROOT / "models" / "fish_detector" / "best_fish.pt"
FISH_ID = CLASS_NAMES.index("animal_fish")
PLASTIC_ID = CLASS_NAMES.index("trash_plastic")
CONF = 0.18
WEAK = ["animal_fish", "trash_plastic", "trash_metal", "trash_paper", "trash_fishing_gear"]


def cached_preds(tag: str, weights: Path, split: str, remap_to: int | None = None) -> dict:
    path = OUT / f"preds_{tag}_{split}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    preds = run_split_inference(YOLO(str(weights)), PROCESSED_DIR / split, conf_floor=0.02, device="cpu")
    if remap_to is not None:  # single-class fish model: class 0 -> TrashCan fish id
        preds = {k: [dict(p, cls=remap_to) for p in v] for k, v in preds.items()}
    path.write_bytes(pickle.dumps(preds))
    return preds


def match_image(gt_boxes: np.ndarray, preds: list[dict]) -> tuple[list[int], list[int], int]:
    """Greedy IoU>=0.5 matching (same rule as all project evaluations). Returns (tp_idx, fp_idx, fn)."""
    order = np.argsort([-p["conf"] for p in preds])
    if len(gt_boxes) == 0:
        return [], [int(i) for i in order], 0
    ious = box_iou_xyxy(np.array([p["box"] for p in preds], np.float32), gt_boxes) if preds else np.zeros((0, len(gt_boxes)))
    used, tp, fp = set(), [], []
    for i in order:
        best, best_iou = -1, 0.5
        for g in range(len(gt_boxes)):
            if g not in used and ious[i, g] >= best_iou:
                best, best_iou = g, ious[i, g]
        (tp if best >= 0 else fp).append(int(i))
        if best >= 0:
            used.add(best)
    return tp, fp, len(gt_boxes) - len(used)


def counts(gt: dict, preds: dict, cls: int, conf: float):
    per_img = {}
    for name, d in gt.items():
        g = np.array([x["box"] for x in d["gts"] if x["cls"] == cls], np.float32).reshape(-1, 4)
        p = [x for x in preds.get(name, []) if x["cls"] == cls and x["conf"] >= conf]
        tp, fp, fn = match_image(g, p)
        per_img[name] = (len(tp), len(fp), fn, len(g))
    return per_img


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def video_of(name: str) -> str:
    m = re.match(r"vid_(\d+)_frame", name)
    return m.group(1) if m else name


def train_instance_counts() -> Counter:
    c = Counter()
    for f in (PROCESSED_DIR / "train" / "labels").glob("*.txt"):
        for line in f.read_text().splitlines():
            if line.strip():
                c[int(line.split()[0])] += 1
    return c


def draw(name: str, split: str, gt: dict, preds: dict, cls: int, conf: float, title: str) -> Path:
    img = cv2.imread(str(PROCESSED_DIR / split / "images" / name))
    d = gt[name]
    g = np.array([x["box"] for x in d["gts"] if x["cls"] == cls], np.float32).reshape(-1, 4)
    p = [x for x in preds.get(name, []) if x["cls"] == cls and x["conf"] >= conf]
    tp, fp, _ = match_image(g, p)
    for b in g:
        cv2.rectangle(img, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:]), (0, 200, 0), 2)
    for i in tp + fp:
        b = p[i]["box"].astype(int)
        cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), (255, 140, 0) if i in tp else (0, 0, 255), 1)
        cv2.putText(img, f"{p[i]['conf']:.2f}", (int(b[0]), max(10, int(b[1]) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1)
    cv2.putText(img, title, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    out = FRAMES / f"{title.split()[0]}_{name}"
    cv2.imwrite(str(out), img)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(exist_ok=True)
    summary = {}

    print("Loading ground truth...", flush=True)
    gt = {s: load_ground_truths_with_sizes(PROCESSED_DIR / s) for s in ("val", "test")}
    print("Predictions (cached after first run)...", flush=True)
    main_val = cached_preds("main", BEST_MODEL_PATH, "val")
    main_test = cached_preds("main", BEST_MODEL_PATH, "test")
    fish_val = cached_preds("fish", FISH_MODEL, "val", remap_to=FISH_ID)

    # ---- 1. fish threshold comparison on VAL (identical samples + criteria for both detectors)
    ths = np.round(np.arange(0.05, 0.71, 0.05), 2)
    rows = []
    for tag, preds in (("old_16_class", main_val), ("new_fish_only", fish_val)):
        p, r, f = sweep_metrics_for_class(gt["val"], preds, FISH_ID, ths)
        rows += [{"detector": tag, "conf": t, "precision": round(a, 3), "recall": round(b, 3), "f1": round(c, 3)}
                 for t, a, b, c in zip(ths, p, r, f)]
    fish_tab = pd.DataFrame(rows)
    fish_tab.to_csv(OUT / "fish_threshold_comparison_val.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for tag, g in fish_tab.groupby("detector"):
        ax.plot(g["recall"], g["precision"], marker="o", label=tag)
    ax.set_xlabel("recall (VAL)"); ax.set_ylabel("precision (VAL)"); ax.set_title("Fish: precision-recall on TrashCan val")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fish_pr_val.png", dpi=140); plt.close(fig)
    fc = json.loads((METRICS_DIR / "fish_detector_comparison.json").read_text())
    summary["fish_test_frozen"] = {
        "old_16_class_at_0.18": {k: fc["old_16_class_model"][k] for k in ("precision", "recall", "f1", "precision_ci", "recall_ci")},
        "new_fish_at_val_chosen_0.35": {k: fc["new_fish_detector_val_tuned"][k] for k in ("precision", "recall", "f1", "precision_ci", "recall_ci")},
        "decision": "keep baseline in pipeline: test F1 0.231 vs 0.229, overlapping 95% CIs (no significant gain)",
    }

    # ---- 2. per-class metrics + counts + error type (VAL and TEST, conf 0.18)
    train_c = train_instance_counts()
    cls_rows = []
    for split, preds in (("val", main_val), ("test", main_test)):
        for cid, cname in enumerate(CLASS_NAMES):
            c = counts(gt[split], preds, cid, CONF)
            tp, fp, fn = (sum(v[i] for v in c.values()) for i in range(3))
            p, r, f = prf(tp, fp, fn)
            cls_rows.append({"split": split, "class": cname, "train_instances": train_c[cid], f"n_{split}": tp + fn,
                             "tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})
    cls_tab = pd.DataFrame(cls_rows)
    cls_tab.to_csv(OUT / "per_class_metrics_conf0.18.csv", index=False)
    weak = cls_tab[(cls_tab["split"] == "test") & cls_tab["class"].isin(WEAK)].copy()
    weak["main_error"] = np.where(weak["fn"] > 2 * weak["fp"], "mostly missed objects",
                                  np.where(weak["fp"] > 2 * weak["fn"], "mostly false positives", "both"))
    weak.to_csv(OUT / "weak_class_diagnosis_test.csv", index=False)

    # ---- 3. per-video ranking on VAL (selection only), plastic (main model) + fish (main model)
    rank_rows = []
    for task, cid in (("plastic", PLASTIC_ID), ("fish", FISH_ID)):
        c = counts(gt["val"], main_val, cid, CONF)
        by_vid = defaultdict(lambda: [0, 0, 0, 0, 0])
        for name, (tp, fp, fn, n) in c.items():
            v = by_vid[video_of(name)]
            v[0] += tp; v[1] += fp; v[2] += fn; v[3] += n; v[4] += 1
        for vid, (tp, fp, fn, n, frames) in by_vid.items():
            if n >= 3:  # need enough annotated objects for a per-video score
                p, r, f = prf(tp, fp, fn)
                rank_rows.append({"task": task, "video": f"vid_{vid}", "frames": frames, "objects": n, "tp": tp, "fp": fp,
                                  "fn": fn, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})
    rank = pd.DataFrame(rank_rows).sort_values(["task", "f1", "objects"], ascending=[True, False, False])
    rank.to_csv(OUT / "val_video_ranking.csv", index=False)

    # ---- 4. demo frames: best 3 videos per task (frame with most TPs) + 1 failure (most missed) per task
    demo = []
    for task, cid in (("plastic", PLASTIC_ID), ("fish", FISH_ID)):
        c = counts(gt["val"], main_val, cid, CONF)
        t = rank[rank["task"] == task]
        for _, row in t.head(3).iterrows():
            vid = row["video"].replace("vid_", "")
            frames = [n for n in c if video_of(n) == vid and c[n][0] > 0]

            def visibility(n, cid=cid):
                # largest correctly detected box, ignoring the top/bottom 12% (camera text overlay zones)
                d, h = gt["val"][n], gt["val"][n]["h"]
                g = np.array([x["box"] for x in d["gts"] if x["cls"] == cid], np.float32).reshape(-1, 4)
                p = [x for x in main_val.get(n, []) if x["cls"] == cid and x["conf"] >= CONF]
                tp, fp, _ = match_image(g, p)
                areas = [(p[i]["box"][2] - p[i]["box"][0]) * (p[i]["box"][3] - p[i]["box"][1]) for i in tp
                         if 0.12 * h < (p[i]["box"][1] + p[i]["box"][3]) / 2 < 0.88 * h]
                return (max(areas) if areas else 0.0, -len(fp))

            best = max(frames, key=visibility)
            demo.append({"kind": "best-case", "task": task, "video": row["video"], "frame": best, "video_f1": row["f1"],
                         "file": draw(best, "val", gt["val"], main_val, cid, CONF, f"BEST-{task} {row['video']} F1={row['f1']}").name})
        worst_frame = max(c, key=lambda n: (c[n][2], c[n][1]))
        demo.append({"kind": "failure", "task": task, "video": f"vid_{video_of(worst_frame)}", "frame": worst_frame,
                     "missed": c[worst_frame][2], "false_pos": c[worst_frame][1],
                     "file": draw(worst_frame, "val", gt["val"], main_val, cid, CONF, f"FAIL-{task} missed={c[worst_frame][2]}").name})
    (OUT / "demo_frames_index.json").write_text(json.dumps(demo, indent=2))

    # ---- 5. summary
    op = json.loads((METRICS_DIR / "test_bootstrap_confidence_intervals.json").read_text())["metrics"]
    summary.update({
        "baseline_record": "baseline_record.txt",
        "plastic_test_conf0.18": op["plastic_018"],
        "per_class_table": "per_class_metrics_conf0.18.csv",
        "weak_class_diagnosis": weak[["class", "train_instances", "n_test", "tp", "fp", "fn", "precision", "recall", "main_error"]].to_dict("records"),
        "demo_frames": "demo_frames/ (green = ground truth, orange = correct detection, red = false alarm); "
                       "best-case frames are selected from the best VAL videos and are NOT representative of the dataset",
    })
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
