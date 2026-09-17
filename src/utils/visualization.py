"""Plotting helpers shared by EDA, evaluation and inference.

All functions save a PNG to `save_path` and also return the matplotlib
Figure, so notebooks can display them inline while scripts just save them.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Fixed, deterministic color per class id so figures are visually consistent
# across the whole project (EDA plots, ground-truth overlays, predictions).
_CMAP = plt.get_cmap("tab20")


def class_color(class_id: int) -> tuple[int, int, int]:
    r, g, b, _ = _CMAP(class_id % 20)
    return int(r * 255), int(g * 255), int(b * 255)


def plot_class_distribution(class_counts: dict, save_path: Path, title: str = "Class distribution") -> plt.Figure:
    names = list(class_counts.keys())
    counts = list(class_counts.values())
    order = np.argsort(counts)[::-1]
    names = [names[i] for i in order]
    counts = [counts[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(names)), counts, color="#2b7a78")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_ylabel("Annotation count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_image_size_distribution(size_counts: dict, save_path: Path) -> plt.Figure:
    labels = list(size_counts.keys())
    counts = list(size_counts.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts, color="#3aafa9")
    ax.set_xlabel("Image size (WxH)")
    ax.set_ylabel("Count")
    ax.set_title("Image size distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_objects_per_image(counter: dict, save_path: Path) -> plt.Figure:
    xs = sorted(counter.keys())
    ys = [counter[x] for x in xs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xs, ys, color="#17252a")
    ax.set_xlabel("Objects per image")
    ax.set_ylabel("Number of images")
    ax.set_title("Objects per image")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_training_curves(results_csv: Path, save_path: Path) -> plt.Figure:
    """Plot train/val loss and key metrics from an Ultralytics results.csv."""
    import pandas as pd

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    loss_cols = [c for c in df.columns if c.endswith("loss")]
    metric_cols = [c for c in df.columns if c.startswith("metrics/")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for c in loss_cols:
        axes[0].plot(df["epoch"], df[c], label=c)
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Loss curves")
    axes[0].legend(fontsize=7)

    for c in metric_cols:
        axes[1].plot(df["epoch"], df[c], label=c.replace("metrics/", ""))
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Validation metrics")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def draw_detections(image: np.ndarray, boxes: np.ndarray, class_ids: np.ndarray,
                     confidences: np.ndarray, class_names: list[str],
                     masks: np.ndarray | None = None, alpha: float = 0.4) -> np.ndarray:
    """Draw boxes (+ optional segmentation masks) with class/confidence labels.

    boxes: (N, 4) xyxy in pixel coords.
    masks: (N, H, W) boolean, same size as `image`, or None.
    """
    out = image.copy()
    overlay = image.copy()

    for i in range(len(boxes)):
        cls_id = int(class_ids[i])
        color = class_color(cls_id)
        if masks is not None:
            overlay[masks[i]] = color
    if masks is not None:
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    for i in range(len(boxes)):
        cls_id = int(class_ids[i])
        color = class_color(cls_id)
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls_id]} {confidences[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def plot_sample_grid(samples: list[tuple[np.ndarray, str]], save_path: Path,
                      n_cols: int = 4, title: str = "") -> plt.Figure:
    """samples: list of (image_bgr, caption)."""
    n = len(samples)
    n_cols = min(n_cols, n) or 1
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)
    for ax, (img, caption) in zip(axes, samples):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(caption, fontsize=9)
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig
