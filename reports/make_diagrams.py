"""Draws the workflow and architecture diagrams used in the FA report (reports/assets/*.png)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#33415c", lw=1.2))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color="#0b2545")
    ax.text(x + w / 2, y + h - 0.5 - 0.27 * title.count("\n"), body, ha="center", va="top", fontsize=7.6, color="#1b263b", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.3, color="#33415c"))


def workflow():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 3.35), 11.8, 2.9, boxstyle="round,pad=0.02", fc="#f4f8fb", ec="#8da9c4", lw=1, ls="--"))
    ax.text(0.3, 6.05, "Model development", fontsize=10.5, fontweight="bold", color="#13315c")
    ax.add_patch(FancyBboxPatch((0.1, 0.15), 11.8, 2.95, boxstyle="round,pad=0.02", fc="#f6fbf4", ec="#9cc5a1", lw=1, ls="--"))
    ax.text(0.3, 2.85, "Application after model integration (Streamlit dashboard, uses the trained models)", fontsize=10.5, fontweight="bold", color="#1e5128")
    w, h, y = 1.75, 2.15, 3.6
    top = [
        ("Problem\ndefinition", "Plastic, fish, coral\nand water quality\nin one estimate", "#fde2e4"),
        ("Data\ncollection", "TrashCan 1.0\nCoralscapes, SUIM\nNOAA CRW, NCEI,\nOBIS", "#e2ece9"),
        ("Pre-processing", "COCO -> YOLO\nsplit by video /\ndive site\nclass grouping", "#fff1e6"),
        ("Model\ntraining", "YOLO11s-seg\nSegFormer MiT-B2\nU-Net, XGBoost", "#dfe7fd"),
        ("Evaluation", "unseen videos\nand dive sites\nP, R, F1, mAP,\nIoU, bootstrap CI", "#e8e8e4"),
    ]
    xs = [0.35 + i * 2.3 for i in range(5)]
    for x, (t, b, c) in zip(xs, top):
        box(ax, x, y, w, h, t, b, c)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w, y + h / 2, b, y + h / 2)
    bottom = [
        ("Input image /\nvideo frame", "survey frame or\nuploaded photo", "#dfe7fd"),
        ("Plastic + fish\ndetection", "YOLO11s-seg masks\nByteTrack tracks", "#fde2e4"),
        ("Coral health", "live / bleached /\ndead coral\n(SegFormer)", "#e2ece9"),
        ("Water quality", "NOAA CRW temp.\npH, DO, salinity,\nturbidity -> WQI", "#fff1e6"),
        ("Impact score +\nexplanation", "XGBoost 0-100\nSHAP, Grad-CAM\nrecommendations", "#e8e8e4"),
    ]
    yb = 0.4
    for x, (t, b, c) in zip(xs, bottom):
        box(ax, x, yb, w, h, t, b, c)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w, yb + h / 2, b, yb + h / 2)
    fig.savefig(OUT / "workflow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.2, 1.3, 1.7, 2.0, "Input", "RGB underwater\nimage\n640 x 640", "#dfe7fd")
    box(ax, 2.4, 1.3, 2.1, 2.0, "Backbone", "Conv + C3k2 blocks\nSPPF, C2PSA\n(attention)\nmulti-scale features", "#e2ece9")
    box(ax, 5.0, 1.3, 2.1, 2.0, "Neck", "upsample + concat\n(FPN / PAN style)\nfuses P3, P4, P5", "#fff1e6")
    box(ax, 7.6, 2.45, 2.1, 1.75, "Detect head", "boxes + class\nscores (16 classes)", "#fde2e4")
    box(ax, 7.6, 0.4, 2.1, 1.75, "Segment head", "32 mask prototypes\n+ coefficients", "#fde2e4")
    box(ax, 10.1, 1.3, 1.75, 2.0, "Output", "NMS (IoU 0.7)\nconf >= 0.18\nmasks -> plastic\ncover %", "#e8e8e4")
    arrow(ax, 1.9, 2.3, 2.4, 2.3); arrow(ax, 4.5, 2.3, 5.0, 2.3)
    arrow(ax, 7.1, 2.6, 7.6, 3.2); arrow(ax, 7.1, 2.0, 7.6, 1.3)
    arrow(ax, 9.7, 3.2, 10.1, 2.6); arrow(ax, 9.7, 1.3, 10.1, 2.0)
    fig.savefig(OUT / "architecture_yolo.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    workflow()
    architecture()
    print("diagrams written to", OUT)
