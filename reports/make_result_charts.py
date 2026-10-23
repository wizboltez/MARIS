"""Report charts: model accuracy vs. majority baseline, read from the saved test metrics.

  reports/assets/coral_vs_baseline.png        (Coralscapes test, 392 images, 5 unseen sites)
  reports/assets/image_level_vs_baseline.png  (TrashCan test, 1,313 images, 53 unseen videos)

Run: python reports/make_result_charts.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "assets"

MODEL = "#2a78d6"
BASELINE = "#c9c7c1"
TEXT = "#0b0b0b"
MUTED = "#52514e"


def grouped_bars(rows, title, subtitle, path):
    """rows: list of (label, model_acc, baseline_acc) as fractions."""
    fig, ax = plt.subplots(figsize=(7.2, max(4.45, 0.95 * len(rows) + 1.6)), dpi=200)
    h = 0.36
    for i, (label, acc, base) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y + h / 2 + 0.01, acc * 100, height=h, color=MODEL, label="Our model" if i == 0 else None)
        ax.barh(y - h / 2 - 0.01, base * 100, height=h, color=BASELINE, label="Always-guess baseline" if i == 0 else None)
        ax.text(acc * 100 + 1, y + h / 2 + 0.01, f"{acc * 100:.1f}%", va="center", fontsize=9, color=TEXT, fontweight="bold")
        ax.text(base * 100 + 1, y - h / 2 - 0.01, f"{base * 100:.1f}%", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=10, color=TEXT)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Image-level accuracy (%)", fontsize=9, color=MUTED)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bdbbb5")
    ax.set_title(f"{title}\n\n", fontsize=11.5, color=TEXT, loc="left", fontweight="bold")
    ax.text(0, 1.09, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, frameon=False, fontsize=8.5,
              handlelength=1.2, columnspacing=1.5)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


coral = json.loads((ROOT / "models/coral_segmenter/coralscapes_segformer_v2_test_metrics.json").read_text())
grouped_bars(
    [
        ("Bleaching present?", coral["image_level_bleaching_present"]["accuracy"],
         coral["image_level_bleaching_present"]["always_majority_baseline"]),
        ("Coral present?", coral["image_level_coral_present"]["accuracy"],
         coral["image_level_coral_present"]["always_majority_baseline"]),
    ],
    "Coral model vs. always-guess baseline",
    f"Coralscapes test split: {coral['n_test_images']} images from 5 dive sites unseen in training",
    OUT / "coral_vs_baseline.png",
)

det = json.loads((ROOT / "outputs/metrics/image_level_detection_test.json").read_text())
grouped_bars(
    [
        ("Underwater robot (ROV) present?", det["rov"]["at_conf_0.18"]["accuracy"], det["rov"]["majority_baseline"]),
        ("Any debris present?", det["any_debris"]["at_conf_0.18"]["accuracy"], det["any_debris"]["majority_baseline"]),
        ("Plastic present?", det["plastic"]["at_conf_0.18"]["accuracy"], det["plastic"]["majority_baseline"]),
    ],
    "Debris detector vs. always-guess baseline",
    f"TrashCan test split: {det['plastic']['n_images']:,} images from 53 unseen videos, confidence 0.18",
    OUT / "image_level_vs_baseline.png",
)
