"""Plain matplotlib (notebook-default style) versions of the accuracy-vs-baseline charts.

  reports/assets/coral_vs_baseline_nb.png
  reports/assets/image_level_vs_baseline_nb.png

Run: python reports/make_result_charts_nb.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "assets"


def plot(labels, model, baseline, title, path):
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    b1 = ax.bar(x - w / 2, model, w, label="Model")
    b2 = ax.bar(x + w / 2, baseline, w, label="Majority baseline")
    ax.bar_label(b1, fmt="%.1f")
    ax.bar_label(b2, fmt="%.1f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


coral = json.loads((ROOT / "models/coral_segmenter/coralscapes_segformer_v2_test_metrics.json").read_text())
keys = ["image_level_bleaching_present", "image_level_coral_present"]
plot(
    ["Bleaching present", "Coral present"],
    [coral[k]["accuracy"] * 100 for k in keys],
    [coral[k]["always_majority_baseline"] * 100 for k in keys],
    f"Coral model - image-level accuracy (Coralscapes test, n={coral['n_test_images']})",
    OUT / "coral_vs_baseline_nb.png",
)

det = json.loads((ROOT / "outputs/metrics/image_level_detection_test.json").read_text())
keys = ["rov", "any_debris", "plastic"]
plot(
    ["ROV present", "Any debris present", "Plastic present"],
    [det[k]["at_conf_0.18"]["accuracy"] * 100 for k in keys],
    [det[k]["majority_baseline"] * 100 for k in keys],
    f"Detector - image-level accuracy (TrashCan test, n={det['plastic']['n_images']}, conf=0.18)",
    OUT / "image_level_vs_baseline_nb.png",
)
