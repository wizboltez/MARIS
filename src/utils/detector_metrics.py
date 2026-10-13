"""Read the plastic detector's held-out test metrics so UI/report text never hard-codes them."""
from __future__ import annotations

import json

from src.config import METRICS_DIR

EVAL_PATH = METRICS_DIR / "eval_metrics_test.json"


def load_plastic_metrics() -> dict | None:
    """trash_plastic test metrics, or None if missing / still the old leaky-split file."""
    if not EVAL_PATH.exists():
        return None
    d = json.loads(EVAL_PATH.read_text())
    if "inflated_metrics_do_not_use" in d or not d.get("leak_free"):
        return None
    keys = ("plastic_precision", "plastic_recall", "plastic_map50", "plastic_map50_95")
    if not all(k in d for k in keys):
        return None
    out = {k.replace("plastic_", ""): round(float(d[k]), 3) for k in keys}
    out["note"] = d.get("note", "")
    return out


BOOTSTRAP_PATH = METRICS_DIR / "test_bootstrap_confidence_intervals.json"


def load_operating_point() -> dict | None:
    """trash_plastic at the pipeline's conf 0.18 on TEST, with 95% CI from a 53-video cluster bootstrap.

    Values are strings like "0.452 [0.315, 0.594]".
    """
    if not BOOTSTRAP_PATH.exists():
        return None
    d = json.loads(BOOTSTRAP_PATH.read_text())
    op = d.get("metrics", {}).get("plastic_018")
    if not op:
        return None
    return {**op, "n_videos": d.get("n_videos"), "n_boot": d.get("n_boot")}


def plastic_summary(short: bool = False) -> str:
    op = load_operating_point()
    if op is not None:
        if short:
            return f"plastic recall {op['recall'].split()[0]}"
        return (f"trash_plastic on unseen test videos at conf 0.18: precision {op['precision']}, "
                f"recall {op['recall']} (95% CI over {op['n_videos']} videos)")
    m = load_plastic_metrics()
    if m is None:
        return "plastic test metrics unavailable"
    if short:
        return f"plastic recall {m['recall']:.2f}"
    return (f"trash_plastic test precision {m['precision']:.2f}, recall {m['recall']:.2f}, "
            f"mAP50 {m['map50']:.2f} (unseen videos)")
