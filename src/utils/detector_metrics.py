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


def plastic_summary(short: bool = False) -> str:
    m = load_plastic_metrics()
    if m is None:
        return "plastic test metrics unavailable"
    if short:
        return f"plastic recall {m['recall']:.2f}"
    return (f"trash_plastic test precision {m['precision']:.2f}, recall {m['recall']:.2f}, "
            f"mAP50 {m['map50']:.2f} (unseen videos)")
