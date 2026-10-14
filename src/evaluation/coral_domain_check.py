"""Deep-sea domain check for a SUIM reef segmenter.

Runs the segmenter on every TrashCan test frame (JAMSTEC deep-sea ROV footage: no coral reef), so any
predicted reef is a false positive. Reports how many frames the segmenter alone marks as reef, and how many
the dashboard's auto mode (segmenter AND colour/texture coral check) would declare as coral.

Usage:  python -m src.evaluation.coral_domain_check [model key] [--sample=N] [--dry-run]
--sample=N checks a fixed random sample of N frames (seed 0), for slow CPU models such as the Coralscapes SegFormer.
Writes models/coral_segmenter/<domain_check file of that model> (see CANDIDATES in coral_segmentation.py).
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from src.config import PROCESSED_DIR
from src.modules.coral_health import CoralHealthAnalyzer
from src.modules.coral_segmentation import CANDIDATES, SEGMENTER_DIR, load_reef_segmenter


def main(key: str | None = None, sample: int | None = None) -> tuple[dict, object]:
    segmenter = load_reef_segmenter(key)
    if segmenter is None:
        raise SystemExit(f"No usable reef segmenter for '{key}' (missing files, failed gate or smp not installed).")
    analyzer = CoralHealthAnalyzer(use_segmenter=False)
    analyzer.segmenter = segmenter

    frames = sorted((PROCESSED_DIR / "test" / "images").glob("*.jpg"))
    n_all = len(frames)
    if sample and sample < len(frames):
        frames = sorted(np.random.default_rng(0).choice(frames, size=sample, replace=False))
    fractions, declared = [], 0
    for i, path in enumerate(frames, 1):
        image = cv2.imread(str(path))
        fractions.append(segmenter.predict(image)["reef_fraction"])
        declared += bool(analyzer.analyze_image(image, habitat="auto")["coral_present"])
        if i % 200 == 0:
            print(f"{i}/{len(frames)}", flush=True)
    fractions = np.array(fractions)

    result = {
        "what": f"{segmenter.dataset} {segmenter.name} reef/coral predictions on {len(frames)} deep-sea TrashCan test frames "
                + (f"(random sample, seed 0, of {n_all}) " if len(frames) < n_all else "") + "(no coral reef in this footage)",
        "model": segmenter.name,
        "n_frames": len(frames),
        "frames_reef_ge_2pct_unet_only": int((fractions >= 0.02).sum()),  # key name kept for the report; = segmenter only
        "frames_reef_ge_5pct_unet_only": int((fractions >= 0.05).sum()),
        "frames_reef_ge_20pct_unet_only": int((fractions >= 0.20).sum()),
        "median_reef_fraction": round(float(np.median(fractions)), 4),
        "frames_coral_declared_auto_mode_unet_AND_colour_check": declared,
    }
    out = SEGMENTER_DIR / CANDIDATES[segmenter.key]["domain_check"]
    print(json.dumps(result, indent=2))
    return result, out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sample = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--sample=")), None)
    result, out = main(args[0] if args else None, sample)
    if "--dry-run" not in sys.argv:
        out.write_text(json.dumps(result, indent=2))
        print("saved", out)
