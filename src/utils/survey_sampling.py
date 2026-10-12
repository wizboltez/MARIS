"""Pick a reproducible, plastic-balanced sample of test images.

Taking the first N test images by filename only covers a few source videos and,
on our split, none of the first 40 contain any ground-truth plastic. That made
every downstream "impact" number come from plastic-free scenes. This helper
mixes images that have ground-truth trash_plastic with ones that don't.
"""
from __future__ import annotations

import random
from pathlib import Path

from src.config import CLASS_NAMES, PLASTIC_CLASSES, PROCESSED_DIR

PLASTIC_ID = CLASS_NAMES.index(PLASTIC_CLASSES[0])


def has_plastic_label(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    with open(label_path) as f:
        return any(line.split()[:1] == [str(PLASTIC_ID)] for line in f)


def select_survey_images(
    n: int = 40,
    plastic_fraction: float = 0.5,
    split: str = "test",
    seed: int = 42,
) -> list[Path]:
    """Return n image paths, ~plastic_fraction of which contain labelled plastic."""
    img_dir = PROCESSED_DIR / split / "images"
    lbl_dir = PROCESSED_DIR / split / "labels"
    images = sorted(img_dir.glob("*.jpg"))
    with_plastic = [p for p in images if has_plastic_label(lbl_dir / f"{p.stem}.txt")]
    plastic_set = set(with_plastic)
    without = [p for p in images if p not in plastic_set]

    rng = random.Random(seed)
    n_plastic = min(len(with_plastic), round(n * plastic_fraction))
    chosen = rng.sample(with_plastic, n_plastic) + rng.sample(without, min(len(without), n - n_plastic))
    return sorted(chosen)
