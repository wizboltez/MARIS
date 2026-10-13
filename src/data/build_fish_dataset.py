"""Build a single-class 'fish' YOLO dataset: FathomNet deep-sea fish + TrashCan.

train = FathomNet fish (all) + TrashCan TRAIN images with fish + a random sample of TrashCan TRAIN
        images without fish (deep-sea negatives)
val   = TrashCan VAL (fish labels only)   -> model selection
test  = TrashCan TEST (fish labels only)  -> reported once, same 151 fish as the 16-class model

Run:  python -m src.data.build_fish_dataset
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import yaml

from src.config import CLASS_NAMES, PROCESSED_DIR, PROJECT_ROOT

FISH_ID = CLASS_NAMES.index("animal_fish")
FN_DIR = PROJECT_ROOT / "data" / "raw" / "fathomnet_fish"
OUT = PROJECT_ROOT / "data" / "processed_fish"
N_NEGATIVES = 1000


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hard link: no extra disk space
    except OSError:
        shutil.copyfile(src, dst)


def _fish_lines(label_path: Path) -> list[str]:
    if not label_path.exists():
        return []
    return ["0 " + " ".join(l.split()[1:]) for l in label_path.read_text().splitlines()
            if l.split()[:1] == [str(FISH_ID)]]


def _add_trashcan(split: str, out_split: str, keep_negatives: int | None) -> tuple[int, int]:
    src_img, src_lbl = PROCESSED_DIR / split / "images", PROCESSED_DIR / split / "labels"
    dst_img, dst_lbl = OUT / out_split / "images", OUT / out_split / "labels"
    positives, negatives = [], []
    for p in sorted(src_img.glob("*.jpg")):
        lines = _fish_lines(src_lbl / f"{p.stem}.txt")
        (positives if lines else negatives).append((p, lines))
    if keep_negatives is not None:
        negatives = random.Random(42).sample(negatives, min(keep_negatives, len(negatives)))
    for p, lines in positives + negatives:
        _link(p, dst_img / p.name)
        (dst_lbl / f"{p.stem}.txt").write_text("\n".join(lines))
    return len(positives), len(negatives)


def main() -> None:
    for s in ("train", "val", "test"):
        (OUT / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT / s / "labels").mkdir(parents=True, exist_ok=True)

    n_fn = 0
    for p in sorted((FN_DIR / "images").glob("*.jpg")):
        lbl = FN_DIR / "labels" / f"{p.stem}.txt"
        if lbl.exists() and lbl.read_text().strip():
            _link(p, OUT / "train" / "images" / p.name)
            shutil.copyfile(lbl, OUT / "train" / "labels" / lbl.name)
            n_fn += 1

    tr_pos, tr_neg = _add_trashcan("train", "train", keep_negatives=N_NEGATIVES)
    va_pos, va_neg = _add_trashcan("val", "val", keep_negatives=None)
    te_pos, te_neg = _add_trashcan("test", "test", keep_negatives=None)

    yaml.safe_dump({"path": str(OUT), "train": "train/images", "val": "val/images", "test": "test/images",
                    "names": {0: "fish"}}, open(OUT / "data.yaml", "w"), sort_keys=False)
    print(f"train: {n_fn} FathomNet + {tr_pos} TrashCan fish images + {tr_neg} TrashCan negatives")
    print(f"val:   {va_pos} images with fish, {va_neg} without (TrashCan val)")
    print(f"test:  {te_pos} images with fish, {te_neg} without (TrashCan test)")


if __name__ == "__main__":
    main()
