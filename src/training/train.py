"""
Train a YOLO instance-segmentation model on the preprocessed TrashCan data.

Run `python -m src.data.prepare_dataset` first to generate data/processed/data.yaml.

Examples
--------
Colab / discrete GPU (recommended defaults, fits a 4GB laptop GPU like an
RTX 3050):
    python -m src.training.train --model yolo11n-seg.pt --imgsz 640 --batch 8 --epochs 100

CPU fallback (slow -- use a small subset / fewer epochs for a smoke test):
    python -m src.training.train --model yolo11n-seg.pt --imgsz 416 --batch 4 --epochs 20 --device cpu

Second model for comparison (larger nano variant / different architecture):
    python -m src.training.train --model yolov8n-seg.pt --name yolov8n_seg_run --epochs 100

If you hit CUDA out-of-memory: lower --batch first (try 4, then 2), then
lower --imgsz (try 512, then 416). Ultralytics also accepts --batch -1 to
auto-select the largest batch size that fits in memory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATA_YAML_PATH, MODELS_DIR, RANDOM_SEED  # noqa: E402
from src.utils.common import get_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=str, default=str(DATA_YAML_PATH), help="Path to data.yaml")
    p.add_argument("--model", type=str, default="yolo11n-seg.pt",
                    help="Ultralytics model config/checkpoint, e.g. yolo11n-seg.pt, yolov8n-seg.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, help="-1 lets Ultralytics auto-pick a batch size that fits in VRAM")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr0", type=float, default=None, help="Initial LR; omit to use the model's default")
    p.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', '0', '0,1', ...")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--project", type=str, default="runs/segment", help="Ultralytics output root")
    p.add_argument("--name", type=str, default="trashcan_yolo_seg", help="Run name (subfolder of --project)")
    p.add_argument("--patience", type=int, default=20, help="Early-stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", type=str, default="ram", choices=["ram", "disk", "none"],
                    help="Cache decoded images after epoch 1 -- avoids re-reading from slow/synced storage")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    from ultralytics import YOLO  # imported here so --help doesn't require torch

    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
        resume=args.resume,
        plots=True,
    )
    if args.lr0 is not None:
        train_kwargs["lr0"] = args.lr0

    results = model.train(**train_kwargs)

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / "best_model.pt"
        dest.write_bytes(best_weights.read_bytes())
        print(f"Copied best checkpoint to {dest}")
    else:
        print(f"WARNING: expected best.pt not found at {best_weights}")

    print(f"Full training run (weights, plots, results.csv) saved under: {run_dir}")


if __name__ == "__main__":
    main()
