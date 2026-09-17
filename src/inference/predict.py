"""
Run the trained plastic detector on one image (or a folder of images).

Examples:
    python -m src.inference.predict --source path/to/image.jpg
    python -m src.inference.predict --source path/to/folder/ --conf 0.3

For every image this:
  1. runs the model,
  2. saves an annotated copy to outputs/predictions/annotated/,
  3. prints plastic count / coverage / confidence,
  4. appends a row to outputs/predictions/plastic_predictions.csv.

--lat/--lon/--timestamp are optional and only make sense for a single
--source image (TrashCan itself has no geotags; this just lets a future
caller attach real coordinates at inference time -- never invented here).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import BEST_MODEL_PATH, CLASS_NAMES, PREDICTIONS_CSV, PREDICTIONS_DIR  # noqa: E402
from src.utils.common import get_device  # noqa: E402
from src.utils.quantify import quantify  # noqa: E402
from src.utils.visualization import draw_detections  # noqa: E402

CSV_COLUMNS = [
    "image_id", "image_path", "plastic_count", "debris_count", "plastic_area",
    "plastic_coverage_percentage", "area_is_estimated_from_bbox",
    "average_confidence", "max_confidence", "class_wise_counts",
    "latitude", "longitude", "timestamp",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=str, required=True, help="Image file or directory of images")
    p.add_argument("--weights", type=str, default=str(BEST_MODEL_PATH))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--lat", type=float, default=None)
    p.add_argument("--lon", type=float, default=None)
    p.add_argument("--timestamp", type=str, default=None)
    return p.parse_args()


def run_on_image(model, image_path: Path, args) -> tuple[dict, np.ndarray]:
    result = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf,
                            device=get_device(args.device), verbose=False)[0]
    stats = quantify(result)

    img = cv2.imread(str(image_path))
    if result.boxes is not None and len(result.boxes) > 0:
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        masks = None
        if result.masks is not None:
            mh, mw = img.shape[:2]
            masks = np.stack([cv2.resize(m, (mw, mh)) > 0.5 for m in result.masks.data.cpu().numpy()])
        annotated = draw_detections(img, xyxy, cls_ids, confs, CLASS_NAMES, masks)
    else:
        annotated = img

    row = {
        "image_id": image_path.stem,
        "image_path": str(image_path),
        "plastic_count": stats["plastic_count"],
        "debris_count": stats["debris_count"],
        "plastic_area": stats["plastic_area"],
        "plastic_coverage_percentage": stats["plastic_coverage_percentage"],
        "area_is_estimated_from_bbox": stats["area_is_estimated_from_bbox"],
        "average_confidence": round(stats["average_confidence"], 4),
        "max_confidence": round(stats["max_confidence"], 4),
        "class_wise_counts": stats["class_wise_counts"],
        "latitude": args.lat,
        "longitude": args.lon,
        "timestamp": args.timestamp,
    }
    return row, annotated


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)

    source = Path(args.source)
    if source.is_dir():
        image_paths = sorted([p for p in source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    else:
        image_paths = [source]

    ann_dir = PREDICTIONS_DIR / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in image_paths:
        row, annotated = run_on_image(model, path, args)
        out_path = ann_dir / f"{path.stem}_annotated.jpg"
        cv2.imwrite(str(out_path), annotated)
        rows.append(row)

        print(f"\n{path.name}")
        print(f"  Plastic objects detected : {row['plastic_count']}")
        area_label = "estimated bbox coverage" if row["area_is_estimated_from_bbox"] else "plastic coverage"
        print(f"  {area_label:<26}: {row['plastic_coverage_percentage']:.2f}%")
        print(f"  Average confidence        : {row['average_confidence']:.2f}")
        print(f"  Annotated image saved to  : {out_path}")

    new_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if PREDICTIONS_CSV.exists():
        existing = pd.read_csv(PREDICTIONS_CSV)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["image_id"], keep="last")
    else:
        combined = new_df
    combined.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\nAppended {len(rows)} row(s) to {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
