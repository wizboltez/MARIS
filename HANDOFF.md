# Handoff to Teammate 2 (Environmental Risk Model)

This is what you need to know to build on top of Teammate 1's output. You should not need to
read or run any of Teammate 1's code — just consume the file below.

## The one file you need

**`outputs/predictions/plastic_predictions.csv`**

One row per underwater image that's been processed. Columns:

| Column | Type | Meaning |
|---|---|---|
| `image_id` | string | filename (no extension) — unique per image |
| `image_path` | string | where the source image file is on disk |
| `plastic_count` | int | number of `trash_plastic`-class objects detected |
| `debris_count` | int | number of *any* trash-material object detected (plastic, metal, rubber, wood, fabric, paper, fishing gear, misc) — broader than `plastic_count` |
| `plastic_area` | int | pixels of plastic detected in the image (real pixel-mask area, not estimated, unless the next column says otherwise) |
| `plastic_coverage_percentage` | float | `plastic_area / (image width × height) × 100` |
| `area_is_estimated_from_bbox` | bool | **check this first.** `False` = `plastic_area`/`plastic_coverage_percentage` come from real segmentation masks (this is the normal case for the shipped model). `True` = they're a bounding-box approximation instead (would only happen if a detection-only model were swapped in later) — treat coverage numbers less strictly if you ever see `True` here |
| `average_confidence` | float 0-1 | mean model confidence across detected plastic objects |
| `max_confidence` | float 0-1 | highest confidence among detected plastic objects |
| `class_wise_counts` | dict (as string) | every detected class and its count in that image, e.g. `{'trash_plastic': 1, 'animal_fish': 2}` — useful if you want debris-type breakdowns beyond just plastic |
| `latitude`, `longitude`, `timestamp` | float/float/string, all nullable | **empty by default.** TrashCan has no geotags, so we never invent these. If you have real location/time metadata for an image (from your own data sources), you can either join it in yourself on `image_id`/`image_path`, or ask to have Teammate 1 re-run `predict.py --lat --lon --timestamp` for that image. |

Load it like any CSV:
```python
import pandas as pd
df = pd.read_csv("outputs/predictions/plastic_predictions.csv")
```

`class_wise_counts` is stored as a Python-dict-formatted string (from `str(dict)`), not JSON —
parse it with `ast.literal_eval(row["class_wise_counts"])` if you need it programmatically.

## What this data does and doesn't mean

- This is **plastic detection and quantification**, not an ecological impact/risk measurement.
  A high `plastic_coverage_percentage` means "a lot of plastic is visible in this photo," not
  "this location is ecologically damaged" — that inference is your model's job.
- Numbers only exist for images that were actually run through the model. If you need
  predictions for new images (e.g. tied to your own biodiversity/coral/ocean data points),
  someone needs to run `python -m src.inference.predict --source <your images>` first — see
  `GUIDE.md` in the Teammate 1 folder, or just ask.
- Detection isn't perfect: on the held-out test set, precision is ~0.83 (few false alarms) but
  recall is ~0.60 (misses roughly 4 in 10 real plastic objects). If your risk model is
  sensitive to undercounting, that's worth knowing — treat `plastic_count`/`plastic_area` as a
  lower bound, not a ground truth.

## If you want to see how good the underlying model is

`outputs/metrics/eval_metrics_test.json` has precision/recall/mAP. `outputs/figures/` has the
confusion matrix, PR curves, and example images showing what it gets right and wrong.

## Getting predictions for new images

```bash
python -m src.inference.predict --source path/to/image_or_folder
```
This appends new rows to `plastic_predictions.csv` automatically (keyed by `image_id`, so
re-running on the same image updates its row rather than duplicating it).
