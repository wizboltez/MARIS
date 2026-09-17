# Teammate 1 — Underwater Plastic Detection & Quantification

University AI/ML project: *AI-Based Marine Plastic Pollution Impact and Risk Assessment System*.
This module is **Teammate 1's part only**: detect and measure plastic/debris in underwater
images using the TrashCan 1.0 dataset. It does **not** compute ecological risk, biodiversity
exposure, habitat sensitivity, or forecasting — that is Teammate 2's module, which consumes
this module's CSV output.

## What this actually measures

TrashCan gives visual evidence of underwater trash; this module reports **plastic detection
and quantification** (count, pixel coverage, confidence) — not "ecological damage" or "risk".
It also does not invent GPS coordinates: TrashCan has none, so `latitude`/`longitude` in the
output CSV are left empty unless you pass them explicitly at inference time.

## Key design decisions (made after inspecting the real dataset, not assumed)

- **Label taxonomy**: `material_version` (16 classes) was used instead of `instance_version`
  (22 classes) because it has an explicit `trash_plastic` category — the other taxonomy splits
  trash by object type (bag, bottle, cup...) with no unified "plastic" label. The model is
  still trained on all 16 classes (rov, plant, 6 animal classes, 8 trash-material classes) so
  it learns to tell trash apart from look-alike marine life; only `trash_plastic` is reported
  as "plastic" downstream.
- **Task**: instance segmentation, not plain bounding-box detection. TrashCan ships real
  polygon masks, and segmentation gives a genuine pixel-based `plastic_coverage_percentage`
  instead of a bounding-box approximation.
- **Model**: `yolo11n-seg` (Ultralytics, ~2.8M params) — fits comfortably on a 4GB laptop GPU
  and trains from a folder of ~7,200 small (480×360) images in a reasonable time. A second
  model for comparison is just a `--model` flag away (e.g. `yolov8n-seg.pt`).
- **Data leakage**: filenames encode a source video id (`vid_<id>_frame<n>.jpg`). We found the
  dataset's *own* published train/val split reuses the same video across both (127 of 133 val
  videos also appear in train) — i.e. it is not leak-free. We therefore pool official
  train+val and re-split ourselves with `GroupShuffleSplit` grouped by video id (70/15/15
  train/val/test), additionally merging any two videos that share a near-duplicate frame
  (detected via perceptual hashing) into one group. Verified zero video-id overlap across our
  three splits. This means results here aren't directly comparable to the original paper's
  reported numbers — that's expected, and is the point.
- **Segmentation label conversion**: when a TrashCan instance's polygon has multiple
  disconnected parts (occlusion), we keep only the largest part as that instance's YOLO
  polygon. This is a documented simplification, not a bug — see the comment in
  `src/data/prepare_dataset.py`.

## Project structure

```
data/raw/trashcan/         downloaded TrashCan 1.0 (not committed — see below)
data/processed/            YOLO-format train/val/test images+labels, data.yaml, split_manifest.csv
src/config.py              single source of truth: paths, class list, split fractions, seed
src/data/
  validate_annotations.py  missing/corrupt image, invalid bbox/polygon, duplicate, class-dist checks
  prepare_dataset.py        validate -> leak-free split -> COCO->YOLO-seg conversion -> EDA figures
src/training/train.py       configurable YOLO-seg training CLI
src/evaluation/evaluate.py  P/R/mAP50/mAP50-95/F1, confusion matrix, PR curves, error analysis
src/inference/predict.py    run on one image/folder -> annotated image + plastic_predictions.csv row
src/utils/
  common.py                seed + device helpers
  quantify.py               plastic count/area/coverage/confidence from one model result
  visualization.py          EDA plots, detection overlay drawing, training-curve plots
app.py                      optional Streamlit demo (Teammate 1 only, no risk/biodiversity content)
models/plastic_detector/best_model.pt   trained weights (produced by train.py)
outputs/figures/            all EDA + evaluation + error-analysis PNGs
outputs/predictions/plastic_predictions.csv   the handoff file for Teammate 2
outputs/metrics/            JSON metrics + validation report
notebooks/                  thin, runnable wrappers around the src/ pipeline, for the report
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on Linux/Mac
pip install -r requirements.txt
```

GPU is used automatically when available (`torch.cuda.is_available()`); otherwise falls back
to CPU. Install PyTorch with the right CUDA build for your GPU first if `pip install torch`
doesn't pick one up automatically, e.g.:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Get the dataset

Download `dataset.zip` (527 MB) from the official University of Minnesota Data Repository —
https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7 (DOI
10.13020/g1gx-y834) — and extract it so you have:
```
data/raw/trashcan/dataset/material_version/{train,val}/  + instances_{train,val}_trashcan.json
```

## Run the pipeline

```bash
# 1. Validate + leak-free split + COCO->YOLO conversion + EDA figures
python -m src.data.prepare_dataset

# 2. Train (defaults fit a 4GB laptop GPU; add --cache ram to avoid re-reading images
#    from slow/cloud-synced storage every epoch)
python -m src.training.train --model yolo11n-seg.pt --imgsz 640 --batch 8 --epochs 60 --cache ram

# Second model, for the required comparison:
python -m src.training.train --model yolov8n-seg.pt --name yolov8n_seg_run --epochs 60 --cache ram

# 3. Evaluate on the held-out test split
python -m src.evaluation.evaluate --weights models/plastic_detector/best_model.pt --split test

# 4. Run inference on a new image
python -m src.inference.predict --source path/to/image.jpg

# 5. Optional demo UI
streamlit run app.py
```

### Recommended settings

| Environment | imgsz | batch | epochs | notes |
|---|---|---|---|---|
| Colab (T4/similar) | 640 | 16 | 60–100 | `--cache ram`; a full run fits comfortably in a session |
| Laptop GPU, 4GB VRAM (e.g. RTX 3050) | 640 | 8 | 30–60 | `--cache ram`; if you hit CUDA OOM, drop `--batch` to 4, then `--imgsz` to 512/416 |
| CPU only | 416 | 4 | 10–20 | for a correctness smoke test, not a real result — expect hours per run |

## The handoff to Teammate 2

`outputs/predictions/plastic_predictions.csv` — one row per processed image:

```
image_id, image_path, plastic_count, debris_count, plastic_area,
plastic_coverage_percentage, area_is_estimated_from_bbox,
average_confidence, max_confidence, class_wise_counts,
latitude, longitude, timestamp
```

- `plastic_*` counts/measures only the `trash_plastic` class; `debris_count` covers all
  trash-material classes for broader context.
- `plastic_coverage_percentage` is real pixel-mask coverage from the segmentation model
  (`area_is_estimated_from_bbox = False`). If a detection-only model were ever swapped in,
  this flag would flip to `True` and the number becomes a bounding-box approximation —
  Teammate 2 should check this flag rather than assume.
- `latitude`/`longitude`/`timestamp` are `null` unless supplied via `predict.py --lat --lon
  --timestamp`; TrashCan has no geotags, so none are invented. Teammate 2 attaches real
  location/time before joining with OBIS/Allen Coral Atlas/Copernicus Marine data.

## Metrics produced

Precision, Recall, mAP@50, mAP@50:95, F1 — separately for boxes and masks (see
`outputs/metrics/eval_metrics_test.json`) — plus confusion matrix and PR curves
(`outputs/figures/`).

## Figures produced

`class_distribution.png`, `image_size_distribution.png`, `objects_per_image.png` (EDA);
`confusion_matrix(.png/_normalized.png)`, `BoxPR_curve.png`, `MaskPR_curve.png` (evaluation);
`sample_predictions.png`, `error_analysis_low_confidence.png`,
`error_analysis_small_objects.png`, `error_analysis_false_negatives_no_detection.png`,
`error_analysis_cluttered_scenes.png` (qualitative).

## Common errors and fixes

- **CUDA out of memory**: lower `--batch` (try 4, then 2), then `--imgsz` (try 512, then 416).
- **`FileNotFoundError` for `data/raw/trashcan/...`**: you haven't downloaded/extracted the
  dataset yet — see "Get the dataset" above.
- **`data/processed/data.yaml` missing** when running `train.py`: run
  `python -m src.data.prepare_dataset` first.
- **"Slow image access" warning during training**: your `data/processed` folder is on
  cloud-synced storage (OneDrive/Dropbox/etc). Pass `--cache ram` (default) so images are only
  read from disk once; for a long run, consider moving the repo out of the synced folder.
- **Streamlit says no model found**: run `train.py` first; it copies the best checkpoint to
  `models/plastic_detector/best_model.pt`.
