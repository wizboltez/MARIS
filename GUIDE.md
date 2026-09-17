# How to Run — Step by Step

This is a practical walkthrough: what to type, what input each step expects, and what output
you should see. For *why* things are built this way (dataset format, class choice, split
strategy), see [README.md](README.md).

## 0. One-time setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Input needed:** none, besides internet access to download packages.
**Output:** a `.venv/` folder with everything installed.

---

## 1. Get the dataset

Download `dataset.zip` (527 MB) from
https://conservancy.umn.edu/items/6dd6a960-c44a-4510-a679-efb8c82ebfb7 and extract it so this
exact path exists:

```
data/raw/trashcan/dataset/material_version/train/   (images)
data/raw/trashcan/dataset/material_version/val/     (images)
data/raw/trashcan/dataset/material_version/instances_train_trashcan.json
data/raw/trashcan/dataset/material_version/instances_val_trashcan.json
```

**Input needed:** nothing else — just place the extracted folder there.
**Output:** ~7,200 `.jpg` images + 2 annotation `.json` files on disk.

---

## 2. Preprocess (validate, split, convert to YOLO format)

```bash
python -m src.data.prepare_dataset
```

**Input needed:** none — it reads directly from `data/raw/trashcan/...`.
**What it does:** checks every image/annotation for problems, splits the data into
train/val/test with no video overlap, converts COCO polygons to YOLO label `.txt` files.

**Output you'll see:**
```
Split sizes (images):
train    4913
test     1313
val       986

Leak-group video overlap check:
  train vs val shared video ids: 0
  ...
Wrote .../data/processed/data.yaml
Wrote .../data/processed/split_manifest.csv
```
Also produces: `data/processed/{train,val,test}/{images,labels}/`,
`outputs/figures/class_distribution.png`, `image_size_distribution.png`,
`objects_per_image.png`, and `outputs/metrics/dataset_validation_report.json`.

Run this once; re-run it any time you want to reshuffle the split with a different seed
(`src/config.py` → `RANDOM_SEED`).

---

## 3. Train the model

### Option A — Google Colab (recommended if your laptop GPU is unstable/overheats)

If local GPU training keeps crashing (driver crash, "GPU is lost", or the GPU running very
hot), train on Colab's free GPU instead — nothing runs on your machine.

1. Upload `data/raw/trashcan/dataset.zip` (527MB, already on your machine from step 1) to
   your Google Drive at `MyDrive/trashcan/dataset.zip`.
2. Open `notebooks/colab_train.ipynb` in Google Colab (colab.research.google.com → File →
   Upload notebook).
3. Runtime → Change runtime type → GPU.
4. Run every cell top to bottom. It mounts your Drive, extracts the dataset, does the same
   leak-free split + COCO→YOLO conversion as the local pipeline, then trains.
5. The last cell copies the trained model to `MyDrive/trashcan/best_model.pt`. Download it
   and place it at `models/plastic_detector/best_model.pt` in this project — everything else
   (`predict.py`, `evaluate.py`, `app.py`) then works locally without needing a GPU at all.

### Option B — locally

**Input needed:** none from you — just monitor for a GPU crash if you're on unreliable
hardware; if `nvidia-smi` ever reports "GPU is lost", reboot and re-run the same command
(training resumes from scratch, not from the crash point).


```bash
python -m src.training.train --model yolo11n-seg.pt --imgsz 640 --batch 8 --epochs 30 --cache disk
```

**Inputs (all optional flags, sensible defaults if omitted):**

| Flag | Meaning | Example values |
|---|---|---|
| `--model` | which YOLO checkpoint to start from | `yolo11n-seg.pt` (default), `yolov8n-seg.pt` (2nd model for comparison) |
| `--imgsz` | training image size | `640` (default), `512`/`416` if low on VRAM |
| `--batch` | images per step | `8` (default), lower if you hit CUDA out-of-memory |
| `--epochs` | how many passes over the data | `30`–`60` typical for this project size |
| `--device` | `auto` (default, picks GPU if present), or `cpu` | |
| `--cache` | `ram`, `disk` (default), or `none` | use `disk` if you don't have ~8GB free RAM |

**Output:**
- Live progress per epoch (loss values, it/s).
- `models/plastic_detector/best_model.pt` — the trained weights, used by every step below.
- `runs/segment/.../` — full Ultralytics run folder (all checkpoints, `results.csv`, plots).

This is the slow step — expect anywhere from ~30 min to a few hours depending on your
hardware (see the table in README.md).

---

## 4. Evaluate on the test set

```bash
python -m src.evaluation.evaluate --weights models/plastic_detector/best_model.pt --split test
```

**Input needed:** just the trained weights path (defaults to the path above, so you can
usually omit `--weights` entirely).

**Output:**
- Console: precision, recall, mAP@50, mAP@50:95, F1 (box + mask), e.g.:
  ```
  {
    "box_precision": 0.71, "box_recall": 0.58, "box_map50": 0.63, "box_map50_95": 0.39,
    "mask_map50": 0.60, ...
  }
  ```
- `outputs/metrics/eval_metrics_test.json` — same numbers, saved.
- `outputs/figures/`: `confusion_matrix.png`, `MaskPR_curve.png`, `BoxPR_curve.png`,
  `sample_predictions.png`, `error_analysis_low_confidence.png`,
  `error_analysis_small_objects.png`, `error_analysis_false_negatives_no_detection.png`,
  `error_analysis_cluttered_scenes.png`.

---

## 5. Run it on your own image (the main deliverable)

```bash
python -m src.inference.predict --source path/to/your_underwater_photo.jpg
```

**Input needed:** `--source` — either:
- a single image file: `--source data/processed/test/images/vid_000052_frame0000001.jpg`
  (use any real file from the test set to try it immediately without your own photo), or
- a folder of images: `--source path/to/a/folder/` (processes every `.jpg`/`.jpeg`/`.png` inside).

Optional, only if you actually have this info for the photo (never guess/invent it):
```bash
python -m src.inference.predict --source photo.jpg --lat 35.02 --lon 139.02 --timestamp 2024-05-01T10:00:00
```

**Output — console, per image:**
```
vid_000052_frame0000001.jpg
  Plastic objects detected : 3
  plastic coverage          : 8.42%
  Average confidence        : 0.87
  Annotated image saved to  : outputs/predictions/annotated/vid_000052_frame0000001_annotated.jpg
```

**Output — files:**
- `outputs/predictions/annotated/<name>_annotated.jpg` — your image with colored outlines
  drawn around every detected object.
- A new row appended to `outputs/predictions/plastic_predictions.csv`:

  | image_id | plastic_count | plastic_coverage_percentage | average_confidence | latitude | longitude |
  |---|---|---|---|---|---|
  | vid_000052_frame0000001 | 3 | 8.42 | 0.87 | (empty unless you passed `--lat`) | |

  This CSV is the file your teammate's module reads.

---

## 6. Try the visual demo (optional)

```bash
streamlit run app.py
```

**Input needed:** none from the command line — it opens a page in your browser. There, you:
1. Upload any underwater `.jpg`/`.png` using the file picker.
2. (Optional) drag the confidence slider.
3. See the original vs. annotated image side by side, plus plastic count / coverage /
   confidence metrics.

Good for showing your professor a live demo instead of terminal output.

---

## Quick reference: minimum path to a result

If you just want to see *something* work end to end without tuning anything:

```bash
python -m src.data.prepare_dataset
python -m src.training.train --epochs 10 --cache disk
python -m src.inference.predict --source data/processed/test/images/<pick any file here>
```

10 epochs won't be a great model, but it proves every stage of the pipeline runs correctly —
useful for a first check before committing to a longer training run.
