# File Structure — What Each File Contains

A map of the whole MARIS system: the Teammate 1 plastic-detection module (TrashCan 1.0) plus the
full assessment layer built on top of it (fish tracking, coral segmentation, real ocean data,
impact scoring, explainability, dashboard). For *why* the detector is designed this way see
[README.md](README.md); for run instructions see [GUIDE.md](GUIDE.md); for the running log of
requests, mistakes and fixes see [prompt.md](prompt.md).

```
aiml fa2/
├── README.md                     project overview + design decisions
├── GUIDE.md                      step-by-step "how to run" walkthrough
├── HANDOFF.md                    what Teammate 2 needs (the output CSV explained)
├── FILE_STRUCTURE.md             this file
├── prompt.md                     project prompt + log of requests, mistakes, failures and fixes
├── context.md                    architecture overview of the full system
├── requirements.txt              Python dependencies
├── app.py                        Streamlit decision-support dashboard (6 pages)
├── .gitignore                    what is NOT committed (data, runs, pretrained weights)
├── .claude/launch.json           launch config for the Streamlit demo
│
├── src/                          all the real code
│   ├── config.py                 single source of truth: paths, classes, split, seed
│   ├── data/
│   │   ├── validate_annotations.py   dataset quality checks + EDA stats
│   │   └── prepare_dataset.py        leak-free split + COCO → YOLO conversion
│   ├── training/
│   │   └── train.py                  model training (YOLO11-seg)
│   ├── evaluation/
│   │   └── evaluate.py               metrics, confusion matrix, PR curves, error analysis
│   ├── inference/
│   │   └── predict.py                run model on new images → CSV row
│   ├── utils/
│   │   ├── common.py                 seeding + CPU/GPU selection
│   │   ├── quantify.py               plastic count / area / coverage % calculation
│   │   ├── survey_sampling.py        plastic-balanced test-image sample for the batch run
│   │   └── visualization.py          all plotting + drawing helpers
│   ├── modules/                      full assessment layer
│   │   ├── gradcam.py                real Grad-CAM on the YOLO neck layers
│   │   ├── fish_tracking.py          ByteTrack over video frames + camera-motion compensation
│   │   ├── bytetrack_trashcan.yaml   tracker thresholds tuned to this detector
│   │   ├── fish_behavior.py          behavioural anomaly index (motion only from real tracks)
│   │   ├── coral_segmentation.py     loads the SUIM U-Net (only if it passed its test gate)
│   │   ├── coral_health.py           reef presence / cover / bleaching (U-Net + colour rule)
│   │   ├── ocean_data.py             NOAA Coral Reef Watch per station (+ Copernicus loader)
│   │   ├── water_quality.py          Water Quality Index + station presets
│   │   ├── impact_scoring.py         image-level impact score (documented expert formula)
│   │   ├── real_impact_data.py       observational model: NCEI microplastics + OBIS + NOAA
│   │   ├── xai_engine.py             SHAP + rule-based recommendations
│   │   └── geospatial_maps.py        station map
│   └── run_assessment.py         batch pipeline -> CSVs, figures, metrics, executive report
│
├── notebooks/                    report-friendly wrappers around src/
│   ├── 01_trashcan_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── colab_train.ipynb         detector training on Colab (+ Trash-ICRA19 fallback)
│   └── colab_train_suim_coral.ipynb   SUIM U-Net reef segmentation on Colab
│
├── models/
│   ├── plastic_detector/best_model.pt     trained YOLO weights
│   ├── coral_segmenter/                   SUIM U-Net: config + TEST metrics + deep-sea domain check
│   │                                      (suim_unet.pt weights are on Drive, git-ignored)
│   └── impact_model/                      XGBoost models (+ observational/ real-data model)
│
├── docs/llm_improvement_prompt.md   ready-made prompt to ask another LLM for improvements
│
├── outputs/
│   ├── figures/                  EDA + evaluation + error-analysis PNGs
│   ├── metrics/                  JSON metrics + dataset validation report
│   └── predictions/
│       └── plastic_predictions.csv   the handoff file for Teammate 2
│
├── data/
│   ├── raw/trashcan/             (not in git) TrashCan 1.0 dataset.zip + extracted files
│   ├── processed/                (not in git) YOLO-format train/val/test + data.yaml + split_manifest.csv
│   └── external/                 cached real data: NOAA station data, NCEI microplastics, OBIS cells
│
└── runs/                         (not in git) Ultralytics training logs/checkpoints
```

---

## Top-level files

| File | What it contains |
|---|---|
| `README.md` | Project scope, key design decisions (why `material_version` labels, why segmentation, why YOLO11n-seg, the data-leakage fix), setup, pipeline commands, recommended settings, common errors. |
| `GUIDE.md` | Practical walkthrough: what to type at each step, what input it needs, what output to expect. |
| `HANDOFF.md` | For Teammate 2: every column of `plastic_predictions.csv` explained, what the numbers do/don't mean, and the honest accuracy caveat. |
| `requirements.txt` | Dependencies: `ultralytics` (YOLO), `torch`, `opencv`, `pandas`, `pycocotools`, `imagehash` (duplicate detection), `scikit-learn`, `matplotlib`, `streamlit`, `jupyter`. |
| `app.py` | Streamlit decision-support dashboard, 6 pages: diagnostic studio (plastic, Grad-CAM, fish, coral, impact score), station map, fish tracking, coral, SHAP + recommendations, survey table + report. Run with `.venv\Scripts\python -m streamlit run app.py`. |
| `.gitignore` | Excludes large/regenerable things: raw & processed data, `runs/`, pretrained `yolo*.pt`. The trained `best_model.pt` and the predictions CSV **are** committed on purpose. |

## `src/` — the core code

### `src/config.py`
Central configuration used by every other file:
- All folder/file paths (raw data, processed data, outputs, model).
- `CLASS_NAMES` — the 16 TrashCan `material_version` classes (rov, plant, 6 animal classes, 8 trash-material classes).
- `PLASTIC_CLASSES = ["trash_plastic"]` and `DEBRIS_CLASSES` (all `trash_*`).
- `RANDOM_SEED = 42` and `SPLIT_FRACTIONS` = 70 / 15 / 15 (train / val / test).
- Top docstring explains the dataset findings (only 2 image sizes, official split leaks videos).

### `src/data/validate_annotations.py`
Checks the raw dataset before anything else uses it:
- Image exists, isn't corrupt, and its size matches the JSON.
- Bounding boxes are in bounds; polygons have ≥ 3 points and positive area.
- Near-duplicate frames via perceptual hashing (`imagehash.phash`).
- Collects EDA stats: class counts, image sizes, objects per image.
- Result: 7,212 images, 12,336 annotations, 0 invalid → `outputs/metrics/dataset_validation_report.json`.

### `src/data/prepare_dataset.py`
End-to-end preprocessing pipeline:
1. Runs validation and saves EDA figures.
2. Pools the official train + val images.
3. Builds "leak groups" by source video id (from `vid_<id>_frame<n>.jpg`), merging videos that share near-duplicate frames (Union-Find).
4. `GroupShuffleSplit` → train 4,913 / val 986 / test 1,313 images with **zero video overlap**.
5. Converts COCO polygons to YOLO segmentation labels (multi-part polygons keep the largest part).
6. Writes `data/processed/data.yaml` and `split_manifest.csv` (which image went to which split).

### `src/training/train.py`
Command-line training script. Loads a pretrained Ultralytics model (default `yolo11n-seg.pt`) and fine-tunes it on `data.yaml`. Flags: `--model --imgsz --batch --epochs --lr0 --patience --device --seed --cache --resume`. Copies the best checkpoint to `models/plastic_detector/best_model.pt`.

### `src/evaluation/evaluate.py`
Evaluates the trained model on the held-out **test** split:
- Box and mask Precision, Recall, mAP@50, mAP@50-95, F1 → `outputs/metrics/eval_metrics_test.json`.
- Confusion matrix and PR / F1 curves → `outputs/figures/`.
- Error analysis image grids: low-confidence, small objects, missed detections, cluttered scenes.
- Sample predictions grid.

### `src/inference/predict.py`
Runs the model on a new image or folder: saves an annotated image, prints plastic count / coverage / confidence, and adds (or updates) a row in `plastic_predictions.csv`. Optional `--lat --lon --timestamp` for real location metadata (never invented).

### `src/utils/`
| File | What it contains |
|---|---|
| `common.py` | `set_seed()` (Python/NumPy/PyTorch reproducibility) and `get_device()` (GPU if available, else CPU). |
| `quantify.py` | `quantify()` — turns one model result into plastic count, debris count, plastic pixel area, coverage %, average/max confidence, per-class counts. Uses real masks; falls back to box area only if no masks (and flags it). Shared by evaluate, predict and app so numbers are identical everywhere. |
| `visualization.py` | Plot helpers: class distribution, image sizes, objects per image, training curves, drawing masks/boxes on images, sample grids. |

## `notebooks/`

| Notebook | What it does |
|---|---|
| `01_trashcan_eda.ipynb` | Runs validation, shows class/size/objects-per-image plots and ground-truth masks on sample images. |
| `02_preprocessing.ipynb` | Runs `prepare_dataset.py`, shows the split summary and **proves zero video overlap** between splits. |
| `03_model_evaluation.ipynb` | Runs `evaluate.py` on the test split and displays the metrics, confusion matrix, PR curves, and error-analysis figures. |
| `colab_train.ipynb` | The actual GPU training on Google Colab. Uses `split_manifest.csv` so it trains on exactly the same leak-free split. Config cell holds MODEL / EPOCHS / PATIENCE / BATCH / IMGSZ / SEED; checkpoints saved to Drive so it can resume after a disconnect; ends with test-set evaluation including `trash_plastic`-only metrics. |

## `models/plastic_detector/`
- `best_model.pt` — the trained YOLO11n-seg weights (~6 MB). Used by `predict.py`, `evaluate.py` and `app.py`.

## `outputs/`

| Path | What it contains |
|---|---|
| `figures/class_distribution.png`, `image_size_distribution.png`, `objects_per_image.png` | EDA plots |
| `figures/confusion_matrix.png`, `confusion_matrix_normalized.png` | Which classes get confused with which |
| `figures/BoxPR_curve.png`, `MaskPR_curve.png`, `BoxF1_curve.png`, `MaskF1_curve.png` | Precision-recall and F1 curves |
| `figures/sample_predictions.png` | Model predictions on test images |
| `figures/error_analysis_*.png` | Failure cases: low confidence, small objects, missed detections, cluttered scenes |
| `metrics/dataset_validation_report.json` | Dataset stats: 7,212 images, 12,336 objects, per-class counts, duplicate groups |
| `metrics/eval_metrics_test.json` | Test metrics. ⚠️ Until the Colab retrain is copied in, the file wraps old values (P 0.83, mAP50 0.68) that came from a leaky split, under an explicit warning key. |
| `metrics/comprehensive_system_metrics.json` | Whole-system summary: detection, tracking, coral, SHAP, provenance of every component. |
| `reports/executive_impact_report.md` | Auto-generated report; every number is computed, with a Data Provenance section. |
| `predictions/marine_biodiversity_assessment.csv`, `fish_tracks.csv`, `station_summary.csv` | Batch-pipeline outputs. |
| `predictions/plastic_predictions.csv` | One row per processed image: plastic/debris count, plastic area, coverage %, confidence, class-wise counts, optional lat/lon/timestamp. |

## `data/` and `runs/` (not committed)
- `data/raw/trashcan/` — `dataset.zip` (527 MB) from the University of Minnesota repository, and the extracted `material_version/` images + COCO JSONs.
- `data/processed/` — generated by `prepare_dataset.py`: `{train,val,test}/{images,labels}/`, `data.yaml`, `split_manifest.csv`.
- `runs/segment/` — Ultralytics logs from local training attempts (`args.yaml`, `results.csv`, weights, batch previews).

---

## How the files connect

```
dataset.zip
   → validate_annotations.py  (quality checks, EDA)
   → prepare_dataset.py       (leak-free split, YOLO labels, data.yaml)
   → train.py / colab_train.ipynb   (fine-tune YOLO11-seg)
   → models/plastic_detector/best_model.pt
   → evaluate.py              (metrics + figures in outputs/)
   → predict.py / app.py      (new images → quantify.py → plastic_predictions.csv)
   → Teammate 2's risk model
```
