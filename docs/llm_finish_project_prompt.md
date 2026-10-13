# Prompt: finish MARIS end to end at the highest honest accuracy

Paste the text block below into another LLM. It is self-contained. Numbers are the real held-out
results as of 2026-10-13; update them if they change.

```text
ROLE
You are a senior computer-vision / ML engineer helping a university team finish a project end to end.
Give an ordered, executable plan with code and exact commands. Every accuracy number must come from a
held-out test split that was never used for training, threshold tuning or model selection. Never
suggest simulated, hard-coded or cherry-picked results. If something cannot be done with public data,
say so and give the honest alternative.

PROJECT
MARIS - AI-based marine plastic pollution impact assessment on biodiversity. Brief: detect underwater
plastic; track fish and behaviour change; assess coral health and bleaching; integrate temperature,
pH, dissolved oxygen, salinity, turbidity; Marine Pollution Impact Score (0-100); biodiversity loss
prediction; explainable AI; pollution maps; decision-support dashboard.

STACK AND COMPUTE
Python 3.12, Ultralytics 8.4 (YOLO11-seg), PyTorch, segmentation-models-pytorch, XGBoost, SHAP,
OpenCV, Streamlit. Training on free Google Colab (Tesla T4 16 GB, ~2 h per run, disconnects) and a
local RTX 3050 4 GB laptop. Repo layout: src/modules/*.py, src/run_assessment.py (batch pipeline +
report), app.py (dashboard), notebooks/colab_train.ipynb (detector), notebooks/colab_train_suim_coral.ipynb.

CURRENT REAL RESULTS (held-out)
1. Plastic detector - YOLO11s-seg on TrashCan 1.0 (16 classes, deep-sea ROV video frames 480x270/360,
   JAMSTEC). Video-disjoint split: train 4913 / val 986 / test 1313 images, 312 videos, 0 overlap.
   batch 16, imgsz 640, AdamW auto lr 0.0005, cos_lr, mosaic 1.0, copy_paste 0, patience 30; best epoch 30,
   early-stopped at 60 (val flat after ~30 while train loss kept falling).
   Test: trash_plastic P 0.611 R 0.374 mAP50 0.489 mAP50-95 0.340. All classes box mAP50 0.252.
   animal_fish P 0.427 R 0.093; animal_shells, trash_rubber, animal_etc ~0; trash_metal R 0.18.
   Inference conf threshold is a fixed 0.18 (not tuned).
   Known failure: false positives on the camera's on-screen text overlay (timestamps, readouts).
2. Grad-CAM on YOLO neck layers 16/19/22, target = sum of class logits over ALL anchors >= conf.
   Only ~6% of heat falls inside detected plastic boxes; heat spreads to ROV equipment, marine snow, HUD.
3. Fish tracking - ByteTrack (custom thresholds 0.15/0.05) + phase-correlation camera-motion
   compensation over consecutive frames. 34 tracks over 53 test videos, 8 usable mobile tracks,
   3 near plastic. Mobile 12.3 px/frame vs 1.75 px/frame noise floor (near-static starfish).
4. Coral - U-Net ResNet-34 on SUIM, official TEST: reef IoU 0.684 (P 0.762, R 0.870), mIoU 0.684.
   No bleaching labels in SUIM; bleaching is a colour rule inside the reef mask. Out of domain on
   deep-sea frames (marked 595/1313 TrashCan frames as reef), so auto mode needs a colour check too.
5. Water - NOAA Coral Reef Watch SST / DHW / bleaching alert per station (real). pH, DO, salinity,
   turbidity are presets until a Copernicus Marine login (loader already written).
6. Image-level impact score - XGBoost reproducing a documented expert formula on synthetic data
   (R2 0.945 of the formula). Labelled as an assumption.
7. Observational model - 698 two-degree ocean cells: NOAA NCEI microplastics (13,460 samples, pieces/m3)
   + OBIS species counts + NOAA SST/DHW. Spatial-block CV R2: effort only 0.822, without plastic 0.847,
   with plastic 0.848 (microplastics add +0.001). Honest finding: no measurable association at this scale.
8. TrashCan images have no geotags; survey images are assigned to 5 stations in rotation.
Datasets allowed: TrashCan 1.0, SUIM, Trash-ICRA19, Fish4Knowledge, Reef Life Survey, NOAA CRW,
Allen Coral Atlas, OBIS, Copernicus Marine, NOAA NCEI microplastics, plus any other public dataset
you can name with a working source (say if unsure).

WHAT I NEED - in this order, each with: goal, exact steps/code, which split is used for what,
acceptance metric, expected gain, time on T4, and a fallback if it fails.

1. Confidence threshold per class: tune on VAL (max F1 or recall at precision >= 0.5 for
   trash_plastic), then report once on TEST. Code to compute the PR curve on val and pick the threshold.
2. HUD / on-screen-text masking: how to find the overlay regions (fixed zones per camera type, or OCR,
   or a mask learned from frame-to-frame invariance), apply at train and inference time, and measure
   false positives on test before/after.
3. Detector round 2 to raise trash_plastic recall without leaking test data:
   - adding Trash-ICRA19 (VOC boxes: plastic/bio/rov) to a segmentation model: box-only labels vs
     separate detection model vs SAM-generated pseudo-masks - which is best and why;
   - perceptual-hash dedupe of ICRA19 against TrashCan val+test (same JAMSTEC archive);
   - augmentations (copy_paste, mixup, hsv, scale), class imbalance (oversampling plastic images,
     cls loss weight), model size (11s vs 11m), imgsz (640 vs 960 for small objects), SAHI tiling;
   - exact Ultralytics train command(s) that fit a free T4 session, with resume.
4. Fish: raise fish recall (0.093). Option A: a separate fish detector on Fish4Knowledge / DeepFish /
   another public underwater fish set, used only for the fish class. Option B: fine-tune with extra
   fish data. How to validate tracking (MOTA / IDF1 / HOTA) on a dataset with track ground truth, and which
   behaviour metrics are defensible.
5. Grad-CAM per detection: target only the anchors whose centres fall inside the chosen detection box
   (map boxes to letterbox coordinates, strides 8/16/32), plus a deletion/insertion faithfulness test.
6. Coral bleaching: a real public healthy-vs-bleached coral dataset (name + link), a small classifier
   trained on Colab, applied inside the SUIM reef mask; validate on a held-out split; cross-check with
   NOAA CRW alerts.
7. Impact score: how to present and, if possible, ground the image-level score; how to use the
   observational null result honestly in the report.
8. Final evaluation and report checklist: one table of all held-out metrics with 95% confidence
   intervals (bootstrap over test images/videos), ablations, limitations, and what a strict
   examiner will attack.

Return numbered sections 1-8. Prefer specific, testable steps over general advice.
```
