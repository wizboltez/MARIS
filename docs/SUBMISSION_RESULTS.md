# MARIS submission results

MARIS detects marine plastic, estimates visible coverage, and combines marine observations with an illustrative ecological scoring model.

## Measured results already stored in this project

| Task | Metric | Result |
|---|---|---|
| Plastic present/absent in an image | Image-level accuracy at confidence 0.18 | 75.1% |
| Plastic present/absent in an image | Majority-class baseline | 62.8% |
| Plastic object detection | Bounding-box mAP50 | 48.9% |
| Plastic object detection at confidence 0.18 | Precision / recall / F1 | 47.5% / 45.2% / 46.3% |
| Coral reef segmentation, SUIM test | Reef IoU / pixel accuracy | 68.4% / 85.4% |

Plastic results use 1,313 held-out images from 53 source videos. Image-level accuracy measures presence/absence; it does not measure correct object locations or segmentation. The coral figures use a separate dataset and task. These figures are existing recorded evaluations, not a new evaluation of today's fixes.

Evidence: `outputs/metrics/image_level_detection_test.json`, `outputs/metrics/eval_metrics_test.json`, `outputs/experiments/2026-10-14/FINAL_EVALUATION.md`, and `models/coral_segmenter/suim_test_metrics.json`.

## Demonstration

Run from the project folder:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Use source images from `data/processed/val/images` for a demonstration. The video ranking in `outputs/experiments/2026-10-14/val_video_ranking.csv` identifies successful validation examples. Already annotated examples are in `outputs/experiments/2026-10-14/demo_frames`; they illustrate outcomes and should not be submitted as fresh raw-image inference inputs. Best-case demonstrations do not establish overall accuracy.

## Changes made for submission

- Prediction CLI now uses the same configured confidence default as the dashboard (fallback 0.18).
- Segmentation coverage restores letterboxed masks and counts their union to prevent double-counting overlapping plastic pixels.
- These fix operating consistency and coverage measurement; they do not establish an increase in detector mAP.

## Scope and limitations

The image-level ecological impact model reproduces a documented expert formula using synthetic training data; its output is an illustrative score, not measured biodiversity loss. Coral bleaching is heuristic. Plastic detection still has substantial misses and false positives. Threshold tuning and a replacement fish model were previously evaluated without a convincing held-out gain. More diverse labeled video data and retraining are needed for stronger generalization.
