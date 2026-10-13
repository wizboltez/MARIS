# MARIS: AI-Based Marine Plastic Pollution Impact Assessment on Biodiversity

**Architecture & status overview** · last updated 2026-10-13 · branch `full-assessment-pipeline`

For the running log of requests, mistakes and fixes see [prompt.md](prompt.md). For file-by-file
detail see [FILE_STRUCTURE.md](FILE_STRUCTURE.md).

---

## 1. Problem statement

> Develop an AI system that detects underwater plastic waste, analyzes fish behavior and coral health,
> integrates water quality data, and estimates the ecological impact of plastic pollution on marine
> biodiversity.

Expected outcomes: plastic pollution maps, fish behaviour analysis, coral health assessment, Marine
Pollution Impact Score (0–100), biodiversity loss prediction, decision-support dashboard.

---

## 2. Components and how real each one is

| Component | Method | Data | Status | Evidence |
|---|---|---|---|---|
| Plastic detection | YOLO11s-seg (16 classes) | TrashCan 1.0, video-disjoint split (312 videos, 0 overlap) | Real: test mAP50 0.489; at conf 0.18 P 0.475, R 0.452 (95% CI over 53 videos) | `outputs/metrics/eval_metrics_test.json`, `test_bootstrap_confidence_intervals.json` |
| Grad-CAM | Gradients of the class logit w.r.t. YOLO neck layers 16/19/22 | Same images | Real | ~32% of heat inside plastic boxes |
| Fish tracking | ByteTrack (tuned thresholds) + phase-correlation camera-motion compensation | TrashCan test videos | Real, small sample | `outputs/predictions/fish_tracks.csv` |
| Coral reef segmentation | U-Net, ResNet-34 encoder | SUIM (official TEST split) | Real: reef IoU 0.684 | `models/coral_segmenter/suim_test_metrics.json` |
| Coral bleaching | Colour rule inside the U-Net reef mask | — | Heuristic (SUIM has no bleaching labels) | |
| Sea temperature, heat stress, bleaching alert | NOAA Coral Reef Watch 5 km daily | 5 survey stations | Real | `data/external/station_ocean_data.json` |
| pH, dissolved oxygen, salinity, turbidity | Copernicus Marine loader | — | Presets until `copernicusmarine login` | `src/modules/ocean_data.py` |
| Image-level impact score + biodiversity loss | XGBoost reproducing a documented expert formula | Synthetic | Assumption, labelled | `models/impact_model/fit_report.json` |
| Observational plastic–biodiversity model | XGBoost, spatial block CV, with/without-plastic ablation | NOAA NCEI microplastics + OBIS + NOAA CRW, per 2° cell | Real data, association only | `models/impact_model/observational/metrics.json` |
| Explainability | SHAP TreeExplainer on every score | — | Real | report Section "SHAP" |
| Station map | 5 real stations with NOAA values | — | Real positions; images assigned by rotation (no geotags) | `outputs/predictions/station_summary.csv` |
| Dashboard | Streamlit, 6 pages | — | Working, 0 errors in headless test | `app.py` |

---

## 3. Data flow

```
Image ─► YOLO11-seg ─► plastic count / area / % cover ──────────────┐
          ├─► Grad-CAM heatmap                                       │
          └─► animal boxes ─► ByteTrack over the video ─► fish index ─┤
Image ─► SUIM U-Net reef mask (+ colour check) ─► coral cover / BSI ──┤
Station ─► NOAA CRW SST, DHW, alert (+ presets / sliders) ─► WQI ─────┤
                                                                      ▼
                         XGBoost impact score (0–100) + biodiversity loss % ─► SHAP ─► recommendations
NCEI microplastics + OBIS species + NOAA CRW (per 2° cell) ─► observational XGBoost ─► real-world evidence
```

---

## 4. Known limitations (stated in the report)

- Plastic detector: misses about half of plastic on unseen videos (recall 0.452, 95% CI 0.32-0.59);
  false positives on the camera's on-screen text. Fish recall is low (0.093), which limits tracking.
- TrashCan has no geotags: survey images are assigned to stations in rotation.
- Fish: only 9 usable tracks of moving animals, 1 near plastic, so no behaviour-change claim.
- SUIM U-Net is out of domain on deep-sea footage (marked 595/1313 TrashCan frames as reef alone),
  so automatic coral detection requires agreement with a colour/texture check.
- No public dataset links plastic exposure to measured biodiversity loss for individual images; the
  image-level score is an expert formula. The observational model shows association, not causation,
  and OBIS species counts depend on sampling effort (controlled for, not removed).

---

## 5. How to run

```bash
.venv\Scripts\python -m src.run_assessment          # batch pipeline -> outputs/
.venv\Scripts\python -m streamlit run app.py        # dashboard
.venv\Scripts\python -m src.modules.ocean_data --refresh       # refresh NOAA station data
.venv\Scripts\python -m src.modules.real_impact_data --refresh # rebuild the observational model
```

Training (GPU): `notebooks/colab_train.ipynb` (detector) and `notebooks/colab_train_suim_coral.ipynb`
(coral) on Google Colab. Each has a quality gate on a held-out test split.
