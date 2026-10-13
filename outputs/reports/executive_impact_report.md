# Executive Environmental Impact Assessment: Marine Plastic Pollution on Biodiversity

**System**: MARIS (Marine AI Risk and Impact System)
**Evaluation Scope**: 40 TrashCan 1.0 test images (20 with labelled plastic, 20 without)
**Core Objective**: Detect underwater plastic waste, track fish behavioral changes, evaluate coral bleaching, integrate water quality telemetry, and forecast ecological biodiversity loss.

> **Read this first.** Real: plastic detection, Grad-CAM, fish tracking, SUIM reef segmentation, NOAA satellite SST and
> bleaching alerts. Assumed: the image-to-station assignment (TrashCan has no geotags), pH / oxygen / salinity /
> turbidity unless Copernicus is configured, and the expert formula behind the impact score and biodiversity
> loss (Section 2, item 5). Treat the scores as a demonstration of the pipeline, not as ecological findings.

---

## 1. Key Results

| Metric | Value | Reference |
|---|---|---|
| **Mean Marine Pollution Impact Score (MPIS)** | **13.9 / 100** | Tiers: Low < 25, Moderate < 50, High < 75, Critical ≥ 75 |
| **Mean Predicted Biodiversity Loss** | **10.8%** | Model output (synthetic training data) |
| **Plastic Items Detected** | **26** items; found in 16 of 20 images with labelled plastic, plus 5 false-alarm images | Detector recall for plastic is ~0.31 |
| **Coral bleaching** | Not assessed | TrashCan frames are deep-sea (no reef); see NOAA station alerts below |
| **Mean Fish Behavioral Anomaly Index** | **17.1 / 100** | 1 of 40 images have tracked moving animals |
| **Transects at High / Critical Risk** | **0 of 40 (0.0%)** | MPIS ≥ 50 |

---

## 2. Pipeline

1. **Plastic detection & segmentation**: YOLO11-seg trained on TrashCan 1.0; coverage % from real pixel masks.
2. **Fish detection & tracking**: the same detector + ByteTrack across consecutive video frames, with ROV camera motion removed by phase correlation; speed, turning and distance to plastic per track.
3. **Coral health**: U-Net reef segmentation trained on SUIM (see Section 4) plus a colour rule for bleaching inside the reef area; skipped for deep-sea TrashCan frames.
4. **Water quality**: Water Quality Index from temperature (NOAA CRW satellite), pH, dissolved oxygen, salinity and turbidity.
5. **Impact fusion**: XGBoost regressors predict the MPIS (0–100) and biodiversity loss % (expert-formula targets; held-out R² 0.945 for MPIS).
6. **Explainability**: SHAP TreeExplainer attributions for every transect, and Grad-CAM on the detector.
   On 6 frames with plastic detections, on average 32% of the
   Grad-CAM heat falls inside the plastic boxes (which cover only a few % of the image).

---

## 3. Fish Tracking

53 test videos tracked; 85 animal tracks, 9 mobile (fish/eel/crab) tracks of 3+ frames ({'eel': 4, 'fish': 3, 'crab': 2}). Median mobile speed 10.28 px/frame vs. 0.9 px/frame for near-static animals (measurement noise floor). Mobile tracks passing within 25% of the image diagonal of plastic: 1 - too few to test whether plastic changes behaviour.

## 4. Coral Reef Segmentation Model

U-Net with a resnet34 encoder trained on SUIM (8 classes). Results on the official SUIM TEST split (110 images, never used for training or model selection):

| Metric | Value |
|---|---|
| Reef IoU (reefs & invertebrates) | **0.684** (gate 0.5) |
| Reef precision / recall | 0.762 / 0.870 |
| Mean IoU (8 classes) | 0.684 |
| Pixel accuracy | 0.854 |

**Domain check on deep-sea footage:** on the 1313 TrashCan test frames (no reef), the U-Net alone marked 595 frames as >=2% reef (mud, sediment and rock). It is therefore only trusted together with the colour/texture coral check in auto mode, which reduces this to 0 frames.

## 5. Survey Stations (NOAA Coral Reef Watch, real satellite data)

| Station | SST (°C) | DHW (°C-weeks) | Bleaching alert | Images assigned | Mean MPIS |
|---|---|---|---|---|---|
| Raja Ampat, Coral Triangle | 29.05 | 0.0 | Bleaching Watch | 8 | 9.2 |
| Cairns Sector, Great Barrier Reef | 25.21 | 0.0 | No Stress | 8 | 7.3 |
| Red Sea Central Channel | 31.79 | 1.19 | Bleaching Watch | 8 | 17.4 |
| Manila Bay Outflow | 30.28 | 4.79 | Alert Level 1 | 8 | 19.3 |
| Gulf Coastal Estuary | 28.49 | 7.46 | Bleaching Watch | 8 | 16.5 |

---

## 6. SHAP Attribution Summary (computed over all 40 transects)

Most influential feature on average: **`plastic_coverage_percentage`**. SHAP values are relative to the model's
average synthetic site, so a feature "lowers" the score when these images have less of it than that
average (e.g. most images have little or no plastic).

| Feature | Mean abs. SHAP (score points) | Mean SHAP | Direction |
|---|---|---|---|
| `plastic_coverage_percentage` | 4.96 | -3.73 | lowers score on average |
| `fish_behavior_anomaly_index` | 4.36 | -4.36 | lowers score on average |
| `water_quality_index` | 3.85 | +2.07 | raises score on average |
| `plastic_count` | 2.48 | -2.48 | lowers score on average |
| `coral_bleaching_severity_index` | 1.66 | -1.66 | lowers score on average |
| `temperature_c` | 0.97 | +0.65 | raises score on average |
| `coral_cover_percentage` | 0.87 | +0.87 | raises score on average |
| `coral_present` | 0.54 | +0.54 | raises score on average |

---

## 7. Data Provenance

| Component | Status |
|---|---|
| Plastic Detection | REAL - YOLO11-seg trained on TrashCan 1.0, run on real test images |
| Plastic Detector Quality | Weak - trash_plastic precision ~0.37, recall ~0.31 on leak-free images (see HANDOFF.md) |
| Fish Behavior | REAL, SMALL SAMPLE - ByteTrack over test video frames with camera-motion compensation; few mobile animals are tracked, so plastic-related behaviour change cannot be concluded |
| Coral Health | TRAINED - U-Net (resnet34) on SUIM; official SUIM TEST reef IoU 0.684 (precision 0.76, recall 0.87). Not applied to TrashCan (deep-sea, no reef; the U-Net is out of domain there). Bleaching itself is still a colour rule inside the reef mask (SUIM has no bleaching labels). |
| Water Temperature And Bleaching Alert | REAL - NOAA Coral Reef Watch 5km daily satellite SST, Degree Heating Weeks and Bleaching Alert Area at each station |
| Ph Oxygen Salinity Turbidity | PRESET - station values (Copernicus Marine loader is implemented but needs a free account: `copernicusmarine login`) |
| Station Assignment | ASSUMED - TrashCan images have no geotags, so images are assigned to stations in rotation |
| Impact Score And Biodiversity Loss | EXPERT FORMULA - XGBoost reproduces a documented hand-specified formula (synthetic training data); weights are assumptions, see src/modules/impact_scoring.py |
| Shap | REAL - SHAP TreeExplainer on the XGBoost model (explains the model, which is trained on synthetic data) |
| Gradcam | REAL - gradient-based Grad-CAM on the YOLO neck layers for the trash_plastic logit |
| Geospatial Map | REAL station positions + NOAA CRW values; per-station scores average the images assigned to each station |

---

## 8. Standard Decision-Support Recommendations

Rule-based guidance templates, triggered by score thresholds:

1. **Debris Extraction**: prioritise cleanup where plastic coverage exceeds 5%.
2. **Thermal Stress Response**: shading and monitoring for reefs at NOAA Alert Level 1 or above.
3. **Runoff Containment**: sediment barriers where turbidity exceeds 5 NTU or oxygen is low.
4. **Zoning**: protective restrictions where MPIS exceeds 50.

---
*Generated by the MARIS assessment pipeline (`python -m src.run_assessment`).*
