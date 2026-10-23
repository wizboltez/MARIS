# MARIS — AI-Based Marine Plastic Pollution Impact Assessment on Biodiversity

University AI/ML project (Semester 5, FA1 + FA2). MARIS detects underwater plastic, analyses fish movement,
assesses coral reef health, adds satellite ocean data, and combines everything into a **Marine Pollution Impact
Score (0–100)** with explanations, shown in an interactive decision-support dashboard.

All results below are measured on **data never used for training** (unseen videos / unseen dive sites).

## Expected outcomes (from the brief) and where they are

| # | Outcome | Dashboard page | Main code |
|---|---|---|---|
| 1 | Plastic pollution maps | 1. 🗺️ Plastic Pollution Maps | `src/modules/geospatial_maps.py`, `ocean_data.py` |
| 2 | Fish behaviour analysis | 2. 🐟 Fish Behaviour Analysis | `src/modules/fish_tracking.py` (ByteTrack) |
| 3 | Coral health assessment | 3. 🪸 Coral Health Assessment | `src/modules/coral_segmentation.py`, `coral_health.py` |
| 4 | Marine Pollution Impact Score | 4. 📊 Marine Pollution Impact Score | `src/modules/impact_scoring.py` (XGBoost) |
| 5 | Biodiversity loss prediction | 5. 📉 Biodiversity Loss Prediction | `impact_scoring.py`, `real_impact_data.py` |
| 6 | Decision-support dashboard | 6. 🧭 Decision Support (+ whole app) | `app.py`, `src/modules/xai_engine.py`, `gradcam.py` |

## Pipeline

```
underwater image / video
   ├─ YOLO11-seg (TrashCan) ............ plastic + debris masks, animals, ROV
   ├─ ByteTrack + camera-motion fix .... fish tracks: speed, turning, distance to plastic
   ├─ SegFormer MiT-B2 (Coralscapes) ... live / bleached / dead coral
   └─ NOAA Coral Reef Watch ............ sea temperature, heat stress, bleaching alert
                    ↓
   XGBoost → Marine Pollution Impact Score (0–100) + biodiversity loss
                    ↓
   SHAP + Grad-CAM explanations → recommendations → Streamlit dashboard
```

## Key results (held-out test data)

| Module | Result | Test data |
|---|---|---|
| Coral: bleaching present? (image level) | **95.7 %** accuracy (baseline 82.9 %) | Coralscapes test, 392 images, 5 unseen dive sites |
| Coral: coral present? (image level) | **95.4 %** accuracy | same |
| Coral vs not-coral (pixels) | **95.2 %** accuracy | same |
| Coral segmentation | pixel acc. 94.3 %, coral IoU 0.767 (P 0.876 / R 0.860) | same |
| Coral false alarms on deep-sea video | 1 of 200 frames | TrashCan test |
| Underwater robot (ROV) present? | **91.5 %** accuracy, recall 90.9 % | TrashCan test, 1,313 images, 53 unseen videos |
| Debris alarms that are correct | **89.3 %** precision (image level) | same |
| Plastic present? (image level) | 75.1 % accuracy (baseline 62.8 %) | same |
| Plastic detection (objects) | P 0.475, R 0.452, F1 0.463 (95 % CI ±~0.12); mAP50 0.489 | same |
| Fish detection | P 0.407, R 0.159, F1 0.229 — main weakness | same |

Full numbers and sources: [REPORT_GUIDE.md](REPORT_GUIDE.md) (report structure + every table),
[outputs/reports/executive_impact_report.md](outputs/reports/executive_impact_report.md) (full assessment report),
[outputs/experiments/2026-10-14/FINAL_EVALUATION.md](outputs/experiments/2026-10-14/FINAL_EVALUATION.md)
(detector evaluation with confidence intervals).

## Datasets

TrashCan 1.0 · Coralscapes · SUIM · SeaClear · FathomNet · NOAA Coral Reef Watch · NOAA NCEI Marine Microplastics ·
OBIS · Copernicus Marine · Allen Coral Atlas — details, licences and citations in [REPORT_GUIDE.md](REPORT_GUIDE.md#5-dataset).

## What is real and what is estimated

- **Real:** plastic / animal detection, fish tracking, coral segmentation, Grad-CAM, NOAA satellite temperature and
  bleaching alerts, the NOAA microplastics + OBIS species analysis.
- **Estimated:** the impact score and biodiversity-loss % follow a documented expert formula (XGBoost reproduces it);
  pH, oxygen, salinity and turbidity are station presets; survey images have no GPS, so they are assigned to
  stations by rotation.
- **Honest evaluation:** splits by video / dive site; thresholds chosen on validation only. An earlier version had
  77 % of test images in training (inflated score 0.68 vs real ~0.17) — found and fixed.

## Run it

Commands are for Windows, run from the project folder.

1. **Create the virtual environment** (first time only):
   ```bash
   python -m venv .venv
   ```
2. **Install dependencies** (first time, or when `requirements.txt` changes):
   ```bash
   .venv\Scripts\pip install -r requirements.txt
   ```
3. **Get the coral model weights.** Files over 100 MB are not in git. Download
   `models/coral_segmenter/coralscapes_segformer_v2/model.safetensors` from Google Drive and put it at that path.
   Without it the app automatically uses a fallback instead.
4. **Start the dashboard.** It opens at http://localhost:8501:
   ```bash
   .venv\Scripts\python.exe -m streamlit run app.py
   ```
5. **Optional: run the batch pipeline.** This writes CSVs, figures and the report to `outputs/`:
   ```bash
   .venv\Scripts\python.exe -m src.run_assessment
   ```

See [GUIDE.md](GUIDE.md) for the full walkthrough.

## Repository map

| Path | Contents |
|---|---|
| `app.py` | Streamlit dashboard (pages follow the 6 expected outcomes) |
| `src/` | data prep, training, evaluation, inference, all analysis modules |
| `notebooks/` | Colab training: plastic detector, coral (SUIM, Coralscapes v1/v2), fish, plastic v3 (SeaClear), environmental risk module |
| `models/` | trained detector, coral model configs + test metrics, impact models |
| `outputs/` | figures, metrics JSON, predictions CSV, report, evaluation package |
| `data/samples/reef/` | sample reef photos for the demo |
| `REPORT_GUIDE.md` | how to write the FA report, with all numbers, tables, figures and references |
| `FILE_STRUCTURE.md`, `context.md`, `prompt.md` | file map, architecture, development log |

## Team

Plastic detection, fish, coral and dashboard: wizboltez. Environmental risk module (Copernicus + Allen Coral Atlas,
Gulf of Mannar): Anannya Jadhav.
