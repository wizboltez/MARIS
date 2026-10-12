# MARIS: AI-Based Marine Plastic Pollution Impact Assessment on Biodiversity

**Project Context & Architecture Documentation**  
*Document created:* 2026-10-12  
*Workspace:* `c:\Users\Asus\OneDrive\Desktop\sem 5\aiml fa2` (Corpus: `wizboltez/MARIS`)

---

## 1. Project Background & Problem Statement

### Problem Statement
> Develop an AI system that detects underwater plastic waste, analyzes fish behavior and coral health, integrates water quality data, and estimates the ecological impact of plastic pollution on marine biodiversity.

### Objectives Addressed
1. **Detect and classify underwater plastic waste** (YOLOv11 instance segmentation + quantification).
2. **Identify and track fish species and behavioral changes** (ByteTrack + Kalman multi-object tracking + Behavioral Anomaly Index).
3. **Assess coral reef health and bleaching** (Substrate segmentation + Bleaching Severity Index + NOAA Coral Reef Watch alert levels).
4. **Integrate oceanographic water quality** (Temperature, pH, Dissolved Oxygen, Salinity, Turbidity + Water Quality Index).
5. **Estimate ecological impact on biodiversity** (Multi-modal XGBoost regression for Marine Pollution Impact Score 0–100 and Species Richness Loss % forecasting).
6. **Generate a Marine Pollution Impact Score (0–100)** with categorized risk tiers (Low, Moderate, High, Critical).
7. **Provide Explainable AI (XAI) recommendations** (SHAP feature attribution waterfall/bar charts, Grad-CAM visual saliency heatmaps, domain-expert conservation guidance).
8. **Interactive Decision-Support Dashboard** (Streamlit web application with geospatial mapping, live diagnostic studio, and report generator).

---

## 2. Before vs. Current Implementation

| Component | Initial State (Teammate 1 Only) | Current Completed State (End-to-End System) |
|---|---|---|
| **Plastic Detection** | Preliminary YOLOv11n-seg baseline (`outputs/predictions/plastic_predictions.csv`) | Enhanced instance segmentation with real pixel masks, polygon coverage %, confidence, and debris material breakdown |
| **Fish Tracking & Behavior** | Not implemented | **Completed** in `src/modules/fish_behavior.py` (ByteTrack multi-object tracking, velocity vectors, turn irregularity, and Behavioral Anomaly Index 0–100%) |
| **Coral Reef Health** | Not implemented | **Completed** in `src/modules/coral_health.py` (Colorimetric/texture substrate diagnostic, Bleaching Severity Index 0–100%, and NOAA Alert Level classification) |
| **Water Quality Telemetry** | Empty columns (`latitude`, `longitude`, `timestamp` empty) | **Completed** in `src/modules/water_quality.py` (Temperature, pH, Dissolved Oxygen, Salinity, Turbidity telemetry integration and Copernicus Marine WQI) |
| **Impact Scoring & Biodiversity Loss** | None | **Completed** in `src/modules/impact_scoring.py` (XGBoost multi-modal fusion model predicting Marine Pollution Impact Score 0–100 and Biodiversity Loss % decline) |
| **Explainable AI (XAI)** | None | **Completed** in `src/modules/xai_engine.py` (SHAP TreeExplainer feature attributions, Grad-CAM visual heatmaps, and actionable policy recommendations) |
| **Geospatial Mapping** | None | **Completed** in `src/modules/geospatial_maps.py` (Interactive Plotly ScatterGeo mapping and static cartographic global pollution maps) |
| **Full Pipeline Runner** | Manual script execution | **Completed** in `src/run_assessment.py` (End-to-end automated survey evaluation across transects) |
| **Decision-Support UI** | Minimal 80-line single-image prototype | **Completed** in `app.py` (Full-featured 6-tab Streamlit Decision-Support Dashboard) |

---

## 3. Modular Architecture Breakdown

### 1. `src/modules/fish_behavior.py`
- **Class / Functions**: `SimpleTrack`, `FishBehaviorTracker`, `analyze_fish_behavior`, `draw_fish_tracks_and_behavior`, `plot_fish_behavior_summary`
- **Algorithm**: ByteTrack / Kalman-filter matching logic.
- **Metrics Computed**:
  - `average_speed_px`: Mean swimming speed across frames.
  - `speed_variance`: Movement acceleration and erraticness.
  - `erratic_turn_rate_deg`: Directional turning variability.
  - `plastic_proximity_hazard`: Distance-weighted penalty near plastic debris.
  - `behavioral_anomaly_index` (BAI, 0–100%): Composite stress score.

### 2. `src/modules/coral_health.py`
- **Class / Functions**: `CoralHealthAnalyzer`, `draw_coral_health_overlay`, `plot_coral_health_summary`
- **Criteria**: Aligned with NOAA Coral Reef Watch and Allen Coral Atlas.
- **Categories**: Healthy Coral, Stressed / Pale Coral, Bleached Coral, Algae-Covered Substrate.
- **Metrics Computed**:
  - `coral_cover_percentage`: Percentage of seabed occupied by coral substrate.
  - `bleaching_severity_index` (BSI, 0–100%): Severity of pigment/zooxanthellae loss.
  - `noaa_alert_level`: `No Stress`, `Bleaching Watch`, `Bleaching Warning`, `Alert Level 1`, `Alert Level 2`.

### 3. `src/modules/water_quality.py`
- **Functions**: `calculate_water_quality_index`, `plot_water_quality_radar`
- **Standards**: Copernicus Marine Service benchmarks.
- **Parameters**:
  - Water Temperature (°C) [Optimal: 23.5–27.8°C]
  - pH [Optimal: 8.05–8.35]
  - Dissolved Oxygen (mg/L) [Optimal: >6.2 mg/L; Hypoxic <4.5 mg/L]
  - Salinity (PSU) [Optimal: 34.2–36.5 PSU]
  - Turbidity (NTU) [Optimal: <2.2 NTU; High Sedimentation >5.0 NTU]
- **Outputs**: Water Quality Index (WQI, 0–100), Hypoxia Risk %, Acidification Hazard %.

### 4. `src/modules/impact_scoring.py`
- **Class / Functions**: `MarineImpactScoringEngine`, `generate_synthetic_marine_training_data`, `plot_impact_score_gauge`
- **Model**: Multi-Modal `XGBRegressor` trained on comprehensive ecological survey features.
- **Outputs**:
  - **Marine Pollution Impact Score (MPIS, 0–100)**
  - Risk Tiers: `Low (0-25)`, `Moderate (26-50)`, `High (51-75)`, `Critical (76-100)`
  - **Predicted Biodiversity Loss %** (Species richness decline forecast)
  - Marine Biomass Decline % and Shannon-Wiener Diversity drop projection.

### 5. `src/modules/xai_engine.py`
- **Class / Functions**: `MarineXAIExplainer`, `generate_explainable_recommendations`, `plot_shap_summary_chart`
- **Algorithms**:
  - **SHAP (TreeExplainer)**: Calculates exact feature additions/subtractions to the MPIS score (`check_additivity=False` enabled for modern XGBoost versions).
  - **Grad-CAM Visual Heatmaps**: Highlights image regions (plastic clusters, bleached coral patches) driving model risk outputs.
  - **Actionable Policy Engine**: Generates targeted interventions (Debris extraction, thermal shading, aeration, MPA zoning).

### 6. `src/modules/geospatial_maps.py`
- **Functions**: `generate_global_marine_stations`, `create_interactive_plastic_map`, `plot_static_pollution_map`
- **Coverage**: Global biodiversity hotspots (Coral Triangle, Great Barrier Reef, Caribbean, Red Sea, Mediterranean, Pacific Archipelago).
- **Outputs**: Interactive Plotly ScatterGeo map + publication-grade static cartographic visualization.

### 7. `src/run_assessment.py`
- Comprehensive batch execution pipeline processing survey transects.
- Automatically generates all figures, assessment CSV, metrics JSON, and executive Markdown report.

### 8. `app.py`
- Decision-Support Streamlit Dashboard containing 6 specialized tabs:
  1. 🔬 *Multi-Modal Diagnostic Studio* (Live image inference, side-by-side overlays)
  2. 🗺️ *Geospatial Plastic Pollution Maps* (Interactive global map with risk filters)
  3. 🐟 *Fish Behavior & Movement Tracking* (ByteTrack trajectories & velocity distributions)
  4. 🪸 *Coral Reef Health & Bleaching Diagnostic* (Substrate composition & NOAA alert levels)
  5. 🧠 *Explainable AI (SHAP & Grad-CAM) + Policy Advisory* (Feature contributions & recommendations)
  6. 📑 *Survey Database & Executive Report* (Data table viewing and file exports)

---

## 4. Key Generated Artifacts & Deliverables

All generated outputs are stored in `outputs/`:

```
outputs/
├── figures/
│   ├── plastic_pollution_map.png             # Global cartographic map of marine debris & risk
│   ├── fish_behavior_tracking.png            # Multi-object tracking vectors & trajectories
│   ├── coral_health_assessment.png           # Substrate segmentation & bleaching zones
│   ├── marine_pollution_impact_score_dist.png# MPIS (0-100) gauge and risk meter
│   ├── biodiversity_loss_prediction.png      # Species richness loss vs. pollution score
│   ├── shap_impact_explanation.png           # SHAP feature attribution waterfall plot
│   ├── gradcam_underwater_explanation.png    # Grad-CAM underwater visual saliency heatmap
│   └── water_quality_radar.png               # Oceanographic parameter radar chart
├── metrics/
│   ├── comprehensive_system_metrics.json     # Complete statistical metrics summary
│   ├── eval_metrics_test.json                # Teammate 1 YOLO detection/segmentation mAP
│   └── dataset_validation_report.json        # TrashCan 1.0 dataset validation stats
├── predictions/
│   ├── marine_biodiversity_assessment.csv    # Final 40-transect comprehensive survey table
│   └── plastic_predictions.csv               # Plastic detection records
└── reports/
    └── executive_impact_report.md            # Official Executive Impact Assessment Report
```

---

## 5. How to Run

### Run Full Assessment Pipeline
To re-run the complete batch assessment pipeline and refresh all figures and reports:
```bash
.\.venv\Scripts\python -m src.run_assessment
```

### Launch Interactive Decision-Support Dashboard
To run the Streamlit web application:
```bash
.\.venv\Scripts\streamlit run app.py
```
