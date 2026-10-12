"""Complete End-to-End Marine Biodiversity & Plastic Pollution Assessment Pipeline.

Executes the entire multi-modal AI assessment pipeline:
1. Underwater Plastic Waste Detection & Quantification (YOLOv11)
2. Fish Species Detection, Tracking & Behavioral Change Analysis
3. Coral Reef Health & Bleaching Diagnostic Assessment
4. Water Quality Integration (Temp, pH, DO, Salinity, Turbidity)
5. Multi-modal Marine Pollution Impact Scoring (0-100) via XGBoost
6. Biodiversity Loss Prediction (% species richness loss)
7. Explainable AI (SHAP Feature Attribution & Grad-CAM Heatmaps)
8. Saves final CSV, figures, metrics, and executive environmental report.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.config import BEST_MODEL_PATH, CLASS_NAMES, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, PROJECT_ROOT
from src.modules.coral_health import CoralHealthAnalyzer, draw_coral_health_overlay, plot_coral_health_summary
from src.modules.fish_behavior import (
    analyze_fish_behavior,
    draw_fish_tracks_and_behavior,
    plot_fish_behavior_summary,
)
from src.modules.geospatial_maps import (
    generate_global_marine_stations,
    plot_static_pollution_map,
)
from src.modules.impact_scoring import (
    MarineImpactScoringEngine,
    plot_impact_score_gauge,
)
from src.modules.water_quality import (
    PRESET_STATIONS,
    calculate_water_quality_index,
    plot_water_quality_radar,
)
from src.modules.xai_engine import (
    MarineXAIExplainer,
    generate_explainable_recommendations,
    plot_shap_summary_chart,
)
from src.utils.quantify import quantify
from src.utils.survey_sampling import has_plastic_label, select_survey_images
from src.utils.visualization import draw_detections

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def run_complete_assessment(sample_limit: int = 40):
    print("=" * 70)
    print("STARTING COMPLETE MARINE BIODIVERSITY & PLASTIC IMPACT ASSESSMENT")
    print("=" * 70)

    # 1. Initialize models
    print("\n[1/6] Loading AI Models & Initializing Engines...")
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained YOLO model not found at {BEST_MODEL_PATH}")

    yolo_model = YOLO(str(BEST_MODEL_PATH))
    coral_analyzer = CoralHealthAnalyzer()
    impact_engine = MarineImpactScoringEngine()
    xai_explainer = MarineXAIExplainer(impact_engine.mpis_model, impact_engine.training_df)
    print("      ✓ YOLOv11 Instance Segmentation Model Loaded")
    print("      ✓ Coral Substrate & Bleaching Diagnostic Engine Ready")
    print("      ✓ Multi-Modal XGBoost Marine Impact Engine Ready")
    print("      ✓ SHAP TreeExplainer & Grad-CAM Visualizer Ready")

    # 2. Collect sample images
    # Half the sample has labelled plastic, half doesn't (the first N by filename had none).
    image_files = select_survey_images(n=sample_limit, plastic_fraction=0.5)
    test_lbl_dir = PROJECT_ROOT / "data" / "processed" / "test" / "labels"
    print(f"\n[2/6] Processing {len(image_files)} Underwater Survey Transects...")

    results_table = []
    shap_rows = []
    station_names = list(PRESET_STATIONS.keys())

    saved_sample_visuals = False

    for idx, img_path in enumerate(image_files):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]

        # A. YOLO Inference
        res = yolo_model(img_bgr, conf=0.18, verbose=False)[0]
        plastic_stats = quantify(res)

        # Extract plastic and fish boxes
        plastic_boxes = []
        fish_detections = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            for box, cid, conf in zip(boxes_xyxy, cls_ids, confs):
                cname = CLASS_NAMES[cid]
                if cname == "trash_plastic":
                    plastic_boxes.append(box)
                elif cname.startswith("animal_"):
                    fish_detections.append(
                        {
                            "bbox": box.tolist(),
                            "cls_name": cname.replace("animal_", ""),
                            "conf": float(conf),
                        }
                    )

        # B. Fish Behavior Analysis
        fish_stats = analyze_fish_behavior(fish_detections, plastic_boxes, image_shape=(h, w))

        # C. Water Quality Telemetry
        preset_stn = PRESET_STATIONS[station_names[idx % len(station_names)]]
        wq_stats = calculate_water_quality_index(
            temperature_c=preset_stn["temperature_c"],
            ph=preset_stn["ph"],
            dissolved_oxygen_mg_l=preset_stn["dissolved_oxygen_mg_l"],
            salinity_psu=preset_stn["salinity_psu"],
            turbidity_ntu=preset_stn["turbidity_ntu"],
        )

        # D. Coral Health Assessment
        coral_stats = coral_analyzer.analyze_image(
            img_bgr,
            temperature_c=wq_stats["temperature_c"],
            turbidity_ntu=wq_stats["turbidity_ntu"],
        )

        # E. Marine Pollution Impact Score & Biodiversity Loss Prediction
        impact_prediction = impact_engine.predict(plastic_stats, fish_stats, coral_stats, wq_stats)

        # F. Explainable AI SHAP Attribution
        shap_res = xai_explainer.explain_prediction(impact_prediction["feature_df"])
        shap_rows.append({c["feature"]: c["shap_value"] for c in shap_res["contributions"]})

        # Store tabular record
        record = {
            "image_id": img_path.stem,
            "image_path": str(img_path.relative_to(PROJECT_ROOT)),
            "ground_truth_has_plastic": has_plastic_label(test_lbl_dir / f"{img_path.stem}.txt"),
            "station_region": preset_stn["location"],
            "latitude": preset_stn["latitude"],
            "longitude": preset_stn["longitude"],
            # Plastic metrics
            "plastic_count": plastic_stats["plastic_count"],
            "debris_count": plastic_stats["debris_count"],
            "plastic_area_px": plastic_stats["plastic_area"],
            "plastic_coverage_percentage": plastic_stats["plastic_coverage_percentage"],
            "plastic_avg_confidence": plastic_stats["average_confidence"],
            # Fish metrics
            "fish_count": fish_stats["fish_count"],
            "fish_avg_speed_px": fish_stats["average_speed_px"],
            "fish_behavior_anomaly_index": fish_stats["behavioral_anomaly_index"],
            "fish_behavior_state": fish_stats["behavior_state"],
            # Coral metrics
            "coral_cover_percentage": coral_stats["coral_cover_percentage"],
            "coral_bleaching_severity_index": coral_stats["bleaching_severity_index"],
            "coral_health_status": coral_stats["health_status"],
            "noaa_alert_level": coral_stats["noaa_alert_level"],
            # Water Quality metrics
            "temperature_c": wq_stats["temperature_c"],
            "ph": wq_stats["ph"],
            "dissolved_oxygen_mg_l": wq_stats["dissolved_oxygen_mg_l"],
            "salinity_psu": wq_stats["salinity_psu"],
            "turbidity_ntu": wq_stats["turbidity_ntu"],
            "water_quality_index": wq_stats["water_quality_index"],
            # Multi-Modal Impact Score & Biodiversity Loss
            "marine_pollution_impact_score": impact_prediction["marine_pollution_impact_score"],
            "risk_tier": impact_prediction["risk_tier"],
            "predicted_biodiversity_loss_pct": impact_prediction["predicted_biodiversity_loss_pct"],
            "predicted_biomass_decline_pct": impact_prediction["predicted_biomass_decline_pct"],
            "top_shap_driver": shap_res["contributions"][0]["feature"],
            "top_shap_driver_impact": round(shap_res["contributions"][0]["shap_value"], 2),
        }
        results_table.append(record)

        # Save representative visual artifact images on the first plastic/active frame
        if not saved_sample_visuals and (plastic_stats["plastic_count"] > 0 or idx == 10):
            print(f"      -> Generating representative XAI & Diagnostic visual for {img_path.name}...")
            # 1. Fish Tracking overlay
            fish_viz = draw_fish_tracks_and_behavior(img_bgr, fish_detections, fish_stats, plastic_boxes)
            cv2.imwrite(str(FIGURES_DIR / "fish_behavior_tracking.png"), fish_viz)

            # 2. Coral Overlay
            coral_viz = draw_coral_health_overlay(img_bgr, coral_stats)
            cv2.imwrite(str(FIGURES_DIR / "coral_health_assessment.png"), coral_viz)

            # 3. Grad-CAM visual saliency
            gradcam_viz = xai_explainer.generate_gradcam_heatmap(img_bgr, plastic_boxes, coral_stats.get("masks"))
            cv2.imwrite(str(FIGURES_DIR / "gradcam_underwater_explanation.png"), gradcam_viz)

            # 4. SHAP summary plot
            plot_shap_summary_chart(shap_res, save_path=str(FIGURES_DIR / "shap_impact_explanation.png"))

            # 5. Score gauge
            plot_impact_score_gauge(
                impact_prediction["marine_pollution_impact_score"],
                impact_prediction["risk_tier"],
                save_path=str(FIGURES_DIR / "marine_pollution_impact_score_dist.png"),
            )

            # 6. Water quality radar
            plot_water_quality_radar(wq_stats, save_path=str(FIGURES_DIR / "water_quality_radar.png"))

            saved_sample_visuals = True

    # 3. Save comprehensive CSV
    print("\n[3/6] Exporting Multi-Modal Assessment CSV...")
    df_results = pd.DataFrame(results_table)
    out_csv = PREDICTIONS_DIR / "marine_biodiversity_assessment.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"      ✓ Assessment CSV saved to: {out_csv} ({len(df_results)} rows)")

    # 4. Generate Geospatial Pollution & Risk Maps
    print("\n[4/6] Generating Geospatial Plastic Pollution Maps...")
    df_stations = generate_global_marine_stations(n_stations=75, random_seed=42)
    map_path = FIGURES_DIR / "plastic_pollution_map.png"
    plot_static_pollution_map(df_stations, save_path=str(map_path))
    print(f"      ✓ Static Geospatial Map saved to: {map_path}")

    # 5. Generate Biodiversity Loss Prediction Chart
    print("\n[5/6] Generating Biodiversity Loss Correlation Visuals...")
    fig_bio, ax_bio = plt.subplots(figsize=(8, 5))
    scatter = ax_bio.scatter(
        df_results["marine_pollution_impact_score"],
        df_results["predicted_biodiversity_loss_pct"],
        c=df_results["coral_bleaching_severity_index"],
        cmap="plasma",
        s=70,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )
    cbar = fig_bio.colorbar(scatter, ax=ax_bio)
    cbar.set_label("Coral Bleaching Severity Index (BSI)", fontsize=9)
    ax_bio.set_xlabel("Marine Pollution Impact Score (0 - 100)", fontsize=10)
    ax_bio.set_ylabel("Predicted Species Richness Loss (%)", fontsize=10)
    ax_bio.set_title("Multi-Modal AI Biodiversity Loss Prediction vs. Pollution Impact", fontsize=11, fontweight="bold")
    ax_bio.grid(True, linestyle="--", alpha=0.5)
    fig_bio.tight_layout()
    fig_bio.savefig(str(FIGURES_DIR / "biodiversity_loss_prediction.png"), dpi=150)
    plt.close(fig_bio)
    print(f"      ✓ Biodiversity Loss Chart saved to: {FIGURES_DIR / 'biodiversity_loss_prediction.png'}")

    # 6. Compile System Metrics & Executive Report
    print("\n[6/6] Compiling Executive Environmental Impact Report...")
    mean_mpis = float(df_results["marine_pollution_impact_score"].mean())
    mean_biodiv = float(df_results["predicted_biodiversity_loss_pct"].mean())
    plastic_detections_total = int(df_results["plastic_count"].sum())
    total_transects = len(df_results)

    high_risk_count = int((df_results["marine_pollution_impact_score"] >= 50.0).sum())
    gt_plastic_images = int(df_results["ground_truth_has_plastic"].sum())
    detected = df_results["plastic_count"] > 0
    gt = df_results["ground_truth_has_plastic"].astype(bool)
    detected_plastic_images = int(detected.sum())
    plastic_images_hit = int((detected & gt).sum())
    plastic_false_alarm_images = int((detected & ~gt).sum())

    # SHAP summary across all transects (computed, not hand-written)
    df_shap = pd.DataFrame(shap_rows)
    shap_summary = (
        pd.DataFrame({"mean_shap": df_shap.mean(), "mean_abs_shap": df_shap.abs().mean()})
        .sort_values("mean_abs_shap", ascending=False)
        .round(2)
    )
    shap_lines = "\n".join(
        f"| `{feat}` | {row.mean_abs_shap:.2f} | {row.mean_shap:+.2f} | "
        f"{'raises score' if row.mean_shap > 0 else 'lowers score'} on average |"
        for feat, row in shap_summary.head(8).iterrows()
    )
    top_feature = shap_summary.index[0]

    data_provenance = {
        "plastic_detection": "REAL - YOLO11-seg trained on TrashCan 1.0, run on real test images",
        "plastic_detector_quality": "Weak - trash_plastic precision ~0.37, recall ~0.31 on leak-free images (see HANDOFF.md)",
        "fish_behavior": "SIMULATED - fish boxes are real detections, but speed/turn values come from box size/position in a single image (no multi-frame tracking)",
        "coral_health": "HEURISTIC - colour/texture rule, not a trained model; TrashCan images are deep-sea ROV footage, mostly without coral",
        "water_quality": "SIMULATED - preset station values assigned to images in rotation, not measured for these images",
        "impact_score_and_biodiversity_loss": "SYNTHETIC TRAINING DATA - XGBoost learned a hand-written formula from generated data",
        "shap": "REAL - SHAP TreeExplainer on the XGBoost model (explains the model, which is trained on synthetic data)",
        "saliency_heatmap": "APPROXIMATE - blurred detection boxes, not gradient-based Grad-CAM",
        "geospatial_map": "SIMULATED - randomly generated stations around real reef regions",
    }

    system_metrics = {
        "assessment_summary": {
            "total_survey_transects": total_transects,
            "transects_with_ground_truth_plastic": gt_plastic_images,
            "transects_with_detected_plastic": detected_plastic_images,
            "labelled_plastic_transects_detected": plastic_images_hit,
            "plastic_false_alarm_transects": plastic_false_alarm_images,
            "total_plastic_items_detected": plastic_detections_total,
            "mean_plastic_coverage_pct": round(float(df_results["plastic_coverage_percentage"].mean()), 2),
            "mean_fish_behavioral_anomaly_index": round(float(df_results["fish_behavior_anomaly_index"].mean()), 2),
            "mean_coral_bleaching_severity_index": round(float(df_results["coral_bleaching_severity_index"].mean()), 2),
            "mean_water_quality_index": round(float(df_results["water_quality_index"].mean()), 2),
            "mean_marine_pollution_impact_score": round(mean_mpis, 2),
            "mean_predicted_biodiversity_loss_pct": round(mean_biodiv, 2),
            "transects_in_critical_or_high_risk": high_risk_count,
            "percentage_at_high_or_critical_risk": round((high_risk_count / total_transects) * 100.0, 1),
        },
        "shap_mean_abs_contribution": shap_summary["mean_abs_shap"].to_dict(),
        "ai_model_architectures": {
            "plastic_detector": "YOLOv11-seg (Instance Segmentation, Ultralytics)",
            "fish_tracking": "IoU track matcher (ByteTrack-style); behaviour metrics simulated from single frames",
            "coral_diagnostic": "Colour & texture heuristic (NOAA alert-level naming)",
            "water_telemetry": "Water Quality Index from preset station values",
            "impact_fusion": "XGBoost Regressor (160 estimators) trained on synthetic data",
            "biodiversity_loss": "XGBoost Regressor (140 estimators) trained on synthetic data",
            "explainable_ai": "SHAP (TreeExplainer) + approximate detection saliency overlay",
        },
        "data_provenance": data_provenance,
    }

    metrics_json_path = METRICS_DIR / "comprehensive_system_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(system_metrics, f, indent=2)
    print(f"      ✓ System Metrics JSON saved to: {metrics_json_path}")

    provenance_lines = "\n".join(f"| {k.replace('_', ' ').title()} | {v} |" for k, v in data_provenance.items())

    # Generate Executive Impact Report Markdown
    report_md = f"""# Executive Environmental Impact Assessment: Marine Plastic Pollution on Biodiversity

**System**: MARIS (Marine AI Risk and Impact System)
**Evaluation Scope**: {total_transects} TrashCan 1.0 test images ({gt_plastic_images} with labelled plastic, {total_transects - gt_plastic_images} without)
**Core Objective**: Detect underwater plastic waste, track fish behavioral changes, evaluate coral bleaching, integrate water quality telemetry, and forecast ecological biodiversity loss.

> **Prototype notice.** Only plastic detection runs on real data. Water quality, fish behaviour,
> the impact-score model and the map use simulated inputs (see Section 4). Treat the scores
> below as a demonstration of the pipeline, not as ecological findings.

---

## 1. Key Results

| Metric | Value | Reference |
|---|---|---|
| **Mean Marine Pollution Impact Score (MPIS)** | **{mean_mpis:.1f} / 100** | Tiers: Low < 25, Moderate < 50, High < 75, Critical ≥ 75 |
| **Mean Predicted Biodiversity Loss** | **{mean_biodiv:.1f}%** | Model output (synthetic training data) |
| **Plastic Items Detected** | **{plastic_detections_total}** items; found in {plastic_images_hit} of {gt_plastic_images} images with labelled plastic, plus {plastic_false_alarm_images} false-alarm images | Detector recall for plastic is ~0.31 |
| **Mean Coral Bleaching Severity (BSI)** | **{df_results['coral_bleaching_severity_index'].mean():.1f} / 100** | Colour heuristic |
| **Mean Fish Behavioral Anomaly Index** | **{df_results['fish_behavior_anomaly_index'].mean():.1f} / 100** | 20 = baseline when no fish detected |
| **Transects at High / Critical Risk** | **{high_risk_count} of {total_transects} ({high_risk_count/total_transects*100:.1f}%)** | MPIS ≥ 50 |

---

## 2. Pipeline

1. **Plastic detection & segmentation**: YOLO11-seg trained on TrashCan 1.0; coverage % from real pixel masks.
2. **Fish detection & behaviour**: fish boxes from the same detector; behaviour index from position/size heuristics (single frame).
3. **Coral health**: colour/texture heuristic producing a Bleaching Severity Index and NOAA-style alert label.
4. **Water quality**: Water Quality Index from temperature, pH, dissolved oxygen, salinity and turbidity.
5. **Impact fusion**: XGBoost regressors predict the MPIS (0–100) and biodiversity loss %.
6. **Explainability**: SHAP TreeExplainer attributions for every transect.

---

## 3. SHAP Attribution Summary (computed over all {total_transects} transects)

Most influential feature on average: **`{top_feature}`**.

| Feature | Mean abs. SHAP (score points) | Mean SHAP | Direction |
|---|---|---|---|
{shap_lines}

---

## 4. Data Provenance

| Component | Status |
|---|---|
{provenance_lines}

---

## 5. Standard Decision-Support Recommendations

Rule-based guidance templates, triggered by score thresholds:

1. **Debris Extraction**: prioritise cleanup where plastic coverage exceeds 5%.
2. **Thermal Stress Response**: shading and monitoring for reefs at NOAA Alert Level 1 or above.
3. **Runoff Containment**: sediment barriers where turbidity exceeds 5 NTU or oxygen is low.
4. **Zoning**: protective restrictions where MPIS exceeds 50.

---
*Generated by the MARIS assessment pipeline (`python -m src.run_assessment`).*
"""
    report_file = REPORTS_DIR / "executive_impact_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"      ✓ Executive Report saved to: {report_file}")

    print("\n" + "=" * 70)
    print("ASSESSMENT COMPLETE. See the report's Data Provenance section for which parts are simulated.")
    print("=" * 70)


if __name__ == "__main__":
    run_complete_assessment()
