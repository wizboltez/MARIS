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
from src.modules.fish_tracking import (
    MOBILE_IDS,
    compute_track_metrics,
    draw_trajectories,
    fish_stats_for_frame,
    group_frames_by_video,
    load_tracking_model,
    track_video,
)
from src.modules.ocean_data import load_cache as load_ocean_cache
from src.modules.ocean_data import refresh_cache as refresh_ocean_cache
from src.modules.ocean_data import station_conditions
from src.modules.gradcam import PLASTIC_ID, YOLOGradCAM, overlay_cam
from src.modules.geospatial_maps import (
    build_station_table,
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


def summarise_tracking(all_tracks: pd.DataFrame, videos: dict, yolo_model) -> dict:
    """Save fish_tracks.csv + trajectory figure; return summary numbers for the report."""
    if all_tracks.empty:
        return {"videos": len(videos), "tracks": 0}
    all_tracks.drop(columns=["image_ids", "path_x", "path_y"]).to_csv(PREDICTIONS_DIR / "fish_tracks.csv", index=False)

    usable = all_tracks[all_tracks["n_frames"] >= 3]
    mobile, static = usable[usable["mobile"]], usable[~usable["mobile"]]
    near = mobile["min_plastic_dist_norm"] < 0.25

    # Figure: the video with the most multi-frame mobile tracks
    multi = all_tracks[all_tracks["mobile"] & (all_tracks["n_frames"] >= 2)]
    if not multi.empty:
        best_vid = multi["video_id"].value_counts().index[0]
        vid_tracks = multi[multi["video_id"] == best_vid]
        frame_path = videos[best_vid][int(vid_tracks["last_frame"].max())]
        frame = cv2.imread(str(frame_path))
        res = yolo_model(frame, conf=0.18, verbose=False)[0]
        plastic = [b for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()) if int(c) == PLASTIC_ID]
        cv2.imwrite(str(FIGURES_DIR / "fish_behavior_tracking.png"), draw_trajectories(frame, vid_tracks, plastic))

    def med(df, col):
        v = df[col].dropna()
        return round(float(v.median()), 2) if len(v) else None

    return {
        "videos": len(videos),
        "tracks": int(len(all_tracks)),
        "tracks_3plus_frames": int(len(usable)),
        "mobile_tracks_3plus_frames": int(len(mobile)),
        "mobile_species": mobile["species"].value_counts().to_dict(),
        "median_speed_mobile_px_per_frame": med(mobile, "mean_speed_px"),
        "median_speed_static_px_per_frame_noise_floor": med(static, "mean_speed_px"),
        "median_turn_mobile_deg": med(mobile, "mean_turn_deg"),
        "mobile_tracks_near_plastic": int(near.sum()),
        "median_speed_near_plastic": med(mobile[near], "mean_speed_px"),
        "median_speed_away_from_plastic": med(mobile[~near], "mean_speed_px"),
    }


def save_gradcam_grid(yolo_model, gradcam, df_plastic_paths, n: int = 6) -> list[float]:
    """Real Grad-CAM for trash_plastic on up to n frames with plastic detections.

    Returns, per frame, the share of Grad-CAM heat that falls inside the plastic boxes
    (a quick sanity check that the explanation points at the detected object).
    """
    tiles, heat_in_box = [], []
    for path in df_plastic_paths[:n]:
        img = cv2.imread(str(path))
        res = yolo_model(img, conf=0.18, verbose=False)[0]
        cam = gradcam.compute(img, class_id=PLASTIC_ID)["cam"]
        overlay = overlay_cam(img, cam)
        inside = np.zeros(img.shape[:2], dtype=bool)
        for box, cid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            if int(cid) == PLASTIC_ID:
                x1, y1, x2, y2 = box.astype(int)
                inside[y1:y2, x1:x2] = True
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
        if cam.sum() > 0:
            heat_in_box.append(float(cam[inside].sum() / cam.sum()))
        tiles.append(cv2.resize(overlay, (480, 300)))
    if tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        cv2.imwrite(str(FIGURES_DIR / "gradcam_underwater_explanation.png"), grid)
    return heat_in_box


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
    gradcam = YOLOGradCAM(yolo_model)
    print("      ✓ YOLOv11 Instance Segmentation Model Loaded")
    print("      ✓ Coral Substrate & Bleaching Diagnostic Engine Ready")
    print("      ✓ Multi-Modal XGBoost Marine Impact Engine Ready")
    print("      ✓ SHAP TreeExplainer & Grad-CAM Visualizer Ready")

    # 2. Collect sample images
    # Half the sample has labelled plastic, half doesn't (the first N by filename had none).
    image_files = select_survey_images(n=sample_limit, plastic_fraction=0.5)
    test_lbl_dir = PROJECT_ROOT / "data" / "processed" / "test" / "labels"

    # Multi-frame fish tracking (ByteTrack) over every test video
    print("\n      Tracking fish across test video sequences (ByteTrack)...")
    tracking_model = load_tracking_model(BEST_MODEL_PATH)
    all_test_images = sorted((PROJECT_ROOT / "data" / "processed" / "test" / "images").glob("*.jpg"))
    videos = group_frames_by_video(all_test_images)
    tracks_by_video, track_tables = {}, []
    for vid, frames in videos.items():
        tracks = compute_track_metrics(track_video(tracking_model, frames))
        tracks_by_video[vid] = tracks
        if not tracks.empty:
            track_tables.append(tracks.assign(video_id=vid))
    all_tracks = pd.concat(track_tables, ignore_index=True) if track_tables else pd.DataFrame()
    print(f"      ✓ {len(videos)} videos tracked, {len(all_tracks)} animal tracks")
    print(f"\n[2/6] Processing {len(image_files)} Underwater Survey Transects...")

    results_table = []
    shap_rows = []
    station_names = list(PRESET_STATIONS.keys())
    ocean_cache = load_ocean_cache()
    if not ocean_cache.get("stations"):
        print("      Fetching NOAA Coral Reef Watch data for stations...")
        try:
            ocean_cache = refresh_ocean_cache(PRESET_STATIONS)
        except Exception as exc:
            print(f"      ! Ocean data fetch failed ({exc}); using station presets")

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
        video_id = img_path.stem.split("_frame")[0].replace("vid_", "")
        fish_stats = fish_stats_for_frame(tracks_by_video.get(video_id), img_path.stem, plastic_boxes, (h, w))
        if fish_stats is None:
            fish_stats = analyze_fish_behavior(fish_detections, plastic_boxes, image_shape=(h, w))

        # C. Water Quality Telemetry
        # TrashCan has no geotags, so each image is assigned a station in rotation;
        # the station's water values are real where NOAA/Copernicus data exists.
        station_name = station_names[idx % len(station_names)]
        preset_stn = PRESET_STATIONS[station_name]
        conditions = station_conditions(station_name, preset_stn, ocean_cache)
        wq_stats = calculate_water_quality_index(
            temperature_c=conditions["temperature_c"],
            ph=conditions["ph"],
            dissolved_oxygen_mg_l=conditions["dissolved_oxygen_mg_l"],
            salinity_psu=conditions["salinity_psu"],
            turbidity_ntu=conditions["turbidity_ntu"],
        )

        # D. Coral Health Assessment
        coral_stats = coral_analyzer.analyze_image(
            img_bgr,
            temperature_c=wq_stats["temperature_c"],
            turbidity_ntu=wq_stats["turbidity_ntu"],
            habitat="non_reef",  # TrashCan = deep-sea ROV footage (JAMSTEC), no coral reef
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
            "station_assignment": "assigned in rotation (TrashCan has no geotags)",
            "temperature_source": conditions["provenance"]["temperature_c"],
            "noaa_dhw": conditions["noaa_dhw"],
            "noaa_bleaching_alert": conditions["noaa_bleaching_alert"],
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
            "fish_measurement": fish_stats.get("measurement", "none"),
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
            # 2. Coral Overlay
            coral_viz = draw_coral_health_overlay(img_bgr, coral_stats)
            cv2.imwrite(str(FIGURES_DIR / "coral_health_assessment.png"), coral_viz)

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

    # Grad-CAM grid over frames where plastic was detected
    gradcam_scores = save_gradcam_grid(yolo_model, gradcam, df_plastic_paths=[
        PROJECT_ROOT / r["image_path"] for r in results_table if r["plastic_count"] > 0
    ])

    # Fish tracking outputs: per-track CSV, trajectory figure, summary
    tracking_summary = summarise_tracking(all_tracks, videos, yolo_model)

    # 3. Save comprehensive CSV
    print("\n[3/6] Exporting Multi-Modal Assessment CSV...")
    df_results = pd.DataFrame(results_table)
    out_csv = PREDICTIONS_DIR / "marine_biodiversity_assessment.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"      ✓ Assessment CSV saved to: {out_csv} ({len(df_results)} rows)")

    # 4. Generate Geospatial Pollution & Risk Maps
    print("\n[4/6] Generating Geospatial Plastic Pollution Maps...")
    df_stations = build_station_table(df_results, PRESET_STATIONS, ocean_cache)
    df_stations.to_csv(PREDICTIONS_DIR / "station_summary.csv", index=False)
    map_path = FIGURES_DIR / "plastic_pollution_map.png"
    plot_static_pollution_map(df_stations, save_path=str(map_path))
    print(f"      ✓ Static Geospatial Map saved to: {map_path}")

    # 5. Generate Biodiversity Loss Prediction Chart
    print("\n[5/6] Generating Biodiversity Loss Correlation Visuals...")
    fig_bio, ax_bio = plt.subplots(figsize=(8, 5))
    scatter = ax_bio.scatter(
        df_results["marine_pollution_impact_score"],
        df_results["predicted_biodiversity_loss_pct"],
        c=df_results["plastic_coverage_percentage"],
        cmap="plasma",
        s=70,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
    )
    cbar = fig_bio.colorbar(scatter, ax=ax_bio)
    cbar.set_label("Plastic coverage (% of image)", fontsize=9)
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

    any_copernicus = any("copernicus" in v for v in ocean_cache.get("stations", {}).values())
    copernicus_status = (
        "REAL - Copernicus Marine model values" if any_copernicus
        else "PRESET - station values (Copernicus Marine loader is implemented but needs a free account: `copernicusmarine login`)"
    )
    seg_dir = PROJECT_ROOT / "models" / "coral_segmenter"
    coral_test = json.loads((seg_dir / "suim_test_metrics.json").read_text()) if (seg_dir / "suim_test_metrics.json").exists() else None
    coral_domain = json.loads((seg_dir / "domain_check_trashcan.json").read_text()) if (seg_dir / "domain_check_trashcan.json").exists() else None
    if coral_test:
        coral_status = (
            f"TRAINED - U-Net ({coral_test['model'].split()[-1]}) on SUIM; official SUIM TEST reef IoU {coral_test['reef_iou']:.3f} "
            f"(precision {coral_test['reef_precision']:.2f}, recall {coral_test['reef_recall']:.2f}). Not applied to TrashCan "
            "(deep-sea, no reef; the U-Net is out of domain there). Bleaching itself is still a colour rule inside the reef mask "
            "(SUIM has no bleaching labels)."
        )
        coral_section = (
            f"U-Net with a {coral_test['model'].split()[-1]} encoder trained on SUIM (8 classes). Results on the official SUIM TEST "
            f"split ({coral_test['n_test_images']} images, never used for training or model selection):\n\n"
            "| Metric | Value |\n|---|---|\n"
            f"| Reef IoU (reefs & invertebrates) | **{coral_test['reef_iou']:.3f}** (gate {coral_test['gate_reef_iou']}) |\n"
            f"| Reef precision / recall | {coral_test['reef_precision']:.3f} / {coral_test['reef_recall']:.3f} |\n"
            f"| Mean IoU (8 classes) | {coral_test['mean_iou']:.3f} |\n"
            f"| Pixel accuracy | {coral_test['pixel_accuracy']:.3f} |\n\n"
            + (
                f"**Domain check on deep-sea footage:** on the {coral_domain['n_frames']} TrashCan test frames (no reef), the U-Net "
                f"alone marked {coral_domain['frames_reef_ge_2pct_unet_only']} frames as >=2% reef (mud, sediment and rock). "
                "It is therefore only trusted together with the colour/texture coral check in auto mode, which reduces this to "
                f"{coral_domain['frames_coral_declared_auto_mode_unet_AND_colour_check']} frames."
                if coral_domain else ""
            )
        )
    else:
        coral_status = "NOT ASSESSED for TrashCan (deep-sea ROV footage, no reef). For other photos: colour/texture heuristic with a coral-presence check"
        coral_section = "No trained coral model found in models/coral_segmenter/; the colour/texture heuristic is used."

    fit_report = getattr(impact_engine, "fit_report", None) or (
        json.loads((PROJECT_ROOT / "models" / "impact_model" / "fit_report.json").read_text())
        if (PROJECT_ROOT / "models" / "impact_model" / "fit_report.json").exists() else {}
    )
    ts = tracking_summary
    tracking_text = (
        f"{ts.get('videos', 0)} test videos tracked; {ts.get('tracks', 0)} animal tracks, "
        f"{ts.get('mobile_tracks_3plus_frames', 0)} mobile (fish/eel/crab) tracks of 3+ frames "
        f"({ts.get('mobile_species', {})}). Median mobile speed {ts.get('median_speed_mobile_px_per_frame')} px/frame vs. "
        f"{ts.get('median_speed_static_px_per_frame_noise_floor')} px/frame for near-static animals (measurement noise floor). "
        f"Mobile tracks passing within 25% of the image diagonal of plastic: {ts.get('mobile_tracks_near_plastic', 0)} - "
        "too few to test whether plastic changes behaviour."
    )
    station_lines = "\n".join(
        f"| {r.region} | {r.noaa_sst_c} | {r.noaa_dhw} | {r.noaa_bleaching_alert} | {r.n_survey_images} | {r.marine_pollution_impact_score} |"
        for r in df_stations.itertuples()
    )

    data_provenance = {
        "plastic_detection": "REAL - YOLO11-seg trained on TrashCan 1.0, run on real test images",
        "plastic_detector_quality": "Weak - trash_plastic precision ~0.37, recall ~0.31 on leak-free images (see HANDOFF.md)",
        "fish_behavior": "REAL, SMALL SAMPLE - ByteTrack over test video frames with camera-motion compensation; few mobile animals are tracked, so plastic-related behaviour change cannot be concluded",
        "coral_health": coral_status,
        "water_temperature_and_bleaching_alert": "REAL - NOAA Coral Reef Watch 5km daily satellite SST, Degree Heating Weeks and Bleaching Alert Area at each station",
        "ph_oxygen_salinity_turbidity": copernicus_status,
        "station_assignment": "ASSUMED - TrashCan images have no geotags, so images are assigned to stations in rotation",
        "impact_score_and_biodiversity_loss": "EXPERT FORMULA - XGBoost reproduces a documented hand-specified formula (synthetic training data); weights are assumptions, see src/modules/impact_scoring.py",
        "shap": "REAL - SHAP TreeExplainer on the XGBoost model (explains the model, which is trained on synthetic data)",
        "gradcam": "REAL - gradient-based Grad-CAM on the YOLO neck layers for the trash_plastic logit",
        "geospatial_map": "REAL station positions + NOAA CRW values; per-station scores average the images assigned to each station",
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
        "fish_tracking": tracking_summary,
        "coral_segmenter_suim_test": coral_test,
        "coral_segmenter_deep_sea_domain_check": coral_domain,
        "shap_mean_abs_contribution": shap_summary["mean_abs_shap"].to_dict(),
        "gradcam_heat_inside_plastic_boxes": {
            "frames": len(gradcam_scores),
            "mean_share": round(float(np.mean(gradcam_scores)), 3) if gradcam_scores else None,
        },
        "ai_model_architectures": {
            "plastic_detector": "YOLOv11-seg (Instance Segmentation, Ultralytics)",
            "fish_tracking": "ByteTrack (Ultralytics) on YOLO11-seg detections + phase-correlation camera-motion compensation",
            "coral_diagnostic": "Colour & texture heuristic (NOAA alert-level naming)",
            "water_telemetry": "Water Quality Index from preset station values",
            "impact_fusion": "XGBoost Regressor (160 estimators) trained on synthetic data",
            "biodiversity_loss": "XGBoost Regressor (140 estimators) trained on synthetic data",
            "explainable_ai": "SHAP (TreeExplainer) + Grad-CAM on YOLO11-seg neck layers",
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

> **Read this first.** Real: plastic detection, Grad-CAM, fish tracking, SUIM reef segmentation, NOAA satellite SST and
> bleaching alerts. Assumed: the image-to-station assignment (TrashCan has no geotags), pH / oxygen / salinity /
> turbidity unless Copernicus is configured, and the expert formula behind the impact score and biodiversity
> loss (Section 2, item 5). Treat the scores as a demonstration of the pipeline, not as ecological findings.

---

## 1. Key Results

| Metric | Value | Reference |
|---|---|---|
| **Mean Marine Pollution Impact Score (MPIS)** | **{mean_mpis:.1f} / 100** | Tiers: Low < 25, Moderate < 50, High < 75, Critical ≥ 75 |
| **Mean Predicted Biodiversity Loss** | **{mean_biodiv:.1f}%** | Model output (synthetic training data) |
| **Plastic Items Detected** | **{plastic_detections_total}** items; found in {plastic_images_hit} of {gt_plastic_images} images with labelled plastic, plus {plastic_false_alarm_images} false-alarm images | Detector recall for plastic is ~0.31 |
| **Coral bleaching** | Not assessed | TrashCan frames are deep-sea (no reef); see NOAA station alerts below |
| **Mean Fish Behavioral Anomaly Index** | **{df_results['fish_behavior_anomaly_index'].mean():.1f} / 100** | {int((df_results['fish_measurement'] == 'multi-frame ByteTrack').sum())} of {total_transects} images have tracked moving animals |
| **Transects at High / Critical Risk** | **{high_risk_count} of {total_transects} ({high_risk_count/total_transects*100:.1f}%)** | MPIS ≥ 50 |

---

## 2. Pipeline

1. **Plastic detection & segmentation**: YOLO11-seg trained on TrashCan 1.0; coverage % from real pixel masks.
2. **Fish detection & tracking**: the same detector + ByteTrack across consecutive video frames, with ROV camera motion removed by phase correlation; speed, turning and distance to plastic per track.
3. **Coral health**: U-Net reef segmentation trained on SUIM (see Section 4) plus a colour rule for bleaching inside the reef area; skipped for deep-sea TrashCan frames.
4. **Water quality**: Water Quality Index from temperature (NOAA CRW satellite), pH, dissolved oxygen, salinity and turbidity.
5. **Impact fusion**: XGBoost regressors predict the MPIS (0–100) and biodiversity loss % (expert-formula targets; held-out R² {fit_report.get('mpis_r2', 'n/a')} for MPIS).
6. **Explainability**: SHAP TreeExplainer attributions for every transect, and Grad-CAM on the detector.
   On {len(gradcam_scores)} frames with plastic detections, on average {(np.mean(gradcam_scores) if gradcam_scores else 0) * 100:.0f}% of the
   Grad-CAM heat falls inside the plastic boxes (which cover only a few % of the image).

---

## 3. Fish Tracking

{tracking_text}

## 4. Coral Reef Segmentation Model

{coral_section}

## 5. Survey Stations (NOAA Coral Reef Watch, real satellite data)

| Station | SST (°C) | DHW (°C-weeks) | Bleaching alert | Images assigned | Mean MPIS |
|---|---|---|---|---|---|
{station_lines}

---

## 6. SHAP Attribution Summary (computed over all {total_transects} transects)

Most influential feature on average: **`{top_feature}`**. SHAP values are relative to the model's
average synthetic site, so a feature "lowers" the score when these images have less of it than that
average (e.g. most images have little or no plastic).

| Feature | Mean abs. SHAP (score points) | Mean SHAP | Direction |
|---|---|---|---|
{shap_lines}

---

## 7. Data Provenance

| Component | Status |
|---|---|
{provenance_lines}

---

## 8. Standard Decision-Support Recommendations

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
