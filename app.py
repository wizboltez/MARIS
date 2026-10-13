"""MARIS Decision-Support Dashboard: Marine Plastic Pollution & Biodiversity Impact.

University AI/ML System: AI-Based Marine Plastic Pollution Impact Assessment on Biodiversity.
Provides an interactive multi-modal decision-support dashboard featuring:
- Underwater plastic debris detection & segmentation (YOLOv11)
- Fish species tracking & behavioral anomaly metrics (ByteTrack)
- Coral reef health & bleaching assessment (NOAA criteria)
- Water quality telemetry integration (Copernicus standards)
- Multi-modal Marine Pollution Impact Score (0-100)
- Biodiversity loss prediction & species richness loss
- Explainable AI: SHAP feature attribution & Grad-CAM visual heatmaps
- Interactive geospatial plastic pollution maps & environmental mitigation advisor
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from src.config import BEST_MODEL_PATH, CLASS_NAMES, FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, PROJECT_ROOT
from src.modules.coral_health import CoralHealthAnalyzer, draw_coral_health_overlay, plot_coral_health_summary
from src.modules.fish_behavior import (
    analyze_fish_behavior,
    draw_fish_tracks_and_behavior,
    plot_fish_behavior_summary,
)
from src.modules.ocean_data import load_cache as load_ocean_cache
from src.modules.ocean_data import station_conditions
from src.modules.fish_tracking import (
    compute_track_metrics,
    draw_trajectories,
    fish_stats_for_frame,
    group_frames_by_video,
    load_tracking_model,
    track_video,
)
from src.modules.gradcam import PLASTIC_ID, YOLOGradCAM, overlay_cam
from src.modules.geospatial_maps import (
    create_interactive_plastic_map,
    build_station_table,
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

# Streamlit Page Config
st.set_page_config(
    page_title="MARIS - Marine Biodiversity Impact System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark oceanic theme, glassmorphism, responsive cards)
st.markdown(
    """
    <style>
    .main {
        background-color: #0b111e;
        color: #f1f5f9;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .badge-critical {
        background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 8px; font-weight: bold;
    }
    .badge-high {
        background-color: #f97316; color: white; padding: 4px 10px; border-radius: 8px; font-weight: bold;
    }
    .badge-mod {
        background-color: #eab308; color: black; padding: 4px 10px; border-radius: 8px; font-weight: bold;
    }
    .badge-low {
        background-color: #22c55e; color: white; padding: 4px 10px; border-radius: 8px; font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models_and_engines():
    yolo_model = YOLO(str(BEST_MODEL_PATH)) if BEST_MODEL_PATH.exists() else None
    coral_analyzer = CoralHealthAnalyzer()
    impact_engine = MarineImpactScoringEngine()
    xai_explainer = MarineXAIExplainer(impact_engine.mpis_model, impact_engine.training_df)
    gradcam = YOLOGradCAM(yolo_model) if yolo_model is not None else None
    return yolo_model, coral_analyzer, impact_engine, xai_explainer, gradcam


@st.cache_resource
def get_tracking_model():
    return load_tracking_model(BEST_MODEL_PATH)


@st.cache_data
def get_test_videos():
    return group_frames_by_video(sorted((PROJECT_ROOT / "data" / "processed" / "test" / "images").glob("*.jpg")))


@st.cache_data(show_spinner=False)
def track_video_cached(video_id: str):
    return compute_track_metrics(track_video(get_tracking_model(), get_test_videos()[video_id]))


def rank_videos_for_tracking(videos: dict) -> list[str]:
    """Videos with the most mobile-animal tracks first (from the batch pipeline output, if present)."""
    tracks_csv = PREDICTIONS_DIR / "fish_tracks.csv"
    if tracks_csv.exists():
        t = pd.read_csv(tracks_csv, dtype={"video_id": str})
        counts = t[t["mobile"] & (t["n_frames"] >= 2)]["video_id"].value_counts()
        best = [v for v in counts.index if v in videos]
        return best + [v for v in videos if v not in best]
    return list(videos)


@st.cache_data
def get_curated_images():
    # Mix of images with and without labelled plastic (the first N by filename had none).
    return select_survey_images(n=40, plastic_fraction=0.5)


SIMULATED_NOTE = (
    "ℹ️ What is real: plastic detection (weak, recall ≈ 0.31), Grad-CAM, fish tracking on test videos, and NOAA "
    "Coral Reef Watch sea temperature / bleaching alerts. Still assumed: pH, oxygen, salinity and turbidity (station "
    "presets unless Copernicus is configured), the expert formula behind the impact score / biodiversity loss, "
    "and the image-to-station assignment (TrashCan has no geotags)."
)

yolo_model, coral_analyzer, impact_engine, xai_explainer, gradcam = load_models_and_engines()

if yolo_model is None:
    st.error(f"Trained model not found at {BEST_MODEL_PATH}. Please verify model weights.")
    st.stop()

# Header
st.title("🌊 MARIS: Marine AI Plastic Pollution & Biodiversity Impact System")
st.caption(
    "AI-Based Marine Plastic Pollution Impact Assessment on Biodiversity | "
    "Underwater Plastic Detection (YOLOv11) • Fish Behavioral Tracking (ByteTrack) • "
    "Coral Bleaching Assessment (NOAA Criteria) • Oceanographic Telemetry • "
    "Marine Pollution Impact Score (0-100) • Explainable AI (SHAP & Grad-CAM)"
)
st.info(SIMULATED_NOTE)

# Sidebar
st.sidebar.image("https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=500&auto=format&fit=crop&q=80", caption="Autonomous Marine Survey Feed")
st.sidebar.header("🕹️ Ocean Telemetry & Control Panel")

# Navigation
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🔬 Multi-Modal Diagnostic Studio",
        "🗺️ Geospatial Plastic Pollution Maps",
        "🐟 Fish Behavior & Movement Tracking",
        "🪸 Coral Reef Health & Bleaching Diagnostic",
        "🧠 Explainable AI (SHAP & Grad-CAM) + Policy Advisory",
        "📑 Survey Database & Executive Report",
    ],
)

# Water quality preset or custom inputs in sidebar
st.sidebar.subheader("🌡️ Oceanographic In-Situ Telemetry")
station_preset = st.sidebar.selectbox("Select Marine Survey Station Preset", list(PRESET_STATIONS.keys()))
preset_data = PRESET_STATIONS[station_preset]
# Real NOAA CRW (and Copernicus, if configured) values merged over the preset
station_data = {**preset_data, **station_conditions(station_preset, preset_data, load_ocean_cache())}

use_custom_sensors = st.sidebar.checkbox("Override with Custom Sensor Telemetry", value=False)
if use_custom_sensors:
    temp = st.sidebar.slider("Water Temperature (°C)", 18.0, 33.0, float(station_data["temperature_c"]), 0.1)
    ph = st.sidebar.slider("Ocean pH (Acidity)", 7.4, 8.6, float(station_data["ph"]), 0.02)
    do = st.sidebar.slider("Dissolved Oxygen (mg/L)", 1.5, 9.5, float(station_data["dissolved_oxygen_mg_l"]), 0.1)
    sal = st.sidebar.slider("Salinity (PSU)", 28.0, 42.0, float(station_data["salinity_psu"]), 0.2)
    turb = st.sidebar.slider("Turbidity (NTU)", 0.1, 16.0, float(station_data["turbidity_ntu"]), 0.2)
else:
    temp = station_data["temperature_c"]
    ph = station_data["ph"]
    do = station_data["dissolved_oxygen_mg_l"]
    sal = station_data["salinity_psu"]
    turb = station_data["turbidity_ntu"]

conf_threshold = st.sidebar.slider("YOLO Detection Confidence Threshold", 0.05, 0.85, 0.18, 0.02)

# Compute WQI from active parameters
wq_stats = calculate_water_quality_index(temp, ph, do, sal, turb)

if use_custom_sensors:
    water_source_note = "Water values: entered manually in the sidebar."
else:
    prov = station_data["provenance"]
    water_source_note = "Water values: " + "; ".join(
        f"{k.split('_')[0]} = {v}" for k, v in prov.items()
    )

# ==============================================================================
# TAB 1: MULTI-MODAL DIAGNOSTIC STUDIO
# ==============================================================================
if page == "🔬 Multi-Modal Diagnostic Studio":
    st.subheader("Underwater Survey Multi-Modal Diagnostic Studio")

    col_input1, col_input2 = st.columns([2, 1])

    with col_input1:
        test_dir = PROJECT_ROOT / "data" / "processed" / "test" / "images"
        sample_images = get_curated_images()
        label_dir = PROJECT_ROOT / "data" / "processed" / "test" / "labels"
        plastic_flags = {p.name: has_plastic_label(label_dir / f"{p.stem}.txt") for p in sample_images}
        sample_names = [p.name for p in sample_images]
        # Default to a frame where the batch pipeline actually detected plastic, else any labelled-plastic frame
        assessment_csv = PREDICTIONS_DIR / "marine_biodiversity_assessment.csv"
        detected_names = set()
        if assessment_csv.exists():
            df_prev = pd.read_csv(assessment_csv)
            if "plastic_count" in df_prev.columns:
                detected_names = {f"{i}.jpg" for i in df_prev.loc[df_prev["plastic_count"] > 0, "image_id"]}
        default_idx = next(
            (i for i, n in enumerate(sample_names) if n in detected_names),
            next((i for i, n in enumerate(sample_names) if plastic_flags[n]), 0),
        )

        input_mode = st.radio("Survey Image Source", ["Curated Underwater Survey Transect", "Upload Custom Underwater Photo/Frame"], horizontal=True)

        if input_mode == "Curated Underwater Survey Transect":
            chosen_name = st.selectbox(
                "Select Survey Image Frame",
                sample_names,
                index=default_idx,
                format_func=lambda n: f"{n}  {'(labelled plastic)' if plastic_flags[n] else '(no plastic label)'}",
            )
            img_path = test_dir / chosen_name
            img_bgr = cv2.imread(str(img_path))
        else:
            uploaded = st.file_uploader("Upload Underwater Image", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                habitat_choice = st.radio(
                    "Habitat of this photo",
                    ["Auto-detect coral", "Coral reef", "Not a reef"],
                    horizontal=True,
                    help="Auto uses a colour/texture check for coral before scoring bleaching.",
                )
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            else:
                img_bgr = None
                st.info("Upload an underwater image to run the assessment.")

    with col_input2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Station Location</h4>
                <p><b>{preset_data['location']}</b></p>
                <p>Coordinates: {preset_data['latitude']:.2f}°, {preset_data['longitude']:.2f}°</p>
                <p>Water Quality Index: <b>{wq_stats['water_quality_index']}/100</b></p>
                <p>Status: <span style="color: {wq_stats['status_color']}; font-weight: bold;">{wq_stats['rating']}</span></p>
                <p>NOAA Coral Reef Watch ({station_data['noaa_date'] or 'no data'}): <b>{station_data['noaa_bleaching_alert'] or 'n/a'}</b>,
                DHW {station_data['noaa_dhw'] if station_data['noaa_dhw'] is not None else 'n/a'} °C-weeks</p>
                <p style="font-size: 0.8em; opacity: 0.8;">{water_source_note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if img_bgr is not None:
        h, w = img_bgr.shape[:2]

        # 1. Run YOLO Segmentation
        yolo_res = yolo_model(img_bgr, conf=conf_threshold, verbose=False)[0]
        plastic_stats = quantify(yolo_res)

        plastic_boxes = []
        fish_detections = []
        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            boxes_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
            cls_ids = yolo_res.boxes.cls.cpu().numpy().astype(int)
            confs = yolo_res.boxes.conf.cpu().numpy()
            for box, cid, c_val in zip(boxes_xyxy, cls_ids, confs):
                cname = CLASS_NAMES[cid]
                if cname == "trash_plastic":
                    plastic_boxes.append(box)
                elif cname.startswith("animal_"):
                    fish_detections.append({"bbox": box.tolist(), "cls_name": cname.replace("animal_", ""), "conf": float(c_val)})

        # 2. Run Fish Behavior Analysis
        fish_stats = None
        if input_mode == "Curated Underwater Survey Transect":
            video_id = chosen_name.split("_frame")[0].replace("vid_", "")
            with st.spinner("Tracking fish across this video's frames..."):
                fish_stats = fish_stats_for_frame(track_video_cached(video_id), Path(chosen_name).stem, plastic_boxes, (h, w))
        if fish_stats is None:
            fish_stats = analyze_fish_behavior(fish_detections, plastic_boxes, image_shape=(h, w))

        # 3. Run Coral Health Diagnostic
        if input_mode == "Curated Underwater Survey Transect":
            habitat = "non_reef"  # TrashCan frames are deep-sea ROV footage
        else:
            habitat = {"Auto-detect coral": "auto", "Coral reef": "reef", "Not a reef": "non_reef"}[habitat_choice]
        coral_stats = coral_analyzer.analyze_image(img_bgr, temperature_c=temp, turbidity_ntu=turb, habitat=habitat)

        # 4. Multi-modal Impact Score & Biodiversity Loss Prediction
        impact_prediction = impact_engine.predict(plastic_stats, fish_stats, coral_stats, wq_stats)
        mpis = impact_prediction["marine_pollution_impact_score"]
        biodiv_loss = impact_prediction["predicted_biodiversity_loss_pct"]

        # Results KPI Metrics Banner
        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Marine Pollution Impact", f"{mpis:.1f} / 100", delta=impact_prediction["urgency"][:15], delta_color="inverse")
        m2.metric("Predicted Biodiversity Loss", f"{biodiv_loss:.1f}%", delta="Species Richness Drop", delta_color="inverse")
        m3.metric("Plastic Debris Count", plastic_stats["plastic_count"], f"{plastic_stats['plastic_coverage_percentage']:.2f}% Cover")
        m4.metric("Fish Anomaly Index (BAI)", f"{fish_stats['behavioral_anomaly_index']:.1f}%", fish_stats["behavior_state"][:18])
        m5.metric("Coral Bleaching (BSI)", f"{coral_stats['bleaching_severity_index']:.1f}%", coral_stats["noaa_alert_level"])
        st.caption(
            "Plastic count/cover: real YOLO detections (plastic recall ≈ 0.31, so treat as a lower bound). "
            f"Fish index: {fish_stats.get('measurement', 'single image')}. "
            "Coral: not assessed for TrashCan (deep-sea) frames. "
            "Impact score & biodiversity loss: XGBoost reproducing a documented expert formula (synthetic training data)."
        )

        # Keep this diagnostic so the Fish / Coral / XAI pages can show it instead of example values
        st.session_state["last_diagnostic"] = {
            "image_bgr": img_bgr,
            "plastic_stats": plastic_stats,
            "plastic_boxes": plastic_boxes,
            "fish_detections": fish_detections,
            "fish_stats": fish_stats,
            "coral_stats": coral_stats,
            "impact_prediction": impact_prediction,
        }

        # Visual Diagnostics Side-by-Side
        st.markdown("### 🔍 Computer Vision Multi-Modal Overlays")
        tab_v1, tab_v2, tab_v3, tab_v4 = st.tabs(
            ["🗑️ Plastic Debris Segmentation", "🐟 Fish Tracking & Trajectories", "🪸 Coral Health & Bleaching", "🔥 Grad-CAM Heatmap"]
        )

        with tab_v1:
            annotated_plastic = img_bgr.copy()
            if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
                cls_ids = yolo_res.boxes.cls.cpu().numpy().astype(int)
                confs = yolo_res.boxes.conf.cpu().numpy()
                xyxy = yolo_res.boxes.xyxy.cpu().numpy()
                masks = None
                if yolo_res.masks is not None:
                    masks = np.stack([cv2.resize(m, (w, h)) > 0.5 for m in yolo_res.masks.data.cpu().numpy()])
                annotated_plastic = draw_detections(img_bgr, xyxy, cls_ids, confs, CLASS_NAMES, masks)
            st.image(cv2.cvtColor(annotated_plastic, cv2.COLOR_BGR2RGB), width="stretch", caption="YOLOv11 Instance Segmentation (Masks & Bounding Boxes)")

        with tab_v2:
            annotated_fish = draw_fish_tracks_and_behavior(img_bgr, fish_detections, fish_stats, plastic_boxes)
            st.image(cv2.cvtColor(annotated_fish, cv2.COLOR_BGR2RGB), width="stretch", caption="Fish Tracking Vectors & Behavioral Stress Analysis")

        with tab_v3:
            annotated_coral = draw_coral_health_overlay(img_bgr, coral_stats)
            st.image(cv2.cvtColor(annotated_coral, cv2.COLOR_BGR2RGB), width="stretch", caption="Coral Substrate & Bleaching Health Diagnostic (Green: Healthy, Yellow: Stressed, White: Bleached)")

        with tab_v4:
            detected_ids = sorted({int(c) for c in yolo_res.boxes.cls.cpu().numpy()}) if yolo_res.boxes is not None else []
            class_options = sorted(set(detected_ids) | {PLASTIC_ID})
            target_id = st.selectbox(
                "Explain the model's evidence for class",
                class_options,
                index=class_options.index(PLASTIC_ID),
                format_func=lambda i: CLASS_NAMES[i] + ("" if i in detected_ids else " (not detected)"),
            )
            cam_res = gradcam.compute(img_bgr, class_id=target_id, conf=conf_threshold)
            cam_img = overlay_cam(img_bgr, cam_res["cam"])
            for box, cid in zip(yolo_res.boxes.xyxy.cpu().numpy(), yolo_res.boxes.cls.cpu().numpy()):
                if int(cid) == target_id:
                    cv2.rectangle(cam_img, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), (255, 255, 255), 2)
            st.image(
                cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB),
                width="stretch",
                caption=(
                    f"Grad-CAM for '{cam_res['class_name']}' (gradients of the class logit w.r.t. YOLO neck layers "
                    f"16/19/22). Max class score {cam_res['max_score']:.2f}; white boxes = detections of this class."
                ),
            )

# ==============================================================================
# TAB 2: GEOSPATIAL MAPS
# ==============================================================================
elif page == "🗺️ Geospatial Plastic Pollution Maps":
    st.subheader("Survey Station Map: Plastic Impact & NOAA Coral Reef Watch")
    st.caption(
        "Station positions and NOAA Coral Reef Watch values (sea temperature, Degree Heating Weeks, bleaching "
        "alert) are real satellite data. Impact scores are the mean of the survey images assigned to each "
        "station by the batch pipeline. TrashCan images have no geotags, so that assignment is by rotation."
    )

    assessment_csv = PREDICTIONS_DIR / "marine_biodiversity_assessment.csv"
    df_assess = pd.read_csv(assessment_csv) if assessment_csv.exists() else pd.DataFrame()
    df_stations = build_station_table(df_assess, PRESET_STATIONS, load_ocean_cache())

    tier_filter = st.multiselect(
        "Filter by Pollution Impact Tier",
        ["Low", "Moderate", "High", "Critical", "No data"],
        default=["Low", "Moderate", "High", "Critical", "No data"],
    )
    filtered_df = df_stations[df_stations["impact_tier"].isin(tier_filter)]

    fig_map = create_interactive_plastic_map(filtered_df)
    st.plotly_chart(fig_map, width="stretch")

    st.markdown("#### 📋 Survey Station Records")
    st.dataframe(filtered_df, width="stretch")

# ==============================================================================
# TAB 3: FISH BEHAVIOR LAB
# ==============================================================================
elif page == "🐟 Fish Behavior & Movement Tracking":
    st.subheader("Fish Detection, Tracking & Behavioral Change Analysis")
    st.caption(
        "ByteTrack tracks the detector's animal detections across consecutive TrashCan video frames. ROV camera "
        "motion is removed by phase correlation. Starfish and shells barely move, so their remaining apparent "
        "speed shows the measurement noise floor."
    )

    videos = get_test_videos()
    ranked = rank_videos_for_tracking(videos)
    vid = st.selectbox(
        "Survey video",
        ranked,
        format_func=lambda v: f"vid_{v} ({len(videos[v])} frames)",
    )
    with st.spinner(f"Tracking {len(videos[vid])} frames..."):
        tracks = track_video_cached(vid)

    if tracks.empty:
        st.info("No animals were tracked in this video.")
    else:
        mobile = tracks[tracks["mobile"] & (tracks["n_frames"] >= 2)]
        static = tracks[~tracks["mobile"] & (tracks["n_frames"] >= 2)]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Animal tracks", len(tracks))
        c2.metric("Mobile tracks (fish/eel/crab, 2+ frames)", len(mobile))
        c3.metric("Median mobile speed", f"{mobile['mean_speed_px'].median():.1f} px/frame" if mobile["mean_speed_px"].notna().any() else "n/a")
        c4.metric("Noise floor (static animals)", f"{static['mean_speed_px'].median():.1f} px/frame" if static["mean_speed_px"].notna().any() else "n/a")

        frame_idx = st.slider("Frame", 0, len(videos[vid]) - 1, int(tracks["last_frame"].max()))
        frame = cv2.imread(str(videos[vid][frame_idx]))
        res = yolo_model(frame, conf=conf_threshold, verbose=False)[0]
        plastic = [b for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()) if int(c) == PLASTIC_ID]
        upto = tracks[(tracks["first_frame"] <= frame_idx) & (tracks["n_frames"] >= 2)].copy()
        # trim each path to the frames seen so far
        for i, t in upto.iterrows():
            keep = [k for k, img_id in enumerate(t["image_ids"]) if int(img_id.split("frame")[-1]) <= int(videos[vid][frame_idx].stem.split("frame")[-1])]
            upto.at[i, "path_x"] = [t["path_x"][k] for k in keep]
            upto.at[i, "path_y"] = [t["path_y"][k] for k in keep]
        upto = upto[upto["path_x"].apply(len) > 0]
        st.image(
            cv2.cvtColor(draw_trajectories(frame, upto, plastic), cv2.COLOR_BGR2RGB),
            caption="Track paths up to this frame (red boxes = plastic detected in this frame)",
            width="stretch",
        )
        st.markdown("#### Track table")
        st.dataframe(
            tracks.drop(columns=["image_ids", "path_x", "path_y"]).round(2).sort_values("n_frames", ascending=False),
            width="stretch",
        )
        st.caption(
            "Speed = camera-compensated displacement per frame step. Turn = angle between successive moves "
            "(only for moves > 2 px). min_plastic_dist_norm = closest approach to plastic as a fraction of "
            "the image diagonal."
        )

# ==============================================================================
# TAB 4: CORAL REEF HEALTH STUDIO
# ==============================================================================
elif page == "🪸 Coral Reef Health & Bleaching Diagnostic":
    st.subheader("Coral Reef Health Assessment & Bleaching Severity")
    seg = coral_analyzer.segmenter
    if seg is not None:
        tm = seg.test_metrics
        st.caption(
            f"Reef area from a U-Net trained on SUIM (official TEST split: reef IoU {tm['reef_iou']:.3f}, precision "
            f"{tm['reef_precision']:.2f}, recall {tm['reef_recall']:.2f}). SUIM has no bleaching labels, so bleaching is "
            "a colour rule applied only inside the predicted reef area. The U-Net is out of domain on deep-sea footage, "
            "so in auto mode it must agree with a colour/texture coral check; TrashCan frames show 'No coral detected'. "
            "Upload a reef photo in the Diagnostic Studio (habitat 'Coral reef' or 'Auto') to try it."
        )
    else:
        st.caption(
            "Colour/texture heuristic with NOAA-style alert labels (no trained coral model found in "
            "models/coral_segmenter/). Bleaching is only scored when a coral-presence check passes."
        )
    last = st.session_state.get("last_diagnostic")

    col1, col2 = st.columns([1, 1])
    with col1:
        if last is not None:
            coral_viz = draw_coral_health_overlay(last["image_bgr"], last["coral_stats"])
            st.image(cv2.cvtColor(coral_viz, cv2.COLOR_BGR2RGB), caption="Coral heuristic on the Diagnostic Studio image", width="stretch")
        else:
            coral_sample = (FIGURES_DIR / "coral_health_assessment.png")
            if coral_sample.exists():
                st.image(str(coral_sample), caption="Sample frame from the batch pipeline", width="stretch")

    with col2:
        if last is not None:
            st.pyplot(plot_coral_health_summary(last["coral_stats"]))
        else:
            st.info("Run an image in the 🔬 Multi-Modal Diagnostic Studio first. Its coral results will show here.")

# ==============================================================================
# TAB 5: EXPLAINABLE AI & POLICY ADVISOR
# ==============================================================================
elif page == "🧠 Explainable AI (SHAP & Grad-CAM) + Policy Advisory":
    st.subheader("Explainable AI (XAI) & Actionable Decision-Support")
    st.caption("SHAP feature attributions explaining *why* the AI predicted high/low impact, accompanied by expert conservation policy actions.")

    st.caption("SHAP explains the impact-score model, which was trained on synthetic data, so it shows what that model learned rather than real ecology.")
    last = st.session_state.get("last_diagnostic")

    if last is None:
        st.info("Run an image in the 🔬 Multi-Modal Diagnostic Studio first. Its SHAP explanation and recommendations will show here.")
        st.stop()

    # SHAP for the current diagnostic (water quality re-read from the sidebar)
    current_prediction = impact_engine.predict(last["plastic_stats"], last["fish_stats"], last["coral_stats"], wq_stats)
    shap_res = xai_explainer.explain_prediction(current_prediction["feature_df"])
    st.pyplot(plot_shap_summary_chart(shap_res))

    st.markdown("### 🛡️ Targeted Ecological Intervention Recommendations")

    recs = generate_explainable_recommendations(
        current_prediction, last["plastic_stats"], last["fish_stats"], last["coral_stats"], wq_stats
    )

    for r in recs:
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="badge-critical">{r['priority']}</span>
                <h4 style="margin-top: 8px;">{r['category']}</h4>
                <p><b>Observed Driver:</b> {r['driver']}</p>
                <p><b>Recommended Action:</b> {r['action']}</p>
                <p><b>Ecological Rationale:</b> <i>{r['explanation']}</i></p>
                <p><b>Target Timeline:</b> {r['timeline']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# TAB 6: SURVEY DATABASE & EXECUTIVE REPORT
# ==============================================================================
elif page == "📑 Survey Database & Executive Report":
    st.subheader("Comprehensive Survey Database & Deliverables")

    csv_path = PREDICTIONS_DIR / "marine_biodiversity_assessment.csv"
    if csv_path.exists():
        df_all = pd.read_csv(csv_path)
        st.markdown(f"#### Multi-Modal Survey Table ({len(df_all)} Transects)")
        st.dataframe(df_all, width="stretch")

        st.download_button(
            label="📥 Download Full Assessment CSV",
            data=df_all.to_csv(index=False),
            file_name="marine_biodiversity_assessment.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("#### 📜 Executive Environmental Impact Assessment Report")
    report_path = PROJECT_ROOT / "outputs" / "reports" / "executive_impact_report.md"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        st.markdown(report_text)
        st.download_button(
            label="📥 Download Executive Impact Report (.md)",
            data=report_text,
            file_name="executive_impact_report.md",
            mime="text/markdown",
        )
