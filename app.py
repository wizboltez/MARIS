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
from src.modules.geospatial_maps import (
    create_interactive_plastic_map,
    generate_global_marine_stations,
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
    return yolo_model, coral_analyzer, impact_engine, xai_explainer


@st.cache_data
def get_curated_images():
    # Mix of images with and without labelled plastic (the first N by filename had none).
    return select_survey_images(n=40, plastic_fraction=0.5)


SIMULATED_NOTE = (
    "⚠️ Prototype: only plastic detection runs on real data. Water quality values are station presets, "
    "fish behaviour is estimated from a single image, coral health is a colour heuristic, and the impact "
    "score / biodiversity-loss models were trained on synthetic data."
)

yolo_model, coral_analyzer, impact_engine, xai_explainer = load_models_and_engines()

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
st.warning(SIMULATED_NOTE)

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

use_custom_sensors = st.sidebar.checkbox("Override with Custom Sensor Telemetry", value=False)
if use_custom_sensors:
    temp = st.sidebar.slider("Water Temperature (°C)", 18.0, 33.0, float(preset_data["temperature_c"]), 0.1)
    ph = st.sidebar.slider("Ocean pH (Acidity)", 7.4, 8.6, float(preset_data["ph"]), 0.02)
    do = st.sidebar.slider("Dissolved Oxygen (mg/L)", 1.5, 9.5, float(preset_data["dissolved_oxygen_mg_l"]), 0.1)
    sal = st.sidebar.slider("Salinity (PSU)", 28.0, 42.0, float(preset_data["salinity_psu"]), 0.2)
    turb = st.sidebar.slider("Turbidity (NTU)", 0.1, 16.0, float(preset_data["turbidity_ntu"]), 0.2)
else:
    temp = preset_data["temperature_c"]
    ph = preset_data["ph"]
    do = preset_data["dissolved_oxygen_mg_l"]
    sal = preset_data["salinity_psu"]
    turb = preset_data["turbidity_ntu"]

conf_threshold = st.sidebar.slider("YOLO Detection Confidence Threshold", 0.05, 0.85, 0.18, 0.02)

# Compute WQI from active parameters
wq_stats = calculate_water_quality_index(temp, ph, do, sal, turb)

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
        fish_stats = analyze_fish_behavior(fish_detections, plastic_boxes, image_shape=(h, w))

        # 3. Run Coral Health Diagnostic
        coral_stats = coral_analyzer.analyze_image(img_bgr, temperature_c=temp, turbidity_ntu=turb)

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
            "Impact score & biodiversity loss: XGBoost trained on synthetic data. "
            "Fish index: single-image estimate. Coral BSI: colour heuristic (unreliable on images without coral)."
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
            ["🗑️ Plastic Debris Segmentation", "🐟 Fish Tracking & Trajectories", "🪸 Coral Health & Bleaching", "🔥 Grad-CAM Saliency Heatmap"]
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
            st.image(cv2.cvtColor(annotated_plastic, cv2.COLOR_BGR2RGB), use_container_width=True, caption="YOLOv11 Instance Segmentation (Masks & Bounding Boxes)")

        with tab_v2:
            annotated_fish = draw_fish_tracks_and_behavior(img_bgr, fish_detections, fish_stats, plastic_boxes)
            st.image(cv2.cvtColor(annotated_fish, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Fish Tracking Vectors & Behavioral Stress Analysis")

        with tab_v3:
            annotated_coral = draw_coral_health_overlay(img_bgr, coral_stats)
            st.image(cv2.cvtColor(annotated_coral, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Coral Substrate & Bleaching Health Diagnostic (Green: Healthy, Yellow: Stressed, White: Bleached)")

        with tab_v4:
            gradcam_img = xai_explainer.generate_gradcam_heatmap(img_bgr, plastic_boxes, coral_stats.get("masks"))
            st.image(cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Approximate saliency: blurred plastic detections + bleached-coral mask. Not gradient-based Grad-CAM.")

# ==============================================================================
# TAB 2: GEOSPATIAL MAPS
# ==============================================================================
elif page == "🗺️ Geospatial Plastic Pollution Maps":
    st.subheader("Global Geospatial Plastic Pollution & Biodiversity Impact Map")
    st.caption(
        "⚠️ Simulated stations: positions are randomly generated around real reef regions and the scores are "
        "synthetic. No real survey, OBIS or NOAA data is loaded yet."
    )

    df_stations = generate_global_marine_stations(n_stations=85, random_seed=42)

    tier_filter = st.multiselect("Filter by Pollution Impact Tier", ["Low", "Moderate", "High", "Critical"], default=["Low", "Moderate", "High", "Critical"])
    filtered_df = df_stations[df_stations["impact_tier"].isin(tier_filter)]

    fig_map = create_interactive_plastic_map(filtered_df)
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("#### 📋 Marine Survey Station Records")
    st.dataframe(
        filtered_df[
            [
                "station_id",
                "region",
                "latitude",
                "longitude",
                "plastic_coverage_pct",
                "marine_pollution_impact_score",
                "impact_tier",
                "biodiversity_loss_pct",
            ]
        ],
        use_container_width=True,
    )

# ==============================================================================
# TAB 3: FISH BEHAVIOR LAB
# ==============================================================================
elif page == "🐟 Fish Behavior & Movement Tracking":
    st.subheader("Fish Detection, Tracking & Behavioral Change Analysis")
    st.caption(
        "⚠️ Fish boxes are real YOLO detections. Speed and turn values are estimated from box size and position "
        "in a single image. Real multi-frame tracking is not implemented yet."
    )
    last = st.session_state.get("last_diagnostic")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            """
            ### Methodology:
            - **Detection**: fish and other animals detected by the YOLO model.
            - **Association (planned)**: ByteTrack-style IoU matching across video frames.
            - **Behavioral Indicators**:
              - **Swimming Velocity**: Baseline foraging vs. rapid erratic bursts.
              - **Turn Angle Irregularity**: High variance indicates disorientation or escape responses.
              - **Plastic Proximity Hazard**: distance from each fish to the nearest detected plastic.
            """
        )

        if last is not None:
            fish_viz = draw_fish_tracks_and_behavior(last["image_bgr"], last["fish_detections"], last["fish_stats"], last["plastic_boxes"])
            st.image(cv2.cvtColor(fish_viz, cv2.COLOR_BGR2RGB), caption="Fish detections from the image in the Diagnostic Studio", use_container_width=True)
        else:
            fish_sample = (FIGURES_DIR / "fish_behavior_tracking.png")
            if fish_sample.exists():
                st.image(str(fish_sample), caption="Sample frame from the batch pipeline", use_container_width=True)

    with col2:
        if last is not None:
            if last["fish_stats"]["fish_count"] == 0:
                st.info("No fish were detected in the current Diagnostic Studio image. Pick a different frame there to see behaviour metrics.")
            st.pyplot(plot_fish_behavior_summary(last["fish_stats"]))
        else:
            st.info("Run an image in the 🔬 Multi-Modal Diagnostic Studio first. Its fish results will show here.")

# ==============================================================================
# TAB 4: CORAL REEF HEALTH STUDIO
# ==============================================================================
elif page == "🪸 Coral Reef Health & Bleaching Diagnostic":
    st.subheader("Coral Reef Health Assessment & Bleaching Severity")
    st.caption(
        "⚠️ Colour/texture heuristic with NOAA-style alert labels, not a trained coral model. TrashCan images are "
        "deep-sea ROV footage that mostly contain no coral, so bleaching values on them are not meaningful."
    )
    last = st.session_state.get("last_diagnostic")

    col1, col2 = st.columns([1, 1])
    with col1:
        if last is not None:
            coral_viz = draw_coral_health_overlay(last["image_bgr"], last["coral_stats"])
            st.image(cv2.cvtColor(coral_viz, cv2.COLOR_BGR2RGB), caption="Coral heuristic on the Diagnostic Studio image", use_container_width=True)
        else:
            coral_sample = (FIGURES_DIR / "coral_health_assessment.png")
            if coral_sample.exists():
                st.image(str(coral_sample), caption="Sample frame from the batch pipeline", use_container_width=True)

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
        st.dataframe(df_all, use_container_width=True)

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
