"""
Minimal Teammate-1-only demo UI.

Run:
    streamlit run app.py

Upload an underwater image -> run plastic detection -> see the annotated
image plus plastic count / coverage / confidence. No biodiversity, coral,
ocean-condition or risk-score content here -- that's Teammate 2's UI.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from src.config import BEST_MODEL_PATH, CLASS_NAMES
from src.utils.quantify import quantify
from src.utils.visualization import draw_detections

st.set_page_config(page_title="TrashCan Plastic Detector", page_icon="🌊")
st.title("Underwater Plastic Detection & Quantification")
st.caption(
    "Teammate 1 module -- detects and measures plastic/debris in a single "
    "underwater image using a YOLO segmentation model trained on TrashCan 1.0. "
    "This reports plastic detection only, not ecological risk."
)


@st.cache_resource
def load_model():
    from ultralytics import YOLO
    if not BEST_MODEL_PATH.exists():
        return None
    return YOLO(str(BEST_MODEL_PATH))


model = load_model()
if model is None:
    st.error(f"No trained model found at {BEST_MODEL_PATH}. Run training first: "
              f"`python -m src.training.train`")
    st.stop()

conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)
uploaded = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    result = model.predict(img, conf=conf, verbose=False)[0]
    stats = quantify(result)

    annotated = img.copy()
    if result.boxes is not None and len(result.boxes) > 0:
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        xyxy = result.boxes.xyxy.cpu().numpy()
        masks = None
        if result.masks is not None:
            mh, mw = img.shape[:2]
            masks = np.stack([cv2.resize(m, (mw, mh)) > 0.5 for m in result.masks.data.cpu().numpy()])
        annotated = draw_detections(img, xyxy, cls_ids, confs, CLASS_NAMES, masks)

    col1, col2 = st.columns(2)
    col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    col2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Plastic objects detected", stats["plastic_count"])
    label = "Est. bbox coverage" if stats["area_is_estimated_from_bbox"] else "Plastic coverage"
    m2.metric(label, f"{stats['plastic_coverage_percentage']:.1f}%")
    m3.metric("Avg. confidence", f"{stats['average_confidence']:.2f}")

    if stats["class_wise_counts"]:
        st.subheader("All detected classes")
        st.json(stats["class_wise_counts"])
