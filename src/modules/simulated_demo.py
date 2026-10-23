"""Illustrated, predefined demo scenes. These are not model predictions."""
from pathlib import Path
import json

SCENES = {
    "Fish school": [("Fish", 180, 190, 130, 65), ("Fish", 430, 250, 130, 65), ("Fish", 660, 170, 130, 65)],
    "Debris on the seabed": [("Debris", 230, 330, 65, 140), ("Debris", 510, 350, 100, 100), ("Fish", 650, 190, 130, 65)],
    "Coral reef habitat": [("Coral", 180, 340, 140, 150), ("Coral", 480, 340, 140, 150), ("Fish", 670, 180, 130, 65)],
    "Mixed marine scene": [("Fish", 170, 170, 130, 65), ("Fish", 590, 180, 130, 65), ("Debris", 380, 340, 65, 140), ("Coral", 650, 340, 140, 150)],
}


def scene_svg(name, annotated=True):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="560" viewBox="0 0 960 560">',
        '<defs><linearGradient id="ocean" x2="0" y2="1"><stop stop-color="#087fa0"/><stop offset="1" stop-color="#052f4e"/></linearGradient></defs>',
        '<rect width="960" height="560" fill="url(#ocean)"/>',
        '<path d="M0 470 Q200 430 430 485 T960 460 V560 H0Z" fill="#88785f"/>',
        '<text x="24" y="35" fill="white" font-size="21" font-family="sans-serif">SIMULATED DEMO — illustrated scene / predefined annotations</text>']
    for kind, x, y, w, h in SCENES[name]:
        if kind == "Fish":
            parts += [f'<ellipse cx="{x+w*.52}" cy="{y+h*.5}" rx="{w*.4}" ry="{h*.4}" fill="#ffc857"/>',
                f'<path d="M{x+20} {y+h*.5} L{x} {y} V{y+h}Z" fill="#ff9f43"/>',
                f'<circle cx="{x+w*.77}" cy="{y+h*.38}" r="5" fill="#102a43"/>']
        elif kind == "Debris":
            parts += [f'<rect x="{x+w*.2}" y="{y+20}" width="{w*.6}" height="{h-25}" rx="12" fill="#cde7ef" stroke="#7dd3fc" stroke-width="3"/>',
                f'<rect x="{x+w*.35}" y="{y}" width="{w*.3}" height="25" fill="#60a5fa"/>',
                f'<rect x="{x+w*.2}" y="{y+h*.5}" width="{w*.6}" height="25" fill="#e87967"/>']
        else:
            parts += [f'<path d="M{x+70} {y+h} V{y+20} M{x+70} {y+70} L{x+25} {y+30} M{x+70} {y+100} L{x+115} {y+50} M{x+25} {y+30} V{y+5} M{x+115} {y+50} V{y+15}" fill="none" stroke="#f598b9" stroke-width="16" stroke-linecap="round"/>']
        if annotated:
            color = {"Fish": "#34d399", "Debris": "#fbbf24", "Coral": "#f9a8d4"}[kind]
            parts += [f'<rect x="{x-8}" y="{y-8}" width="{w+16}" height="{h+16}" fill="none" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x-8}" y="{y-17}" fill="{color}" font-size="18" font-family="sans-serif">{kind} · simulated annotation</text>']
    parts += ['<rect y="520" width="960" height="40" fill="#102a43"/>', '<text x="24" y="547" fill="white" font-size="17" font-family="sans-serif">Illustration only. No inference, confidence scores, or measured accuracy.</text>', '</svg>']
    return ''.join(parts)


def render_simulated_demo(st, root: Path):
    st.subheader("Simulated marine scene demo")
    st.warning("SIMULATED DEMO: illustrated images and predefined annotations. These are not detector predictions or evidence of model accuracy.")
    name = st.selectbox("Choose an illustrated scene", list(SCENES))
    annotated = st.checkbox("Show predefined annotations", value=True)
    svg = scene_svg(name, annotated)
    st.image(svg, width="stretch")
    counts = {k: sum(obj[0] == k for obj in SCENES[name]) for k in ("Fish", "Debris", "Coral")}
    columns = st.columns(3)
    for column, (kind, count) in zip(columns, counts.items()):
        column.metric(f"Simulated {kind.lower()} count", count)
    st.caption("Counts describe the objects drawn in this illustration. Live model performance is available in Data & Report.")
    st.download_button("Download labeled demo image (SVG)", svg, file_name="SIMULATED_marine_demo.svg", mime="image/svg+xml")
    st.download_button("Download simulated annotations", json.dumps({"simulated": True, "source": "predefined illustration", "scene": name, "objects": [{"class": k, "bbox_xywh": [x,y,w,h]} for k,x,y,w,h in SCENES[name]]}, indent=2), file_name="SIMULATED_annotations.json", mime="application/json")
    st.markdown("#### Existing validation examples and reef photographs")
    st.caption("The fish/debris examples below are existing annotated best-case validation frames. Reef photographs are sample inputs; no coral prediction is claimed here.")
    examples = [
        (root / "outputs/experiments/2026-10-14/demo_frames/BEST-fish_vid_000430_frame0000010.jpg", "Fish: annotated best-case validation frame"),
        (root / "outputs/experiments/2026-10-14/demo_frames/BEST-plastic_vid_000029_frame0000017.jpg", "Plastic debris: annotated best-case validation frame"),
        (root / "data/samples/reef/coralscapes_val_010.jpg", "Reef photograph: sample input, not a detection result"),
    ]
    for path, caption in examples:
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
