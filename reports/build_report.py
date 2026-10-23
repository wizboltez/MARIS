"""Builds reports/MARIS_FA_Report.docx in the faculty's FA report layout.

All numbers come from measured result files in the repo (see REPORT_GUIDE.md for each source).
Run:  .venv\\Scripts\\python.exe reports\\make_diagrams.py  then  .venv\\Scripts\\python.exe reports\\build_report.py
"""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "reports" / "assets"
FIG = ROOT / "outputs" / "figures"
OUT = ROOT / "reports" / "MARIS_FA_Report.docx"

TITLE = "MARIS: AI-Based Marine Plastic Pollution Impact Assessment on Biodiversity"
TEAM = [("[PRN 1]", "[Student name 1]"), ("[PRN 2]", "[Student name 2]"), ("[PRN 3]", "[Student name 3]")]
GUIDE = "[Project guide name]"

doc = Document()
for s in doc.sections:
    s.top_margin, s.bottom_margin, s.left_margin, s.right_margin = Cm(2.2), Cm(2.0), Cm(2.5), Cm(2.2)
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15


# ---------------------------------------------------------------- helpers
def para(text="", bold=False, size=12, align=None, color=None, italic=False, after=6, before=0):
    p = doc.add_paragraph()
    if align == "c":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align != "l":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.space_before = Pt(before)
    if text:
        r = p.add_run(text)
        r.bold, r.italic, r.font.size = bold, italic, Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
    return p


def rich(parts, align=None):
    """parts: list of (text, bold) tuples in one justified paragraph."""
    p = para(align=align)
    for t, b in parts:
        r = p.add_run(t)
        r.bold = b
    return p


def h1(text):
    p = para(text, bold=True, size=16, align="l", after=8, before=4)
    return p


def h2(text):
    return para(text, bold=True, size=13.5, align="l", after=4, before=8)


def bullets(items, numbered=False):
    for i, it in enumerate(items, 1):
        p = doc.add_paragraph(style="List Number" if numbered else "List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        if isinstance(it, tuple):
            r = p.add_run(it[0]); r.bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(caption, header, rows, widths=None, size=10.5):
    para(caption, bold=True, size=11, align="c", after=3, before=6)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(size)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shade(c, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            bold = isinstance(v, str) and v.startswith("**")
            r = cells[i].paragraphs[0].add_run(str(v).strip("*")); r.font.size = Pt(size); r.bold = bold
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    para(after=4)
    return t


def figure(path, caption, width=15.5):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width))
    para(caption, bold=True, size=11, align="c", after=10)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tag)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = text
        r._r.append(e)


# ---------------------------------------------------------------- cover
logo = next(A.glob("logo_*.png"), None)
if logo:
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(logo), width=Cm(3.6))
para("Pimpri Chinchwad Education Trust's", bold=True, size=12, align="c", after=0)
para("Pimpri Chinchwad College of Engineering", bold=True, size=20, align="c", after=0)
para("Sector No. 26, Pradhikaran, Nigdi, Pune 411044", size=11, align="c", after=4)
para("An Autonomous Institute Approved By AICTE and Affiliated To SPPU, Pune", size=10, align="c", after=14)
para("Department Of Information Technology", bold=True, size=20, align="c", after=12)
para("Artificial Intelligence and Machine Learning", bold=True, size=14, align="c", after=4)
para("[Course: BIT25PC01]", bold=True, size=13, align="c", after=10)
para("Project Report", bold=True, size=14, align="c", after=2)
para("of", size=12, align="c", after=2)
para("T. Y. B. Tech", bold=True, size=15, align="c", after=2)
para("Academic Year: 2026-27", bold=True, size=12, align="c", after=0)
para("Semester - V", size=12, align="c", after=10)
para("On", size=12, align="c", after=6)
para(TITLE, bold=True, size=16, align="c", color="C00000", after=14)
para("By", size=12, align="c", after=6)
t = doc.add_table(rows=1, cols=2); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(("PRN No.", "Name of the Student")):
    t.rows[0].cells[i].text = ""
    r = t.rows[0].cells[i].paragraphs[0].add_run(h); r.bold = True
    t.rows[0].cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
for prn, name in TEAM:
    c = t.add_row().cells
    for i, v in enumerate((prn, name)):
        c[i].text = v; c[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
for row in t.rows:
    row.cells[0].width, row.cells[1].width = Cm(5.5), Cm(7.5)
page_break()

# ---------------------------------------------------------------- certificate
para("CERTIFICATE", bold=True, size=18, align="c", after=24, before=24)
p = para(after=12)
p.paragraph_format.line_spacing = 1.6
p.add_run("This is to certify that the project report entitled ")
r = p.add_run(TITLE); r.bold = True
p.add_run(", has been completed by ")
r = p.add_run(", ".join(n for _, n in TEAM)); r.bold = True
p.add_run(" as part of the Artificial Intelligence and Machine Learning project course in the Third Year of "
          "the Information Technology programme. The work was carried out by the group during Semester V and "
          "satisfies the course requirements for applying Artificial Intelligence and Machine Learning concepts.")
for _ in range(6):
    para()
para(f"Project Guide: {GUIDE}", bold=True, align="r" if False else "l", after=0).alignment = WD_ALIGN_PARAGRAPH.RIGHT
page_break()

# new section so page numbers start after the certificate
sec = doc.add_section(WD_SECTION.CONTINUOUS)
sec.footer.is_linked_to_previous = False
add_page_number(sec)
doc.sections[0].footer.is_linked_to_previous = False

# ---------------------------------------------------------------- abstract
para("ABSTRACT", bold=True, size=16, align="c", after=10)
para(
    "Plastic waste is now found from shallow coral reefs to the deep sea, yet most reef and seabed monitoring is "
    "still done by divers or by people watching hours of underwater video. This project presents MARIS, a "
    "multi-model system that examines underwater imagery and turns it into an estimate of ecological pressure. "
    "Debris is detected and segmented by a YOLO11s-seg network trained on TrashCan 1.0; animals are followed "
    "across video frames with ByteTrack; coral is split into live, bleached and dead regions by a SegFormer "
    "(MiT-B2) model fine-tuned on the Coralscapes dataset; and sea temperature and bleaching alerts are read from "
    "NOAA Coral Reef Watch. These signals, together with a water quality index built from temperature, pH, "
    "dissolved oxygen, salinity and turbidity, are merged by an XGBoost model into a Marine Pollution Impact "
    "Score between 0 and 100. SHAP and Grad-CAM explain each result, and everything is presented in a Streamlit "
    "decision-support dashboard. Every model was tested only on footage it had never seen: 53 held-out videos "
    "for debris and five unseen dive sites for coral. The coral model recognised bleaching in 95.7% of test "
    "images, separated coral from background with 95.2% pixel accuracy and reached a coral IoU of 0.767. The "
    "debris model decided correctly whether plastic was present in 75.1% of test images (majority baseline "
    "62.8%) and reached 48.9% mAP@50 for plastic objects. Fish detection in dark deep-sea video remained the "
    "weakest part. The impact score follows a documented expert weighting and should be read as an estimate, "
    "not a field measurement. The work shows that one pipeline can link plastic, fauna, coral and water "
    "conditions, and that larger and more varied labelled data is the main route to better accuracy."
)
rich([("Keywords: ", True), ("marine plastic pollution, coral bleaching, YOLO11, SegFormer, object tracking, "
                             "explainable AI", False)])
page_break()

# ---------------------------------------------------------------- 1 introduction
h1("1. INTRODUCTION")
h2("1.1 Background")
para("Every year millions of tonnes of plastic reach the ocean from land [1]. Once there, it drifts, sinks and "
     "settles on reefs and on the seabed, where it tangles animals, smothers coral and carries microbes. Field "
     "work on Asia-Pacific reefs found that coral touching plastic was far more likely to be diseased [2], while "
     "repeated marine heatwaves have already caused mass bleaching across entire reef systems [3]. Managers "
     "therefore need to know three things together: how much litter is present, how marine life is behaving "
     "around it, and how healthy the coral is.")
para("Today this information is mostly gathered by divers with clipboards or by analysts reviewing underwater "
     "video frame by frame. The process is slow, costly and hard to repeat over large areas, and different "
     "teams often look at only one of these questions. This project uses the image analysis side of the problem: "
     "it builds trained models that read underwater images and video, and connects their outputs to ocean "
     "condition data and a single impact estimate shown in a dashboard.")
h2("1.2 Importance of AI/ML/Deep Learning")
para("Underwater images are difficult for hand-written rules. Light fades quickly with depth, colours shift "
     "towards blue and green, suspended particles blur the view, and a plastic bag can look very similar to a "
     "jellyfish or a patch of sand. Deep learning models learn these visual patterns directly from labelled "
     "examples instead of relying on fixed thresholds.")
para("Different questions in this project need different kinds of model. Object detection with instance masks "
     "(YOLO11-seg [4]) finds each piece of debris and measures its area. Semantic segmentation (SegFormer [5]) "
     "labels every pixel of a reef image as live, bleached or dead coral. Multi-object tracking (ByteTrack [15]) "
     "links detections of the same animal across frames so that speed and turning can be measured. A gradient "
     "boosted tree model (XGBoost [6]) combines these numerical outputs, and explanation methods (SHAP [7] and "
     "Grad-CAM [8]) show which inputs and image regions drove a result. Because models can still be wrong, every "
     "one of them was evaluated on data kept completely separate from training.")
h2("1.3 Problem Definition")
para("The problem addressed is to estimate the ecological pressure caused by underwater plastic from imagery, "
     "by detecting debris, observing fish movement and assessing coral health, and combining these with water "
     "conditions into one interpretable score. Existing tools usually solve one of these sub-tasks in isolation, "
     "and many published accuracies are measured on images that closely resemble the training images, which "
     "overstates real-world performance.")
para("MARIS answers this with a single pipeline: trained detectors and segmenters process each image, satellite "
     "and station data supply water conditions, an XGBoost model produces a Marine Pollution Impact Score "
     "(0-100) and a projected biodiversity-loss percentage, and explanations plus recommended actions are shown "
     "in a dashboard. The system is a research prototype; its impact estimate follows an expert-defined "
     "weighting and has not yet been validated against field biodiversity surveys.")
h2("1.4 Objectives")
para("The objectives of the project, taken from the problem brief, are:", after=2)
bullets([
    "To detect and classify underwater plastic and other debris in images and video frames.",
    "To identify and track fish and other animals and measure behavioural change near plastic.",
    "To assess coral reef health and the extent of bleaching.",
    "To integrate water quality parameters: temperature, pH, dissolved oxygen, salinity and turbidity.",
    "To estimate the impact of plastic pollution on marine biodiversity.",
    "To produce a Marine Pollution Impact Score on a 0-100 scale.",
    "To provide explainable results and recommendations through a decision-support dashboard.",
], numbered=True)
h2("1.5 Scope")
para("The system works on still images and short video sequences from underwater cameras and remotely operated "
     "vehicles. It detects 16 object classes from TrashCan (including plastic, other debris types, the ROV and "
     "several animal groups), segments coral into live, bleached and dead regions, and reads sea temperature and "
     "bleaching alerts for five reference stations from NOAA Coral Reef Watch.")
para("Some parts are outside the current scope or are estimates. The survey images carry no GPS position, so "
     "stations are assigned to them in rotation for demonstration. pH, dissolved oxygen, salinity and turbidity "
     "use station reference values rather than live sensors. Fish species are not identified. The impact score "
     "and biodiversity-loss figure follow a documented expert weighting reproduced by XGBoost, so they are "
     "indicative estimates rather than measured ecological outcomes. Real deployment would require field "
     "validation and more diverse labelled footage.")
page_break()

# ---------------------------------------------------------------- 2 literature
h1("2. Literature Survey")
para("Research on marine debris, reef imagery and fish monitoring has grown quickly, helped by public datasets "
     "and modern detectors. Table 2.1 summarises the studies and resources that shaped this project and what each "
     "leaves open.")
table("Table 2.1. Comparative review of related studies and resources",
      ["Year [Ref]", "Method", "Dataset", "Reported finding", "Limitation / relevance"],
      [
          ["2019 [9]", "YOLOv2, SSD, Faster R-CNN", "Trash-ICRA19", "Deep detectors can locate underwater litter",
           "Detection only; no link to ecology"],
          ["2020 [10]", "Mask R-CNN baselines", "TrashCan 1.0", "Instance masks for deep-sea debris and fauna",
           "Benchmark only; our main debris data"],
          ["2020 [11]", "SUIM-Net, U-Net", "SUIM", "8-class underwater scene segmentation",
           "Reef class has no bleaching label"],
          ["2024 [12]", "Faster R-CNN, YOLOv6", "SeaClear", "68.3% mAP on random split, ~20% on unseen sites",
           "Shows the cross-site drop we guard against"],
          ["2025 [13]", "SegFormer, DINOv2", "Coralscapes", "39-class reef segmentation; MiT-B2 mIoU 0.547",
           "Coral only; we connect it to plastic"],
          ["2022 [14]", "Expert-annotated archive", "FathomNet", "Large labelled deep-sea image database",
           "Used in our fish-detector experiment"],
          ["2018 [2]", "Field statistics", "Reef surveys", "Plastic contact raises coral disease risk",
           "Manual surveys; motivates automation"],
          ["2022 [15]", "ByteTrack", "MOT benchmarks", "Keeps low-confidence boxes for tracking",
           "Adopted for fish tracking"],
      ], widths=[2.0, 3.1, 2.6, 4.2, 4.2], size=9.5)
para("The reviewed work shows that deep learning can find debris [9], [10], segment underwater scenes [11], [13] "
     "and follow moving targets [15]. It also shows a recurring weakness: results measured on random image "
     "splits can fall sharply when the model meets a new site [12]. Each study addresses a single task, and none "
     "connects debris, animal behaviour, coral condition and water quality into one assessment.")
h2("2.1 Research Gap and Project Contribution")
para("The gap addressed here is the lack of an integrated framework that brings plastic detection, fish "
     "behaviour, coral health and water quality together to estimate the overall ecological effect of "
     "underwater plastic. MARIS contributes (i) one pipeline and dashboard covering all four signals, (ii) a coral "
     "model that separates live, bleached and dead coral, (iii) leak-free evaluation on unseen videos and dive "
     "sites, which during development exposed and removed a 77% train-test overlap in an earlier data split, and "
     "(iv) per-result explanations with SHAP and Grad-CAM. The contribution is the integration and the honest "
     "evaluation, not a new detection algorithm.")
page_break()

# ---------------------------------------------------------------- 3 dataset
h1("3. Dataset Used")
para("Several public datasets were used, each for the task it suits best. Image datasets train and test the "
     "models; ocean data products supply environmental conditions and a real-world check.")
h2("3.1 Dataset Sources")
bullets([
    ("TrashCan 1.0 [10]: ", "underwater ROV video frames of debris and marine animals with instance masks. "
                            "https://conservancy.umn.edu/handle/11299/214366"),
    ("Coralscapes [13]: ", "densely labelled Red Sea reef images used for live, bleached and dead coral. "
                           "https://huggingface.co/datasets/EPFL-ECEO/coralscapes"),
    ("SUIM [11]: ", "underwater scene segmentation, used for the first U-Net reef model. "
                    "https://irvlab.cs.umn.edu/resources/suim-dataset"),
    ("FathomNet [14]: ", "expert-labelled deep-sea fish images used to try to improve fish detection. "
                         "https://fathomnet.org"),
    ("NOAA Coral Reef Watch [16]: ", "daily 5 km sea-surface temperature, heat stress and bleaching alerts. "
                                     "https://coralreefwatch.noaa.gov"),
    ("NOAA NCEI Marine Microplastics [17] and OBIS [18]: ", "measured surface microplastics and species records, "
                                                           "used for the plastic map and a real-data check."),
])
h2("3.2 Dataset Description")
table("Table 3.1. Summary of datasets",
      ["Dataset", "No. of samples", "Features", "Classes / target", "Split used"],
      [
          ["TrashCan 1.0", "7,212 images", "RGB pixels, masks, boxes", "16 classes (debris, ROV, animals)",
           "4,913 / 986 / 1,313 by video"],
          ["Coralscapes", "2,075 images", "RGB pixels, pixel labels", "39 classes grouped to 4",
           "1,517 / 166 / 392 by dive site"],
          ["SUIM", "~1,635 images", "RGB pixels, pixel labels", "8 scene classes", "official, 110 test"],
          ["FathomNet subset", "2,720 images", "RGB pixels, boxes", "fish (4,415 boxes)", "training only"],
          ["NOAA CRW", "5 stations, daily", "SST, DHW, alert level", "water / heat stress", "live lookup"],
          ["NCEI + OBIS", "13,460 samples, 698 cells", "density, species counts", "biodiversity check",
           "cross-validation"],
      ], widths=[2.8, 2.8, 3.3, 3.6, 3.5], size=9.5)
h2("3.3 Classes Used")
para("The TrashCan detector predicts 16 classes. Table 3.2 lists the annotation counts that matter most for this "
     "project; an image may contain several objects, so these are object counts, not image counts.")
table("Table 3.2. Main TrashCan classes and annotation counts",
      ["Class", "Description", "Annotations"],
      [
          ["trash_plastic", "Plastic bags, bottles and fragments", "1,864"],
          ["trash_metal", "Cans and metal debris", "1,126"],
          ["trash_etc", "Other or unclear debris", "2,039"],
          ["animal_fish", "Fish of any species", "764"],
          ["rov", "Parts of the recording robot", "3,317"],
          ["Other 11 classes", "Paper, fabric, rubber, wood, fishing gear, crabs, eels, starfish, shells, plants, other animals", "3,226"],
          ["**Total**", "", "**12,336**"],
      ], widths=[3.2, 9.0, 3.0], size=10)
para("For coral, the 39 Coralscapes classes were grouped into four targets: other, live coral, bleached coral "
     "and dead coral. Bleached coral is rare, so class weights were applied during training.")
h2("3.4 Data Preprocessing and Quality")
para("A validation script checked all 7,212 TrashCan images: none were missing or corrupt, and no boxes or "
     "polygons were invalid. COCO polygon annotations were converted into YOLO segmentation format. Frames "
     "from one video look almost identical, so the split was made by source video: no video appears in more than "
     "one of train, validation and test. An earlier split had placed 77% of test images next to near-copies in "
     "training, which inflated mAP to about 0.68; after the fix the honest figure was far lower, and all results "
     "in this report use the corrected split. Coralscapes was used with its official split by dive site, so test "
     "reefs were never seen in training.")
page_break()

# ---------------------------------------------------------------- 4 methodology
h1("4. Proposed Methodology")
para("MARIS joins several trained models and data sources into one assessment. The work covers data preparation, "
     "model training, held-out evaluation and integration into a dashboard.")
h2("4.1 Overall Approach")
para("Each image passes through four analysis branches: the YOLO11s-seg detector finds debris, animals and the "
     "ROV; ByteTrack links animal detections across video frames; the SegFormer model labels coral pixels; and a "
     "water quality module combines NOAA temperature with station values for pH, dissolved oxygen, salinity and "
     "turbidity into a Water Quality Index. The outputs become a feature vector that an XGBoost model maps to the "
     "Marine Pollution Impact Score and a projected biodiversity loss. SHAP explains the score, Grad-CAM shows "
     "where the detector looked, and the dashboard turns the result into suggested actions.")
h2("4.2 Workflow Diagram")
figure(A / "workflow.png", "Figure 4.1: Workflow of the proposed MARIS system.", width=16)
h2("4.3 Preprocessing")
para("TrashCan annotations were converted to YOLO format and split by video, as described in Section 3.4. "
     "Training images were resized to 640 pixels with augmentation (mosaic, horizontal flips and colour jitter). "
     "Coralscapes labels were remapped to four classes and images were processed at full 1024 x 2048 resolution "
     "with flip test-time augmentation. NOAA products were sampled per station, and NCEI and OBIS records were "
     "grouped into 2-degree ocean cells.")
h2("4.4 Model Selection")
bullets([
    ("YOLO11s-seg: ", "fast, single-stage, and gives masks, so plastic area can be measured and not only counted."),
    ("SegFormer MiT-B2: ", "the strongest model family in the Coralscapes benchmark [13]; a U-Net baseline scored lower."),
    ("ByteTrack: ", "keeps low-confidence detections when linking tracks, which suits faint fish in dark video."),
    ("XGBoost: ", "accurate on small tabular inputs and directly compatible with SHAP explanations."),
    ("SHAP and Grad-CAM: ", "explain the score and the detector's visual attention respectively."),
])
h2("4.5 Implementation Details")
para("The detector was initialised from COCO-pretrained YOLO11s-seg weights and trained on a Google Colab T4 GPU "
     "with automatic mixed precision. The coral model started from the Coralscapes authors' released SegFormer "
     "weights and was fine-tuned with Dice plus cross-entropy loss. All thresholds were chosen on validation "
     "data only; the plastic confidence threshold is 0.18. The dashboard is written in Streamlit and runs every "
     "model on a selected or uploaded image, then passes the result to the impact, map, coral and decision "
     "pages.")
page_break()

# ---------------------------------------------------------------- 5 architecture
h1("5. Model Architecture")
para("The central model is YOLO11s-seg, which detects and segments the 16 TrashCan classes. Its outputs feed the "
     "rest of the pipeline.")
h2("5.1 Architecture Overview")
bullets([
    ("Backbone: ", "stacked convolution and C3k2 blocks with SPPF pooling and a C2PSA attention block extract edges, "
                   "textures and shapes at several scales."),
    ("Neck: ", "upsampling and concatenation (FPN/PAN style) merge fine and coarse features so that both small "
               "fragments and large objects can be found."),
    ("Heads: ", "a detection head predicts boxes and class scores; a segmentation head predicts 32 mask prototypes "
                "and per-object coefficients that form each instance mask."),
], numbered=True)
figure(A / "architecture_yolo.png", "Figure 5.1: YOLO11s-seg architecture as used in MARIS.", width=16)
h2("5.2 Model Working and Application Integration")
para("An input frame is resized to 640 x 640. The network predicts boxes, classes, confidences and masks; "
     "non-maximum suppression removes duplicate detections, and detections below 0.18 confidence are dropped. "
     "Plastic masks are restored to the original image size and their union gives the percentage of the image "
     "covered by plastic, so overlapping masks are not counted twice. In parallel, the SegFormer model "
     "(27.4 M parameters) labels coral pixels and a bleaching severity index is computed from the share of "
     "bleached and dead coral. These values, the fish behaviour index and the Water Quality Index are then "
     "passed to the XGBoost impact model.")
page_break()

# ---------------------------------------------------------------- 6 maths
h1("6. Mathematical Model")
para("The detectors and segmenters are evaluated with standard overlap and counting metrics, and the impact "
     "score is defined as a weighted combination of the four signals.")
h2("6.1 Intersection over Union (IoU)")
para("IoU compares a predicted region P with the true region G. A detection counts as correct when IoU >= 0.5.")
para("IoU = |P ∩ G| / |P ∪ G|", bold=True, align="c")
h2("6.2 Precision, Recall and F1")
para("With TP true positives, FP false positives and FN missed objects:")
para("Precision = TP / (TP + FP)      Recall = TP / (TP + FN)", bold=True, align="c")
para("F1 = 2 × Precision × Recall / (Precision + Recall)", bold=True, align="c")
h2("6.3 Mean Average Precision (mAP)")
para("Average Precision (AP) is the area under the precision-recall curve of one class. mAP@50 averages AP over "
     "classes at IoU 0.5; mAP@50-95 also averages over IoU thresholds from 0.50 to 0.95.")
para("mAP = (1/N) × Σ APᵢ ,  i = 1 … N", bold=True, align="c")
h2("6.4 Water Quality Index")
para("Each parameter is scored from 0 to 100 against marine reference ranges, then combined with fixed weights:")
para("WQI = 0.30·S_DO + 0.25·S_T + 0.20·S_pH + 0.15·S_turb + 0.10·S_sal", bold=True, align="c")
h2("6.5 Marine Pollution Impact Score")
para("For reef sites the score combines plastic load P, the fish behavioural anomaly index F, the bleaching "
     "severity index B and poor water quality (100 - WQI), each on a 0-100 scale. For sites without coral the "
     "coral weight is spread over the other three terms (0.42, 0.29 and 0.29).")
para("MPIS = 0.32·P + 0.22·F + 0.24·B + 0.22·(100 - WQI)", bold=True, align="c")
para("An XGBoost regressor is trained to reproduce this score (R² = 0.945 on held-out synthetic samples) so that "
     "SHAP can attribute every prediction to its inputs. The weights are expert assumptions, which is why the "
     "score is reported as an estimate.")
page_break()

# ---------------------------------------------------------------- 7 results
h1("7. Results and Experiments")
para("This section lists the training setup and the held-out test results. Results are ordered from the "
     "strongest module to the weakest.")
h2("7.1 Experimental Setup")
para("Debris detection was tested on 1,313 TrashCan images from 53 videos that were never used for training, "
     "threshold choice or model selection. Coral segmentation was tested on 392 Coralscapes images from five dive "
     "sites absent from training. 95% confidence intervals for the detector were obtained by bootstrap over the "
     "53 test videos (1,000 resamples).")
h2("7.2 Hyperparameter Details")
table("Table 7.1. Training configuration of the plastic detector",
      ["Parameter", "Value"],
      [
          ["Model", "YOLO11s-seg (COCO-pretrained)"],
          ["Input image size", "640 x 640 pixels"],
          ["Epochs", "120 maximum, early stopping (patience 30)"],
          ["Batch size", "16"],
          ["Initial learning rate / optimiser", "0.01 / auto (AdamW)"],
          ["Augmentation", "Mosaic (off for final 10 epochs), horizontal flip 0.5, HSV jitter"],
          ["Hardware", "Google Colab, NVIDIA Tesla T4 GPU, mixed precision (AMP)"],
          ["Confidence threshold at inference", "0.18 (chosen on validation)"],
      ], widths=[6, 9.5])
h2("7.3 Evaluation Metrics")
bullets([
    ("Precision, recall, F1: ", "share of correct detections, share of objects found, and their balance."),
    ("mAP@50 and mAP@50-95: ", "detection quality across confidence levels and box overlaps."),
    ("IoU and pixel accuracy: ", "overlap and per-pixel correctness for segmentation."),
    ("Image-level accuracy: ", "whether the model correctly says an object type is present in an image, always "
                               "compared with the majority-class baseline."),
])
h2("7.4 Experimental Results")
para("Table 7.2 shows the coral model on unseen dive sites. It was the strongest module: bleaching was "
     "recognised in 95.7% of test images against a baseline of 82.9%.")
table("Table 7.2. Coral health results on the Coralscapes test split (392 images, 5 unseen sites)",
      ["Metric", "Value"],
      [
          ["Image-level bleaching detection accuracy", "**95.7%** (P 84.7%, R 91.0%; baseline 82.9%)"],
          ["Image-level coral detection accuracy", "**95.4%** (P 97.5%, R 96.9%; baseline 82.7%)"],
          ["Coral vs not-coral pixel accuracy", "95.2%"],
          ["Pixel accuracy (4 classes)", "94.3%"],
          ["Coral IoU (precision / recall)", "0.767 (0.876 / 0.860)"],
          ["IoU: live / bleached / dead coral", "0.806 / 0.694 / 0.328"],
          ["Mean IoU (4 classes)", "0.693"],
      ], widths=[8, 7.5])
para("Four coral models were compared on the same official test split (Table 7.3). Fine-tuning the published "
     "Coralscapes model at full resolution gave the best mean IoU, although the gain over the published weights "
     "is small, which we report openly.")
table("Table 7.3. Coral model experiments",
      ["Experiment", "Model", "Coral IoU", "Mean IoU"],
      [
          ["1", "U-Net (ResNet-34) on SUIM, reef class", "0.684", "0.684"],
          ["2", "SegFormer v1, ImageNet start, 512 x 1024", "0.704", "0.647"],
          ["3", "Published Coralscapes SegFormer, no fine-tuning", "0.768", "0.688"],
          ["4", "**SegFormer v2, fine-tuned, full resolution (ours)**", "**0.767**", "**0.693**"],
      ], widths=[2.4, 8.2, 2.4, 2.4])
para("Table 7.4 asks whether each image contains a given object. ROV detection was strong, debris alarms were "
     "correct 89.3% of the time, and plastic presence was judged correctly in 75.1% of images, well above the "
     "62.8% obtained by always answering 'no plastic'.")
table("Table 7.4. Image-level detection on the TrashCan test split (1,313 images, confidence 0.18)",
      ["Question", "Accuracy", "Precision", "Recall", "Majority baseline"],
      [
          ["Underwater robot (ROV) present?", "**91.5%**", "81.5%", "90.9%", "71.5%"],
          ["Any debris present?", "76.8%", "**89.3%**", "79.9%", "77.9%"],
          ["Plastic present?", "**75.1%**", "69.7%", "58.4%", "62.8%"],
      ], widths=[5.5, 2.4, 2.4, 2.4, 3.0])
para("Object-level plastic detection is harder because each item must be located precisely (Table 7.5). A "
     "validation-tuned threshold of 0.47 was also tried; it raised precision but lowered F1, so 0.18 was kept.")
table("Table 7.5. Plastic detector experiments on the TrashCan test split (unseen videos)",
      ["Experiment", "Threshold", "Precision (95% CI)", "Recall (95% CI)", "F1", "mAP@50"],
      [
          ["1 - default", "0.18", "47.5% [33.7, 60.2]", "45.2% [31.5, 59.4]", "**46.3%**", "**48.9%**"],
          ["2 - val-tuned", "0.47", "61.5% [45.8, 75.8]", "33.8% [19.0, 49.6]", "43.6%", "-"],
      ], widths=[2.8, 2.0, 3.4, 3.4, 1.8, 1.8], size=10)
para("Plastic mAP@50-95 was 34.0%. Among the other classes, the ROV reached F1 61.2%, eels 48.2% and starfish "
     "43.7%. Figure 7.1 shows sample predictions and Figure 7.2 the coral health overlay.")
figure(FIG / "sample_predictions.png", "Figure 7.1: Sample detections on held-out TrashCan test frames.", width=15)
figure(A / "coral_overlay.png", "Figure 7.2: Coral health overlay on Coralscapes reef images (left: input, right: model output; green = live, white = bleached coral).", width=15.5)
para("Fish detection was the weakest module (Table 7.6). A second detector trained with extra FathomNet fish "
     "images did not improve test F1, and an external fish model that reports 99.3% mAP@50 on its own "
     "Brackish data scored only 0.5% AP@50 on DeepFish, which shows how poorly fish detectors transfer to new "
     "water.")
table("Table 7.6. Fish detection experiments",
      ["Experiment", "Test data", "Precision", "Recall", "F1"],
      [
          ["1 - 16-class detector (ours)", "TrashCan test", "40.7%", "15.9%", "22.9%"],
          ["2 - fish detector + FathomNet (ours)", "TrashCan test", "26.5%", "20.5%", "23.1%"],
          ["3 - Community Fish Detector (pretrained)", "OzFish test", "76.7%", "59.3%", "66.9%"],
          ["4 - Brackish YOLOv8s (pretrained)", "DeepFish", "4.7%", "2.3%", "3.1%"],
      ], widths=[6.0, 3.0, 2.2, 2.2, 2.2], size=10)
para("Tracking over the 53 test videos produced 34 animal tracks, 8 of them usable moving tracks. Moving animals "
     "travelled about 12.3 pixels per frame against 1.75 for near-static ones, but only three tracks passed near "
     "plastic, which is too few to draw a behavioural conclusion.")
page_break()

# ---------------------------------------------------------------- 8 discussion
h1("8. Results Discussion")
para("The results show a clear ranking. Coral health can be assessed reliably from images: on reefs the model had "
     "never seen it found bleaching in 95.7% of images and reached a coral IoU of 0.767. Dead coral (IoU 0.328) "
     "was the hardest coral class because it looks like rubble and rock.")
para("Large and distinctive objects are detected well; the ROV was recognised in 91.5% of images and debris "
     "alarms were usually right. Small plastic items are harder, giving an object-level F1 of 46.3%. This is a "
     "reasonable figure for unseen videos: the SeaClear authors report a drop from 68% to about 20% mAP when "
     "testing on new sites [12]. Error analysis showed that plastic, metal and fishing gear suffer from both "
     "misses and false alarms, whereas fish and paper are mostly missed. Rare classes such as shells, fabric and "
     "rubber had too few training examples.")
para("Fish detection in dark deep-sea footage remains the main weakness, so the fish behaviour analysis is "
     "currently descriptive. The impact score and biodiversity-loss estimate follow an expert weighting; a "
     "real-data check across 698 ocean cells found that microplastic density added only 0.001 to the R² of "
     "predicting recorded species (0.848 against 0.847), so the link is an association, not proof of cause. "
     "All six expected outcomes of the brief are delivered in the dashboard, with objectives 2 and 4 met "
     "partially, as explained in Section 1.5.")
page_break()

# ---------------------------------------------------------------- 9 conclusion
h1("9. Conclusion and Future Scope")
h2("9.1 Conclusion")
para("MARIS combines plastic detection, animal tracking, coral health assessment and water quality data into a "
     "single Marine Pollution Impact Score with explanations, delivered through a working decision-support "
     "dashboard. Evaluated only on unseen videos and dive sites, the system reached 95.7% accuracy in recognising "
     "bleaching, a coral IoU of 0.767, 91.5% accuracy for ROV presence, 75.1% accuracy for plastic presence and "
     "48.9% mAP@50 for plastic objects. The project meets the integration goal identified in the research gap, "
     "while fish detection and field validation of the impact score remain open.")
h2("9.2 Future Scope")
bullets([
    ("More debris data: ", "retrain a larger YOLO11m-seg on TrashCan plus SeaClear to improve plastic detection."),
    ("Better fish detection: ", "collect and label more deep-sea fish video and add species identification."),
    ("Live water data: ", "read pH, oxygen and salinity from Copernicus Marine instead of station reference values."),
    ("Field validation: ", "calibrate the impact score against biodiversity surveys such as Reef Life Survey."),
    ("Model comparison: ", "test RT-DETR, DeepSORT and temporal models against the current choices."),
], numbered=True)
page_break()

# ---------------------------------------------------------------- 10 references
h1("10. References")
refs = [
    'J. R. Jambeck et al., "Plastic waste inputs from land into the ocean," Science, vol. 347, no. 6223, pp. 768-771, 2015.',
    'J. B. Lamb et al., "Plastic waste associated with disease on coral reefs," Science, vol. 359, no. 6374, pp. 460-462, 2018.',
    'T. P. Hughes et al., "Global warming and recurrent mass bleaching of corals," Nature, vol. 543, pp. 373-377, 2017.',
    'G. Jocher and J. Qiu, "Ultralytics YOLO11," 2024. [Online]. Available: https://github.com/ultralytics/ultralytics',
    'E. Xie et al., "SegFormer: Simple and efficient design for semantic segmentation with transformers," in Proc. NeurIPS, 2021.',
    'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. ACM SIGKDD, 2016, pp. 785-794.',
    'S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Proc. NeurIPS, 2017.',
    'R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in Proc. IEEE ICCV, 2017, pp. 618-626.',
    'M. Fulton, J. Hong, M. J. Islam, and J. Sattar, "Robotic detection of marine litter using deep visual detection models," in Proc. IEEE ICRA, 2019, pp. 5752-5758.',
    'J. Hong, M. Fulton, and J. Sattar, "TrashCan: A semantically-segmented dataset towards visual detection of marine debris," arXiv:2007.08097, 2020.',
    'M. J. Islam et al., "Semantic segmentation of underwater imagery: Dataset and benchmark," in Proc. IEEE/RSJ IROS, 2020.',
    'A. Đuraš, B. J. Wolf, A. Ilioudi, I. Palunko, and B. De Schutter, "A dataset for detection and segmentation of underwater marine debris in shallow waters," Scientific Data, vol. 11, art. 921, 2024.',
    'J. Sauder, V. Domazetoski, G. Banc-Prandi, G. Perna, A. Meibom, and D. Tuia, "The Coralscapes dataset: Semantic scene understanding in coral reefs," arXiv:2503.20000, 2025.',
    'K. Katija et al., "FathomNet: A global image database for enabling artificial intelligence in the ocean," Scientific Reports, vol. 12, art. 15914, 2022.',
    'Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every detection box," in Proc. ECCV, 2022.',
    'W. Skirving et al., "CoralTemp and the Coral Reef Watch coral bleaching heat stress product suite version 3.1," Remote Sensing, vol. 12, no. 23, art. 3856, 2020.',
    'NOAA National Centers for Environmental Information, "Marine Microplastics Database." [Online]. Available: https://www.ncei.noaa.gov/products/microplastics',
    'OBIS, "Ocean Biodiversity Information System," IOC of UNESCO. [Online]. Available: https://obis.org',
]
for i, r in enumerate(refs, 1):
    p = para(f"[{i}] {r}", size=11, after=4)
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)
page_break()

# ---------------------------------------------------------------- annexures
para("ANNEXURE", bold=True, size=16, align="c", after=10)
h2("Annexure A: Code Repository")
para("The source code is kept in the project Git repository. It contains the data preparation, training and "
     "evaluation scripts, the Colab notebooks, the analysis modules and the Streamlit dashboard.")
para("GitHub Repository: [add repository link]", bold=True, align="l")
bullets([
    "src/modules/ - plastic quantification, fish tracking, coral health, water quality, impact scoring, SHAP and Grad-CAM",
    "src/evaluation/ - held-out evaluation scripts with bootstrap confidence intervals",
    "notebooks/ - Colab training notebooks for the detector, coral and fish models",
    "app.py - Streamlit decision-support dashboard (run: streamlit run app.py)",
    "outputs/metrics/ - every reported number as JSON / CSV",
])
h2("Annexure B: Additional Results")
para("Supplementary graphs produced during training and evaluation.")
figure(FIG / "training_curves_yolo11s_seg_v2.png", "Figure B.1: Training and validation curves of the YOLO11s-seg detector.", width=15)
figure(FIG / "BoxPR_curve.png", "Figure B.2: Precision-recall curve of the detector.", width=13)
figure(FIG / "confusion_matrix_normalized.png", "Figure B.3: Normalised confusion matrix of the detector.", width=13)
figure(FIG / "shap_impact_explanation.png", "Figure B.4: SHAP explanation of the Marine Pollution Impact Score.", width=13)
figure(FIG / "gradcam_underwater_explanation.png", "Figure B.5: Grad-CAM views of where the detector looks.", width=15)
page_break()
h2("Annexure C: Implementation Evidence")
para("Screenshots and outputs from the MARIS dashboard covering the six expected outcomes of the brief.")
for path, cap in [
    (A / "dashboard.png", "Figure C.1: MARIS dashboard - image analysis page with the six outcome pages."),
    (FIG / "plastic_pollution_map.png", "Figure C.2: Plastic pollution map from NOAA NCEI measurements."),
    (FIG / "fish_behavior_tracking.png", "Figure C.3: Fish tracks and behaviour near plastic."),
    (FIG / "water_quality_radar.png", "Figure C.4: Water quality parameters against reference ranges."),
    (FIG / "marine_pollution_impact_score_dist.png", "Figure C.5: Marine Pollution Impact Score across survey images."),
    (FIG / "biodiversity_loss_prediction.png", "Figure C.6: Projected biodiversity loss."),
    (FIG / "real_world_plastic_biodiversity.png", "Figure C.7: Real-data check: microplastics against recorded species."),
]:
    if path.exists():
        figure(path, cap, width=14.5)
page_break()
h2("Annexure D: Team Contributions")
para("Contributions of each team member to the MARIS project.")
table("Table D.1. Team contributions", ["No.", "Team member", "Contribution"],
      [[str(i), n, "[describe contribution]"] for i, (_, n) in enumerate(TEAM, 1)], widths=[1.5, 5, 9])

doc.save(OUT)
print("saved", OUT)
