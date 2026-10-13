# Prompt for an LLM to help improve MARIS

Copy everything inside the box below into another LLM (ChatGPT, Gemini, Claude, ...). Update the
"Current results" numbers first if they have changed. Attach `prompt.md` and
`outputs/reports/executive_impact_report.md` too if the tool accepts files.

---

```text
You are a senior ML engineer and marine-science reviewer helping a 3rd-year university team improve
their AI/ML project. Be concrete, honest and skeptical. Never suggest faking, simulating or
hard-coding results; every number must come from real data and a held-out evaluation, and
anything that is an assumption must be labelled as one.

PROJECT
"MARIS: AI-Based Marine Plastic Pollution Impact Assessment on Biodiversity."
Brief: detect underwater plastic; identify and track fish and behaviour change; assess coral health
and bleaching; integrate temperature, pH, dissolved oxygen, salinity, turbidity; estimate the
ecological impact on biodiversity; output a Marine Pollution Impact Score (0-100); give explainable
AI recommendations; deliver plastic pollution maps and a decision-support dashboard.
Datasets named in the brief: TrashCan 1.0, SUIM, Trash-ICRA19, Fish4Knowledge, Reef Life Survey,
NOAA Coral Reef Watch, Allen Coral Atlas, OBIS, Copernicus Marine.
Algorithms named: YOLOv11, RT-DETR, ViT, U-Net, DeepSORT, ByteTrack, XGBoost, Temporal Fusion
Transformer, GNNs, SHAP, Grad-CAM.

STACK
Python, Ultralytics YOLO11-seg, PyTorch, segmentation-models-pytorch, XGBoost, SHAP, OpenCV,
Streamlit dashboard (6 pages), training on Google Colab (free T4) and a 4 GB RTX 3050 laptop.

WHAT EXISTS AND IS REAL
1. Plastic detection: YOLO11-seg on TrashCan 1.0 (16 classes, deep-sea ROV footage, JAMSTEC).
   Video-disjoint train/val/test split (312 videos, 0 overlap). Current model: trash_plastic
   (superseded: see docs/llm_finish_fast_prompt.md for current numbers) old model precision ~0.37,
   recall ~0.31 on unseen videos. A YOLO11s-seg retrain is running on Colab
   (overall val mAP50 ~0.21 at epoch 54/120). Fallback: add Trash-ICRA19 (phash-deduplicated).
   Known failure: false "plastic" on the camera's on-screen text overlay.
2. Grad-CAM: real gradients of the class logit w.r.t. YOLO neck layers; ~32% of heat falls inside
   plastic boxes that cover a few % of the image.
3. Fish tracking: ByteTrack over consecutive TrashCan video frames with phase-correlation camera
   motion compensation. Mobile animals 10.3 px/frame vs 0.9 px/frame for near-static starfish
   (noise floor). Only 9 usable mobile tracks, 1 near plastic, so no behaviour-change claim.
4. Coral: U-Net (ResNet-34) trained on SUIM. Official TEST: reef IoU 0.684, precision 0.762,
   recall 0.870, mIoU 0.684. SUIM has no bleaching labels, so bleaching is a colour rule inside the
   reef mask. Out of domain on deep-sea footage (marked 595/1313 TrashCan frames as reef);
   now requires agreement with a colour/texture check (0/1313).
5. Water: NOAA Coral Reef Watch satellite SST, Degree Heating Weeks and bleaching alert for 5
   stations (real). pH, dissolved oxygen, salinity, turbidity are presets until Copernicus login.
6. Impact score + biodiversity loss (image level): XGBoost reproducing a documented expert formula
   (synthetic data, held-out R2 0.945 of the formula). Weights are assumptions.
7. Observational model (new): per 2-degree ocean cell, NOAA NCEI microplastics (14,580 surface
   samples, pieces/m3) + OBIS species counts + NOAA SST/DHW; XGBoost with sampling-effort and
   latitude controls, spatial block CV, ablation with/without the plastic feature.
8. SHAP on every score, rule-based recommendations, station map, CSV + auto-generated report.

LIMITS WE KNOW ABOUT
- TrashCan images have no geotags; images are assigned to stations in rotation.
- No public dataset links plastic exposure to measured biodiversity loss at image level.
- Detector weak on plastic; data imbalance; HUD text false positives.
- Free Colab: one GPU session, ~12 h max, disconnects.

WHAT I WANT FROM YOU
A. Rank the 5 highest-impact improvements for marks and scientific honesty, each with: the exact
   change, the dataset/tool, expected effort (hours), how to measure success (metric + held-out
   split), and the risk if it fails.
B. Plastic detector: concrete recipe to raise trash_plastic recall on unseen TrashCan videos
   (model size, image size, augmentations, class weighting or focal loss, tiling, SAHI,
   copy-paste augmentation, HUD masking, combining with Trash-ICRA19 without test leakage).
   Give an Ultralytics training command.
C. Coral bleaching: which public healthy-vs-bleached datasets exist, how to train and validate a
   classifier inside the SUIM reef mask, and how to cross-check with NOAA bleaching alerts.
D. Fish behaviour: how to use Fish4Knowledge (or DeepFish) to validate tracking (MOTA/IDF1) and
   what behaviour metrics are defensible.
E. Impact score: how to move from the expert formula to something grounded in real data
   (the NCEI + OBIS + NOAA cell model), and how to present the correlation-vs-causation limits.
F. Anything in our evaluation that a strict examiner would attack, and how to pre-empt it.

Answer with numbered sections A-F. Prefer specific, testable steps over general advice. If you are
unsure whether a dataset or API exists, say so instead of guessing.
```
