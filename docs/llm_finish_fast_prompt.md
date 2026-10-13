# Prompt: finish MARIS fast, improve only what needs accuracy

Paste the block below into another LLM (works best in a coding agent that can see the repo).

```text
You are the lead engineer finishing a university AI project in the shortest time possible. Work only on
the items below that still need accuracy; everything else is DONE and must not be redone. Be brief.

=== HARD RULES (a previous answer broke these; do not) ===
1. Never write a metric, confidence interval, ablation gain, dataset size or timing you did not measure
   or cannot cite. If a number is a target, label it "TARGET". If unknown, write "TO MEASURE".
2. Every reported accuracy comes from a held-out TEST split never used for training, threshold tuning
   or model selection. Tune on VAL only.
3. Only name datasets/APIs you are sure exist, with a link. If unsure, say "verify first".
4. Code must handle TrashCan images of BOTH 480x270 and 480x360 (read the real image size).
5. Prefer the smallest change that moves a metric. No rewrites of working modules.

=== PROJECT ===
MARIS: AI marine plastic pollution impact assessment. Streamlit dashboard (6 pages, 0 errors),
batch pipeline src/run_assessment.py -> auto-generated report. Python 3.12, Ultralytics 8.4, PyTorch,
XGBoost, SHAP, segmentation-models-pytorch. GPU: free Colab T4 (~2 min/epoch for YOLO11s-seg at 640,
sessions disconnect) and a 4 GB RTX 3050 laptop.

=== DONE (measured, keep as is) ===
- Video-disjoint TrashCan split: train 4913 / val 986 / test 1313 images; 312 videos, test = 53 videos.
- Coral reef U-Net (SUIM official TEST): reef IoU 0.684, P 0.762, R 0.870. Deep-sea domain gate in place
  (U-Net AND colour check -> 0/1313 false reef on TrashCan).
- NOAA Coral Reef Watch SST / heat stress / bleaching alert per station (real).
- Observational model (698 ocean cells, NCEI microplastics + OBIS + NOAA): spatial-CV R2 0.848 with
  plastic vs 0.847 without (+0.001). Honest null result; keep.
- Impact score = documented expert formula (MCDA surrogate), labelled as such. SHAP on every score.
- Fish tracking pipeline (ByteTrack + camera-motion compensation) works; it is starved of detections.

=== NEEDS ACCURACY (in priority order) ===
A. Plastic detector (most important; everything downstream uses it).
   YOLO11s-seg, TrashCan TEST: trash_plastic P 0.611, R 0.374, mAP50 0.489, mAP50-95 0.340.
   Best epoch 30, early stop at 60 (val flat after 30 = overfitting). Inference conf fixed at 0.18
   (untuned). Known false positives on the camera's on-screen text (timestamps/telemetry).
   Note: Ultralytics' reported P/R are at its own best-F1 confidence, not at 0.18.
   Goal: raise plastic recall without losing precision below ~0.5.
B. Fish detection: animal_fish test R 0.093 (P 0.427) -> only 8 usable fish tracks, 3 near plastic.
C. Grad-CAM: target = all anchors scoring >= conf for the class, so only ~6% of heat falls inside the
   detected plastic box. Needs per-detection targeting (anchor centres at (i+0.5)*stride, strides 8/16/32,
   map box to letterbox coords) plus a deletion/insertion faithfulness check.
D. Coral bleaching: colour rule inside the reef mask (SUIM has no bleaching labels). Optional if a
   verified public healthy-vs-bleached dataset exists.
E. Water: pH, oxygen, salinity, turbidity are presets until a Copernicus Marine login (loader written).
   This is a user action, not engineering.

=== WHAT TO RETURN ===
1. A 1-day plan: ordered steps for A -> B -> C (D only if time), each with
   (a) the exact change, (b) VAL used for, TEST used for, (c) success = which metric must improve,
   (d) T4 or CPU time as a range you are confident in, (e) a stop rule ("if no gain after X, keep current").
2. Ready-to-run code only for the steps that need new code (thresholds on VAL, on-screen-text handling
   with a measured count of real detections it would remove, per-detection Grad-CAM, cluster bootstrap
   95% CI over the 53 test videos). Keep each under ~80 lines.
3. Exact Colab commands for one detector retrain that adds Trash-ICRA19 (deduplicated against TrashCan
   val+test by perceptual hash) with copy-paste augmentation, resumable from Drive.
4. The final report table layout with every cell marked TO MEASURE (not filled in).
5. A cut list: what to skip if time runs out, and how to state each limitation honestly.
```
