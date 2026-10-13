# MARIS — Prompts, Mistakes, Failures & Fixes

Living log for this project. Updated as work happens: what was asked, what went wrong, and how it was
fixed or how to avoid it next time. Newest entries go at the bottom of each section.

---

## 1. Project prompt (from the FA1/FA2 brief)

> **AI-Based Marine Plastic Pollution Impact Assessment on Biodiversity.**
> Develop an AI system that detects underwater plastic waste, analyzes fish behavior and coral health,
> integrates water quality data, and estimates the ecological impact of plastic pollution on marine
> biodiversity.

**Objectives:** detect & classify underwater plastic · identify & track fish and behavioural change ·
assess coral health & bleaching · integrate temperature, pH, dissolved oxygen, salinity, turbidity ·
estimate impact on biodiversity · Marine Pollution Impact Score (0–100) · explainable AI recommendations.

**Datasets named in the brief:** TrashCan 1.0, SUIM, Trash-ICRA19, Fish4Knowledge, Reef Life Survey,
NOAA Coral Reef Watch, Allen Coral Atlas, OBIS, Copernicus Marine.

**Algorithms named:** YOLOv11, RT-DETR, ViT, U-Net, DeepSORT, ByteTrack, XGBoost, Temporal Fusion
Transformer, GNNs, SHAP, Grad-CAM.

**Expected outcomes:** plastic pollution maps · fish behaviour analysis · coral health assessment ·
Marine Pollution Impact Score · biodiversity loss prediction · decision-support dashboard.

---

## 2. Requests made during development (session log)

| Date | Request | Outcome |
|---|---|---|
| 2026-10-12 | "Does this project work as expected now?" (against the brief) | Ran pipeline + dashboard: ran without errors, but much of it was simulated (see §3). |
| 2026-10-12 | "What will the user enter and what should they get?" | Documented dashboard inputs/outputs and which outputs were real. |
| 2026-10-12 | "Quick fixes" | Plastic-balanced sampling, SHAP background fix, computed SHAP report, inflated-metric warning, provenance labels. |
| 2026-10-12 | "Complete the stages that don't need training" | Real Grad-CAM, ByteTrack fish tracking, coral-presence check, NOAA CRW data, honest impact model, real station map. |
| 2026-10-13 | "Keep everything real; train on Colab. If accuracy is low, use another dataset." | Colab notebooks for detector retrain (+ Trash-ICRA19 fallback) and SUIM U-Net, each with a quality gate. |
| 2026-10-13 | "Keep updating prompt.md with the prompt, mistakes, failures and fixes." | This file. |
| 2026-10-13 | Run the two jobs simultaneously or separately? Can job 2 (SUIM) go first? | Independent jobs: run in parallel on different machines (Colab + laptop RTX 3050 4 GB, or two accounts), not two GPU sessions on one free Colab account. Order doesn't matter technically; detector matters more. |
| 2026-10-13 | Put the SUIM direct-download option into the notebook. | Added `SUIM_GDRIVE_FILE_ID` config. |
| 2026-10-13 | SUIM U-Net training progress check (epoch 28). | Val reef IoU 0.70–0.75 (gate 0.50); val mIoU plateaued ~0.63 from epoch 21 while train loss keeps falling (mild overfitting; early stopping keeps best checkpoint). Val set is small (~10%), so epoch-to-epoch noise is ±0.05. Final verdict pending on the official SUIM TEST split. Warnings seen (HF_TOKEN, GradScaler deprecation) are harmless. |
| 2026-10-13 | SUIM progress check (epoch 50). | Val reef IoU stabilised at 0.75–0.76 from epoch ~36; best val mIoU 0.663 (epoch 42, saved checkpoint). Gains in later epochs came from the cosine LR decay; no val degradation. Expect early stop at ~54 or the 60-epoch cap. |
| 2026-10-13 | **SUIM U-Net final result.** | **PASS — reef IoU 0.684 on the official SUIM TEST split** (gate 0.50). Lower than val (~0.75) as expected for a held-out split; val was also used for checkpoint selection. Next: copy `suim_unet.pt`, `suim_unet_config.json`, `suim_test_metrics.json` into `models/coral_segmenter/`. |
| 2026-10-13 | SUIM model files copied into `models/coral_segmenter/`; asked what 0.68 means and what's next. | Explained IoU. Model integrated: TEST reef IoU 0.684, precision 0.762, recall 0.870, mIoU 0.684, pixel acc 0.854. Report + dashboard now show these numbers from the metrics JSON. |

---

## 3. Mistakes & failures found in the project (and fixes)

| # | Problem | How it was found | Fix / how to avoid |
|---|---|---|---|
| P1 | **Train/test leakage** in the detector: an earlier Colab notebook re-derived the split, 77% of local test images were in training → reported precision 0.83 / recall 0.60 (real plastic: ~0.37 / ~0.31). | Teammate re-check (HANDOFF.md). | Colab now reads `split_manifest.csv`; notebook asserts every video is in exactly one split (verified: 312 videos, 0 shared). `eval_metrics_test.json` flagged as inflated until retrain. |
| P2 | **Survey sample had zero plastic**: pipeline took the first 40 test images by filename; none contained labelled plastic, yet the report said plastic was the top risk driver. | Inspected `marine_biodiversity_assessment.csv` (plastic_count all 0). | `src/utils/survey_sampling.py`: 20 images with labelled plastic + 20 without, seeded. |
| P3 | **Hard-coded report text**: SHAP drivers ("+18.4 to +26.1") typed into the report template. | Code review of `run_assessment.py`. | SHAP summary now computed over all transects. Rule: no numbers in report templates that aren't computed. |
| P4 | **SHAP had no background data** after loading saved models (`training_df = None`), and when it had it, the target columns were included. | Smoke test printed `training_df loaded: False`. | Regenerate the seeded training data on load; pass only model feature columns. Check: base value + Σ SHAP = prediction. |
| P5 | **Fake Grad-CAM**: blurred circles drawn on detection boxes. | Code review of `xai_engine.py`. | `src/modules/gradcam.py`: real gradients of the class logit w.r.t. YOLO neck layers 16/19/22. Verified heat concentrates in plastic boxes (~32% of heat in boxes covering a few % of image). |
| P6 | **Fake fish behaviour**: speed/turn computed from box size and list index; trails/arrows drawn from a formula; random polar plot. | Code review of `fish_behavior.py`. | `src/modules/fish_tracking.py`: ByteTrack over consecutive video frames + phase-correlation camera-motion compensation. Fakes removed. |
| P7 | **Coral bleaching reported on deep-sea images with no coral** (10/40 "Alert Level 2"). Pale sediment / marine snow matches the "bleached" colour rule. | Assessment CSV + knowing TrashCan is JAMSTEC deep-sea footage. | Coral-presence check (0/1313 deep-sea false passes); TrashCan frames marked non-reef. SUIM U-Net to replace the heuristic. |
| P8 | **Made-up water quality and random map**: presets cycled per image; 75 random map stations. | Code review. | Real NOAA Coral Reef Watch SST / DHW / bleaching alert per station (`ocean_data.py`); map built from the 5 real stations. pH/DO/salinity still presets until Copernicus login. |
| P9 | **Misleading impact-model docstring**: claimed synthetic data was "calibrated against" papers. | Code review. | Rewritten as a documented expert formula (papers motivate directions only); `coral_present` feature so "no coral" ≠ healthy; held-out R² reported (0.945) with a note that it measures formula reproduction, not ecology. |
| P10 | **Detector reacts to on-screen camera text** (timestamps/HUD) → false "plastic". | Grad-CAM heat on HUD text; false plastic box on HUD in tracking figure. | Report it. Mitigation: retraining (Colab); optionally mask the HUD regions at inference. |
| P11 | **Too little fish data near plastic**: 9 usable mobile tracks, 1 near plastic. | Tracking summary. | State the limitation; behaviour-change claims not supported by this dataset. Fish4Knowledge (brief) could add fish video if needed. |
| P12 | **SUIM U-Net is out of domain on deep-sea footage**: on the 1313 TrashCan test frames (no reef) it marked 595 frames as >=2% reef — mud, sediment and rock coloured as reef. | Ran the U-Net over every TrashCan test frame and looked at the top predictions. | Auto mode requires U-Net AND the colour/texture coral check -> 0/1313 false positives. TrashCan frames stay 'non_reef'. Saved as `models/coral_segmenter/domain_check_trashcan.json` and reported. Lesson: a good test score on one dataset does not transfer to a different domain; always check on the target imagery. |

---

## 4. Mistakes made while building (by the assistant) and how they were handled

| # | Mistake / failure | Effect | Fix / lesson |
|---|---|---|---|
| A1 | Report said plastic found "in 21 of the 20 images with labelled plastic". | Wrong wording (counted false alarms as hits). | Split into hits on labelled images vs false-alarm images. Lesson: sanity-check every ratio in generated text. |
| A2 | Dashboard default image had labelled plastic but the detector missed it → first view showed 0 plastic. | Poor first impression. | Default to a frame where the pipeline actually detected plastic. |
| A3 | ByteTrack with Ultralytics defaults produced **0 tracks** on video 080. | Fish scored ~0.1–0.4, below the default 0.25 track-start threshold. | Custom `bytetrack_trashcan.yaml` (start 0.15, low 0.05); detection conf 0.05 for tracking, plastic kept at 0.18. |
| A4 | Starfish/shells dominated tracks with noisy ~100° "turns". | Would have inflated behaviour metrics. | Behaviour uses mobile classes only (fish/eel/crab); static animals used as the speed noise floor (0.9 vs 10.3 px/frame). Turn angles ignored for steps < 2 px. |
| A5 | Risk: `model.track()` registers tracker callbacks on the model, which would alter later `predict()` calls. | Could silently drop detections elsewhere. | Separate YOLO instance for tracking (`load_tracking_model`). Grad-CAM also uses its own deep copy. |
| A6 | NOAA ERDDAP `coastwatch.pfeg` redirected to a mirror that **timed out**. | No data. | Use `coastwatch.noaa.gov/erddap` datasets `noaacrwsstDaily`, `noaacrwdhwDaily`, `noaacrwbaa7dDaily`; cache to `data/external/station_ocean_data.json`. |
| A7 | Shell heredocs turned `\n` inside Python f-strings into real newlines → **SyntaxError** (twice). | Broken files until fixed. | Write patch scripts to files with the Write tool, or fix with exact edits; always `py_compile` after patching. |
| A8 | Running Streamlit server kept **stale `src/` modules** → `ImportError: build_station_table`. | Dashboard error after edits. | Restart the server after changing anything under `src/`. |
| A9 | An editor temp file `fish_tracking.py.tmp.*` got **committed**. | Junk in the repo. | Removed and amended. Lesson: check `git status` before committing. |
| A10 | Test harness used the wrong notebook cell index. | False failure in local notebook test. | Fixed the index; the notebook itself was fine. |
| A11 | Could not verify the coral-presence thresholds on real reef images (none in the project). | Unknown true-positive rate on reefs. | SUIM TEST evaluation will measure it; documented as "not validated on reef imagery" in code. |
| A12 | First Trash-ICRA19 dedupe draft compared hashes in a pure-Python double loop (~13M comparisons). | Would have been very slow on Colab. | Vectorised Hamming distance with NumPy. |
| A14 | SUIM is a single 4.9 GB `SUIM.zip` on Google Drive (shows "can't scan for viruses" — normal for files >100 MB). Downloading to the laptop and re-uploading would waste time. | Slow setup. | Notebook config `SUIM_GDRIVE_FILE_ID`: downloads straight into Colab with gdown. |
| A15 | My first gdown snippet used `--id`, which newer gdown versions removed; it also didn't handle Drive's "too many users" download block. | Would fail on current gdown / give a confusing crash. | Use the `https://drive.google.com/uc?id=...` URL form; assert the zip exists and is >100 MB with a clear message and fallback (put zip in `MyDrive/suim/`). |
| A16 | Headless dashboard test (Streamlit AppTest) failed: relative `app.py` path resolves against the test script's folder. | Test didn't start. | Pass the absolute path. All 6 pages then ran with 0 exceptions. |
| A17 | Dashboard used `use_container_width`, deprecated (removal date 2025-12-31 already passed). | Warnings in the server log; would break on a newer Streamlit. | Replaced all 11 uses with `width="stretch"`. |
| A18 | `suim_unet.pt` is ~98 MB (GitHub warns >50 MB, hard limit 100 MB). | Risky to commit. | Git-ignored the `.pt`; config + metrics JSONs committed; weights live on Drive (`MyDrive/suim/`). Use Git LFS if teammates need it in the repo. |
| A13 | Notebooks can't be run locally end-to-end (no GPU / Colab Drive). | Risk of discovering bugs mid-session on Colab. | Ran all non-GPU cells locally: split + leakage check on the real TrashCan data, COCO→YOLO conversion, ICRA19 VOC parser, phash dedupe, SUIM mask decoding + TEST/train separation on synthetic folders. Also tested that Grad-CAM/quantify work with a detection-only model and that the coral analyser works with/without a U-Net. |

---

## 5. Current status & open issues

- **Real:** plastic detection (weak until retrain), Grad-CAM, fish tracking, NOAA CRW temperature / heat stress / alerts, SHAP, station map.
- **Assumed:** image→station assignment (TrashCan has no geotags); pH / DO / salinity / turbidity (until `copernicusmarine login`); impact-score & biodiversity-loss formula weights.
- **Training status:**
  - `notebooks/colab_train.ipynb`: YOLO11s-seg on TrashCan; gate `trash_plastic` test mAP50 ≥ 0.40, else Phase 2 adds Trash-ICRA19 (deduplicated against TrashCan val/test) and keeps whichever is better on the same test split.
  - `notebooks/colab_train_suim_coral.ipynb`: U-Net (ResNet-34) on SUIM — **DONE, PASS: TEST reef IoU 0.684** (gate 0.50). Integrated; out of domain on deep-sea footage (P12). Note: SUIM has no bleached/healthy labels — it gives reef location; bleaching stays a colour rule inside the reef mask.
- **Fallback datasets if gates fail:** detector → Trash-ICRA19 (built in); coral → larger encoder / resolution first, then a healthy-vs-bleached coral image dataset for bleaching specifically.
