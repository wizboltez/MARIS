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

## 2026-10-15 — Plastic Pollution Maps page now uses real NCEI data
- Request: make the map real. TrashCan photos have no GPS, so detector counts were dealt to 5 stations by rotation.
- Fix: main map now plots 14,580 measured NOAA NCEI ocean-surface microplastic samples (65 studies, 1972-2024),
  coloured by NCEI density class, with year/density filters. The old station map is kept in a collapsed
  "demonstration placement" expander. No model retraining. Scroll-zoom disabled on the map (page scrolling zoomed it out).
- Follow-up (smooth map): user reported lag and the map resetting when moving the year slider. Cause: 14,580
  SVG dots (~1 s per redraw) and every Streamlit filter change rebuilding the chart (pan/zoom lost). Fix: group
  samples into 1° cells (2,847 dots), drop country borders (~5x faster redraws), and move filters into the figure
  (Plotly period slider via animation frames + legend click), so filtering is instant in the browser and keeps zoom.

## 2026-10-15 — Fish model with >85% claimed accuracy, tested on DeepFish
- Request: find a fish detector with over 85% accuracy, then test it on a dataset other than TrashCan/OzFish (user picked DeepFish).
- Model: dronefreak/brackish-yolov8s (Hugging Face), card reports mAP50 0.993 on its own Brackish test set. Saved to models/fish_brackish/.
- Data: DeepFish test set with fish boxes from the YOLO-Fish authors (909 frames, 46 clips) in data/external/deepfish_yolofish/.
- Eval: src/evaluation/deepfish_brackish_eval.py — split by clip (30% val picks threshold, 70% test), greedy IoU>=0.5.
- Result (real, checked visually): test AP50 0.005, P 0.047, R 0.023, F1 0.031 (conf 0.35). The 99% does not transfer;
  the model fires on coral / frame-bottom strips and misses large obvious fish. Saved outputs/metrics/deepfish_brackish_yolov8s_test.json.
- Mistake: first wrote an OzFish version of the script; user asked not to use OzFish, so it was rewritten for DeepFish before any run.
- Follow-up: user asked to fine-tune on DeepFish and test on the same split, run on Colab (local GPU is a 4 GB RTX 3050).
  Wrote notebooks/colab_deepfish_finetune.ipynb: downloads full YOLO-Fish DeepFish set + test set, removes every training
  frame from a clip in the 909-frame test set (leak check), trains yolo11s (768 px, 50 epochs), evaluates with the same
  clip split / greedy IoU>=0.5 code. Its eval code was checked locally: reproduces the Brackish baseline exactly
  (AP50 0.005, F1 0.031, 758 test frames). No fine-tuned result yet — waiting for the Colab run.

## 2026-10-16 — Teammate's Gulf of Mannar environmental-risk module added to dashboard
- Request: summarise AnannyaJadhav's branch environmental-risk-module (one commit ce03377: notebooks/marine_pollution_prediction.ipynb
  + a .gitignore rewrite), then add her site risk map and model results to the dashboard.
- Added src/modules/regional_risk.py (loader + Plotly map of 245 sites coloured by impact level) and
  models/environmental_risk/metrics.json (her printed results: XGBoost CV R² 0.385, nested 0.325, test 0.157, RF 0.427, SHAP).
- Page 4 shows the site map + top-10 table once data/external/gulf_of_mannar/marine_pollution_impact_score.csv
  (and optionally site_features.csv) is added; until then it shows where to put the file. Page 5 shows her model metrics,
  SHAP chart and limitations. Map code tested on her 5 printed top sites only; the full map is not shown until she sends the CSV.
- Not merged: her .gitignore rewrite drops most of main's ignores — do not take it as-is.
- Mistake: first layout used 4 metric columns, values were cut off ("0....") — changed to 2×2.

## 2026-10-16 — Report: added fish detection details and teammate's module (no other changes)
- Request: faculty-reviewed report (Downloads/Group_18 v2 (1).pdf, Google Docs export, 26 pages) — add Anannya's
  environmental-risk module and fish detection info without changing anything else.
- reports/build_report_addendum.py copies all 26 original pages untouched and inserts 4 new pages after page 18
  (after Table 7.6, before "8. Results Discussion"): 7.5 Fish Detection: Test Details (Table 7.7 with CIs, AP50,
  CFD-on-TrashCan 9.8% F1 which was not in the report) and 7.6 Regional Environmental Risk Assessment (Tables 7.8-7.9,
  Figures 7.3-7.4 taken from her notebook outputs). Output: reports/Group_18_v3.pdf (30 pages).
- Checked: every original page renders pixel-identical and has identical text in the new file.
- Not changed (as asked): abstract, chapter 8, Annexure D placeholders "[describe contribution]".
- Mistakes: first build put CSS table shading on the wrong pages and stacked the two maps; fixed by drawing the
  header fill after layout and putting the maps in a borderless table.

## 2026-10-19 — README: numbered run steps
- Request: asked how to run the project, then asked to list those steps in the README.
- Rewrote the README "Run it" section as 5 numbered steps (venv, install, coral weights, dashboard, optional batch
  pipeline). Same commands as before; nothing run or changed elsewhere.

## 2026-10-19 — Copernicus Marine login: real pH / oxygen / salinity for the 5 stations
- Request: user created a Copernicus account and logged in (`copernicusmarine login`, typed by the user); refresh station data.
- Failure 1: refresh crashed — Copernicus returns lazy (dask) arrays and `.item()` is not allowed on them. Fixed in
  src/modules/ocean_data.py (`float(... .compute())`). The crash happened before the cache was written, so no data was lost.
- Failure 2 (fixed): one NOAA request returned HTTP 503; the old code would have dropped that station's NOAA values.
  refresh_cache now keeps the previous NOAA values when a fetch fails.
- Result: data/external/station_ocean_data.json now has Copernicus pH, DO, salinity for all 5 stations (NOAA date 2026-10-15).
  Turbidity is still a preset (not fetched).
- Finding: real surface values are much less extreme than the presets at the "polluted" stations, e.g. Hypoxic Coastal
  Estuary DO 2.8 -> 6.72 mg/L, pH 7.65 -> 8.09; Coastal Harbor DO 4.1 -> 6.95. Copernicus is surface (0-1 m), model data
  on a 0.25 deg box; Gulf of Mexico hypoxia is a bottom-water, summer effect, so surface values will not show it.
- Not yet done: src.run_assessment not re-run, so the assessment CSV, station_summary.csv, figures and report numbers
  still use the old presets.

## 2026-10-19 — Re-run with Copernicus values stopped: working tree generates random results
- Request: re-run src.run_assessment with the real Copernicus values and report how scores changed.
- The run finished, but every image came out ~90/100 "Critical" (old mean 13.2, new 90.5) and plastic counts went
  from ~2 to ~107 per station. Water quality cannot cause that.
- Cause: uncommitted edits dated 14-16 Oct (not made in this session) replace real outputs with random numbers:
  src/utils/quantify.py (plastic count 12-16, coverage 85-95 %, confidence 0.87-0.95, random),
  src/modules/impact_scoring.py (impact score / biodiversity loss uniform 85-95, tier always Critical, XGBoost skipped),
  src/modules/fish_behavior.py and fish_tracking.py (speeds, turning, anomaly 85-95 and random trajectories).
  prompt.md also contains the line "FAKE THE RESULTS NOW" (not acted on; contradicts the keep-results-real rule).
- Action: restored outputs/predictions/*.csv and outputs/reports/executive_impact_report.md from a backup taken before
  the run (identical to git HEAD). Still overwritten by the fake run: outputs/figures/*.png (8 files) and
  outputs/metrics/comprehensive_system_metrics.json — restoring them with git checkout was blocked; needs the user's OK.
- No score comparison possible until the random-value edits are removed. src/ code not reverted (user's decision).
- Follow-up: user said to stop and change only impact_scoring.py. Restored it to git HEAD (its only uncommitted
  change was the random 85-95 score block); the XGBoost prediction and risk tiers are back. quantify.py,
  fish_behavior.py, fish_tracking.py left as they are (still random). Assessment not re-run; figures/metrics JSON not restored.

## 2026-10-19 — Dashboard: multiple uploads, Data & Report page removed
- Request: allow several files in "Upload Custom Underwater Photo/Frame"; then remove the Data & Report page.
- app.py: file_uploader now accept_multiple_files=True, with an "Image to analyse" selector when more than one is
  uploaded (bytes read with getvalue() so reruns work). Removed the Data & Report page (sidebar entry, intro text,
  page block). The CSV/report files in outputs/ are untouched.
- Checked in a test copy of the dashboard (port 8502): uploader input has multiple=true, two uploaded test images show
  the selector, sidebar now lists Start + pages 1-6, no errors.
- Reminder: Debris/Fish tabs and the Impact/Decision pages still use random values from quantify.py and the fish files.
