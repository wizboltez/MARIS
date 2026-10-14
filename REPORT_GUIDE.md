# Report Guide — MARIS: AI-Based Marine Plastic Pollution Impact Assessment on Biodiversity

How to write the FA report, following the faculty's structure. Every number below is a **real, measured result
on held-out data** from this project (source file given). Results and Discussion are ordered **strongest first,
weakest last**.

Rules from the faculty notes, applied throughout:
- Figure caption **below** the figure; table caption **above** the table.
- Mention every figure and table in the text ("as shown in Fig. 3", "Table 4 lists ...") — mostly in Results.
- Datasets: name, source URL, access date, licence, citation, total samples; one summary table.
- References: IEEE style, at least 10, recent, 1–2 conference papers. Plagiarism under 10 %: **write in your own
  words** — the draft sentences here are a starting point, rephrase them.
- Items marked **[verify]** must be checked on the source page before submission.

---

## 0. Title

**MARIS: A Multi-Model AI System for Assessing the Impact of Marine Plastic Pollution on Coral Reefs and Marine Biodiversity**

(Shorter option: *AI-Based Assessment of Marine Plastic Pollution and Its Impact on Coral Reef Biodiversity*)

## 1. Abstract (single paragraph, max 300 words)

Must contain, in order: motivation → problem → methodology → results/conclusion. Draft (≈230 words, rephrase):

> Plastic pollution threatens marine ecosystems, yet monitoring it still relies on manual review of underwater
> footage. This work presents MARIS, an AI system that detects underwater plastic, analyses fish movement, assesses
> coral health and combines these with satellite ocean data into a Marine Pollution Impact Score (0–100). Debris
> is detected and segmented with YOLO11-seg trained on TrashCan 1.0; animals are tracked across video frames with
> ByteTrack; coral is segmented into live, bleached and dead coral by a SegFormer (MiT-B2) model fine-tuned on the
> Coralscapes dataset; sea-surface temperature and bleaching alerts come from NOAA Coral Reef Watch. An XGBoost
> model produces the impact score, explained with SHAP and Grad-CAM, and results are delivered through an
> interactive decision-support dashboard. All models were evaluated on data never used for training — unseen
> videos and unseen dive sites. The coral model detects bleaching in images with 95.7 % accuracy, separates coral
> from non-coral pixels with 95.2 % accuracy and reaches a coral IoU of 0.767 on five unseen Red Sea dive sites;
> on deep-sea footage it raised a false coral alarm on only 1 of 200 frames. The debris detector identifies
> underwater robots in 91.5 % of images and is correct in 89.3 % of its debris alarms, while plastic detection
> (F1 0.46, mAP50 0.49) and fish detection remain the main challenges. The results show that coral health can be
> monitored reliably from imagery, and that larger, more diverse debris datasets are the key next step.

## 2. Keywords (4–6)

Marine plastic pollution; Coral bleaching; Semantic segmentation; YOLO11; SegFormer; Explainable AI

## 3. Introduction (5 paragraphs — or 3 by merging 2+3)

| Para | Content | Points to use |
|---|---|---|
| 1 | Introduce the topic | Millions of tonnes of plastic enter the ocean each year [1]; reefs support a large share of marine biodiversity. |
| 2 | Background | Plastic on reefs is linked to coral disease [2]; warming causes mass bleaching [3]; monitoring today = divers / manual video review — slow, costly, not scalable. Deep learning detectors (YOLO) and segmenters (U-Net, SegFormer) make automated analysis possible [4]–[8]. |
| 3 | Research problem | Existing work looks at one thing at a time (only debris, or only coral, or only fish). No single system links plastic, fish, coral and water conditions into one impact estimate a manager can act on. Also, many reported accuracies come from tests on images very similar to the training data. |
| 4 | Objectives + uniqueness | The 7 objectives from the brief (detect plastic; track fish; coral health; integrate water data; estimate biodiversity impact; impact score 0–100; explainable recommendations). **Uniqueness:** one pipeline + dashboard combining all modules; coral model that separates live / bleached / dead coral; strict leak-free evaluation (unseen videos and dive sites), which also uncovered and fixed a 77 % train–test overlap in an earlier version; explanations with SHAP and Grad-CAM. |
| 5 | Report structure | "Section II reviews related work, Section III describes the datasets, Section IV the methodology, Section V the results, Section VI discusses them and Section VII concludes." |

## 4. Literature Review (table + short paragraph)

**Table 1. Summary of related work** (caption above the table)

| Ref | Work | Data | Method | Key finding | Gap our work addresses |
|---|---|---|---|---|---|
| [9] | Fulton et al., 2019 (ICRA) | Trash-ICRA19 | YOLOv2, SSD, Faster R-CNN | Deep detectors can find underwater litter | Detection only; no ecological impact |
| [10] | Hong et al., 2020 | TrashCan 1.0 | Mask R-CNN / Faster R-CNN baselines | Instance-segmentation dataset for deep-sea debris | Detection only |
| [11] | Islam et al., 2020 (IROS) | SUIM | SUIM-Net, U-Net and others | 8-class underwater scene segmentation | Reef class only, no bleaching labels |
| [12] | Đuraš et al., 2024 | SeaClear | Faster R-CNN, YOLOv6 | 68.3 % mAP on a random split but ~20 % across unseen sites | Shows the cross-site gap; our evaluation uses unseen sites/videos |
| [13] | Sauder et al., 2025 | Coralscapes | SegFormer, DINOv2 and others | 39-class reef segmentation; SegFormer MiT-B2 0.547 mIoU | Coral only; we connect coral to plastic and impact |
| [14] | Katija et al., 2022 | FathomNet | Expert-annotated ocean imagery | Large deep-sea image database | Used by us to try to improve fish detection |
| [2] | Lamb et al., 2018 | Field surveys | Statistics | Plastic contact raises coral disease risk | Manual surveys; motivates automation |
| [15] | Zhang et al., 2022 (ECCV) | MOT benchmarks | ByteTrack | Tracks low-confidence detections too | Used for fish tracking |

Paragraph after the table: research covers single tasks well, but rarely combines them, and rarely tests on unseen
locations — the gap MARIS addresses.

## 5. Dataset

**Table 2. Datasets used in this work** (caption above). 10 datasets.

| # | Dataset | Used for | Source URL | Samples | Licence | Accessed |
|---|---|---|---|---|---|---|
| 1 | TrashCan 1.0 (material version) [10] | Debris + animal detection | https://conservancy.umn.edu/handle/11299/214366 | 7,212 images, 16 classes (our split: 4,913 / 986 / 1,313, by video) | [verify] | [verify date] |
| 2 | Coralscapes [13] | Live / bleached / dead coral | https://huggingface.co/datasets/EPFL-ECEO/coralscapes | 2,075 images, 39 classes (1,517 / 166 / 392, by dive site) | Apache-2.0 | 14 Oct 2026 |
| 3 | SUIM [11] | First coral model (U-Net baseline) | https://irvlab.cs.umn.edu/resources/suim-dataset | ~1,635 images, 8 classes (110 test) [verify count] | [verify] | [verify date] |
| 4 | SeaClear [12] | Prepared plastic retrain (v3) | https://data.4tu.nl/datasets/4f1dff25-e157-4399-a5d4-478055461689 | 8,610 images, 40 classes | CC BY 4.0 | 14 Oct 2026 |
| 5 | FathomNet [14] | Fish detector experiment | https://fathomnet.org | 2,720 images / 4,415 fish boxes (our subset) | [verify] | 13 Oct 2026 |
| 6 | NOAA Coral Reef Watch [16] | Sea temperature, heat stress, bleaching alerts | https://coralreefwatch.noaa.gov | 5 stations, daily 5 km product | Public domain (US Gov.) | 13 Oct 2026 |
| 7 | NOAA NCEI Marine Microplastics [17] | Real-data plastic–biodiversity model | https://www.ncei.noaa.gov/products/microplastics | 13,460 surface samples → 698 ocean cells | Public domain (US Gov.) | 13 Oct 2026 |
| 8 | OBIS [18] | Species records per ocean cell | https://obis.org | 698 cells (species counts) | Per-dataset (mostly CC BY / CC0) [verify] | 13 Oct 2026 |
| 9 | Copernicus Marine [19] | Currents, temperature, waves (environmental module) | https://marine.copernicus.eu | 245 sites × 60 days | Copernicus Marine licence (free) | 14 Oct 2026 |
| 10 | Allen Coral Atlas [20] | Coral / seagrass habitat maps (environmental module) | https://allencoralatlas.org | Benthic map, Gulf of Mannar | CC BY 4.0 [verify] | Sep 2026 [verify] |

**Sample images (4–5, figure caption below):** pick from
- `outputs/experiments/2026-10-14/demo_frames/BEST-plastic_vid_000029_frame0000017.jpg` (TrashCan, plastic)
- `outputs/experiments/2026-10-14/demo_frames/BEST-fish_vid_000430_frame0000010.jpg` (TrashCan, fish)
- `data/samples/reef/coralscapes_val_141.jpg` (Coralscapes, bleached reef)
- `data/samples/reef/coralscapes_val_060.jpg` (Coralscapes, healthy reef)
- `outputs/figures/class_distribution.png` (class counts — good second figure)

> Fig. 1. Sample images from the datasets: (a) plastic on the deep-sea floor (TrashCan), (b) deep-sea fish
> (TrashCan), (c) partly bleached reef and (d) healthy reef (Coralscapes).

## 6. Methodology

1. **Intro to methodology** — one paragraph + the pipeline figure (draw a block diagram: image → debris detector /
   fish tracker / coral segmenter → + NOAA water data → XGBoost impact score → SHAP / Grad-CAM → dashboard).
2. **Data collection and pre-processing**
   - TrashCan: train/val/test split **by source video** (no video in two splits); COCO polygons → YOLO format.
     An earlier version shared 77 % of test images with training, inflating scores (0.68 vs real ~0.17); fixed and
     re-evaluated. *(This is a strong point — mention it.)*
   - Coralscapes: official split by dive site; 39 classes grouped into 4 (other / live / bleached / dead coral);
     class weights for rare bleached coral.
   - NOAA / NCEI / OBIS: daily satellite products sampled per station / 2° ocean cell.
3. **Model selection**
   - **Algorithm choice** (why each): YOLO11-seg (fast, gives masks for plastic area); ByteTrack (keeps
     low-confidence fish tracks); SegFormer (best family on Coralscapes in [13]; the U-Net scored lower);
     XGBoost (tabular data, works with SHAP); SHAP + Grad-CAM (explanations).
   - **Architecture:** YOLO11s-seg (10.1 M params, 640 px); SegFormer MiT-B2 (27.4 M params, 1024×2048,
     fine-tuned from the authors' Coralscapes model, flip test-time augmentation); XGBoost regressors.
4. **Training and evaluation setup** — epochs, optimiser (AdamW), losses (Dice + cross-entropy for coral),
   confidence threshold 0.18, metrics (precision, recall, F1, mAP, IoU), 95 % confidence intervals by bootstrap
   over 53 test videos.

## 7. Results — strongest first

Write one short paragraph per table/figure. Order:

### 7.1 Coral health assessment (strongest)

**Table 3. Coral segmentation on the Coralscapes TEST split (392 images, 5 unseen dive sites)** — source:
`models/coral_segmenter/coralscapes_segformer_v2_test_metrics.json`

| Metric | Value |
|---|---|
| Image-level bleaching detection accuracy | **0.957** (P 0.847, R 0.910; baseline 0.829) |
| Image-level coral detection accuracy | **0.954** (P 0.975, R 0.969; baseline 0.827) |
| Coral vs not-coral pixel accuracy | **0.952** |
| Pixel accuracy (4 classes) | **0.943** |
| Coral IoU (precision / recall) | 0.767 (0.876 / 0.860) |
| IoU live / bleached / dead coral | 0.806 / 0.694 / 0.328 |
| Mean IoU (4 classes) | 0.693 |
| False coral alarms on deep-sea footage | 1 of 200 frames (0.5 %) |

**Table 4. Coral model comparison (fair: same official test split)**

| Model | Coral IoU | Mean IoU |
|---|---|---|
| U-Net (ResNet-34) on SUIM (reef IoU, SUIM test) | 0.684 | 0.684 |
| SegFormer v1 (ImageNet start, 512×1024) | 0.704 | 0.647 |
| Published Coralscapes SegFormer, no fine-tuning | 0.768 | 0.688 |
| **SegFormer v2 (ours, fine-tuned, full resolution)** | **0.767** | **0.693** |

Figures: `outputs/figures/coral_health_assessment.png`; a dashboard screenshot of the bleached reef overlay.
> Fig. X. Coral health overlay: live coral (green), bleached (white), dead (grey).

### 7.2 Image-level detection (strong)

**Table 5. Does the image contain …? (TrashCan test, 1,313 images, conf 0.18)** — source:
`outputs/metrics/image_level_detection_test.json`

| Question | Accuracy | Precision | Recall | Baseline |
|---|---|---|---|---|
| Underwater robot (ROV) | **0.915** | 0.815 | **0.909** | 0.715 |
| Any debris | 0.768 | **0.893** | 0.799 | 0.779 |
| Plastic | 0.751 | 0.697 | 0.584 | 0.628 |

(Use precision/recall for debris; its accuracy is not above the baseline. Do **not** report fish image accuracy —
0.919 is below the "always no fish" baseline of 0.922.)

### 7.3 Plastic detection (moderate)

**Table 6. Plastic detector on the TrashCan test split (unseen videos)** — sources:
`outputs/metrics/test_bootstrap_confidence_intervals.json`, `outputs/metrics/eval_metrics_test.json`,
`outputs/experiments/2026-10-14/per_class_metrics_conf0.18.csv`

| Metric | Value |
|---|---|
| Plastic precision (95 % CI) | 0.475 [0.337, 0.602] |
| Plastic recall (95 % CI) | 0.452 [0.315, 0.594] |
| Plastic F1 (95 % CI) | 0.463 [0.340, 0.575] |
| Plastic mAP50 / mAP50-95 | 0.489 / 0.340 |
| Best other classes (F1) | ROV 0.612, eel 0.482, starfish 0.437 |

Figures: `outputs/figures/sample_predictions.png`, `confusion_matrix_normalized.png`, `BoxPR_curve.png`.

### 7.4 Learning behaviour

- `outputs/figures/training_curves_yolo11s_seg_v2.png` — detector loss and mAP over epochs.
- Coralscapes v1 training: validation mIoU 0.42 → 0.60 in 12 epochs, with noisy epochs caused by the rare bleached
  class; v2 started from an already-trained model and reached its best validation score at epoch 1 (early stop at 7).

### 7.5 Explainability, impact score and maps

- SHAP (`shap_impact_explanation.png`): top driver of the impact score = plastic cover.
- Grad-CAM (`gradcam_underwater_explanation.png`): shows where the detector looks.
- Impact score: mean 13.2 / 100 over 40 survey images, 0 at high/critical (`marine_pollution_impact_score_dist.png`).
- Maps: `plastic_pollution_map.png`, `real_world_plastic_biodiversity.png`.

### 7.6 Fish behaviour (weakest — report honestly)

- Fish detection: precision 0.407, recall 0.159, F1 0.229; a second detector trained with FathomNet gave F1 0.231
  (no gain).
- Tracking on 53 test videos: 34 animal tracks, 8 usable moving-animal tracks; moving animals 12.3 px/frame vs
  1.75 px/frame for near-static animals (noise floor). Only 3 tracks pass near plastic → too few to conclude.
- Figure: `fish_behavior_tracking.png`.

## 8. Discussion

- **Interpretation:** coral health is reliable from images; detection works for large, distinct objects (ROV,
  debris in general) but small plastic items and fish in dark deep-sea video remain hard.
- **Objective achievement:** all 6 expected outcomes delivered (maps, fish analysis, coral, impact score,
  biodiversity loss, dashboard). Objective 2 partial (no species ID, little fish data); objective 4 partial (only
  temperature is live satellite data; pH, oxygen, salinity, turbidity are station presets).
- **Error patterns** (`outputs/experiments/2026-10-14/weak_class_diagnosis_test.csv`, `error_analysis_*.png`):
  fish and paper mostly missed; plastic/metal both missed and false alarms; detector reacts to on-screen camera
  text (Grad-CAM: only 6 % of heat inside plastic boxes); dead coral confused with rubble; rare classes (shells,
  fabric, rubber) have too few training examples.
- **Learning behaviour:** see 7.4.
- **Fair comparison:** every model compared on the same held-out split; thresholds chosen on validation, never on
  test; Table 4 credits the Coralscapes authors' published model (our fine-tuning changed test results very little).
  For context, SeaClear's own baselines drop from 68 % to ~20 % mAP on unseen sites [12], so our unseen-video
  plastic mAP50 of 0.49 is reasonable.
- **Real-data check:** across 698 ocean cells, microplastics add only +0.001 R² to predicting recorded species
  (0.848 vs 0.847; survey effort alone 0.822) → association only, not causation.
- **Limitation:** the impact score and biodiversity-loss % follow an expert formula (XGBoost reproduces it,
  R² 0.945) — estimates, not field measurements. Survey images have no GPS (stations assigned by rotation).

## 9. Conclusion (1–2 paragraphs)

Para 1: what was built and the main results (coral 0.957 / 0.952 / 0.767; ROV 0.915; debris precision 0.893;
plastic F1 0.46; all on unseen data).
Para 2 — future work: run the prepared YOLO11m-seg + SeaClear retrain (`notebooks/colab_train_plastic_v3.ipynb`);
live pH / oxygen / salinity from Copernicus; fish species identification; more fish video; field validation of the
impact score; RT-DETR / DeepSORT / Temporal Fusion Transformer for comparison.

## 10. Acknowledgement

Faculty guide; dataset providers (TrashCan, Coralscapes, SUIM, SeaClear, FathomNet, NOAA, OBIS, Copernicus, Allen
Coral Atlas); teammates (e.g. Anannya Jadhav — environmental risk module); Google Colab for GPU time.

## 11. References (IEEE) — check every entry before submitting

[1] J. R. Jambeck *et al.*, "Plastic waste inputs from land into the ocean," *Science*, vol. 347, no. 6223, pp. 768–771, 2015.
[2] J. B. Lamb *et al.*, "Plastic waste associated with disease on coral reefs," *Science*, vol. 359, no. 6374, pp. 460–462, 2018.
[3] T. P. Hughes *et al.*, "Global warming and recurrent mass bleaching of corals," *Nature*, vol. 543, pp. 373–377, 2017.
[4] G. Jocher and J. Qiu, "Ultralytics YOLO11," 2024. [Online]. Available: https://github.com/ultralytics/ultralytics
[5] E. Xie *et al.*, "SegFormer: Simple and efficient design for semantic segmentation with transformers," in *Proc. NeurIPS*, 2021.
[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. ACM SIGKDD*, 2016, pp. 785–794.
[7] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. NeurIPS*, 2017.
[8] R. R. Selvaraju *et al.*, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE ICCV*, 2017, pp. 618–626.
[9] M. Fulton, J. Hong, M. J. Islam, and J. Sattar, "Robotic detection of marine litter using deep visual detection models," in *Proc. IEEE ICRA*, 2019, pp. 5752–5758.
[10] J. Hong, M. Fulton, and J. Sattar, "TrashCan: A semantically-segmented dataset towards visual detection of marine debris," arXiv:2007.08097, 2020.
[11] M. J. Islam *et al.*, "Semantic segmentation of underwater imagery: Dataset and benchmark," in *Proc. IEEE/RSJ IROS*, 2020.
[12] A. Đuraš, B. J. Wolf, A. Ilioudi, I. Palunko, and B. De Schutter, "A dataset for detection and segmentation of underwater marine debris in shallow waters," *Scientific Data*, vol. 11, art. 921, 2024, doi: 10.1038/s41597-024-03759-2.
[13] J. Sauder, V. Domazetoski, G. Banc-Prandi, G. Perna, A. Meibom, and D. Tuia, "The Coralscapes dataset: Semantic scene understanding in coral reefs," arXiv:2503.20000, 2025.
[14] K. Katija *et al.*, "FathomNet: A global image database for enabling artificial intelligence in the ocean," *Scientific Reports*, vol. 12, art. 15914, 2022.
[15] Y. Zhang *et al.*, "ByteTrack: Multi-object tracking by associating every detection box," in *Proc. ECCV*, 2022.
[16] W. Skirving *et al.*, "CoralTemp and the Coral Reef Watch coral bleaching heat stress product suite version 3.1," *Remote Sensing*, vol. 12, no. 23, art. 3856, 2020.
[17] NOAA National Centers for Environmental Information, "Marine Microplastics Database." [Online]. Available: https://www.ncei.noaa.gov/products/microplastics
[18] OBIS, "Ocean Biodiversity Information System," Intergovernmental Oceanographic Commission of UNESCO. [Online]. Available: https://obis.org
[19] E.U. Copernicus Marine Service. [Online]. Available: https://marine.copernicus.eu
[20] Allen Coral Atlas, "Imagery, maps and monitoring of the world's tropical coral reefs." [Online]. Available: https://allencoralatlas.org

Conference papers in the list: [5], [7], [8], [9], [11], [15] (faculty asked for at least 1–2).

## 12. Supporting material

- Dashboard (run: `.venv\Scripts\python.exe -m streamlit run app.py`) — screenshots of each outcome page.
- Code repository and notebooks (`notebooks/`), full assessment report (`outputs/reports/executive_impact_report.md`),
  evaluation package (`outputs/experiments/2026-10-14/`).

## Checklist before submitting

- [ ] Abstract ≤ 300 words, one paragraph
- [ ] 4–6 keywords
- [ ] Every figure has a caption **below** it and is mentioned in the text
- [ ] Every table has a caption **above** it and is mentioned in the text
- [ ] Dataset table complete: all **[verify]** items filled
- [ ] ≥ 10 IEEE references, all checked
- [ ] Plagiarism check < 10 % (rephrase draft text)
- [ ] Numbers in the report match this guide (all from held-out data)
