# MARIS — Final Evaluation (2026-10-14)

All numbers are measured. Aggregate results use the held-out TrashCan **test** split (1,313 images,
53 videos, never used for training, threshold choice or model selection). Selections (thresholds,
demo videos) used the **validation** split only. Baseline record: `baseline_record.txt`
(commit `f0cba0d`, checkpoint hashes). Reproduce: `python -m src.evaluation.final_eval_2026_10_14`.

## 1. Baseline vs optimized

| Component | Baseline (test) | Optimized attempt (test) | Kept |
|---|---|---|---|
| Plastic detector, conf 0.18 | P 0.475 [0.337, 0.602], R 0.452 [0.315, 0.594], F1 0.463 | Val-tuned threshold 0.47: P 0.615, R 0.338, F1 0.436 | **Baseline** (tuning did not help; val has only 170 plastic items) |
| Fish, old 16-class model, conf 0.18 | P 0.407, R 0.159, F1 0.229 | New FathomNet+TrashCan fish detector, val-chosen conf 0.35: P 0.265 [0.080, 0.464], R 0.205 [0.088, 0.387], F1 0.231 | **Baseline** (no significant gain; CIs overlap) |
| Coral reef U-Net (SUIM test) | P 0.762, R 0.870, IoU 0.684, pixel accuracy 0.854 | — | Baseline |

95% intervals: cluster bootstrap over the 53 test videos (1,000 resamples).

## 2. Fish confidence-threshold comparison (validation, same samples and matching for both)

See `fish_threshold_comparison_val.csv` and `fish_pr_val.png`. On validation the new fish detector
reaches F1 0.325 at conf 0.35 vs at most 0.19 for the old model; on test the gain disappeared
(0.231 vs 0.229). Reported as a validation-to-test generalisation gap.

## 3. Per-class results (test, conf 0.18, IoU 0.5)

| Class | Train instances | Test instances | Precision | Recall | F1 |
|---|---|---|---|---|---|
| rov | 2457 | 554 | 0.526 | 0.731 | 0.612 |
| trash_plastic | 1194 | 500 | 0.475 | 0.452 | 0.463 |
| animal_eel | 212 | 100 | 0.442 | 0.530 | 0.482 |
| animal_starfish | 317 | 81 | 0.371 | 0.531 | 0.437 |
| animal_crab | 248 | 30 | 0.588 | 0.333 | 0.426 |
| trash_paper | 157 | 36 | 0.412 | 0.194 | 0.264 |
| trash_etc | 1392 | 400 | 0.203 | 0.305 | 0.244 |
| animal_fish | 461 | 151 | 0.407 | 0.159 | 0.229 |
| trash_metal | 768 | 166 | 0.272 | 0.187 | 0.221 |
| trash_fishing_gear | 170 | 18 | 0.214 | 0.167 | 0.188 |
| plant | 433 | 52 | 0.019 | 0.019 | 0.019 |
| trash_fabric | 137 | 123 | 0.018 | 0.008 | 0.011 |
| animal_shells | 42 | 183 | 0.000 | 0.000 | 0.000 |
| animal_etc | 193 | 22 | 0.000 | 0.000 | 0.000 |
| trash_rubber | 130 | 2 | 0.000 | 0.000 | 0.000 |
| trash_wood | 302 | 0 | — | — | — |
| **All classes (micro)** | | 2,418 | **0.371** | **0.383** | |

Weak-class diagnosis (`weak_class_diagnosis_test.csv`): fish and paper errors are **mostly missed
objects**; plastic, metal and fishing gear have **both** misses and false alarms. Shells have only 42
training instances but 183 in test (insufficient training data); fabric and rubber are similar.

## 4. Best annotated validation videos (selection only; `val_video_ranking.csv`)

| Task | Video | Objects | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Plastic | vid_000029 | 7 | 1.000 | 1.000 | 1.000 |
| Plastic | vid_000161 | 11 | 0.909 | 0.909 | 0.909 |
| Plastic | vid_000045 | 14 | 0.818 | 0.643 | 0.720 |
| Fish | vid_000430 | 6 | 1.000 | 0.833 | 0.909 |
| Fish | vid_000433 | 17 | 1.000 | 0.588 | 0.741 |

Only 8 validation videos have >= 3 plastic objects and 7 have >= 3 fish; the third-ranked fish video
(vid_000440) scores F1 0.043.

## 5. Demo frames (`demo_frames/`, index in `demo_frames_index.json`)

Green = ground truth, orange = correct detection, red = false alarm, number = confidence.
`BEST-*` frames come from the best validation videos above and are **best-case examples, not
representative of the dataset**. `FAIL-*` frames show typical failures: fish at the image edge and
plastic among seabed clutter missed.

## 6. What changed today

- Recorded baseline (no original results or checkpoints overwritten).
- Fish detector trained on FathomNet deep-sea fish + TrashCan; validated and tested; not adopted.
- Added `src/evaluation/final_eval_2026_10_14.py` producing everything in this folder.
- No changes to the deployed pipeline or dashboard models.
