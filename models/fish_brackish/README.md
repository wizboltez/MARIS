---
license: agpl-3.0

pipeline_tag: object-detection

library_name: ultralytics

datasets:
  - dronefreak/Brackish

tags:
  - object-detection
  - detectionbench
  - ultralytics
  - pytorch
  - computer-vision
  - underwater
  - marine-biology
  - brackish-water
metrics:
  - map50
  - map50-95
  - precision
  - recall
  - f1

base_model: "Ultralytics/YOLOv8"

model-index:
  - name: YOLOv8s Finetuned on Brackish Underwater
    results:
      - task:
          type: object-detection
          name: Object Detection
        dataset:
          name: Brackish Underwater
          type: brackish
        metrics:
          - type: mAP50
            value: 99.3
            name: mAP@50 (test split)
          - type: mAP50-95
            value: 85.65
            name: mAP@50-95 (test split)
          - type: precision
            value: 99.36
            name: Precision (test split)
          - type: recall
            value: 98.53
            name: Recall (test split)
        source:
          url: https://github.com/dronefreak/DetectionBench
          name: DetectionBench
---


# YOLOv8s Finetuned on Brackish Underwater

Fine-tuned YOLOv8s object detector on the **Brackish Underwater** benchmark dataset, trained and evaluated as part of [DetectionBench](https://github.com/dronefreak/DetectionBench) -- a framework for reproducibly benchmarking modern object detectors with identical training recipes and evaluation metrics across multiple real-world datasets.

<p align="center">
  <img src="brackish_yolov8s_showcase.jpg" alt="Brackish Underwater Detection Demo" width="900">
</p>

<br>

<!-- ROW 1: Identity & Tech Stack -->
<div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">
  <img src="https://img.shields.io/badge/Task-Object_Detection-blue?style=flat-square" alt="Task">
  <img src="https://img.shields.io/badge/Framework-Ultralytics_YOLO-0aa1a7?style=flat-square" alt="Framework">
  <img src="https://img.shields.io/badge/Base_Model-YOLOv8s-purple?style=flat-square" alt="Base Model">
</div>

<!-- ROW 2: Performance Metrics -->
<div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">
  <img src="https://img.shields.io/badge/mAP@50-99.3%25-success?style=flat-square" alt="mAP@50">
  <img src="https://img.shields.io/badge/mAP@50:95-85.65%25-orange?style=flat-square" alt="mAP@50:95">
  <img src="https://img.shields.io/badge/Params-11.2M-lightgrey?style=flat-square" alt="Params">
</div>

<!-- ROW 3: Metadata -->
<div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin-bottom: 24px; flex-wrap: wrap;">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-lightgrey?style=flat-square" alt="License">
  <a href="https://github.com/dronefreak/DetectionBench"><img src="https://img.shields.io/badge/Source-DetectionBench-black?style=flat-square" alt="Source"></a>
</div>

---

## Usage

### Install Dependencies

```bash
pip install ultralytics huggingface_hub
```

### Load Model from Hugging Face

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(
    repo_id="dronefreak/brackish-yolov8s",
    filename="best.pt"
)

model = YOLO(weights)
```

### Run Inference

```python
results = model.predict(
    source="image.jpg",
    conf=0.25
)

results[0].show()
```
---

## Performance

Evaluated on the Brackish Underwater **test** split, using DetectionBench's standard evaluation pipeline (`detectionbench-evaluate`).

| Metric     | Score (%)       |
| ---------- | --------------- |
| mAP@50     | 99.3     |
| mAP@50-95  | 85.65   |
| Precision  | 99.36 |
| Recall     | 98.53    |
| F1 Score   | 98.95        |
| Parameters | 11.2M    |
| FLOPs      | 28.6B (at 640 px)     |

---

## Brackish Underwater Model Zoo

Every model DetectionBench has trained and evaluated on Brackish Underwater so far, for full transparency -- see [DetectionBench](https://github.com/dronefreak/DetectionBench) for the smaller, curated comparison set used on the project README.

| Model                | mAP@50        | mAP@50-95       | Precision         | Recall         |
| --------------------- | ------------- | --------------- | ----------------- | -------------- |
| **YOLOv8s** | **99.3** | **85.65** | **99.36** | **98.53** |
| YOLOv8m | 99.19 | 86.04 | 98.69 | 98.71 |
| YOLO26s | 99.1 | 85.77 | 99.45 | 97.83 |
| YOLO26n | 98.95 | 83.66 | 98.17 | 97.23 |
| YOLO11x | 98.89 | 86.3 | 98.84 | 98.25 |
| YOLO11n | 98.87 | 83.27 | 98.76 | 96.69 |
| YOLO26m | 98.74 | 85.93 | 98.74 | 97.32 |
| YOLOv8n | 98.41 | 83.01 | 99.51 | 96.95 |
---

## Per-Class Performance

| Class                      | mAP@50          | mAP@50-95         |
| -------------------------- | --------------- | ----------------- |
| crab       | 99.5 | 93.93 |
| fish       | 99.48 | 89.69 |
| jellyfish       | 98.77 | 74.97 |
| shrimp       | 99.5 | 79.72 |
| small_fish       | 99.08 | 76.31 |
| starfish       | 99.5 | 99.29 |

![Normalized Confusion Matrix](confusion_matrix_normalized.png)

---

## Dataset

This model was trained on **Brackish Underwater**. For the full dataset description, provenance, license, and citation, see the dataset card:

https://huggingface.co/datasets/dronefreak/Brackish

### Classes

* crab
* fish
* jellyfish
* shrimp
* small_fish
* starfish
---

## Training Configuration

| Setting          | Value                           |
| ---------------- | -------------------------------- |
| Dataset          | Brackish Underwater       |
| Framework        | Ultralytics YOLO                  |
| Training Toolkit | DetectionBench                   |
| Epochs (configured max) | 500 |
| Epochs (actually trained) | 500 |
| Early Stopping Patience | 100 |
| Batch Size | 32 |
| Image Size | 640 |
| Optimizer | auto |
| Initial Learning Rate | 0.001 |
| Seed | 0 |
---

## Repository Contents

```text
best.pt
results.csv
args.yaml
BoxPR_curve.png
BoxF1_curve.png
BoxP_curve.png
BoxR_curve.png
confusion_matrix.png
confusion_matrix_normalized.png
val_batch0_pred.jpg
brackish_yolov8s_showcase.jpg
README.md
```

---

## Related Resources

* [Brackish Underwater dataset card](https://huggingface.co/datasets/dronefreak/Brackish) on Hugging Face
* [DetectionBench](https://github.com/dronefreak/DetectionBench) -- reproducible benchmarks for modern object detectors on real-world datasets

---

## Training Framework

This model was trained using [DetectionBench](https://github.com/dronefreak/DetectionBench), an open-source framework for benchmarking object detectors across multiple real-world datasets with a common pipeline.

Features include:

* A dataset-adapter registry for converting real-world datasets into a canonical format
* Identical training/evaluation recipes across model families (Ultralytics YOLO/RT-DETR, RF-DETR)
* Hardware profiling (latency, FPS, VRAM, parameters, FLOPs)
* One-command reproducibility via versioned Hydra configs

If you find this model useful, please consider starring the repository.

---

## Known Limitations

* Severe class imbalance: `crab` (34.6%) and `small_fish` (30.4%) account for roughly two-thirds of all annotated boxes in the training set, while `shrimp` (1.46%) and `jellyfish` (1.82%) are rare -- per-class accuracy on the minority classes is measured on comparatively few examples.
* Roughly 15% of images have no annotated objects at all (background-only frames, by design -- part of the dataset's varying-visibility setup, not a data-quality issue).
* Single-location, single-camera capture: all footage comes from one fixed camera 9 meters below the surface on the Limfjords bridge, Denmark -- generalization to other underwater cameras, locations, or water types (brackish vs. marine/freshwater) is untested.
* Two-hop provenance: this dataset was converted to YOLO format via a third-party Roboflow export, not sourced directly from the original annotated-video release; images are pre-resized to 1920x1080 by that export.
---

## Citation

If you use this model in your research, please consider citing the dataset and the model architecture:

```
@InProceedings{pedersen2019brackish,
  title = {Detection of Marine Animals in a New Underwater Dataset with Varying Visibility},
  author = {Pedersen, Malte and Haurum, Joakim Bruslund and Gade, Rikke and Moeslund, Thomas B. and Madsen, Niels},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2019}
}
```

```bibtex
No official YOLOv8 research paper has been published by Ultralytics; this is their own recommended software citation instead:

@software{jocher2023yolov8,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```