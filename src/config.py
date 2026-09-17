"""
Central config for Teammate 1 (plastic detection) module.

Design notes (from actually inspecting the downloaded TrashCan 1.0 dataset,
data/raw/trashcan/dataset/ -- see README files shipped inside dataset.zip):

- The dataset ships as standard COCO instance-segmentation JSON
  (keys: info, licenses, images, annotations, categories; annotations carry
  polygon `segmentation`, `bbox`, `area`, `category_id`, `image_id`).
- There are two label taxonomies: `material_version` (16 classes) and
  `instance_version` (22 classes, trash split by object type e.g. trash_bag,
  trash_bottle). We use `material_version` because it has an explicit
  `trash_plastic` category, which is what lets us report a scientifically
  defensible "plastic" count/area distinct from other debris materials
  (metal, rubber, wood, fabric, paper, fishing gear, etc). We still train on
  the full 16-class taxonomy (not just trash_plastic) so the model learns to
  discriminate trash from rov/plant/animal look-alikes, then filter to
  plastic-only at the quantification stage.
- Images are 480x360 or 480x270 (only two distinct sizes found).
- Filenames encode a source video id: vid_<id>_frame<n>.jpg. The dataset's
  own train/val split reuses the same video in both splits (127 of 133 val
  videos also appear in train), i.e. the official split is NOT leak-free.
  We therefore pool official train+val and re-split ourselves by video id
  (GroupShuffleSplit) into train/val/test with zero video overlap. This
  means our numbers aren't directly comparable to the original paper, which
  is expected and documented here rather than silently inherited.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "trashcan" / "dataset" / "material_version"
RAW_TRAIN_JSON = RAW_DATASET_DIR / "instances_train_trashcan.json"
RAW_VAL_JSON = RAW_DATASET_DIR / "instances_val_trashcan.json"
RAW_TRAIN_IMAGES = RAW_DATASET_DIR / "train"
RAW_VAL_IMAGES = RAW_DATASET_DIR / "val"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_YAML_PATH = PROCESSED_DIR / "data.yaml"
SPLIT_MANIFEST_PATH = PROCESSED_DIR / "split_manifest.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_CSV = PREDICTIONS_DIR / "plastic_predictions.csv"

MODELS_DIR = PROJECT_ROOT / "models" / "plastic_detector"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"

# Category order/ids exactly as they appear in the TrashCan material_version
# COCO json (category id 1..16). Index in this list = YOLO class id (0-based).
CLASS_NAMES = [
    "rov",
    "plant",
    "animal_fish",
    "animal_starfish",
    "animal_shells",
    "animal_crab",
    "animal_eel",
    "animal_etc",
    "trash_etc",
    "trash_fabric",
    "trash_fishing_gear",
    "trash_metal",
    "trash_paper",
    "trash_plastic",
    "trash_rubber",
    "trash_wood",
]

# All human-made debris, regardless of material -- used for a "debris"-level
# summary. TrashCan does not separately flag "plastic vs non-plastic" beyond
# the trash_plastic category, so DEBRIS_CLASSES is the union of all trash_*.
DEBRIS_CLASSES = [c for c in CLASS_NAMES if c.startswith("trash_")]

# The one class this project is scientifically entitled to call "plastic".
PLASTIC_CLASSES = ["trash_plastic"]

RANDOM_SEED = 42

# GroupShuffleSplit fractions (must sum to 1.0), grouped by source video id.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
