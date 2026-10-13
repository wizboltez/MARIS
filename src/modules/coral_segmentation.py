"""Reef segmentation with the U-Net trained on SUIM (notebooks/colab_train_suim_coral.ipynb).

Expects, in models/coral_segmenter/:
  suim_unet.pt               state_dict of smp.Unet
  suim_unet_config.json      encoder, classes, reef_class, input_hw, mean, std
  suim_test_metrics.json     official SUIM TEST metrics (incl. passed_gate)

The model is only used if those files exist AND the test metrics passed the quality gate;
otherwise load_reef_segmenter() returns None and CoralHealthAnalyzer keeps its colour heuristic.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from src.config import PROJECT_ROOT

SEGMENTER_DIR = PROJECT_ROOT / "models" / "coral_segmenter"
MIN_REEF_FRACTION = 0.02  # reef must cover >= 2% of the image to count as "coral present"


class ReefSegmenter:
    def __init__(self, weights_path, config: dict, test_metrics: dict):
        import segmentation_models_pytorch as smp
        import torch

        self.torch = torch
        self.cfg = config
        self.test_metrics = test_metrics
        self.model = smp.Unet(config["encoder"], encoder_weights=None, classes=len(config["classes"]))
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.eval()

    def predict(self, image_bgr: np.ndarray) -> dict:
        """Per-pixel class map at the original resolution + the reef mask."""
        h0, w0 = image_bgr.shape[:2]
        h, w = self.cfg["input_hw"]
        rgb = cv2.resize(image_bgr, (w, h))[:, :, ::-1].astype(np.float32) / 255.0
        x = (rgb - np.array(self.cfg["mean"], np.float32)) / np.array(self.cfg["std"], np.float32)
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(x.transpose(2, 0, 1).copy())[None])
        classes = logits.argmax(1)[0].numpy().astype(np.uint8)
        classes = cv2.resize(classes, (w0, h0), interpolation=cv2.INTER_NEAREST)
        reef = classes == self.cfg["reef_class"]
        return {"classes": classes, "reef_mask": reef, "reef_fraction": float(reef.mean())}


def load_reef_segmenter() -> ReefSegmenter | None:
    weights = SEGMENTER_DIR / "suim_unet.pt"
    cfg_path = SEGMENTER_DIR / "suim_unet_config.json"
    metrics_path = SEGMENTER_DIR / "suim_test_metrics.json"
    if not (weights.exists() and cfg_path.exists() and metrics_path.exists()):
        return None
    metrics = json.loads(metrics_path.read_text())
    if not metrics.get("passed_gate", False):
        print(f"Reef segmenter found but failed its quality gate (reef IoU {metrics.get('reef_iou')}); using heuristic.")
        return None
    try:
        return ReefSegmenter(weights, json.loads(cfg_path.read_text()), metrics)
    except ImportError:
        print("segmentation-models-pytorch not installed; using coral heuristic.")
        return None
