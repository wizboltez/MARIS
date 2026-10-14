"""Coral reef segmentation models (all optional; files live in models/coral_segmenter/).

Coralscapes models (live / bleached / dead coral learned from expert labels; preferred when available):
  coralscapes_v2:  coralscapes_segformer_v2/ (Hugging Face SegFormer folder), coralscapes_segformer_v2_config.json,
                   coralscapes_segformer_v2_test_metrics.json   (notebooks/colab_train_coralscapes_v2.ipynb)
  coralscapes:     coralscapes_segformer.pt, coralscapes_segformer_config.json,
                   coralscapes_segformer_test_metrics.json      (notebooks/colab_train_coralscapes.ipynb)
SUIM models (reef location only, no bleaching labels; notebooks/colab_train_suim_coral.ipynb):
  segformer:       suim_segformer.pt, suim_segformer_config.json, suim_segformer_test_metrics.json
  unet:            suim_unet.pt,      suim_unet_config.json,      suim_test_metrics.json

A model is only used if its files exist AND its TEST metrics passed the quality gate. Order of preference:
coralscapes_v2, coralscapes (the v2 notebook only writes its config when v2 beat v1 on VAL), then the SUIM model
with the higher SUIM TEST reef IoU. REEF_SEGMENTER=<key> forces one. If none qualifies, load_reef_segmenter()
returns None and CoralHealthAnalyzer keeps its colour heuristic.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from src.config import PROJECT_ROOT

SEGMENTER_DIR = PROJECT_ROOT / "models" / "coral_segmenter"
MIN_REEF_FRACTION = 0.02  # reef must cover >= 2% of the image to count as "coral present"

CANDIDATES = {
    "coralscapes_v2": {
        "weights": "coralscapes_segformer_v2",  # Hugging Face model folder
        "config": "coralscapes_segformer_v2_config.json",
        "metrics": "coralscapes_segformer_v2_test_metrics.json",
        "domain_check": "domain_check_trashcan_coralscapes_v2.json",
    },
    "coralscapes": {
        "weights": "coralscapes_segformer.pt",
        "config": "coralscapes_segformer_config.json",
        "metrics": "coralscapes_segformer_test_metrics.json",
        "domain_check": "domain_check_trashcan_coralscapes.json",
    },
    "segformer": {
        "weights": "suim_segformer.pt",
        "config": "suim_segformer_config.json",
        "metrics": "suim_segformer_test_metrics.json",
        "domain_check": "domain_check_trashcan_segformer.json",
    },
    "unet": {
        "weights": "suim_unet.pt",
        "config": "suim_unet_config.json",
        "metrics": "suim_test_metrics.json",
        "domain_check": "domain_check_trashcan.json",
    },
}
PREFERRED = ["coralscapes_v2", "coralscapes"]  # tried in this order before the SUIM models
SUIM_KEYS = ["segformer", "unet"]


def _read_json(name: str) -> dict | None:
    p = SEGMENTER_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def select_segmenter_key(require_weights: bool = True) -> str | None:
    """Which candidate to use (no torch import): forced by REEF_SEGMENTER, else Coralscapes, else best SUIM model."""
    usable = {}
    for key, files in CANDIDATES.items():
        metrics = _read_json(files["metrics"])
        has_files = (SEGMENTER_DIR / files["config"]).exists() and (
            not require_weights or (SEGMENTER_DIR / files["weights"]).exists()
        )
        if metrics and has_files:
            if metrics.get("passed_gate", False):
                usable[key] = metrics["reef_iou"]
            else:
                print(f"Reef segmenter '{key}' failed its quality gate (reef IoU {metrics.get('reef_iou')}); skipped.")
    forced = os.environ.get("REEF_SEGMENTER", "").strip().lower()
    if forced:
        if forced in usable:
            return forced
        print(f"REEF_SEGMENTER={forced} is not available or failed its gate; choosing automatically.")
    for key in PREFERRED:
        if key in usable:
            return key
    suim = {k: v for k, v in usable.items() if k in SUIM_KEYS}  # same SUIM TEST split -> comparable
    return max(suim, key=suim.get) if suim else None


def segmenter_info(key: str) -> dict:
    """Test metrics and deep-sea domain check (if run) for a candidate."""
    files = CANDIDATES[key]
    return {"test_metrics": _read_json(files["metrics"]), "domain_check": _read_json(files["domain_check"])}


class ReefSegmenter:
    def __init__(self, key: str, weights_path, config: dict, test_metrics: dict, domain_check: dict | None = None):
        import torch

        self.torch = torch
        self.key = key
        self.cfg = config
        self.test_metrics = test_metrics
        self.domain_check = domain_check
        self.dataset = config.get("dataset", "SUIM")
        self.backend = config.get("backend", "smp")
        self.name = test_metrics.get("model") or f"{config.get('arch', 'Unet')} {config.get('encoder', '')}"
        if self.backend == "hf_segformer":
            self.name = f"SegFormer MiT-B2 ({self.name})"
        self.reef_classes = config.get("reef_classes", [config.get("reef_class")])
        # Coralscapes models separate live / bleached / dead coral; SUIM models only give "reef"
        self.learns_bleaching = "bleached_class" in config
        if self.backend == "hf_segformer":
            from transformers import SegformerForSemanticSegmentation

            self.model = SegformerForSemanticSegmentation.from_pretrained(str(weights_path))
            groups = config["group_of_channel"]  # model channel -> group index (None = unlabelled, unused)
            n_groups = len(config["classes"])
            self.group_channels = [[c for c, g in enumerate(groups) if g == k] for k in range(n_groups)]
        else:
            import segmentation_models_pytorch as smp

            self.model = smp.create_model(
                config.get("arch", "Unet"), encoder_name=config["encoder"], encoder_weights=None, classes=len(config["classes"])
            )
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        # GPU if available (full-resolution SegFormer takes ~1 min per image on CPU), half precision on GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def _probs(self, x):
        """Per-class probabilities [1, n_classes, h, w] at the model input size."""
        F = self.torch.nn.functional
        if self.backend == "hf_segformer":
            # Group the 39 class probabilities at the model's 1/4 output resolution, then upsample only the
            # 4 group maps (the notebook upsampled all 39 logits first; same result up to interpolation, far less memory)
            logits = self.model(pixel_values=x).logits.float()
            used = sorted(c for chans in self.group_channels for c in chans)
            p = F.softmax(logits[:, used], dim=1)
            pos = {c: i for i, c in enumerate(used)}
            groups = self.torch.stack([p[:, [pos[c] for c in chans]].sum(1) for chans in self.group_channels], dim=1)
            return F.interpolate(groups, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return F.softmax(self.model(x), dim=1)

    def predict(self, image_bgr: np.ndarray) -> dict:
        """Per-pixel class map at the original resolution + reef mask (+ live/bleached/dead masks for Coralscapes)."""
        h0, w0 = image_bgr.shape[:2]
        h, w = self.cfg["input_hw"]
        rgb = cv2.resize(image_bgr, (w, h))[:, :, ::-1].astype(np.float32) / 255.0
        x = (rgb - np.array(self.cfg["mean"], np.float32)) / np.array(self.cfg["std"], np.float32)
        x = self.torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(self.device)
        with self.torch.no_grad(), self.torch.autocast(self.device, enabled=self.device == "cuda"):
            probs = self._probs(x).float()
            if self.cfg.get("tta_flip"):
                probs = (probs + self._probs(x.flip(-1)).float().flip(-1)) / 2
        classes = probs.argmax(1)[0].cpu().numpy().astype(np.uint8)
        classes = cv2.resize(classes, (w0, h0), interpolation=cv2.INTER_NEAREST)
        reef = np.isin(classes, self.reef_classes)
        out = {"classes": classes, "reef_mask": reef, "reef_fraction": float(reef.mean())}
        if self.learns_bleaching:
            out["live_mask"] = classes == self.cfg["live_class"]
            out["bleached_mask"] = classes == self.cfg["bleached_class"]
            out["dead_mask"] = classes == self.cfg["dead_class"]
        return out


def load_reef_segmenter(key: str | None = None) -> ReefSegmenter | None:
    key = key or select_segmenter_key()
    if key is None:
        return None
    files = CANDIDATES[key]
    info = segmenter_info(key)
    try:
        return ReefSegmenter(
            key, SEGMENTER_DIR / files["weights"], _read_json(files["config"]), info["test_metrics"], info["domain_check"]
        )
    except ImportError as e:
        print(f"Reef segmenter '{key}' needs a missing package ({e.name}); using coral heuristic.")
        return None
