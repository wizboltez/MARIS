"""Gradient-based Grad-CAM for the YOLO11-seg plastic detector.

Hooks the three neck feature maps that feed the Segment head (P3/P4/P5, layers
16/19/22), back-propagates the raw class logit of the target class at the anchors
that fire for it, and builds a standard Grad-CAM map per scale:

    weights_k = mean_{x,y} d(score) / d(A_k)        (global-average-pooled gradients)
    cam       = ReLU( sum_k weights_k * A_k )

The three scale maps are upsampled to the input size and averaged. The map shows
which image regions increased the network's confidence in the target class.
"""
from __future__ import annotations

import copy

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox

from src.config import CLASS_NAMES

PLASTIC_ID = CLASS_NAMES.index("trash_plastic")


class YOLOGradCAM:
    def __init__(self, yolo, layer_ids: tuple[int, ...] | None = None, imgsz: int = 640):
        # Own copy, so the hooks never touch the model used for normal inference
        self.model = copy.deepcopy(yolo.model).float().cpu()
        self.model.eval()
        head = self.model.model[-1]
        # Default: the layers feeding the detection/segment head
        self.layer_ids = layer_ids or tuple(head.f)
        self.imgsz = imgsz
        self.stride = int(max(self.model.stride))
        self._acts: dict[int, torch.Tensor] = {}
        for i in self.layer_ids:
            self.model.model[i].register_forward_hook(self._save_activation(i))

    def _save_activation(self, idx: int):
        def hook(_module, _inp, out):
            if out.requires_grad:
                out.retain_grad()
            self._acts[idx] = out
        return hook

    def _preprocess(self, image_bgr: np.ndarray):
        lb = LetterBox((self.imgsz, self.imgsz), auto=True, stride=self.stride)
        img = lb(image=image_bgr)
        h0, w0 = image_bgr.shape[:2]
        h1, w1 = img.shape[:2]
        r = min(h1 / h0, w1 / w0)
        new_w, new_h = round(w0 * r), round(h0 * r)
        pad_x, pad_y = (w1 - new_w) / 2, (h1 - new_h) / 2
        crop = (int(round(pad_y - 0.1)), int(round(pad_y - 0.1)) + new_h, int(round(pad_x - 0.1)), int(round(pad_x - 0.1)) + new_w)
        x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        return x, crop

    def compute(self, image_bgr: np.ndarray, class_id: int = PLASTIC_ID, conf: float = 0.18) -> dict:
        """Return {'cam': HxW float map in [0,1], 'class_id', 'class_name', 'n_anchors', 'max_score'}."""
        x, (y0, y1, x0, x1) = self._preprocess(image_bgr)
        x.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        self._acts.clear()

        with torch.enable_grad():
            _, aux = self.model(x)
            logits = aux["scores"][0]  # (nc, n_anchors), raw class logits
            probs = logits.sigmoid()

            # Target: anchors where the class fires; else the single strongest anchor
            cls_probs = probs[class_id]
            mask = cls_probs >= conf
            if mask.sum() == 0:
                mask = cls_probs == cls_probs.max()
            target = logits[class_id][mask].sum()
            target.backward()

        h_in, w_in = x.shape[2:]
        cam_total = np.zeros((h_in, w_in), dtype=np.float32)
        for i in self.layer_ids:
            act = self._acts[i]
            grad = act.grad
            if grad is None:
                continue
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * act).sum(dim=1))[0].detach().cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()
            cam_total += cv2.resize(cam, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        cam_total /= max(1, len(self.layer_ids))

        # Remove letterbox padding and map back to the original image size
        cam_total = cam_total[y0:y1, x0:x1]
        cam_total = cv2.resize(cam_total, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        if cam_total.max() > 0:
            cam_total = cam_total / cam_total.max()

        return {
            "cam": cam_total,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "n_anchors": int(mask.sum().item()),
            "max_score": float(cls_probs.max().item()),
        }


def overlay_cam(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, color, alpha, 0)
