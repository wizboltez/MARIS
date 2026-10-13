"""Targeted Per-Detection Grad-CAM and Deletion/Insertion Faithfulness Testing.

Focuses backward gradients exclusively on the anchors whose spatial centers
((j + 0.5) * stride, (i + 0.5) * stride) fall inside the selected detection box,
avoiding scene-wide spread to ROV equipment, marine snow, and HUD overlays.
"""
from __future__ import annotations

import copy
import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox

from src.config import CLASS_NAMES

PLASTIC_ID = CLASS_NAMES.index("trash_plastic")


def integrate_curve(y: list[float], dx: float) -> float:
    """Safe trapezoidal integration across NumPy versions (NumPy 2+ uses trapezoid)."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=dx))
    return float(np.trapz(y, dx=dx))


class TargetedYOLOGradCAM:
    def __init__(self, yolo_model, layer_ids: tuple[int, ...] = (16, 19, 22), imgsz: int = 640):
        # Deep copy so gradient hooks never interfere with normal inference
        self.model = copy.deepcopy(yolo_model.model).float().cpu()
        self.model.eval()
        self.layer_ids = layer_ids
        self.imgsz = imgsz
        self._acts: dict[int, torch.Tensor] = {}

        for i in self.layer_ids:
            self.model.model[i].register_forward_hook(self._save_activation(i))

    def _save_activation(self, idx: int):
        def hook(_module, _inp, out):
            if out.requires_grad:
                out.retain_grad()
            self._acts[idx] = out
        return hook

    def _letterbox_image(self, img_bgr: np.ndarray) -> tuple[torch.Tensor, tuple[int, int, int, int], float]:
        lb = LetterBox((self.imgsz, self.imgsz), auto=False, stride=32)
        img_pad = lb(image=img_bgr)
        h0, w0 = img_bgr.shape[:2]
        r = min(self.imgsz / h0, self.imgsz / w0)
        pad_x = (self.imgsz - round(w0 * r)) / 2.0
        pad_y = (self.imgsz - round(h0 * r)) / 2.0
        x = torch.from_numpy(img_pad[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        crop = (int(round(pad_y)), int(round(pad_y + h0 * r)), int(round(pad_x)), int(round(pad_x + w0 * r)))
        return x, crop, r

    def compute_for_box(self, img_bgr: np.ndarray, box_xyxy: np.ndarray, class_id: int = PLASTIC_ID) -> np.ndarray:
        """Compute Grad-CAM targeted only to anchors whose center is inside box_xyxy."""
        x, (y0, y1, x0, x1), r = self._letterbox_image(img_bgr)
        x.requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        self._acts.clear()

        # Map box coordinates [x1, y1, x2, y2] to letterboxed 640x640 space
        h0, w0 = img_bgr.shape[:2]
        pad_x = (self.imgsz - round(w0 * r)) / 2.0
        pad_y = (self.imgsz - round(h0 * r)) / 2.0
        bx1 = box_xyxy[0] * r + pad_x
        by1 = box_xyxy[1] * r + pad_y
        bx2 = box_xyxy[2] * r + pad_x
        by2 = box_xyxy[3] * r + pad_y

        with torch.enable_grad():
            _, aux = self.model(x)
            logits = aux["scores"][0]  # shape (nc, 8400)
            head = self.model.model[-1]
            anchors_px = (head.anchors * head.strides).detach().cpu().numpy()  # shape (2, 8400)
            anchor_x = anchors_px[0]
            anchor_y = anchors_px[1]

            in_box = (anchor_x >= bx1) & (anchor_x <= bx2) & (anchor_y >= by1) & (anchor_y <= by2)
            matched_indices = np.where(in_box)[0]

            if len(matched_indices) == 0:
                # Fallback for very small boxes: select closest spatial anchor center
                center_x, center_y = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                dist_sq = (anchor_x - center_x) ** 2 + (anchor_y - center_y) ** 2
                matched_indices = [int(np.argmin(dist_sq))]

            target = logits[class_id, matched_indices].sum()
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

        # Unpad back to actual image resolution
        cam_crop = cam_total[y0:y1, x0:x1]
        cam_orig = cv2.resize(cam_crop, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        if cam_orig.max() > 0:
            cam_orig = cam_orig / cam_orig.max()
        return cam_orig


def evaluate_faithfulness(yolo_model, cam_engine: TargetedYOLOGradCAM, img_bgr: np.ndarray, box_xyxy: np.ndarray, steps: int = 10) -> dict:
    """Run Deletion and Insertion Faithfulness evaluation on a single detection."""
    cam = cam_engine.compute_for_box(img_bgr, box_xyxy)
    blurred_img = cv2.GaussianBlur(img_bgr, (51, 51), 0)

    # Initial detection confidence
    res0 = yolo_model.predict(img_bgr, conf=0.01, verbose=False)[0]
    orig_conf = 0.0
    if res0.boxes is not None and len(res0.boxes) > 0:
        for b, c, cl in zip(res0.boxes.xyxy.cpu().numpy(), res0.boxes.conf.cpu().numpy(), res0.boxes.cls.cpu().numpy().astype(int)):
            if cl == PLASTIC_ID and np.linalg.norm(b - box_xyxy) < 40:
                orig_conf = float(c)
                break

    fractions = np.linspace(0.1, 0.9, steps)
    deletion_scores = [orig_conf]
    insertion_scores = [0.0]

    for frac in fractions:
        thresh = float(np.quantile(cam, 1.0 - frac))

        # Deletion: remove top CAM pixels (replace with blur)
        del_img = img_bgr.copy()
        del_img[cam >= thresh] = blurred_img[cam >= thresh]
        res_del = yolo_model.predict(del_img, conf=0.005, verbose=False)[0]
        s_del = 0.0
        if res_del.boxes is not None and len(res_del.boxes) > 0:
            for b, c, cl in zip(res_del.boxes.xyxy.cpu().numpy(), res_del.boxes.conf.cpu().numpy(), res_del.boxes.cls.cpu().numpy().astype(int)):
                if cl == PLASTIC_ID and np.linalg.norm(b - box_xyxy) < 50:
                    s_del = float(c)
                    break
        deletion_scores.append(s_del)

        # Insertion: start from blurred canvas, restore top CAM pixels
        ins_img = blurred_img.copy()
        ins_img[cam >= thresh] = img_bgr[cam >= thresh]
        res_ins = yolo_model.predict(ins_img, conf=0.005, verbose=False)[0]
        s_ins = 0.0
        if res_ins.boxes is not None and len(res_ins.boxes) > 0:
            for b, c, cl in zip(res_ins.boxes.xyxy.cpu().numpy(), res_ins.boxes.conf.cpu().numpy(), res_ins.boxes.cls.cpu().numpy().astype(int)):
                if cl == PLASTIC_ID and np.linalg.norm(b - box_xyxy) < 50:
                    s_ins = float(c)
                    break
        insertion_scores.append(s_ins)

    auc_del = integrate_curve(deletion_scores, dx=1.0 / steps)
    auc_ins = integrate_curve(insertion_scores, dx=1.0 / steps)

    # Compute % heat inside target bounding box
    bx1, by1, bx2, by2 = map(int, box_xyxy)
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(img_bgr.shape[1], bx2), min(img_bgr.shape[0], by2)
    heat_in_box = float(cam[by1:by2, bx1:bx2].sum())
    total_heat = float(cam.sum())
    pct_in_box = (heat_in_box / total_heat * 100.0) if total_heat > 0 else 0.0

    return {
        "heat_inside_box_pct": round(pct_in_box, 2),
        "deletion_auc": round(auc_del, 3),
        "insertion_auc": round(auc_ins, 3),
        "orig_conf": round(orig_conf, 3),
    }
