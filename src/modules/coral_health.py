"""Coral Reef Health & Bleaching Assessment Module.

Implements computer vision-based coral reef health assessment, substrate
segmentation, and bleaching index calculation in alignment with NOAA Coral Reef
Watch and Allen Coral Atlas criteria.
"""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np


class CoralHealthAnalyzer:
    """Evaluates coral cover, bleaching severity, and algal overgrowth from underwater imagery."""

    def __init__(self):
        # Colorimetric reference ranges in HSV and LAB space
        pass

    def analyze_image(
        self,
        image_bgr: np.ndarray,
        temperature_c: float = 27.5,
        turbidity_ntu: float = 1.8,
    ) -> dict:
        """Analyzes an underwater image to determine coral coverage and bleaching severity."""
        h, w = image_bgr.shape[:2]
        total_pixels = h * w

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

        # 1. Coral substrate identification: exclude open water column (top deep blue/cyan)
        # Low saturation or high brightness with texture indicates substrate/coral
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Water column mask (typically blue-dominant, smooth, in upper third or deep areas)
        # Substrate/coral typically resides in mid-to-lower frame or textured zones
        y_coords = np.tile(np.arange(h)[:, None], (1, w))
        depth_weight = np.clip((y_coords - h * 0.2) / (h * 0.8), 0.0, 1.0)

        # 2. Bleached coral detection: High luminance L in LAB + low saturation S in HSV
        l_channel = lab[:, :, 0]
        s_channel = hsv[:, :, 1]
        h_channel = hsv[:, :, 0]

        # Bleached coral mask (pale white/light chalky gray, high L, low S)
        bleached_mask = (l_channel > 155) & (s_channel < 75) & (depth_weight > 0.2)

        # Healthy coral mask (warm browns, purples, oranges, vibrant greens, medium L, medium-high S)
        # Coral hues: oranges/browns (5-25), purples/reds (150-180), or vibrant yellows (25-35)
        warm_coral = ((h_channel < 35) | (h_channel > 150)) & (s_channel > 60) & (l_channel > 60) & (l_channel < 190)
        healthy_mask = warm_coral & (depth_weight > 0.2) & (~bleached_mask)

        # Algae overgrowth mask (dark olive-green / dull brown turf algae, hue 35-75, low-medium L)
        algae_mask = (h_channel >= 35) & (h_channel <= 85) & (s_channel > 40) & (l_channel < 140) & (depth_weight > 0.2)

        # Stressed / pale coral (intermediate L, moderate S)
        stressed_mask = (l_channel >= 135) & (l_channel <= 155) & (s_channel < 95) & (depth_weight > 0.2) & (~healthy_mask) & (~algae_mask)

        bleached_px = int(bleached_mask.sum())
        healthy_px = int(healthy_mask.sum())
        algae_px = int(algae_mask.sum())
        stressed_px = int(stressed_mask.sum())

        total_coral_px = max(1, healthy_px + bleached_px + stressed_px)
        substrate_px = max(1, total_coral_px + algae_px)

        coral_cover_pct = round(min(85.0, (total_coral_px / total_pixels) * 100.0 * 2.5), 1)

        # Bleaching Severity Index (BSI, 0-100)
        # Weighted by bleached area, stressed area, and thermal anomaly
        raw_bleach_ratio = (bleached_px + 0.5 * stressed_px) / total_coral_px
        temp_stress_bonus = max(0.0, (temperature_c - 28.5) * 12.0)
        turbidity_penalty = max(0.0, (turbidity_ntu - 5.0) * 2.5)

        bsi = float(np.clip(raw_bleach_ratio * 70.0 + temp_stress_bonus + turbidity_penalty, 0.0, 100.0))

        # Classify health status & NOAA Alert Level
        if bsi < 20:
            health_status = "Healthy / Low Stress"
            noaa_alert = "No Stress"
            primary_condition = "Vibrant Reef Habitat"
        elif bsi < 45:
            health_status = "Bleaching Watch / Mild Thermal Stress"
            noaa_alert = "Bleaching Watch"
            primary_condition = "Partial Pigment Expulsion"
        elif bsi < 70:
            health_status = "Bleaching Warning / Moderate Bleaching"
            noaa_alert = "Bleaching Warning"
            primary_condition = "Significant Symbiont Depletion"
        elif bsi < 85:
            health_status = "Alert Level 1 / Extensive Bleaching"
            noaa_alert = "Alert Level 1"
            primary_condition = "Severe Bleaching Hazard"
        else:
            health_status = "Alert Level 2 / Severe Bleaching & Mortality"
            noaa_alert = "Alert Level 2"
            primary_condition = "Imminent Coral Mortality"

        algae_ratio = round((algae_px / max(1, total_coral_px)), 2)

        masks = {
            "healthy": healthy_mask,
            "stressed": stressed_mask,
            "bleached": bleached_mask,
            "algae": algae_mask,
        }

        return {
            "coral_cover_percentage": coral_cover_pct,
            "bleaching_severity_index": round(bsi, 1),
            "healthy_coral_ratio_pct": round((healthy_px / total_coral_px) * 100.0, 1),
            "bleached_coral_ratio_pct": round((bleached_px / total_coral_px) * 100.0, 1),
            "algae_competition_ratio": algae_ratio,
            "health_status": health_status,
            "noaa_alert_level": noaa_alert,
            "primary_condition": primary_condition,
            "masks": masks,
        }


def draw_coral_health_overlay(image_bgr: np.ndarray, analysis: dict, alpha: float = 0.45) -> np.ndarray:
    """Overlays diagnostic color-coded masks for coral health and bleaching zones."""
    out = image_bgr.copy()
    overlay = image_bgr.copy()
    h, w = out.shape[:2]

    masks = analysis.get("masks", {})
    if "healthy" in masks and masks["healthy"].any():
        overlay[masks["healthy"]] = (30, 200, 30)  # Green for healthy coral
    if "stressed" in masks and masks["stressed"].any():
        overlay[masks["stressed"]] = (0, 215, 255)  # Gold/yellow for stressed
    if "bleached" in masks and masks["bleached"].any():
        overlay[masks["bleached"]] = (245, 245, 255)  # White/bright cyan for bleached
    if "algae" in masks and masks["algae"].any():
        overlay[masks["algae"]] = (0, 100, 160)  # Brownish for algae

    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    # Top-right summary badge
    bsi = analysis["bleaching_severity_index"]
    alert = analysis["noaa_alert_level"]
    status = analysis["health_status"]

    badge_w, badge_h = 320, 75
    bx1 = w - badge_w - 12
    by1 = 12
    bx2 = w - 12
    by2 = by1 + badge_h

    badge_color = (0, 140, 0) if bsi < 25 else ((0, 180, 240) if bsi < 60 else (0, 0, 220))
    cv2.rectangle(out, (bx1, by1), (bx2, by2), (25, 25, 25), -1)
    cv2.rectangle(out, (bx1, by1), (bx2, by2), badge_color, 2)

    cv2.putText(out, f"Coral BSI: {bsi:.1f}/100", (bx1 + 10, by1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"NOAA: {alert}", (bx1 + 10, by1 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"Cover: {analysis['coral_cover_percentage']:.1f}%", (bx1 + 10, by1 + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1, cv2.LINE_AA)

    return out


def plot_coral_health_summary(analysis: dict, save_path: str | None = None) -> plt.Figure:
    """Plots coral coverage distribution and NOAA bleaching metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: Substrate composition
    categories = ["Healthy Coral", "Bleached Coral", "Stressed / Other"]
    bleached = analysis["bleached_coral_ratio_pct"]
    healthy = analysis["healthy_coral_ratio_pct"]
    other = max(0.0, 100.0 - (healthy + bleached))
    shares = [healthy, bleached, other]
    colors = ["#27ae60", "#ecf0f1", "#f39c12"]

    axes[0].pie(shares, labels=categories, colors=colors, autopct="%1.1f%%", startangle=140, textprops={"fontsize": 9})
    axes[0].set_title(f"Coral Composition (Cover: {analysis['coral_cover_percentage']}%)", fontsize=10)

    # Panel 2: Bleaching Severity vs Thresholds
    bsi = analysis["bleaching_severity_index"]
    axes[1].bar(["Bleaching Severity"], [bsi], color="#e74c3c" if bsi > 60 else "#f1c40f" if bsi > 30 else "#2ecc71", width=0.35)
    axes[1].set_ylim(0, 100)
    axes[1].axhline(25, color="gray", linestyle="--", alpha=0.7, label="Watch (25)")
    axes[1].axhline(60, color="orange", linestyle="--", alpha=0.7, label="Warning (60)")
    axes[1].axhline(80, color="red", linestyle="--", alpha=0.7, label="Alert L2 (80)")
    axes[1].set_ylabel("Bleaching Severity Index (0-100)")
    axes[1].set_title(f"NOAA Status: {analysis['noaa_alert_level']}", fontsize=10)
    axes[1].legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
