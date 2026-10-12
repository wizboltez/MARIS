"""Explainable AI (XAI) Engine & Recommendation System.

Implements:
1. SHAP (SHapley Additive exPlanations) feature attribution for the impact model.
2. Grad-CAM / Visual Saliency heatmap generator for underwater imagery.
3. Domain-specific, explainable AI ecological recommendations for decision-makers.
"""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class MarineXAIExplainer:
    """Computes SHAP explanations and generates visual Grad-CAM overlays."""

    def __init__(self, mpis_model, background_data: pd.DataFrame | None = None):
        self.model = mpis_model
        if background_data is not None:
            # Keep only the model's input columns (training data also carries the targets).
            features = list(self.model.feature_names_in_) if hasattr(self.model, "feature_names_in_") else background_data.columns
            background = background_data[features].sample(min(100, len(background_data)), random_state=42)
            self.explainer = shap.TreeExplainer(self.model, data=background)
        else:
            self.explainer = shap.TreeExplainer(self.model)

    def explain_prediction(self, feature_df: pd.DataFrame) -> dict:
        """Computes SHAP values and feature contributions for a single survey prediction."""
        shap_values = self.explainer(feature_df, check_additivity=False)
        base_val = float(shap_values.base_values[0])
        values = shap_values.values[0]
        feature_names = feature_df.columns.tolist()

        contributions = []
        for name, val, f_val in zip(feature_names, values, feature_df.iloc[0]):
            contributions.append(
                {
                    "feature": name,
                    "feature_value": float(f_val),
                    "shap_value": float(val),
                    "impact_direction": "Increases Risk" if val > 0 else "Lowers Risk / Protective",
                    "abs_impact": abs(float(val)),
                }
            )

        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

        return {
            "base_value": round(base_val, 2),
            "predicted_score": round(base_val + float(np.sum(values)), 2),
            "contributions": contributions,
            "shap_explanation": shap_values,
        }

    def generate_gradcam_heatmap(
        self,
        image_bgr: np.ndarray,
        plastic_boxes: list[np.ndarray] | None = None,
        coral_masks: dict | None = None,
    ) -> np.ndarray:
        """Approximate saliency overlay highlighting detected plastic and bleached-coral regions.

        NOTE: this is NOT gradient-based Grad-CAM -- it blurs the detection boxes and the
        coral colour mask into a heatmap. It shows *where* detections are, not what the
        network attended to.
        """
        h, w = image_bgr.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        # 1. Plastic debris heat activation centers
        if plastic_boxes:
            for box in plastic_boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                bw, bh = max(10, x2 - x1), max(10, y2 - y1)
                radius = int(max(bw, bh) * 0.8)
                cv2.circle(heatmap, (cx, cy), radius, 1.0, -1)

        # 2. Bleached coral activation
        if coral_masks and "bleached" in coral_masks:
            bleached = coral_masks["bleached"].astype(np.float32)
            if bleached.any():
                heatmap = np.maximum(heatmap, bleached * 0.75)

        # Smooth and Gaussian blur to create smooth Grad-CAM aesthetic
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_uint8 = np.uint8(255 * heatmap)
        color_cam = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Alpha blend with original underwater image
        blended = cv2.addWeighted(image_bgr, 0.55, color_cam, 0.45, 0)
        return blended


def generate_explainable_recommendations(
    mpis_data: dict,
    plastic_stats: dict,
    fish_stats: dict,
    coral_stats: dict,
    water_quality_stats: dict,
) -> list[dict]:
    """Generates structured, actionable, and explainable recommendations for marine managers."""
    recs = []
    mpis = mpis_data["marine_pollution_impact_score"]
    p_cov = plastic_stats.get("plastic_coverage_percentage", 0.0)
    p_count = plastic_stats.get("plastic_count", 0)
    bsi = coral_stats.get("bleaching_severity_index", 0.0)
    bai = fish_stats.get("behavioral_anomaly_index", 20.0)
    do = water_quality_stats.get("dissolved_oxygen_mg_l", 7.0)
    temp = water_quality_stats.get("temperature_c", 26.5)
    turb = water_quality_stats.get("turbidity_ntu", 1.0)

    # 1. Plastic Waste Recommendations
    if p_count > 0 or p_cov > 1.0:
        priority = "Critical Priority" if p_cov > 8.0 or p_count >= 3 else "High Priority"
        recs.append(
            {
                "category": "Plastic Pollution Abatement",
                "priority": priority,
                "driver": f"Detected {p_count} plastic items ({p_cov:.1f}% benthic coverage).",
                "action": "Deploy targeted marine autonomous cleanup ROVs and boom barriers.",
                "explanation": "High benthic plastic density causes physical smothering of sessile organisms, light attenuation, and ingestion hazard for reef fish.",
                "timeline": "Immediate (within 48-72 hours)",
            }
        )

    # 2. Coral Bleaching Interventions
    if bsi > 40.0:
        priority = "Critical Priority" if bsi > 70.0 else "High Priority"
        recs.append(
            {
                "category": "Coral Reef Conservation",
                "priority": priority,
                "driver": f"Bleaching Severity Index is {bsi:.1f}/100 with water temp at {temp:.1f}°C.",
                "action": "Deploy artificial reef shade canopies and initiate larval reseeding.",
                "explanation": "Elevated thermal stress coupled with macroplastic debris accelerates zooxanthellae expulsion and triggers coral tissue necrosis.",
                "timeline": "Urgent (within 1-2 weeks)",
            }
        )

    # 3. Water Quality & Hypoxia Mitigation
    if do < 5.0 or turb > 5.0:
        priority = "Critical Priority" if do < 3.5 else "Moderate Priority"
        recs.append(
            {
                "category": "Oceanographic Water Quality",
                "priority": priority,
                "driver": f"Dissolved oxygen at {do:.1f} mg/L (hypoxic) and turbidity at {turb:.1f} NTU.",
                "action": "Activate localized subsurface bubble aeration and inspect upstream sediment runoff.",
                "explanation": "Oxygen depletion below 5.0 mg/L induces severe respiratory stress and behavioral disorientation in teleost fishes.",
                "timeline": "Immediate to Medium-term",
            }
        )

    # 4. Marine Protected Area (MPA) Zoning
    if mpis > 50.0 or bai > 55.0:
        recs.append(
            {
                "category": "Ecosystem Protection & Policy",
                "priority": "High Priority",
                "driver": f"Marine Pollution Impact Score ({mpis:.1f}/100) indicates acute biodiversity degradation.",
                "action": "Implement a temporary No-Take Zone (NTZ) and restrict motorized vessel transit.",
                "explanation": "Mitigates compounding anthropogenic pressure, allowing juvenile fish populations and stressed coral colonies to recover resilience.",
                "timeline": "Policy Advisory (Next regulatory cycle)",
            }
        )

    # Baseline preservation if healthy
    if len(recs) == 0:
        recs.append(
            {
                "category": "Sanctuary Maintenance",
                "priority": "Routine / Preservation",
                "driver": f"Ecosystem parameters within optimal bounds (MPIS: {mpis:.1f}/100).",
                "action": "Maintain quarterly autonomous visual transects and in-situ sensor telemetry.",
                "explanation": "Habitat exhibits robust ecological equilibrium, healthy coral cover, and low plastic intrusion.",
                "timeline": "Ongoing quarterly schedule",
            }
        )

    return recs


def plot_shap_summary_chart(shap_result: dict, save_path: str | None = None) -> plt.Figure:
    """Renders a clean, aesthetic horizontal bar chart of top SHAP feature drivers."""
    contributions = shap_result["contributions"][:8]
    names = [c["feature"].replace("_", " ").title() for c in contributions][::-1]
    values = [c["shap_value"] for c in contributions][::-1]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(names, values, color=colors, height=0.6)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("SHAP Impact on Marine Pollution Score (+ increases risk / - decreases risk)", fontsize=9)
    ax.set_title(
        f"Explainable AI: Key Drivers of Pollution Impact Score\n(Base: {shap_result['base_value']} -> Predicted: {shap_result['predicted_score']})",
        fontsize=10,
        pad=10,
    )

    for bar, val in zip(bars, values):
        x_pos = val + (0.5 if val >= 0 else -0.5)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}", va="center", ha=ha, fontsize=8, fontweight="bold")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
