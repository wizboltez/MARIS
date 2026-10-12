"""Water Quality & Oceanographic Telemetry Integration Module.

Integrates real-time / simulated oceanographic data (temperature, pH, dissolved oxygen,
salinity, turbidity) inspired by Copernicus Marine Service & NOAA In-Situ Stations.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# Marine water quality standard baselines (Copernicus / NOAA marine benchmarks)
OCEAN_BASELINES = {
    "temperature_c": {"opt_min": 23.5, "opt_max": 27.8, "crit_min": 18.0, "crit_max": 31.5, "unit": "°C"},
    "ph": {"opt_min": 8.05, "opt_max": 8.35, "crit_min": 7.60, "crit_max": 8.60, "unit": "pH"},
    "dissolved_oxygen_mg_l": {"opt_min": 6.2, "opt_max": 8.5, "crit_min": 3.0, "crit_max": 11.0, "unit": "mg/L"},
    "salinity_psu": {"opt_min": 34.2, "opt_max": 36.5, "crit_min": 28.0, "crit_max": 41.0, "unit": "PSU"},
    "turbidity_ntu": {"opt_min": 0.2, "opt_max": 2.2, "crit_min": 0.0, "crit_max": 12.0, "unit": "NTU"},
}

PRESET_STATIONS = {
    "Pristine Reef Sanctuary (Coral Triangle)": {
        "temperature_c": 26.2,
        "ph": 8.22,
        "dissolved_oxygen_mg_l": 7.4,
        "salinity_psu": 35.1,
        "turbidity_ntu": 0.6,
        "location": "Raja Ampat, Coral Triangle",
        "latitude": -0.52,
        "longitude": 130.65,
    },
    "Great Barrier Reef - Outer Reef": {
        "temperature_c": 27.6,
        "ph": 8.16,
        "dissolved_oxygen_mg_l": 6.8,
        "salinity_psu": 35.4,
        "turbidity_ntu": 1.2,
        "location": "Cairns Sector, Great Barrier Reef",
        "latitude": -16.85,
        "longitude": 146.22,
    },
    "Thermal Stress Hotspot (Bleaching Risk)": {
        "temperature_c": 30.8,
        "ph": 8.08,
        "dissolved_oxygen_mg_l": 5.4,
        "salinity_psu": 36.1,
        "turbidity_ntu": 2.8,
        "location": "Red Sea Central Channel",
        "latitude": 21.45,
        "longitude": 38.80,
    },
    "Coastal Harbor & Runoff Zone": {
        "temperature_c": 28.5,
        "ph": 7.78,
        "dissolved_oxygen_mg_l": 4.1,
        "salinity_psu": 32.2,
        "turbidity_ntu": 9.4,
        "location": "Manila Bay Outflow",
        "latitude": 14.52,
        "longitude": 120.90,
    },
    "Hypoxic Coastal Estuary (Dead Zone Warning)": {
        "temperature_c": 29.2,
        "ph": 7.65,
        "dissolved_oxygen_mg_l": 2.8,
        "salinity_psu": 29.8,
        "turbidity_ntu": 13.5,
        "location": "Gulf Coastal Estuary",
        "latitude": 28.70,
        "longitude": -90.50,
    },
}


def calculate_water_quality_index(
    temperature_c: float,
    ph: float,
    dissolved_oxygen_mg_l: float,
    salinity_psu: float,
    turbidity_ntu: float,
) -> dict:
    """Computes parameter sub-indices and overall Water Quality Index (WQI 0-100)."""
    # 1. Temperature score (opt: 24-27.5°C)
    if 24.0 <= temperature_c <= 27.8:
        t_score = 100.0
    elif temperature_c > 27.8:
        t_score = max(0.0, 100.0 - (temperature_c - 27.8) * 22.0)
    else:
        t_score = max(0.0, 100.0 - (24.0 - temperature_c) * 16.0)

    # 2. pH score (opt: 8.1 - 8.35; acidification penalty)
    if 8.1 <= ph <= 8.35:
        ph_score = 100.0
    elif ph < 8.1:
        ph_score = max(0.0, 100.0 - (8.1 - ph) * 180.0)
    else:
        ph_score = max(0.0, 100.0 - (ph - 8.35) * 150.0)

    # 3. Dissolved Oxygen score (opt: >= 6.5 mg/L; hypoxia hazard below 4.5)
    if dissolved_oxygen_mg_l >= 6.5:
        do_score = 100.0
    elif dissolved_oxygen_mg_l >= 4.5:
        do_score = 65.0 + ((dissolved_oxygen_mg_l - 4.5) / 2.0) * 35.0
    elif dissolved_oxygen_mg_l >= 2.0:
        do_score = 25.0 + ((dissolved_oxygen_mg_l - 2.0) / 2.5) * 40.0
    else:
        do_score = max(0.0, (dissolved_oxygen_mg_l / 2.0) * 25.0)

    # 4. Salinity score (opt: 34.0 - 36.5 PSU)
    if 34.0 <= salinity_psu <= 36.5:
        sal_score = 100.0
    elif salinity_psu < 34.0:
        sal_score = max(0.0, 100.0 - (34.0 - salinity_psu) * 14.0)
    else:
        sal_score = max(0.0, 100.0 - (salinity_psu - 36.5) * 18.0)

    # 5. Turbidity score (opt: < 1.5 NTU; sediment pollution above 5.0)
    if turbidity_ntu <= 1.5:
        turb_score = 100.0
    elif turbidity_ntu <= 5.0:
        turb_score = max(50.0, 100.0 - (turbidity_ntu - 1.5) * 14.0)
    else:
        turb_score = max(0.0, 50.0 - (turbidity_ntu - 5.0) * 6.5)

    # Weighted aggregate WQI (Weights: DO 30%, Temp 25%, pH 20%, Turbidity 15%, Salinity 10%)
    wqi = (
        0.30 * do_score
        + 0.25 * t_score
        + 0.20 * ph_score
        + 0.15 * turb_score
        + 0.10 * sal_score
    )
    wqi = float(np.clip(wqi, 0.0, 100.0))

    if wqi >= 85:
        rating = "Excellent / Pristine Marine Condition"
        status_color = "#2ecc71"
    elif wqi >= 65:
        rating = "Good / Low Environmental Stress"
        status_color = "#3498db"
    elif wqi >= 45:
        rating = "Fair / Moderate Environmental Hazard"
        status_color = "#f39c12"
    elif wqi >= 25:
        rating = "Poor / High Environmental Degradation"
        status_color = "#e67e22"
    else:
        rating = "Critical / Severe Hypoxia & Acidification"
        status_color = "#e74c3c"

    # Environmental stress metrics
    hypoxia_risk = max(0.0, (5.0 - dissolved_oxygen_mg_l) * 20.0) if dissolved_oxygen_mg_l < 5.0 else 0.0
    acidification_risk = max(0.0, (8.1 - ph) * 200.0) if ph < 8.1 else 0.0
    thermal_stress_deg_c = max(0.0, temperature_c - 27.8)

    return {
        "temperature_c": temperature_c,
        "ph": ph,
        "dissolved_oxygen_mg_l": dissolved_oxygen_mg_l,
        "salinity_psu": salinity_psu,
        "turbidity_ntu": turbidity_ntu,
        "water_quality_index": round(wqi, 1),
        "rating": rating,
        "status_color": status_color,
        "hypoxia_risk_pct": round(min(100.0, hypoxia_risk), 1),
        "acidification_risk_pct": round(min(100.0, acidification_risk), 1),
        "thermal_anomaly_c": round(thermal_stress_deg_c, 2),
        "sub_scores": {
            "Dissolved Oxygen": round(do_score, 1),
            "Temperature": round(t_score, 1),
            "pH (Acidification)": round(ph_score, 1),
            "Turbidity (Clarity)": round(turb_score, 1),
            "Salinity": round(sal_score, 1),
        },
    }


def plot_water_quality_radar(wq_data: dict, save_path: str | None = None) -> plt.Figure:
    """Plots a radar / spider chart of oceanographic sub-scores."""
    sub_scores = wq_data["sub_scores"]
    categories = list(sub_scores.keys())
    values = list(sub_scores.values())

    # Complete radar loop
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#00a8ff", linewidth=2.5, linestyle="solid")
    ax.fill(angles, values, color="#00a8ff", alpha=0.35)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 100)

    ax.set_title(f"Oceanographic Quality Index: {wq_data['water_quality_index']}/100\n({wq_data['rating']})", fontsize=10, pad=15)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
