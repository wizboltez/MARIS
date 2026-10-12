"""Geospatial Marine Plastic Pollution & Biodiversity Risk Mapping Module.

Generates interactive geospatial maps and static publication-grade cartographic
visualizations synthesizing:
- Plastic pollution density heatmaps
- Coral bleaching alerts (NOAA Coral Reef Watch integration)
- Survey stations & Marine Pollution Impact Scores (0-100)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def tier_of(mpis: float) -> str:
    return "Low" if mpis < 25 else "Moderate" if mpis < 50 else "High" if mpis < 75 else "Critical"


def build_station_table(assessment: pd.DataFrame, stations: dict, ocean_cache: dict) -> pd.DataFrame:
    """One row per survey station: real NOAA Coral Reef Watch values + the mean scores of
    the survey images assigned to that station by the assessment pipeline.

    (TrashCan images carry no geotags, so the image-to-station assignment is by rotation;
    the station positions and NOAA values are real.)
    """
    records = []
    for name, st in stations.items():
        noaa = ocean_cache.get("stations", {}).get(name, {}).get("noaa_crw", {})
        rows = assessment[assessment["station_region"] == st["location"]] if len(assessment) else assessment
        mpis = float(rows["marine_pollution_impact_score"].mean()) if len(rows) else np.nan
        records.append({
            "station_id": name,
            "region": st["location"],
            "latitude": st["latitude"],
            "longitude": st["longitude"],
            "n_survey_images": int(len(rows)),
            "plastic_items_detected": int(rows["plastic_count"].sum()) if len(rows) else 0,
            "plastic_coverage_pct": round(float(rows["plastic_coverage_percentage"].mean()), 2) if len(rows) else 0.0,
            "water_quality_index": round(float(rows["water_quality_index"].mean()), 1) if len(rows) else np.nan,
            "marine_pollution_impact_score": round(mpis, 1),
            "impact_tier": tier_of(mpis) if not np.isnan(mpis) else "No data",
            "biodiversity_loss_pct": round(float(rows["predicted_biodiversity_loss_pct"].mean()), 1) if len(rows) else np.nan,
            "noaa_sst_c": noaa.get("sea_surface_temperature_c"),
            "noaa_dhw": noaa.get("degree_heating_weeks"),
            "noaa_bleaching_alert": noaa.get("bleaching_alert_level", "n/a"),
            "noaa_date": noaa.get("date"),
        })
    return pd.DataFrame(records)


def create_interactive_plastic_map(df_stations: pd.DataFrame) -> go.Figure:
    """Creates a high-performance interactive Plotly Scattergeo map."""
    color_map = {
        "Low": "#2ecc71",
        "Moderate": "#f1c40f",
        "High": "#e67e22",
        "Critical": "#e74c3c",
    }

    fig = px.scatter_geo(
        df_stations,
        lat="latitude",
        lon="longitude",
        color="impact_tier",
        color_discrete_map=color_map,
        size=df_stations["plastic_items_detected"] + 3,
        hover_name="region",
        hover_data={
            "marine_pollution_impact_score": True,
            "plastic_items_detected": True,
            "n_survey_images": True,
            "noaa_sst_c": True,
            "noaa_dhw": True,
            "noaa_bleaching_alert": True,
            "noaa_date": True,
            "latitude": False,
            "longitude": False,
        },
        title="<b>Survey Stations: Impact Score & NOAA Coral Reef Watch Status</b>",
        size_max=22,
        projection="natural earth",
    )

    fig.update_geos(
        showocean=True,
        oceancolor="#0d1b2a",
        showland=True,
        landcolor="#1b263b",
        showlakes=False,
        showrivers=False,
        showcountries=True,
        countrycolor="#415a77",
        bgcolor="#070d16",
    )

    fig.update_layout(
        paper_bgcolor="#070d16",
        plot_bgcolor="#070d16",
        font=dict(color="#e0e1dd"),
        title_font=dict(size=18, color="#00d2d3"),
        legend=dict(
            title="Pollution Impact Tier",
            bgcolor="rgba(13, 27, 42, 0.8)",
            bordercolor="#415a77",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        height=540,
    )

    return fig


def plot_static_pollution_map(df_stations: pd.DataFrame, save_path: str | None = None) -> plt.Figure:
    """Generates a cartographic 2D plot for offline publication and report inclusion."""
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="#0e1726")
    ax.set_facecolor("#0e1726")

    # Scatter points colored by MPIS
    scatter = ax.scatter(
        df_stations["longitude"],
        df_stations["latitude"],
        c=df_stations["marine_pollution_impact_score"],
        s=df_stations["plastic_items_detected"] * 25 + 80,
        cmap="turbo",
        alpha=0.88,
        edgecolors="white",
        linewidths=0.6,
    )

    for _, r in df_stations.iterrows():
        ax.annotate(
            f"{r['region']}\nMPIS {r['marine_pollution_impact_score']} | {r['noaa_bleaching_alert']}",
            (r["longitude"], r["latitude"]), xytext=(6, 6), textcoords="offset points", color="white", fontsize=7,
        )

    cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.12, fraction=0.04)
    cbar.set_label("Marine Pollution Impact Score (0 - 100)", color="white", fontsize=10)
    cbar.ax.tick_params(colors="white")

    ax.set_title("Survey Stations: Impact Score (colour), Plastic Items (size), NOAA CRW Alert (label)", color="white", fontsize=13, pad=15)
    ax.set_xlabel("Longitude (°)", color="#a0aec0", fontsize=9)
    ax.set_ylabel("Latitude (°)", color="#a0aec0", fontsize=9)
    ax.tick_params(colors="#a0aec0")
    ax.grid(color="#2d3748", linestyle="--", alpha=0.5)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 65)
    scatter.set_clim(0, 100)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
    return fig
