"""Gulf of Mannar site risk (environmental-risk-module, notebooks/marine_pollution_prediction.ipynb).

The notebook scores 245 ocean grid cells (~9 km) inside the Allen Coral Atlas polygon from
Copernicus currents/temperature/waves, OBIS species records and Atlas habitat, and writes
two tables that the dashboard reads from data/external/gulf_of_mannar/:
- marine_pollution_impact_score.csv: site_id, lat, lon, value_score, exposure_score,
  biodiversity_deficit, marine_pollution_impact_score (0-100), impact_rank
- site_features.csv (optional): per-site features used for hover details
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go

from src.config import PROJECT_ROOT
from src.modules.geospatial_maps import tier_of

REGION_DIR = PROJECT_ROOT / "data" / "external" / "gulf_of_mannar"
IMPACT_CSV = REGION_DIR / "marine_pollution_impact_score.csv"
FEATURES_CSV = REGION_DIR / "site_features.csv"
METRICS_JSON = PROJECT_ROOT / "models" / "environmental_risk" / "metrics.json"

REQUIRED_COLS = ["site_id", "lat", "lon", "value_score", "exposure_score",
                 "biodiversity_deficit", "marine_pollution_impact_score", "impact_rank"]
DETAIL_COLS = ["species_richness", "habitat_sens", "coral_dist_km", "current_speed", "wave_height", "temperature"]

# Same boundary as POLY in the notebook (lon, lat)
ATLAS_POLYGON = [(78.356112, 9.17125), (80.06426, 10.021422), (80.419466, 9.335989), (79.818754, 7.719526)]

TIER_COLORS = {"Low": "#2a9d8f", "Moderate": "#e9c46a", "High": "#f4a261", "Critical": "#e63946"}


def load_metrics() -> dict | None:
    return json.loads(METRICS_JSON.read_text()) if METRICS_JSON.exists() else None


def load_sites() -> pd.DataFrame | None:
    """Impact table joined with site features when present. None if the CSV is missing."""
    if not IMPACT_CSV.exists():
        return None
    sites = pd.read_csv(IMPACT_CSV)
    missing = [c for c in REQUIRED_COLS if c not in sites.columns]
    if missing:
        raise ValueError(f"{IMPACT_CSV.name} is missing columns: {', '.join(missing)}")
    if FEATURES_CSV.exists():
        feats = pd.read_csv(FEATURES_CSV)
        keep = ["site_id"] + [c for c in DETAIL_COLS if c in feats.columns]
        sites = sites.merge(feats[keep], on="site_id", how="left")
    sites["tier"] = sites["marine_pollution_impact_score"].map(tier_of)
    return sites.sort_values("impact_rank").reset_index(drop=True)


def create_site_risk_map(sites: pd.DataFrame) -> go.Figure:
    details = [c for c in DETAIL_COLS if c in sites.columns]
    hover = ("<b>%{customdata[0]}</b> — score %{customdata[1]:.1f} (rank %{customdata[2]})<br>"
             "value %{customdata[3]:.2f} · exposure %{customdata[4]:.2f} · deficit %{customdata[5]:.2f}")
    hover += "".join(f"<br>{c.replace('_', ' ')}: %{{customdata[{6 + i}]:.2f}}" for i, c in enumerate(details))
    cols = ["site_id", "marine_pollution_impact_score", "impact_rank",
            "value_score", "exposure_score", "biodiversity_deficit"] + details

    fig = go.Figure()
    lons, lats = zip(*(ATLAS_POLYGON + [ATLAS_POLYGON[0]]))
    fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode="lines", name="Coral Atlas study area",
                                line=dict(color="#00d2d3", width=1.5, dash="dash"), hoverinfo="skip"))
    for tier, color in TIER_COLORS.items():
        d = sites[sites["tier"] == tier]
        if d.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lon=d["lon"], lat=d["lat"], mode="markers", name=f"{tier} ({len(d)})",
            marker=dict(size=9, color=color, line=dict(width=0.5, color="#0d1b2a")),
            customdata=d[cols].to_numpy(), hovertemplate=hover + "<extra></extra>",
        ))
    fig.update_geos(
        projection_type="mercator", resolution=50, fitbounds="locations",
        showocean=True, oceancolor="#0d1b2a", showland=True, landcolor="#1b263b",
        showcountries=True, countrycolor="#415a77", coastlinecolor="#778da9", bgcolor="#070d16",
    )
    fig.update_layout(
        paper_bgcolor="#070d16", font=dict(color="#e0e1dd"), height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(title="Impact level (click to hide/show)", orientation="h",
                    yanchor="bottom", y=1.0, xanchor="center", x=0.5),
    )
    return fig
