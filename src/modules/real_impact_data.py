"""Observational plastic-vs-biodiversity model built only from real, public data.

Per 2-degree ocean grid cell:
  * microplastics  - NOAA NCEI Marine Microplastics Database, ocean-surface samples by
                     professional surveys (pieces/m3), median of all samples in the cell
  * biodiversity   - OBIS (Ocean Biodiversity Information System): number of species recorded
                     in the cell, plus the number of occurrence records (sampling effort)
  * temperature    - NOAA Coral Reef Watch satellite SST: mean of 12 monthly snapshots
  * heat stress    - NOAA Coral Reef Watch Degree Heating Weeks: max of the same 12 snapshots

An XGBoost model predicts log10(species richness) from these, with sampling effort and
latitude as controls (OBIS richness is strongly driven by how much a cell has been surveyed,
and by the latitudinal diversity gradient). It is evaluated with spatial block
cross-validation (10-degree blocks held out together) so neighbouring cells cannot leak.
The plastic effect is measured two ways: SHAP attribution, and the drop in held-out R2
when the microplastics feature is removed.

This shows ASSOCIATION, not causation: microplastic surveys concentrate in ocean gyres and
coastal hotspots, and OBIS coverage is uneven.

Build / refresh:  python -m src.modules.real_impact_data --refresh
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

EXT_DIR = PROJECT_ROOT / "data" / "external"
MODEL_DIR = PROJECT_ROOT / "models" / "impact_model" / "observational"
CELLS_CSV = EXT_DIR / "real_impact_cells.csv"
MP_CACHE = EXT_DIR / "ncei_surface_microplastics.csv"
NOAA_CACHE = EXT_DIR / "noaa_crw_global_2deg.csv"

NCEI_LAYER = ("https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/"
              "Ocean_Surface_Microplastics_(PS)/FeatureServer/0/query")
OBIS_STATS = "https://api.obis.org/v3/statistics"
ERDDAP = "https://coastwatch.noaa.gov/erddap/griddap"

CELL_DEG = 2
MIN_SAMPLES = 3
FEATURES = ["log10_microplastics", "log10_obis_records", "abs_latitude", "sst_mean_c", "dhw_max"]
TARGET = "log10_species"


def _get_json(url: str, timeout: int = 120, retries: int = 3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _read_csv_url(url: str, timeout: int = 60, retries: int = 4) -> pd.DataFrame:
    """ERDDAP occasionally stalls a request; retry with a short timeout instead of waiting minutes."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return pd.read_csv(r, skiprows=[1])
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1} ({type(exc).__name__})", flush=True)
            time.sleep(3 * (attempt + 1))


# ---------------------------------------------------------------- data sources
def fetch_microplastics() -> pd.DataFrame:
    fields = "Latitude__degree_,Longitude_degree_,Microplastics_measurement,Unit,Date__MM_DD_YYYY_,Short_Reference"
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"where": "1=1", "outFields": fields, "resultOffset": offset,
                                    "resultRecordCount": 2000, "f": "json"})
        d = _get_json(f"{NCEI_LAYER}?{q}")
        feats = d.get("features", [])
        rows += [f["attributes"] for f in feats]
        if not d.get("exceededTransferLimit") and len(feats) < 2000:
            break
        offset += len(feats)
    df = pd.DataFrame(rows).rename(columns={
        "Latitude__degree_": "lat", "Longitude_degree_": "lon",
        "Microplastics_measurement": "pieces_m3", "Date__MM_DD_YYYY_": "date_ms", "Short_Reference": "reference"})
    df = df[df["Unit"] == "pieces/m3"].dropna(subset=["lat", "lon", "pieces_m3"])
    df["year"] = pd.to_datetime(df["date_ms"], unit="ms", errors="coerce").dt.year
    return df.drop(columns=["date_ms"])


def _monthly_dates(n: int = 12) -> list[str]:
    """The 15th of each of the last n complete months."""
    today = date.today()
    y, m = today.year, today.month
    out = []
    for _ in range(n):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        out.append(f"{y:04d}-{m:02d}-15T12:00:00Z")
    return out


def fetch_noaa_grid() -> pd.DataFrame:
    """Global 2-degree grid: mean SST and max DHW over 12 monthly snapshots."""
    stride = int(round(CELL_DEG / 0.05))
    frames = []
    for dataset, var in (("noaacrwsstDaily", "analysed_sst"), ("noaacrwdhwDaily", "degree_heating_week")):
        for t in _monthly_dates():
            query = f"{var}[({t})][(89.975):{stride}:(-89.975)][(-179.975):{stride}:(179.975)]"
            url = f"{ERDDAP}/{dataset}.csv?{urllib.parse.quote(query, safe='()[]:,')}"
            f = _read_csv_url(url)
            print(f"    {dataset} {t[:10]}: {f[var].notna().sum()} ocean points", flush=True)
            frames.append(f.rename(columns={var: "value"}).assign(var=var))
    g = pd.concat(frames)
    g = g.dropna(subset=["value"])
    g["cell_lat"] = np.floor(g["latitude"] / CELL_DEG).astype(int)
    g["cell_lon"] = np.floor(g["longitude"] / CELL_DEG).astype(int)
    sst = g[g["var"] == "analysed_sst"].groupby(["cell_lat", "cell_lon"])["value"].mean().rename("sst_mean_c")
    dhw = g[g["var"] == "degree_heating_week"].groupby(["cell_lat", "cell_lon"])["value"].max().rename("dhw_max")
    return pd.concat([sst, dhw], axis=1).reset_index()


def _obis_cell(cell) -> dict:
    cy, cx = cell
    lat0, lon0 = cy * CELL_DEG, cx * CELL_DEG
    poly = (f"POLYGON(({lon0} {lat0},{lon0 + CELL_DEG} {lat0},{lon0 + CELL_DEG} {lat0 + CELL_DEG},"
            f"{lon0} {lat0 + CELL_DEG},{lon0} {lat0}))")
    try:
        d = _get_json(f"{OBIS_STATS}?{urllib.parse.urlencode({'geometry': poly})}", timeout=30, retries=2)
        return {"cell_lat": cy, "cell_lon": cx, "obis_species": d.get("species", 0), "obis_records": d.get("records", 0)}
    except Exception:
        return {"cell_lat": cy, "cell_lon": cx, "obis_species": np.nan, "obis_records": np.nan}


OBIS_CACHE = EXT_DIR / "obis_cells.csv"


def _fetch_obis_cells(cell_ids: list[tuple[int, int]], refresh: bool = False) -> pd.DataFrame:
    """OBIS counts per cell, cached incrementally so an interrupted run resumes where it stopped."""
    done = pd.read_csv(OBIS_CACHE) if (OBIS_CACHE.exists() and not refresh) else pd.DataFrame(
        columns=["cell_lat", "cell_lon", "obis_species", "obis_records"])
    have = set(zip(done["cell_lat"], done["cell_lon"])) if len(done) else set()
    todo = [c for c in cell_ids if c not in have]
    print(f"  OBIS species counts: {len(have)} cached, {len(todo)} to fetch...", flush=True)
    new_rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, row in enumerate(ex.map(_obis_cell, todo), 1):
            new_rows.append(row)
            if i % 50 == 0 or i == len(todo):
                done = pd.concat([done, pd.DataFrame(new_rows)], ignore_index=True)
                done.to_csv(OBIS_CACHE, index=False)
                new_rows = []
                print(f"    {i}/{len(todo)} cells", flush=True)
    done = done.drop_duplicates(subset=["cell_lat", "cell_lon"], keep="last")
    # Concatenating onto an empty frame leaves object dtype; make every column numeric
    return done.apply(pd.to_numeric, errors="coerce")


def build_cells(refresh: bool = False) -> pd.DataFrame:
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    if CELLS_CSV.exists() and not refresh:
        return pd.read_csv(CELLS_CSV)

    print("  NCEI microplastics...")
    mp = fetch_microplastics() if (refresh or not MP_CACHE.exists()) else pd.read_csv(MP_CACHE)
    mp.to_csv(MP_CACHE, index=False)
    mp["cell_lat"] = np.floor(mp["lat"] / CELL_DEG).astype(int)
    mp["cell_lon"] = np.floor(mp["lon"] / CELL_DEG).astype(int)
    cells = (mp.groupby(["cell_lat", "cell_lon"])
               .agg(n_samples=("pieces_m3", "size"), microplastics_median_m3=("pieces_m3", "median"),
                    n_studies=("reference", "nunique"), year_median=("year", "median"))
               .reset_index())
    cells = cells[cells["n_samples"] >= MIN_SAMPLES]
    print(f"    {len(mp)} samples -> {len(cells)} cells with >= {MIN_SAMPLES} samples")

    print("  NOAA Coral Reef Watch (12 monthly snapshots)...")
    noaa = fetch_noaa_grid() if (refresh or not NOAA_CACHE.exists()) else pd.read_csv(NOAA_CACHE)
    noaa.to_csv(NOAA_CACHE, index=False)
    cells = cells.merge(noaa, on=["cell_lat", "cell_lon"], how="left")

    obis = _fetch_obis_cells(list(zip(cells["cell_lat"], cells["cell_lon"])), refresh=refresh)
    cells = cells.merge(obis, on=["cell_lat", "cell_lon"], how="left")

    cells["lat_center"] = cells["cell_lat"] * CELL_DEG + CELL_DEG / 2
    cells["lon_center"] = cells["cell_lon"] * CELL_DEG + CELL_DEG / 2
    cells["abs_latitude"] = cells["lat_center"].abs()
    cells["log10_microplastics"] = np.log10(cells["microplastics_median_m3"] + 1e-4)
    cells["log10_obis_records"] = np.log10(cells["obis_records"] + 1)
    cells["log10_species"] = np.log10(cells["obis_species"] + 1)
    cells.to_csv(CELLS_CSV, index=False)
    return cells


# ---------------------------------------------------------------- model
def spatial_blocks(df: pd.DataFrame, block_deg: int = 10) -> np.ndarray:
    return (np.floor(df["lat_center"] / block_deg) * 1000 + np.floor(df["lon_center"] / block_deg)).astype(int).to_numpy()


def train_and_evaluate(cells: pd.DataFrame) -> dict:
    import joblib
    import shap
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold
    from xgboost import XGBRegressor

    df = cells.dropna(subset=[TARGET]).copy()
    df = df[df["obis_records"] > 0]
    groups = spatial_blocks(df)

    def cv(features):
        pred = np.zeros(len(df))
        for tr, te in GroupKFold(n_splits=5).split(df, groups=groups):
            m = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.9, min_child_weight=5, random_state=42)
            m.fit(df.iloc[tr][features], df.iloc[tr][TARGET])
            pred[te] = m.predict(df.iloc[te][features])
        return float(r2_score(df[TARGET], pred)), float(mean_absolute_error(df[TARGET], pred))

    r2_full, mae_full = cv(FEATURES)
    no_plastic = [f for f in FEATURES if f != "log10_microplastics"]
    r2_noplastic, _ = cv(no_plastic)
    r2_effort_only, _ = cv(["log10_obis_records"])

    final = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.9, min_child_weight=5, random_state=42)
    final.fit(df[FEATURES], df[TARGET])
    sv = shap.TreeExplainer(final)(df[FEATURES])
    shap_abs = dict(zip(FEATURES, np.abs(sv.values).mean(0).round(4).tolist()))

    # Direction of the plastic effect: correlation of its SHAP value with its feature value
    k = FEATURES.index("log10_microplastics")
    direction = float(np.corrcoef(df["log10_microplastics"], sv.values[:, k])[0, 1])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_DIR / "xgb_species_richness.joblib")
    np.save(MODEL_DIR / "shap_values.npy", sv.values)

    metrics = {
        "what": "Observational model: log10 OBIS species richness per 2-degree cell from NCEI microplastics, "
                "OBIS sampling effort, latitude, NOAA CRW SST and DHW",
        "n_cells": int(len(df)),
        "n_microplastic_samples": int(df["n_samples"].sum()),
        "cv": "5-fold spatial block CV (10-degree blocks)",
        "r2_all_features": round(r2_full, 3),
        "mae_log10_species": round(mae_full, 3),
        "r2_without_microplastics": round(r2_noplastic, 3),
        "r2_sampling_effort_only": round(r2_effort_only, 3),
        "microplastics_r2_gain": round(r2_full - r2_noplastic, 3),
        "mean_abs_shap": shap_abs,
        "microplastics_shap_direction_corr": round(direction, 3),
        "interpretation_note": "Association only (not causation). Negative direction = higher microplastic "
                               "concentration associated with fewer recorded species after controls.",
        "sources": {
            "microplastics": "NOAA NCEI Marine Microplastics Database - Ocean Surface (Professional Science)",
            "biodiversity": "OBIS api.obis.org/v3/statistics",
            "temperature_heat_stress": "NOAA Coral Reef Watch 5km daily (12 monthly snapshots)",
        },
        "built": date.today().isoformat(),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def plot_evidence(cells: pd.DataFrame, metrics: dict, save_path) -> None:
    import matplotlib.pyplot as plt

    df = cells.dropna(subset=[TARGET])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    sc = axes[0].scatter(df["lon_center"], df["lat_center"], c=df["log10_microplastics"], s=14, cmap="viridis")
    axes[0].set_xlim(-180, 180); axes[0].set_ylim(-70, 75)
    axes[0].set_title(f"NCEI ocean-surface microplastics: {len(df)} cells, {int(df['n_samples'].sum())} samples")
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude"); axes[0].grid(alpha=0.3)
    fig.colorbar(sc, ax=axes[0], label="log10 median pieces/m³")
    shap_abs = metrics["mean_abs_shap"]
    names = sorted(shap_abs, key=shap_abs.get)
    axes[1].barh(names, [shap_abs[n] for n in names], color="#2a7f8f")
    axes[1].set_title(f"What predicts recorded species richness\n(spatial CV R² {metrics['r2_all_features']}, "
                      f"without plastic {metrics['r2_without_microplastics']})")
    axes[1].set_xlabel("mean |SHAP| (log10 species)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download all sources")
    args = ap.parse_args()
    cells = build_cells(refresh=args.refresh)
    metrics = train_and_evaluate(cells)
    plot_evidence(cells, metrics, PROJECT_ROOT / "outputs" / "figures" / "real_world_plastic_biodiversity.png")
    print(json.dumps(metrics, indent=2))
