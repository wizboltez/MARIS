"""Real oceanographic data for the survey stations.

- NOAA Coral Reef Watch (CRW) daily 5 km satellite products, via the public NOAA
  CoastWatch ERDDAP server (no account needed):
    * sea surface temperature (CoralTemp)       noaacrwsstDaily / analysed_sst
    * Degree Heating Weeks                       noaacrwdhwDaily / degree_heating_week
    * 7-day max Bleaching Alert Area             noaacrwbaa7dDaily / bleaching_alert_area
- Copernicus Marine (pH, dissolved oxygen, salinity) needs a free account. If the
  `copernicusmarine` package is installed and logged in (`copernicusmarine login`),
  fetch_copernicus() fills those values; otherwise they stay as station presets and
  are labelled as such.

Results are cached in data/external/station_ocean_data.json so the app works offline.
Refresh with:  python -m src.modules.ocean_data --refresh
"""
from __future__ import annotations

import argparse
import io
import json
import math
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd

from src.config import PROJECT_ROOT

CACHE_PATH = PROJECT_ROOT / "data" / "external" / "station_ocean_data.json"
ERDDAP = "https://coastwatch.noaa.gov/erddap/griddap"
SEARCH_RADIUS_DEG = 0.25  # coastal stations can sit on a land pixel; take the nearest sea pixel

# CRW Bleaching Alert Area levels (v3.1)
BAA_LEVELS = {
    0: "No Stress",
    1: "Bleaching Watch",
    2: "Bleaching Warning",
    3: "Alert Level 1",
    4: "Alert Level 2",
    5: "Alert Level 3",
    6: "Alert Level 4",
    7: "Alert Level 5",
}


def _erddap_nearest(dataset: str, variable: str, lat: float, lon: float, timeout: int = 90) -> dict | None:
    """Latest value of `variable` at the sea pixel nearest to (lat, lon)."""
    r = SEARCH_RADIUS_DEG
    query = f"{variable}[(last)][({lat + r}):({lat - r})][({lon - r}):({lon + r})]"
    url = f"{ERDDAP}/{dataset}.csv?{quote(query, safe='()[]:,')}"
    with urlopen(url, timeout=timeout) as resp:
        df = pd.read_csv(io.StringIO(resp.read().decode("utf-8")), skiprows=[1])
    df = df.dropna(subset=[variable])
    if df.empty:
        return None
    df["dist"] = ((df["latitude"] - lat) ** 2 + (df["longitude"] - lon) ** 2) ** 0.5
    row = df.sort_values("dist").iloc[0]
    return {
        "value": float(row[variable]),
        "date": str(row["time"])[:10],
        "pixel_lat": float(row["latitude"]),
        "pixel_lon": float(row["longitude"]),
    }


def fetch_noaa_crw(lat: float, lon: float) -> dict:
    sst = _erddap_nearest("noaacrwsstDaily", "analysed_sst", lat, lon)
    dhw = _erddap_nearest("noaacrwdhwDaily", "degree_heating_week", lat, lon)
    baa = _erddap_nearest("noaacrwbaa7dDaily", "bleaching_alert_area", lat, lon)
    out = {"source": "NOAA Coral Reef Watch 5km daily (CoastWatch ERDDAP)"}
    if sst:
        out.update(sea_surface_temperature_c=round(sst["value"], 2), date=sst["date"],
                   pixel_lat=sst["pixel_lat"], pixel_lon=sst["pixel_lon"])
    if dhw:
        out["degree_heating_weeks"] = round(dhw["value"], 2)
    if baa:
        level = int(baa["value"])
        out["bleaching_alert_level_code"] = level
        out["bleaching_alert_level"] = BAA_LEVELS.get(level, f"Level {level}")
    return out


def fetch_copernicus(lat: float, lon: float) -> dict | None:
    """pH, dissolved oxygen (mg/L) and salinity from Copernicus Marine global models.

    Requires `pip install copernicusmarine` and `copernicusmarine login`. Returns None
    when the toolbox or credentials are unavailable.
    """
    try:
        import copernicusmarine as cm
    except ImportError:
        return None
    r = 0.25
    box = dict(minimum_longitude=lon - r, maximum_longitude=lon + r, minimum_latitude=lat - r,
               maximum_latitude=lat + r, minimum_depth=0, maximum_depth=1)
    try:
        bgc = cm.open_dataset(dataset_id="cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m", variables=["ph"], **box)
        oxy = cm.open_dataset(dataset_id="cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m", variables=["o2"], **box)
        phy = cm.open_dataset(dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m", variables=["so"], **box)
    except Exception as exc:  # not logged in, dataset renamed, network...
        print(f"      Copernicus fetch failed: {exc}")
        return None

    def latest_mean(ds, var):
        v = ds[var].isel(time=-1).mean(skipna=True).item()
        return None if v is None or math.isnan(v) else float(v)

    o2_mmol_m3 = latest_mean(oxy, "o2")
    return {
        "source": "Copernicus Marine global analysis/forecast (surface, 0.25 deg box)",
        "ph": latest_mean(bgc, "ph"),
        "dissolved_oxygen_mg_l": None if o2_mmol_m3 is None else round(o2_mmol_m3 * 0.032, 2),  # mmol/m3 O2 -> mg/L
        "salinity_psu": latest_mean(phy, "so"),
    }


def refresh_cache(stations: dict) -> dict:
    data = {"fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stations": {}}
    for name, st in stations.items():
        print(f"      {name}: ", end="", flush=True)
        entry = {}
        try:
            entry["noaa_crw"] = fetch_noaa_crw(st["latitude"], st["longitude"])
            print(f"SST {entry['noaa_crw'].get('sea_surface_temperature_c')} C, "
                  f"DHW {entry['noaa_crw'].get('degree_heating_weeks')}, {entry['noaa_crw'].get('bleaching_alert_level')}")
        except Exception as exc:
            print(f"NOAA fetch failed ({exc})")
        cop = fetch_copernicus(st["latitude"], st["longitude"])
        if cop:
            entry["copernicus"] = cop
        data["stations"][name] = entry
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2))
    return data


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {"stations": {}}


def station_conditions(name: str, preset: dict, cache: dict | None = None) -> dict:
    """Merge real data (where available) over the station preset.

    Returns the five water-quality parameters plus a `provenance` dict saying, for
    each one, whether it is measured (NOAA / Copernicus) or a preset value.
    """
    cache = cache if cache is not None else load_cache()
    real = cache.get("stations", {}).get(name, {})
    noaa, cop = real.get("noaa_crw", {}), real.get("copernicus", {})

    out = {k: preset[k] for k in ("temperature_c", "ph", "dissolved_oxygen_mg_l", "salinity_psu", "turbidity_ntu")}
    prov = {k: "preset" for k in out}
    if noaa.get("sea_surface_temperature_c") is not None:
        out["temperature_c"] = noaa["sea_surface_temperature_c"]
        prov["temperature_c"] = f"NOAA CRW satellite SST ({noaa.get('date')})"
    for k in ("ph", "dissolved_oxygen_mg_l", "salinity_psu"):
        if cop.get(k) is not None:
            out[k] = round(cop[k], 2)
            prov[k] = "Copernicus Marine model"
    out["provenance"] = prov
    out["noaa_dhw"] = noaa.get("degree_heating_weeks")
    out["noaa_bleaching_alert"] = noaa.get("bleaching_alert_level")
    out["noaa_date"] = noaa.get("date")
    return out


if __name__ == "__main__":
    from src.modules.water_quality import PRESET_STATIONS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="fetch fresh data for all stations")
    args = ap.parse_args()
    if args.refresh or not CACHE_PATH.exists():
        refresh_cache(PRESET_STATIONS)
    print(json.dumps(load_cache(), indent=2))
