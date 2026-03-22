# utils/agrotech_data_connector.py
"""
Real-World Data Connector — GAGS Agrotech Equity Module v3.2

Sources (all free, no API key required unless noted)
─────────────────────────────────────────────────────
  1. FAOSTAT  — crop yield, production, harvested area for Nigeria
               https://fenixservices.fao.org/faostat/api/v1
               No key required. CC-BY 4.0.

  2. World Bank Open Data — agricultural indicators for Nigeria
               https://api.worldbank.org/v2
               No key required. CC BY 4.0.

  3. Open-Meteo Historical Weather — rainfall, temperature, soil moisture
               https://archive-api.open-meteo.com/v1/archive
               No key required. CC BY 4.0.
               Covers Nigeria FCT (Abuja 9.05°N 7.49°E) and
               Plateau State (Jos 9.92°N 8.89°E).

  4. CSV / Excel upload — user's own farm or survey data.

Blending strategy
──────────────────
  fetch_real_world_data() returns a RealWorldBundle containing:
    • a calibrated NormStats dict (mean/std of each feature from real data)
    • a time-series DataFrame for charts
    • a coverage report of which sources succeeded
  The bundle is passed to blend_real_and_synthetic() which replaces
  synthetic feature distributions with real ones while preserving the
  sample count needed by the ML pipeline.
"""

from __future__ import annotations

import io
import time
import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Geographic constants — Nigeria locations
# ─────────────────────────────────────────────────────────────────────────────

NIGERIA_LOCATIONS: Dict[str, Dict[str, float]] = {
    "Abuja FCT":    {"lat": 9.0579, "lon": 7.4951},
    "Jos (Plateau State)": {"lat": 9.9167, "lon": 8.8906},
    "Kaduna":       {"lat": 10.5222, "lon": 7.4383},
    "Kano":         {"lat": 12.0022, "lon": 8.5920},
    "Ibadan":       {"lat": 7.3775,  "lon": 3.9470},
    "Enugu":        {"lat": 6.4584,  "lon": 7.5464},
    "Maiduguri":    {"lat": 11.8311, "lon": 13.1510},
}

# Nigeria FAOSTAT country code
NIGERIA_FAOSTAT_CODE = "159"

# World Bank Nigeria country code
NIGERIA_WB_CODE = "NG"

# ─────────────────────────────────────────────────────────────────────────────
# World Bank indicator codes → agrotech features
# ─────────────────────────────────────────────────────────────────────────────

WB_INDICATORS: Dict[str, str] = {
    "AG.CON.FERT.ZS":   "fertilizer_kg_per_ha",       # fertilizer consumption
    "AG.LND.ARBL.ZS":   "arable_land_pct",            # arable land % of total
    "AG.LND.IRIG.AG.ZS":"irrigation_pct_cropland",    # irrigated land %
    "SL.AGR.EMPL.ZS":   "agr_employment_pct",         # agricultural employment
    "AG.PRD.FOOD.XD":   "food_production_index",      # food production index
    "SP.RUR.TOTL.ZS":   "rural_population_pct",       # rural population %
    "AG.LND.TRAC.ZS":   "tractors_per_100sqkm",       # tractors per 100 km²
}

# ─────────────────────────────────────────────────────────────────────────────
# FAOSTAT crop codes for Nigeria priority crops
# ─────────────────────────────────────────────────────────────────────────────

FAO_CROPS: Dict[str, str] = {
    "Maize":         "56",
    "Sorghum":       "83",
    "Cassava":       "125",
    "Yams":          "137",
    "Cowpeas":       "176",
    "Groundnuts":    "242",
    "Rice, paddy":   "27",
    "Millet":        "79",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RealWorldBundle:
    """Container for all real-world data fetched for one simulation run."""

    # Normalisation statistics derived from real data
    # Format: {feature_name: {"mean": float, "std": float, "min": float, "max": float}}
    norm_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Time series DataFrames for charting
    fao_yield_df:   Optional[pd.DataFrame] = None   # year, crop, yield_hg_ha
    wb_indicators_df: Optional[pd.DataFrame] = None # year, indicator, value
    weather_df:     Optional[pd.DataFrame] = None   # date, precip_mm, temp_max, etc.

    # Per-source success/error report
    coverage: Dict[str, str] = field(default_factory=dict)

    # Raw upload (user CSV/Excel)
    uploaded_df: Optional[pd.DataFrame] = None

    # Metadata
    location:   str  = "Abuja FCT"
    year_start: int  = 2015
    year_end:   int  = 2023


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — FAOSTAT
# ─────────────────────────────────────────────────────────────────────────────

_FAOSTAT_BASE = "https://fenixservices.fao.org/faostat/api/v1/en/data"

def fetch_fao_crop_yields(
    year_start: int = 2015,
    year_end:   int = 2023,
    crops: Optional[List[str]] = None,
    timeout: int = 20,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fetch crop yield data for Nigeria from FAOSTAT.

    Returns (DataFrame, status_message).
    DataFrame columns: year, crop, area_harvested_ha, production_tonnes, yield_hg_ha
    """
    selected_crops = crops or list(FAO_CROPS.keys())
    crop_codes     = ",".join(FAO_CROPS[c] for c in selected_crops if c in FAO_CROPS)

    params = {
        "area":      NIGERIA_FAOSTAT_CODE,
        "element":   "5419,5510,5312",   # yield (hg/ha), production (t), area (ha)
        "item":      crop_codes,
        "year":      ",".join(str(y) for y in range(year_start, year_end + 1)),
        "output_type": "csv",
    }

    try:
        resp = requests.get(
            _FAOSTAT_BASE + "/QCL",  # Crops and livestock products domain
            params=params, timeout=timeout
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))

        if df.empty:
            return None, "FAOSTAT returned empty dataset"

        # Normalise columns
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={
            "year":  "year",
            "item":  "crop",
            "element": "measure",
            "value": "value",
        })

        # Pivot: one row per (year, crop), columns per measure
        if "measure" in df.columns:
            pivot = df.pivot_table(
                index=["year", "crop"], columns="measure",
                values="value", aggfunc="first"
            ).reset_index()
            pivot.columns.name = None
            col_map = {
                "Yield": "yield_hg_ha",
                "Production": "production_tonnes",
                "Area harvested": "area_harvested_ha",
            }
            pivot = pivot.rename(columns=col_map)
        else:
            pivot = df

        return pivot, "OK"

    except requests.exceptions.ConnectionError:
        return None, "Connection error — check internet access"
    except requests.exceptions.Timeout:
        return None, "Timeout — FAOSTAT may be temporarily unavailable"
    except Exception as exc:
        logger.warning(f"FAOSTAT fetch error: {exc}")
        return None, f"Error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — World Bank
# ─────────────────────────────────────────────────────────────────────────────

_WB_BASE = "https://api.worldbank.org/v2"

def fetch_worldbank_indicators(
    year_start: int = 2010,
    year_end:   int = 2023,
    timeout: int = 20,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fetch agricultural indicators for Nigeria from World Bank Open Data API.

    Returns (DataFrame, status_message).
    DataFrame columns: year, indicator_code, indicator_name, value
    """
    rows = []
    errors = []

    for code, friendly_name in WB_INDICATORS.items():
        url = (f"{_WB_BASE}/country/{NIGERIA_WB_CODE}/indicator/{code}"
               f"?format=json&date={year_start}:{year_end}&per_page=100")
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            if len(data) < 2 or not data[1]:
                continue

            for entry in data[1]:
                if entry.get("value") is not None:
                    rows.append({
                        "year":            int(entry["date"]),
                        "indicator_code":  code,
                        "indicator_name":  friendly_name,
                        "value":           float(entry["value"]),
                    })
        except requests.exceptions.ConnectionError:
            errors.append(f"{code}: connection error")
            continue
        except Exception as exc:
            errors.append(f"{code}: {exc}")
            continue

        time.sleep(0.15)   # be polite to the API

    if not rows:
        return None, f"World Bank: no data retrieved. Errors: {'; '.join(errors[:3])}"

    df = pd.DataFrame(rows)
    status = "OK" if not errors else f"OK (partial: {len(errors)} indicators failed)"
    return df, status


def worldbank_norm_stats(wb_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Derive normalisation statistics from World Bank data for use in
    seeding synthetic feature distributions.
    """
    stats: Dict[str, Dict[str, float]] = {}
    pivot = wb_df.pivot_table(index="year", columns="indicator_name",
                              values="value", aggfunc="mean")

    mapping = {
        "fertilizer_kg_per_ha":    "fertilizer_access",
        "irrigation_pct_cropland": "irrigation_access",
        "arable_land_pct":         "soil_quality",        # proxy
        "rural_population_pct":    "market_distance_km",  # inverse proxy
    }

    for wb_col, feat_name in mapping.items():
        if wb_col in pivot.columns:
            col = pivot[wb_col].dropna()
            if len(col) > 0:
                v = col.values.astype(float)
                # Normalise to [0,1] where sensible
                if wb_col == "fertilizer_kg_per_ha":
                    # Nigeria: 4–16 kg/ha historically
                    normed = np.clip(v / 50.0, 0, 1)
                elif wb_col in ("irrigation_pct_cropland", "arable_land_pct",
                                "rural_population_pct"):
                    normed = np.clip(v / 100.0, 0, 1)
                else:
                    normed = np.clip((v - v.min()) / (v.ptp() + 1e-9), 0, 1)

                stats[feat_name] = {
                    "mean": float(normed.mean()),
                    "std":  float(normed.std()),
                    "min":  float(normed.min()),
                    "max":  float(normed.max()),
                    "source": "World Bank",
                }

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — Open-Meteo Historical Weather
# ─────────────────────────────────────────────────────────────────────────────

_OPENMETEO_BASE = "https://archive-api.open-meteo.com/v1/archive"

_WEATHER_VARIABLES = [
    "precipitation_sum",           # mm/day
    "temperature_2m_max",          # °C
    "temperature_2m_min",          # °C
    "soil_moisture_0_to_7cm_mean", # m³/m³
    "et0_fao_evapotranspiration",  # mm/day — crop reference ET
]

def fetch_weather_history(
    location_name: str = "Abuja FCT",
    year_start: int = 2018,
    year_end:   int = 2023,
    timeout: int = 30,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Fetch daily historical weather for a Nigerian location via Open-Meteo.

    Returns (DataFrame, status_message).
    Columns: date, precipitation_sum, temperature_2m_max, temperature_2m_min,
             soil_moisture_0_to_7cm_mean, et0_fao_evapotranspiration
    """
    coords = NIGERIA_LOCATIONS.get(location_name, NIGERIA_LOCATIONS["Abuja FCT"])

    params = {
        "latitude":  coords["lat"],
        "longitude": coords["lon"],
        "start_date": f"{year_start}-01-01",
        "end_date":   f"{year_end}-12-31",
        "daily":     ",".join(_WEATHER_VARIABLES),
        "timezone":  "Africa/Lagos",
    }

    try:
        resp = requests.get(_OPENMETEO_BASE, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        if "daily" not in data:
            return None, f"Open-Meteo: unexpected response — {data.get('reason','')}"

        df = pd.DataFrame(data["daily"])
        df["date"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"], errors="ignore")
        df = df.rename(columns={v: v for v in _WEATHER_VARIABLES})

        return df, "OK"

    except requests.exceptions.ConnectionError:
        return None, "Connection error — check internet access"
    except requests.exceptions.Timeout:
        return None, "Timeout — Open-Meteo unavailable"
    except Exception as exc:
        logger.warning(f"Open-Meteo fetch error: {exc}")
        return None, f"Error: {exc}"


def weather_norm_stats(weather_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Derive normalisation statistics from historical weather data.
    Maps weather variables to agrotech simulation features.
    """
    stats: Dict[str, Dict[str, float]] = {}

    # Rainfall index: annual sum normalised to [0,1]
    if "precipitation_sum" in weather_df.columns:
        weather_df["year"] = weather_df["date"].dt.year
        annual = weather_df.groupby("year")["precipitation_sum"].sum()
        # Nigeria annual rainfall: 400–2500 mm depending on region
        normed = np.clip(annual.values / 2500.0, 0, 1)
        stats["rainfall_index"] = {
            "mean":   float(normed.mean()),
            "std":    float(normed.std()),
            "min":    float(normed.min()),
            "max":    float(normed.max()),
            "source": "Open-Meteo (ERA5)",
        }

    # Climate stress: high temp + low rainfall → stress
    if ("temperature_2m_max" in weather_df.columns
            and "precipitation_sum" in weather_df.columns):
        temp_norm  = np.clip(weather_df["temperature_2m_max"] / 45.0, 0, 1)
        rain_norm  = np.clip(weather_df["precipitation_sum"]  / 30.0, 0, 1)
        stress_idx = (temp_norm * (1 - rain_norm)).values
        stats["climate_stress"] = {
            "mean":   float(stress_idx.mean()),
            "std":    float(stress_idx.std()),
            "min":    float(stress_idx.min()),
            "max":    float(stress_idx.max()),
            "source": "Open-Meteo (derived)",
        }

    # Soil moisture → irrigation need proxy
    if "soil_moisture_0_to_7cm_mean" in weather_df.columns:
        sm = weather_df["soil_moisture_0_to_7cm_mean"].dropna().values
        # ERA5 soil moisture range: 0.01–0.57 m³/m³ typically
        normed = np.clip(sm / 0.57, 0, 1)
        stats["irrigation_access"] = {
            "mean":   float(normed.mean()),
            "std":    float(normed.std()),
            "min":    float(normed.min()),
            "max":    float(normed.max()),
            "source": "Open-Meteo soil moisture",
        }

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Source 4 — User CSV / Excel upload
# ─────────────────────────────────────────────────────────────────────────────

# Expected column aliases: map possible user column names → our feature names
_UPLOAD_ALIASES: Dict[str, str] = {
    "age":               "age",
    "farmer_age":        "age",
    "income":            "income_level",
    "income_level":      "income_level",
    "monthly_income":    "income_level",
    "connectivity":      "connectivity",
    "internet_access":   "connectivity",
    "land_size":         "land_size_ha",
    "land_area":         "land_size_ha",
    "farm_size":         "land_size_ha",
    "hectares":          "land_size_ha",
    "soil_quality":      "soil_quality",
    "soil_type":         "soil_quality",
    "rainfall":          "rainfall_index",
    "rainfall_index":    "rainfall_index",
    "rain":              "rainfall_index",
    "market_distance":   "market_distance_km",
    "distance_market":   "market_distance_km",
    "km_to_market":      "market_distance_km",
    "fertilizer":        "fertilizer_access",
    "fertilizer_access": "fertilizer_access",
    "fertilizer_use":    "fertilizer_access",
    "irrigation":        "irrigation_access",
    "irrigation_access": "irrigation_access",
    "yield":             "prior_yield_kg",
    "prior_yield":       "prior_yield_kg",
    "crop_yield":        "prior_yield_kg",
    "climate_stress":    "climate_stress",
    "climate_risk":      "climate_stress",
    "gender":            "gender",
    "sex":               "gender",
}

_TARGET_ALIASES: Dict[str, str] = {
    "failed":           "crop_failure",
    "crop_failure":     "crop_failure",
    "failure":          "crop_failure",
    "outcome":          "outcome",
    "target":           "target",
    "label":            "label",
    "market_access":    "market_access",
    "fertilizer_need":  "fertilizer_need",
    "high_risk":        "high_risk",
}

def parse_uploaded_file(
    file_obj,
    file_name: str,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Parse a user-uploaded CSV or Excel file.

    Columns are matched to our feature names via _UPLOAD_ALIASES.
    Returns (normalised_DataFrame, status_message).
    """
    try:
        if file_name.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(file_obj)
        else:
            df_raw = pd.read_csv(file_obj)
    except Exception as exc:
        return None, f"Could not read file: {exc}"

    if df_raw.empty:
        return None, "Uploaded file is empty"

    # Normalise column names
    df_raw.columns = [str(c).lower().strip().replace(" ", "_") for c in df_raw.columns]

    # Map to our feature names
    rename_map = {col: _UPLOAD_ALIASES[col]
                  for col in df_raw.columns if col in _UPLOAD_ALIASES}
    df = df_raw.rename(columns=rename_map)

    # Keep only columns we recognise
    known_features = list(set(_UPLOAD_ALIASES.values()) | set(_TARGET_ALIASES.values()))
    kept = [c for c in df.columns if c in known_features]

    if len(kept) < 3:
        return None, (
            f"Only {len(kept)} recognisable columns found. "
            f"Expected at least 3 of: {', '.join(sorted(set(_UPLOAD_ALIASES.values())))}"
        )

    df = df[kept].copy()

    # Convert numeric columns
    for col in df.columns:
        if col not in ("gender", "sex"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Encode gender if present
    if "gender" in df.columns:
        df["gender_encoded"] = df["gender"].map(
            {"female": 0, "male": 1, "f": 0, "m": 1, "0": 0, "1": 1}
        ).fillna(0.5)

    df = df.dropna(subset=[c for c in kept if c not in ("gender",)])

    if df.empty:
        return None, "All rows were dropped after removing NaN values"

    return df, f"OK — {len(df):,} rows, {len(kept)} columns recognised"


def upload_norm_stats(upload_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute normalisation stats from user upload."""
    stats: Dict[str, Dict[str, float]] = {}
    numeric_features = [
        "age", "income_level", "connectivity", "land_size_ha",
        "soil_quality", "rainfall_index", "market_distance_km",
        "fertilizer_access", "irrigation_access", "prior_yield_kg",
        "climate_stress",
    ]
    for feat in numeric_features:
        if feat in upload_df.columns:
            v = upload_df[feat].dropna().values.astype(float)
            if len(v) > 5:
                stats[feat] = {
                    "mean":   float(v.mean()),
                    "std":    float(v.std()),
                    "min":    float(v.min()),
                    "max":    float(v.max()),
                    "source": "User upload",
                }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def fetch_real_world_data(
    location_name:  str  = "Abuja FCT",
    year_start:     int  = 2015,
    year_end:       int  = 2023,
    use_fao:        bool = True,
    use_worldbank:  bool = True,
    use_weather:    bool = True,
    selected_crops: Optional[List[str]] = None,
    upload_file=None,
    upload_name:    str  = "",
    timeout:        int  = 25,
) -> RealWorldBundle:
    """
    Fetch all configured real-world data sources and return a RealWorldBundle.

    Parameters
    ----------
    location_name   : one of NIGERIA_LOCATIONS keys
    year_start/end  : date range for historical data
    use_fao         : fetch FAOSTAT crop yields
    use_worldbank   : fetch World Bank agricultural indicators
    use_weather     : fetch Open-Meteo historical weather
    selected_crops  : FAOSTAT crop list (None = all priority crops)
    upload_file     : Streamlit UploadedFile object or None
    upload_name     : filename string (needed to detect .xlsx)
    timeout         : HTTP request timeout in seconds
    """
    bundle = RealWorldBundle(
        location=location_name,
        year_start=year_start,
        year_end=year_end,
    )
    combined_stats: Dict[str, Dict[str, float]] = {}

    # ── FAO ──────────────────────────────────────────────────────────────────
    if use_fao:
        fao_df, fao_status = fetch_fao_crop_yields(
            year_start, year_end, selected_crops, timeout
        )
        bundle.fao_yield_df = fao_df
        bundle.coverage["FAOSTAT"] = fao_status
        if fao_df is not None and "yield_hg_ha" in fao_df.columns:
            # Real Nigeria yield range: 1000–30000 hg/ha depending on crop
            yields = fao_df["yield_hg_ha"].dropna().values.astype(float)
            if len(yields) > 0:
                normed = np.clip(yields / 30000.0, 0, 1)
                combined_stats["prior_yield_kg"] = {
                    "mean":   float(normed.mean()),
                    "std":    float(normed.std()),
                    "min":    float(normed.min()),
                    "max":    float(normed.max()),
                    "source": "FAOSTAT",
                }

    # ── World Bank ────────────────────────────────────────────────────────────
    if use_worldbank:
        wb_df, wb_status = fetch_worldbank_indicators(year_start, year_end, timeout)
        bundle.wb_indicators_df = wb_df
        bundle.coverage["World Bank"] = wb_status
        if wb_df is not None:
            wb_stats = worldbank_norm_stats(wb_df)
            combined_stats.update(wb_stats)

    # ── Open-Meteo ────────────────────────────────────────────────────────────
    if use_weather:
        weather_df, wx_status = fetch_weather_history(
            location_name, year_start, year_end, timeout
        )
        bundle.weather_df = weather_df
        bundle.coverage["Open-Meteo Weather"] = wx_status
        if weather_df is not None:
            wx_stats = weather_norm_stats(weather_df)
            combined_stats.update(wx_stats)

    # ── User upload ───────────────────────────────────────────────────────────
    if upload_file is not None:
        upload_df, up_status = parse_uploaded_file(upload_file, upload_name)
        bundle.uploaded_df = upload_df
        bundle.coverage["User upload"] = up_status
        if upload_df is not None:
            up_stats = upload_norm_stats(upload_df)
            # Upload stats take priority — they are the most specific data
            combined_stats.update(up_stats)

    bundle.norm_stats = combined_stats
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Blending: replace synthetic distributions with real-world calibrated ones
# ─────────────────────────────────────────────────────────────────────────────

def blend_real_and_synthetic(
    X_synthetic: np.ndarray,
    feat_names:  List[str],
    bundle:      RealWorldBundle,
    blend_weight: float = 0.7,
    random_state: int   = 42,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Replace synthetic feature columns with real-world calibrated distributions.

    For each feature that has real-world stats in bundle.norm_stats:
      new_col = blend_weight × real_sample + (1−blend_weight) × synthetic_col

    If the user uploaded their own raw data, use that directly for matched
    columns instead of sampling from the distribution.

    Returns (X_blended, blend_report)
    """
    rng          = np.random.default_rng(random_state)
    X            = X_synthetic.copy().astype(np.float64)
    n_samples    = X.shape[0]
    blend_report: Dict[str, str] = {}
    norm_stats   = bundle.norm_stats

    # If user uploaded raw data with enough rows, use it directly
    upload_df = bundle.uploaded_df
    if upload_df is not None and len(upload_df) >= 50:
        for fi, feat in enumerate(feat_names):
            if feat in upload_df.columns and fi < X.shape[1]:
                raw_vals = upload_df[feat].dropna().values.astype(float)
                sampled  = rng.choice(raw_vals, size=n_samples, replace=True)
                X[:, fi] = (blend_weight * sampled
                             + (1 - blend_weight) * X[:, fi])
                blend_report[feat] = f"blended {blend_weight:.0%} from user upload"
        return X, blend_report

    # Otherwise use the derived norm_stats to re-sample each feature
    for fi, feat in enumerate(feat_names):
        if feat not in norm_stats or fi >= X.shape[1]:
            continue

        ns     = norm_stats[feat]
        mean   = ns["mean"]
        std    = max(ns["std"], 0.01)
        fmin   = ns.get("min", 0.0)
        fmax   = ns.get("max", 1.0)
        source = ns.get("source", "real data")

        # Sample from a truncated-normal approximating the real distribution
        raw_sample = rng.normal(mean, std, size=n_samples)
        raw_sample = np.clip(raw_sample, fmin, fmax)

        # Blend with the existing synthetic column
        X[:, fi] = (blend_weight * raw_sample
                    + (1 - blend_weight) * X[:, fi])
        blend_report[feat] = f"blended {blend_weight:.0%} from {source}"

    return X, blend_report


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers for UI display
# ─────────────────────────────────────────────────────────────────────────────

def bundle_summary(bundle: RealWorldBundle) -> Dict[str, Any]:
    """Return a plain dict suitable for display in Streamlit metrics."""
    n_fao    = len(bundle.fao_yield_df)    if bundle.fao_yield_df    is not None else 0
    n_wb     = len(bundle.wb_indicators_df) if bundle.wb_indicators_df is not None else 0
    n_wx     = len(bundle.weather_df)       if bundle.weather_df       is not None else 0
    n_upload = len(bundle.uploaded_df)     if bundle.uploaded_df      is not None else 0

    return {
        "fao_rows":        n_fao,
        "worldbank_rows":  n_wb,
        "weather_rows":    n_wx,
        "upload_rows":     n_upload,
        "calibrated_features": len(bundle.norm_stats),
        "coverage":        bundle.coverage,
        "location":        bundle.location,
        "year_range":      f"{bundle.year_start}–{bundle.year_end}",
    }


def validate_upload_template() -> pd.DataFrame:
    """Return a template DataFrame showing expected column names for uploads."""
    return pd.DataFrame({
        "age":              [32, 45, 28],
        "income_level":     [45000, 78000, 32000],
        "connectivity":     [0.45, 0.82, 0.21],
        "land_size_ha":     [1.2, 3.4, 0.8],
        "soil_quality":     [0.65, 0.72, 0.48],
        "rainfall_index":   [0.71, 0.55, 0.83],
        "market_distance_km": [12, 45, 8],
        "fertilizer_access":  [0, 1, 0],
        "irrigation_access":  [0, 0, 1],
        "prior_yield_kg":   [1200, 2400, 800],
        "climate_stress":   [0.42, 0.61, 0.35],
        "gender":           ["female", "male", "female"],
    })