# pages/1_🏥_Healthcare_Equity.py
"""
Healthcare Equity Simulation — GAGS Framework v1.0 | Article 1 Research v8

Refactored to integrate all five feature modules from simulation_core.py:
  Feature 1 — AI Agent Economy Sandbox  (resource auction in healthcare domain)
  Feature 2 — Multimodal Red Teaming    (text/image/deepfake attack surface)
  Feature 3 — Africa-Centric / Gender   (Abuja presets + UNESCO equity audit)
  Feature 4 — Hybrid Governance Layer   (citizen vote + blockchain ledger)
  Feature 5 — Strategic Social Arena    (negotiation / coalition game)
"""

import json
from datetime import datetime, date
import warnings
import time
import re
import hashlib
import uuid
from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from openai import OpenAI
import streamlit as st

# ── Page Config (Must be initial Streamlit command) ───────────────────────────
st.set_page_config(page_title="Healthcare Equity • Platform", layout="wide", page_icon="🏥")

from fpdf import FPDF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset

from components.translate import install_auto_translate, tx, tx_plotly, language_switcher

install_auto_translate()
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="specific_module"
)

# ── GAGS Core Components ───────────────────────────────────────────────────────
try:
    from components.pdf_report import generate_pdf_compliance_report

    PDF_OK = True
except (ImportError, ModuleNotFoundError):
    PDF_OK = False


    def generate_pdf_compliance_report(*a, **kw):
        return None

from components.governance_logic import (
    generate_synthetic_data,
    generate_africa_centric_data,
    AFRICA_SCENARIO_PRESETS,
    apply_bias,
    simulate_data_poisoning,
    calculate_fairness_metrics,
    run_gender_equity_audit,
    run_simple_simulation,
    simulate_longitudinal_bias,
    simulate_federated_learning,
)
from utils.config import simulation_config, settings
from components.i18n import t, get_lang, language_badge
from components.ussd_simulator import ussd_interface, accessibility_gap_report, format_sms_result
from components.nigeria_regulatory import nigeria_compliance_panel
from components.ux_utils import (
    guided_tour_banner, preset_selector,
    metric_glossary_expander, history_browser, save_to_history,
    share_url_panel, load_config_from_url, apply_url_config,
    annotation_panel,
    role_switcher, get_active_role, role_banner,
    get_role_algo, get_role_tabs, get_role_defaults,
    role_algo_banner, role_brief_banner,
    board_member_summary, ROLE_TAB_VISIBILITY,
)
from components.live_data import national_live_banner
from components.gags_gender_ui import render_gender_equity_audit
from components.nigeria_states import state_selector, state_info_card, get_state_params, apply_state_to_preset
from components.gags_interactive import (
    progress_tracker, scenario_story_banner,
    domain_challenge_panel, benchmark_challenge_panel,
    what_if_explorer, bias_detective_panel, track_run, award_points,
    _reset_render_guards
)
from components.healthcare_real_models import (
    run_real_model_comparison,
    render_real_model_tab,
    real_model_sidebar_controls,
)

_reset_render_guards()

from components.ai_safety import run_ai_safety_suite
from components.gags_lifecycle import run_lifecycle_suite
from components.gags_lifecycle_ui import render_lifecycle_tab, render_eco_tab
from components.gags_dynamic_systems import run_dynamic_systems_suite, derive_ds_params
from components.gags_dynamic_ui import render_dynamic_systems_tab
from components.gags_safety_ui import render_safety_tab
from components.gags_features_full import (
    run_agent_economy_simulation, run_redteam_simulation, run_arena_simulation,
)
from components.gags_feature_modules import (
    feature_modules_tab, multi_challenge_panel,
    admin_challenge_panel, feature_module_sidebar
)
from components.health.clinical_impact import (
    compute_clinical_impact_matrix,
    apply_sample_reweighing,
    render_national_compliance_panel,
)
from components.health.equity_audit_engine import (
    build_intersectional_features,
    calculate_facility_access,
    audit_ai_recommendation,
    generate_pdf_report,
    generate_llm_vignette,
    generate_clinical_recommendation,
    get_osrm_route_distance_and_time,
    run_multi_agent_triage_pipeline,
render_mitigation_workbench,
render_medical_compliance_panel,
)
from components.healthcare_sim_runner import (
    run_simulation_step,
    compute_live_metrics,
)
from components.ussd_triage_engine import render_ussd_triage_section

# ── Advanced chart & benchmark libraries ──────────────────────────────────────
try:
    from components.gags_charts import (
        waterfall_feature_contributions, benchmark_comparison_bar,
        lollipop_gap_chart, radar_with_benchmark, fairness_heatmap,
        multi_run_distribution, animated_bias_drift, gauge_cluster,
        ai_bias_incident_timeline, make_economic_sankey,
    )

    CHARTS_OK = True
except ImportError:
    CHARTS_OK = False

try:
    from components.gags_benchmarks import (
        REAL_WORLD_BENCHMARKS, get_benchmarks_for_domain, compare_to_benchmark,
    )

    BENCHMARKS_OK = True
except ImportError:
    BENCHMARKS_OK = False
    REAL_WORLD_BENCHMARKS = {}

# ── Preset Data Constants ──────────────────────────────────────────────────────
CLINICAL_VIGNETTES_PRESETS = {
    "🧠 Acute Ischemic Stroke": {
        "text": "45yo male presenting with sudden onset right-sided hemiparesis, facial drooping, and severe dysarthria. Onset 45 minutes prior to arrival. BP: 175/100 mmHg.",
        "category": "Neurology / Emergency",
        "acuity": "CRITICAL",
        "time_window": "< 4.5 Hours",
        "required_tech": ["CT/MRI", "tPA Thrombolysis"]
    },
    "🩸 Severe Postpartum Hemorrhage": {
        "text": "28yo female, P3L3, delivered 2 hours ago at home, presenting with profuse vaginal bleeding, tachycardia (HR 130 bpm), and confusion. Pale conjunctivae.",
        "category": "Obstetrics / Hemorrhage",
        "acuity": "CRITICAL",
        "time_window": "Immediate (< 30 min)",
        "required_tech": ["Blood Transfusion", "Emergency OR"]
    },
    "👶 Neonatal Sepsis & Respiratory Distress": {
        "text": "4-day-old neonate presenting with severe chest indrawing, grunting, lethargy, and inability to feed. Temperature: 38.8°C.",
        "category": "Pediatric Emergency",
        "acuity": "HIGH",
        "time_window": "< 2 Hours",
        "required_tech": ["Oxygen Therapy", "NICU/Specialized Nursery"]
    },
    "🦟 Severe Falciparum Malaria": {
        "text": "12yo female presenting with high fever, jaundice, recurrent convulsions, and hemoglobin level of 4.2 g/dL.",
        "category": "Infectious Disease",
        "acuity": "HIGH",
        "time_window": "< 4 Hours",
        "required_tech": ["IV Artesunate", "Blood Transfusion"]
    }
}

CLINICAL_SPECIALTIES = [
    "🧠 Neurology & Stroke Care",
    "🤰 Obstetrics & High-Risk Maternal Care",
    "👶 Pediatrics & Neonatal Intensive Care",
    "🫀 Cardiovascular Medicine & Acute Care",
    "🦟 Infectious Diseases (Malaria, TB, HIV, Viral Fevers)",
    "🩻 Orthopedic Trauma & Acute Fractures",
    "🩸 Hematology & Transfusion Medicine",
    "🫁 Pulmonology & Acute Respiratory Care",
    "🔪 General & Emergency Surgery",
    "🧪 Nephrology & Acute Renal Failure",
    "🧠 Psychiatry & Behavioral Health Emergencies",
    "🩺 General Primary Care & Community Health"
]

CLINICAL_ENVIRONMENTS = [
    "🪵 Rural Primary Health Post (No Onsite Lab/Imaging)",
    "🩺 Community Health Center (Basic Rapid Tests, Basic Ward)",
    "🏥 District General Hospital (Secondary Care, Basic Surgical/Lab)",
    "🏛️ University Teaching Hospital (Tertiary, Full ICU/Advanced Imaging)",
    "🚑 Mobile Field Unit / Humanitarian Emergency Post",
    "🏘️ Peri-Urban Health Post (High Volume, Low Supplies)",
    "⛺ Remote Outpost Clinic (Off-Grid, Limited Electricity/Cold Chain)"
]

COMPLEXITY_TIERS = [
    "🚨 Critical Emergency (Immediate Life-Threat)",
    "⚠️ High Acuity (Requires Urgent Transfer/Intervention)",
    "🟡 Moderate / Subacute Presentation",
    "🟢 Diagnostic Dilemma / Non-Standard Symptoms"
]

# ── Styling & Page Config ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .med-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .dark .med-card {
        background-color: #0F172A;
        border-color: #1E293B;
    }
    .status-badge-active {
        background-color: #E0F2FE; color: #0369A1;
        font-weight: 600; padding: 2px 8px; border-radius: 4px; font-size: 0.82rem;
    }
    .status-badge-warning {
        background-color: #FEF3C7; color: #92400E;
        font-weight: 600; padding: 2px 8px; border-radius: 4px; font-size: 0.82rem;
    }
    .status-badge-success {
        background-color: #DCFCE7; color: #15803D;
        font-weight: 600; padding: 2px 8px; border-radius: 4px; font-size: 0.82rem;
    }
    .sidebar-section-hdr {
        font-size: 0.85rem; font-weight: 700; text-transform: uppercase;
        color: #64748B; letter-spacing: 0.05em; margin-top: 15px; margin-bottom: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ── Helper Data Loaders ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Processing Facility Data...")
def load_local_facilities(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        df = pd.read_excel(uploaded_file)
        df = df.loc[:, ~df.columns.duplicated()]
        lat_candidates = ['latitude', 'lat', 'y', 'y_coord']
        lon_candidates = ['longitude', 'lon', 'lng', 'long', 'x', 'x_coord']
        rename_dict = {}
        for col in df.columns:
            clean_col = str(col).lower().strip()
            if clean_col in lat_candidates and 'lat' not in rename_dict.values():
                rename_dict[col] = 'lat'
            elif clean_col in lon_candidates and 'lon' not in rename_dict.values():
                rename_dict[col] = 'lon'
        df = df.rename(columns=rename_dict)
        if 'lat' in df.columns and 'lon' in df.columns:
            if isinstance(df['lat'], pd.DataFrame):
                df['lat'] = df['lat'].iloc[:, 0]
            if isinstance(df['lon'], pd.DataFrame):
                df['lon'] = df['lon'].iloc[:, 0]
            df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df = df.dropna(subset=['lat', 'lon'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner="Loading AfriMed-QA Dataset...")
def load_afrimed_qa() -> pd.DataFrame:
    try:
        dataset = load_dataset("afrimedqa/afrimedqa_v2", split="train")
        df = dataset.to_pandas()
        if not df.empty:
            df = df.rename(columns={"question": "clinical_prompt", "answer": "expected_diagnosis"})
        return df
    except Exception as e:
        st.error(f"Failed to load AfriMed-QA: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner="Loading HDX Health Facilities...")
def load_hdx_facilities(country_iso3=None) -> pd.DataFrame:
    try:
        api_url = "https://data.humdata.org/api/3/action/package_show?id=health-facilities-in-sub-saharan-africa"
        response = requests.get(api_url, timeout=15)
        response.raise_for_status()
        data = response.json()
        resources = data.get("result", {}).get("resources", [])
        xlsx_url = next((res["url"] for res in resources if res["format"].lower() in ["xlsx", "excel"]), None)
        if xlsx_url:
            df = pd.read_excel(xlsx_url)
            if country_iso3 and 'Country' in df.columns:
                df = df[df['Country'] == country_iso3]
            return df
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_user_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()
    return pd.DataFrame()


def detect_or_select_columns(df: pd.DataFrame, key_prefix: str):
    if df.empty:
        return {}
    cols = list(df.columns)

    def find_idx(keywords, default_idx=0):
        for i, c in enumerate(cols):
            if any(k in str(c).lower() for k in keywords):
                return i
        return default_idx if default_idx < len(cols) else 0

    st.markdown("##### 🛠️ Column Mapping Assistant")
    c1, c2, c3 = st.columns(3)
    with c1:
        lat_col = st.selectbox("Latitude Column", cols, index=find_idx(['lat', 'latitude', 'y'], 0),
                               key=f"{key_prefix}_lat")
        lon_col = st.selectbox("Longitude Column", cols,
                               index=find_idx(['lon', 'lng', 'longitude', 'x'], min(1, len(cols) - 1)),
                               key=f"{key_prefix}_lon")
    with c2:
        name_col = st.selectbox("Facility Name Column", cols,
                                index=find_idx(['name', 'facility', 'hospital', 'title'], 0), key=f"{key_prefix}_name")
        tier_col = st.selectbox("Tier / Type Column", cols,
                                index=find_idx(['type', 'tier', 'level', 'category', 'ownership'], 0),
                                key=f"{key_prefix}_tier")
    with c3:
        state_col = st.selectbox("State Column", cols, index=find_idx(['state', 'region', 'province'], 0),
                                 key=f"{key_prefix}_state")
        lga_col = st.selectbox("LGA / District Column", cols, index=find_idx(['lga', 'district', 'county', 'city'], 0),
                               key=f"{key_prefix}_lga")

    return {
        "lat": lat_col, "lon": lon_col, "name": name_col,
        "tier": tier_col, "state": state_col, "lga": lga_col
    }


# ── Reproducible Experiment Contracts ──────────────────────────────────────────
@dataclass(frozen=True)
class ExperimentConfig:
    data_source: str
    n_samples: int
    n_runs: int
    selected_biases: tuple
    bias_intensity: float
    poison_rate: float
    access_inequality: float
    low_income_ratio: float
    uninsured_ratio: float
    mitigation_strategy: str
    threshold: float
    selected_state: str
    healthcare_setting: str
    prediction_task: str
    region: str
    seed: int = 42

    def validate(self):
        if not 0 <= self.bias_intensity <= float(simulation_config.MAX_BIAS_FACTOR):
            raise ValueError("Bias intensity is outside the configured range.")
        if not 0 <= self.poison_rate <= 0.5:
            raise ValueError("Poison rate must be between 0 and 0.5.")
        if not 0 <= self.access_inequality <= 1:
            raise ValueError("Access inequality must be between 0 and 1.")
        if not 0 <= self.low_income_ratio <= 1 or not 0 <= self.uninsured_ratio <= 1:
            raise ValueError("Demographic ratios must be between 0 and 1.")
        if not 0.05 <= self.threshold <= 0.95:
            raise ValueError("Decision threshold must be between 0.05 and 0.95.")
        if self.n_samples < 500 or self.n_runs < 1:
            raise ValueError("Sample size must be >= 500 and runs must be >= 1.")


@dataclass
class ModelArtifact:
    model: object
    scaler: object
    feature_names: list
    X_train: object
    X_test: object
    y_train: object
    y_test: object
    demo_train: object
    demo_test: object
    y_probability: object
    y_prediction: object
    baseline_prediction: object
    threshold: float


@dataclass(frozen=True)
class Article1ReferralConfig:
    """Research configuration for Article 1 referral-access coupling.

    Article 1 uses a prespecified screen-to-referral simulation policy: patients
    with the benchmark-positive disease label are considered eligible for
    higher-level diagnostic assessment. This is a research policy assumption,
    not an emergency-triage rule and not a patient-specific clinical referral
    recommendation.
    """
    enabled: bool = False
    travel_threshold_min: float = 60.0
    eligible_types: tuple = ("Secondary", "Tertiary")
    functional_statuses: tuple = ("Functional",)
    routing_method: str = "geodesic_proxy"  # geodesic_proxy | osrm
    assumed_speed_kmh: float = 40.0
    origin_mode: str = "primary_facility_proxy"
    group_location_coupling: float = 0.0
    referral_policy: str = "screen_positive_to_diagnostic_assessment"

    def validate(self):
        if self.travel_threshold_min <= 0:
            raise ValueError("Article 1 travel threshold must be > 0 minutes.")
        if self.routing_method not in {"geodesic_proxy", "osrm"}:
            raise ValueError("routing_method must be 'geodesic_proxy' or 'osrm'.")
        if self.assumed_speed_kmh <= 0:
            raise ValueError("assumed_speed_kmh must be > 0.")
        if not 0 <= self.group_location_coupling <= 1:
            raise ValueError("group_location_coupling must be between 0 and 1.")
        if self.referral_policy != "screen_positive_to_diagnostic_assessment":
            raise ValueError("Article 1 currently implements only the prespecified screen-to-referral diagnostic-assessment policy.")


def _normalize_grid3_for_article1(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the nationwide GRID3 facility schema used by Article 1.

    This function does not infer service capability. Eligibility is based only on
    the uploaded facility level/type and functional-status fields.
    """
    if df is None or df.empty:
        raise ValueError("Article 1 requires a non-empty facility dataset.")
    out = df.copy()
    lookup = {str(c).strip().lower(): c for c in out.columns}

    def pick(*names):
        for name in names:
            if name.lower() in lookup:
                return lookup[name.lower()]
        return None

    lat = pick("latitude", "lat", "y")
    lon = pick("longitude", "lon", "lng", "x")
    typ = pick("type", "facility_tier", "tier", "level")
    status = pick("func_stats", "functional_status", "status")
    name = pick("prmry_name", "facility_name", "name", "alt_name")
    state = pick("statename", "state")
    lga = pick("lganame", "lga")
    uid = pick("uniq_id", "fid", "globalid")
    required = {"latitude": lat, "longitude": lon, "type": typ, "functional_status": status}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Facility dataset is missing required Article 1 fields: {', '.join(missing)}")

    norm = pd.DataFrame({
        "facility_id": out[uid].astype(str) if uid else np.arange(len(out)).astype(str),
        "facility_name": out[name].astype(str) if name else "Unnamed facility",
        "lat": pd.to_numeric(out[lat], errors="coerce"),
        "lon": pd.to_numeric(out[lon], errors="coerce"),
        "facility_type": out[typ].astype(str).str.strip(),
        "functional_status": out[status].astype(str).str.strip(),
        "state": out[state].astype(str) if state else "Unknown",
        "lga": out[lga].astype(str) if lga else "Unknown",
    }).dropna(subset=["lat", "lon"])
    norm = norm[norm["lat"].between(-90, 90) & norm["lon"].between(-180, 180)].copy()
    if norm.empty:
        raise ValueError("No valid facility coordinates remain after normalization.")
    return norm.reset_index(drop=True)


def _haversine_vector_km(lat, lon, lat_arr, lon_arr):
    lat1 = np.radians(float(lat))
    lon1 = np.radians(float(lon))
    lat2 = np.radians(np.asarray(lat_arr, dtype=float))
    lon2 = np.radians(np.asarray(lon_arr, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * (2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))


def _article1_origin_pool(facilities: pd.DataFrame, cfg: Article1ReferralConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = facilities[
        facilities["facility_type"].str.casefold().isin({x.casefold() for x in cfg.eligible_types})
        & facilities["functional_status"].str.casefold().isin({x.casefold() for x in cfg.functional_statuses})
    ].copy()
    origins = facilities[
        facilities["facility_type"].str.casefold().eq("primary")
        & facilities["functional_status"].str.casefold().isin({x.casefold() for x in cfg.functional_statuses})
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible functional secondary/tertiary referral facilities were found.")
    if origins.empty:
        raise ValueError("No functional primary facilities were found for patient-origin proxies.")
    return origins.reset_index(drop=True), eligible.reset_index(drop=True)


def _precompute_origin_access(origins: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Attach nearest eligible higher-level facility using geodesic preselection."""
    e_lat = eligible["lat"].to_numpy()
    e_lon = eligible["lon"].to_numpy()
    rows = []
    for row in origins.itertuples(index=False):
        d = _haversine_vector_km(row.lat, row.lon, e_lat, e_lon)
        j = int(np.argmin(d))
        dest = eligible.iloc[j]
        rows.append({
            "origin_facility_id": row.facility_id,
            "origin_facility_name": row.facility_name,
            "origin_lat": float(row.lat), "origin_lon": float(row.lon),
            "origin_state": row.state, "origin_lga": row.lga,
            "referral_facility_id": dest["facility_id"],
            "referral_facility_name": dest["facility_name"],
            "referral_facility_type": dest["facility_type"],
            "referral_lat": float(dest["lat"]), "referral_lon": float(dest["lon"]),
            "straight_line_km": float(d[j]),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Precomputing Article 1 referral geography...")
def _prepare_article1_facility_access_cached(
        facilities_df: pd.DataFrame, eligible_types: tuple, functional_statuses: tuple
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize/filter GRID3 and cache the expensive primary-to-referral lookup.

    Geography is invariant across simulation seeds when the uploaded facility
    dataset and eligibility definition are unchanged, so recomputing this table
    for every run wastes time without changing the experiment.
    """
    cache_cfg = Article1ReferralConfig(
        enabled=True,
        eligible_types=tuple(eligible_types),
        functional_statuses=tuple(functional_statuses),
    )
    facilities = _normalize_grid3_for_article1(facilities_df)
    origins, eligible = _article1_origin_pool(facilities, cache_cfg)
    access = _precompute_origin_access(origins, eligible)
    return origins, eligible, access


def _article1_route_time(row: pd.Series, cfg: Article1ReferralConfig) -> tuple[float, str]:
    if cfg.routing_method == "geodesic_proxy":
        # Development/sensitivity proxy only; never label this as road-network time.
        return float(row["straight_line_km"] / cfg.assumed_speed_kmh * 60.0), "geodesic_speed_proxy"

    try:
        routed = get_osrm_route_distance_and_time(
            float(row["origin_lat"]), float(row["origin_lon"]),
            float(row["referral_lat"]), float(row["referral_lon"]),
        )
    except Exception as exc:
        raise RuntimeError(f"OSRM routing failed for Article 1; no proxy fallback was substituted: {exc}") from exc

    if isinstance(routed, dict):
        for key in ("travel_time_min", "duration_min", "time_min", "minutes"):
            if key in routed:
                return float(routed[key]), "osrm"
        if "duration" in routed:
            val = float(routed["duration"])
            return (val / 60.0 if val > 300 else val), "osrm"
    if isinstance(routed, (tuple, list)) and len(routed) >= 2:
        val = float(routed[1])
        return (val / 60.0 if val > 300 else val), "osrm"
    if np.isscalar(routed):
        val = float(routed)
        return (val / 60.0 if val > 300 else val), "osrm"
    raise RuntimeError("OSRM helper returned an unrecognized result; Article 1 did not substitute a proxy.")


def _run_article1_joint_burden(
        artifact: ModelArtifact, facilities_df: pd.DataFrame,
        cfg: Article1ReferralConfig, seed: int) -> tuple[dict, pd.DataFrame]:
    """Couple model misses to referral geography for Article 1.

    The primary Article 1 estimand is defined under a prespecified
    screen-to-referral policy: y_test==1 denotes eligibility for higher-level
    diagnostic assessment, while a model-negative prediction represents a missed
    referral opportunity. This is not an emergency-triage or urgency rule.
    """
    cfg.validate()
    origins, eligible, access = _prepare_article1_facility_access_cached(
        facilities_df, tuple(cfg.eligible_types), tuple(cfg.functional_statuses)
    )
    if access.empty:
        raise ValueError("Article 1 origin-access table is empty.")

    # Rank primary-care origins by distance to higher-level care. Coupling=0 gives
    # independent random geography. Higher values intentionally create a controlled
    # structural-access scenario and are recorded as simulation assumptions.
    access = access.sort_values("straight_line_km").reset_index(drop=True)
    half = max(1, len(access) // 2)
    better_access = access.iloc[:half]
    worse_access = access.iloc[half:] if len(access) > half else access.iloc[:half]
    rng = np.random.default_rng(int(seed) + 701)

    y_true = np.asarray(artifact.y_test).astype(int)
    y_pred = np.asarray(artifact.y_prediction).astype(int)
    y_prob = np.asarray(artifact.y_probability, dtype=float)
    groups = np.asarray(artifact.demo_test)

    # Common-random-number design for coupling experiments. For a given seed,
    # every lambda scenario receives the same latent random draws. Lambda changes
    # only whether a patient uses the structural pool versus the all-origin pool;
    # it does not generate a fresh stochastic realization for each scenario.
    n_patients = len(groups)
    u_structural = rng.random(n_patients)
    u_all_origin = rng.random(n_patients)
    u_structural_origin = rng.random(n_patients)
    assigned = []
    for idx, g in enumerate(groups):
        use_structural = u_structural[idx] < cfg.group_location_coupling
        if use_structural:
            pool = worse_access if str(g) in {"0", "0.0"} else better_access
            u_pick = u_structural_origin[idx]
        else:
            pool = access
            u_pick = u_all_origin[idx]
        j = min(int(u_pick * len(pool)), len(pool) - 1)
        assigned.append(pool.iloc[j])
    assigned_df = pd.DataFrame(assigned).reset_index(drop=True)

    travel = []
    methods = []
    # Route each unique origin-destination pair once per run.
    route_cache = {}
    for _, row in assigned_df.iterrows():
        key = (row["origin_facility_id"], row["referral_facility_id"], cfg.routing_method)
        if key not in route_cache:
            route_cache[key] = _article1_route_time(row, cfg)
        t, method = route_cache[key]
        travel.append(t); methods.append(method)

    rec = pd.DataFrame({
        "patient_index": np.arange(len(y_true), dtype=int),
        "group": groups,
        "clinical_target_y": y_true,
        "referral_eligible_r": y_true,  # prespecified screen-to-referral policy
        "predicted_positive": y_pred,
        "prediction_probability": y_prob,
        "missed_referral_opportunity_m": ((y_true == 1) & (y_pred == 0)).astype(int),
        "travel_time_min": np.asarray(travel, dtype=float),
        "geographic_constraint_c": (np.asarray(travel, dtype=float) > cfg.travel_threshold_min).astype(int),
        "routing_method": methods,
    })
    rec = pd.concat([rec, assigned_df.reset_index(drop=True)], axis=1)
    rec["joint_burden_j"] = ((rec["missed_referral_opportunity_m"] == 1) & (rec["geographic_constraint_c"] == 1)).astype(int)

    subgroup = {}
    for g in pd.unique(rec["group"]):
        d = rec[(rec["group"] == g) & (rec["referral_eligible_r"] == 1)].copy()
        if d.empty:
            continue
        fnr = float(d["missed_referral_opportunity_m"].mean())
        c_rate = float(d["geographic_constraint_c"].mean())
        joint = float(d["joint_burden_j"].mean())
        expected = fnr * c_rate
        excess = joint - expected
        c1 = d[d["geographic_constraint_c"] == 1]
        c0 = d[d["geographic_constraint_c"] == 0]
        d_gap = float(c1["missed_referral_opportunity_m"].mean() - c0["missed_referral_opportunity_m"].mean()) if (not c1.empty and not c0.empty) else np.nan
        subgroup[str(g)] = {
            "n_referral_eligible": int(len(d)), "fnr": fnr, "constraint_rate": c_rate,
            "joint_burden": joint, "independence_expected": expected,
            "excess_beyond_independence": excess, "dependence_gap": d_gap,
        }

    group_keys = list(subgroup)
    delta_joint = None
    if len(group_keys) >= 2:
        delta_joint = float(subgroup[group_keys[0]]["joint_burden"] - subgroup[group_keys[1]]["joint_burden"])

    summary = {
        "enabled": True,
        "analysis_population": "R=1: benchmark-positive patients eligible for higher-level diagnostic assessment under the prespecified screen-to-referral simulation policy",
        "referral_policy": "screen_positive_to_diagnostic_assessment",
        "referral_policy_scope": "research simulation policy; not emergency triage and not an individualized clinical recommendation",
        "travel_threshold_min": float(cfg.travel_threshold_min),
        "routing_method_requested": cfg.routing_method,
        "routing_method_observed": sorted(set(methods)),
        "eligible_referral_types": list(cfg.eligible_types),
        "functional_statuses": list(cfg.functional_statuses),
        "n_primary_origin_facilities": int(len(origins)),
        "n_eligible_referral_facilities": int(len(eligible)),
        "facility_access_precompute_cached": True,
        "group_location_coupling": float(cfg.group_location_coupling),
        "subgroups": subgroup,
        "delta_joint_first_minus_second": delta_joint,
        "warning": "Article 1 is a simulation/audit testbed. R=1 is defined by a prespecified screen-to-referral diagnostic-assessment policy; it must not be interpreted as emergency referral urgency.",
    }
    return summary, rec


# ── Hybrid Data Pipeline ───────────────────────────────────────────────────────
class HealthcareHybridPipeline:
    def __init__(self):
        self.available_datasets = {
            "Synthetic Only": self.generate_synthetic_only,
            "Heart Disease (UCI)": self.load_heart_disease_uci,
            "Diabetes (PIMA)": self.load_pima_diabetes,
            "Breast Cancer (Wisconsin)": self.load_breast_cancer,
            "Hybrid (Synthetic + Real)": self.generate_hybrid_data,
            "Nigeria Multilingual Healthcare": self._load_africa_healthcare,
        }



    def _load_africa_healthcare(self, n_samples: int = 5000):
        X, y, demo, preset = generate_africa_centric_data("multilingual_healthcare", n_samples)
        return self._arrays_to_df(X, y, demo, preset.get("description", "Multilingual Healthcare Preset"))

    @staticmethod
    def _arrays_to_df(X, y, demo, description: str) -> pd.DataFrame:
        cols = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y
        df["demographic_group"] = demo
        rng = np.random.default_rng(42)
        df["income_level"] = X[:, 1] if X.shape[1] > 1 else rng.uniform(0, 1, len(X))
        df["_africa_centric"] = True
        df["_description"] = description
        return df

    @st.cache_data(show_spinner=False)
    def load_heart_disease_uci(_self, n_samples: int = 5000):
        rng = np.random.default_rng(42)
        try:
            url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                   "heart-disease/processed.cleveland.data")
            cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
            df = pd.read_csv(url, names=cols, na_values="?").dropna()
            df["target"] = (df["target"] > 0).astype(int)
            n = min(len(df), n_samples)
            if len(df) > n:
                df = df.iloc[rng.choice(len(df), size=n, replace=False)].copy()
            else:
                df = df.copy()
            df["income_level"] = rng.uniform(0, 1, len(df))
            n = len(df)
            df["access_score"] = rng.uniform(0.3, 1, n)
            df["education_level"] = rng.choice([1, 2, 3, 4], n, p=[0.2, 0.3, 0.3, 0.2])
            df["demographic_group"] = df["sex"].astype(int)
            return df
        except Exception as exc:
            raise RuntimeError(f"UCI Heart Disease dataset could not be loaded: {exc}") from exc

    @st.cache_data(show_spinner=False)
    def load_pima_diabetes(_self, n_samples: int = 5000):
        rng = np.random.default_rng(42)
        try:
            url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
            cols = ["preg", "glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age", "target"]
            df = pd.read_csv(url, names=cols)
            n = min(len(df), n_samples)
            if len(df) > n:
                df = df.iloc[rng.choice(len(df), size=n, replace=False)].copy()
            else:
                df = df.copy()
            df["income_level"] = rng.uniform(0, 1, len(df))
            n = len(df)
            df["access_score"] = rng.uniform(0.3, 1, n)
            df["demographic_group"] = (df["income_level"] < 0.5).astype(int)
            return df
        except Exception as exc:
            raise RuntimeError(f"PIMA Diabetes dataset could not be loaded: {exc}") from exc

    @st.cache_data(show_spinner=False)
    def load_breast_cancer(_self, n_samples: int = 5000):
        rng = np.random.default_rng(42)
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        n = min(len(df), n_samples)
        if len(df) > n:
            df = df.iloc[rng.choice(len(df), size=n, replace=False)].copy()
        else:
            df = df.copy()
        n = len(df)
        df["age"] = rng.normal(55, 15, n).clip(25, 90)
        df["income_level"] = rng.uniform(0, 1, n)
        df["insurance"] = rng.choice([0, 1], n, p=[0.2, 0.8])
        df["demographic_group"] = (df["income_level"] < 0.5).astype(int)
        return df

    def generate_synthetic_only(self, n_samples: int = 5000, seed: int = 42, low_income_ratio: float = 0.40, uninsured_ratio: float = 0.35) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = n_samples
        data = {
            "age": rng.normal(55, 15, n).clip(18, 100),
            "sex": rng.choice([0, 1], n, p=[0.45, 0.55]),
            "bmi": rng.normal(27, 6, n).clip(15, 50),
            "blood_pressure": rng.normal(130, 20, n).clip(80, 200),
            "cholesterol": rng.normal(200, 40, n).clip(100, 350),
            "glucose": rng.normal(110, 30, n).clip(60, 300),
            "chronic_conditions": rng.poisson(1.5, n).clip(0, 8),
            "previous_hospitalizations": rng.poisson(0.8, n),
            "smoking": rng.binomial(1, 0.25, n),
            "exercise_frequency": rng.uniform(0, 1, n),
            "income_level": np.where(rng.random(n) < low_income_ratio, rng.uniform(0.05, 0.49, n), rng.uniform(0.50, 1.00, n)),
            "education": rng.choice([1, 2, 3, 4], n, p=[0.15, 0.35, 0.35, 0.15]),
            "insurance": rng.binomial(1, 1-uninsured_ratio, n),
            "access_score": rng.uniform(0.3, 1, n),
        }
        rs = (
                data["age"] / 100 * 0.2
                + (data["bmi"] - 25) / 25 * 0.15
                + (data["blood_pressure"] - 120) / 80 * 0.15
                + (data["cholesterol"] - 200) / 150 * 0.1
                + data["chronic_conditions"] / 8 * 0.2
                + (1 - data["exercise_frequency"]) * 0.1
                + data["smoking"] * 0.05
        )
        data["target"] = (rs + rng.normal(0, 0.1, n) > np.percentile(rs, 60)).astype(int)
        df = pd.DataFrame(data)
        df["demographic_group"] = df["sex"].astype(int)
        return df

    def generate_hybrid_data(self, n_samples: int = 5000) -> pd.DataFrame:
        try:
            real = self.load_heart_disease_uci(n_samples // 2)
            synth = self.generate_synthetic_only(n_samples // 2, seed=4242)
            common = [c for c in real.columns if c in synth.columns]
            hybrid = pd.concat([real[common], synth[common]], ignore_index=True)
            if "target" not in hybrid.columns:
                hybrid["target"] = np.random.choice([0, 1], len(hybrid))
            if "demographic_group" not in hybrid.columns:
                hybrid["demographic_group"] = np.random.choice([0, 1], len(hybrid))
            return hybrid.sample(frac=1, random_state=4242).reset_index(drop=True)
        except Exception as exc:
            raise RuntimeError(f"Hybrid healthcare dataset construction failed: {exc}") from exc


_pipeline = HealthcareHybridPipeline()


def _summarize_article1_records_at_threshold(records: pd.DataFrame, threshold_min: float) -> list[dict]:
    """Recompute Article 1 estimands at a travel-time threshold without rerouting.

    This is intentionally a post-routing sensitivity calculation. It reuses each
    patient's already-computed travel_time_min, changes only the definition of
    geographic constraint C, and then recomputes J, J_ind, E, and D.
    """
    if records is None or records.empty:
        return []
    if "travel_time_min" not in records.columns:
        raise ValueError("Article 1 threshold sensitivity requires travel_time_min in patient records.")
    threshold_min = float(threshold_min)
    if threshold_min <= 0:
        raise ValueError("Article 1 sensitivity thresholds must be > 0 minutes.")

    d = records.copy()
    d = d[d["referral_eligible_r"] == 1].copy()
    if d.empty:
        return []
    d["sensitivity_constraint_c"] = (pd.to_numeric(d["travel_time_min"], errors="coerce") > threshold_min).astype(int)
    d["sensitivity_joint_j"] = (
        (d["missed_referral_opportunity_m"].astype(int) == 1)
        & (d["sensitivity_constraint_c"] == 1)
    ).astype(int)

    rows = []
    group_values = list(d["group"].drop_duplicates())
    for group_value in group_values + ["ALL"]:
        g = d if group_value == "ALL" else d[d["group"] == group_value]
        if g.empty:
            continue
        fnr = float(g["missed_referral_opportunity_m"].mean())
        constraint = float(g["sensitivity_constraint_c"].mean())
        joint = float(g["sensitivity_joint_j"].mean())
        independence = fnr * constraint
        constrained = g[g["sensitivity_constraint_c"] == 1]
        accessible = g[g["sensitivity_constraint_c"] == 0]
        dependence = (
            float(constrained["missed_referral_opportunity_m"].mean())
            - float(accessible["missed_referral_opportunity_m"].mean())
        ) if len(constrained) and len(accessible) else np.nan
        rows.append({
            "threshold_min": threshold_min,
            "group": group_value,
            "n_referral_eligible": int(len(g)),
            "fnr_missed_referral": fnr,
            "geographic_constraint": constraint,
            "joint_burden_j": joint,
            "independence_j_ind": independence,
            "excess_e": joint - independence,
            "dependence_d": dependence,
        })
    return rows



def _article1_summarize_counterfactual(records: pd.DataFrame, scenario: str) -> list[dict]:
    """Summarize one S0-S3 counterfactual scenario on the R=1 population."""
    if records is None or records.empty:
        return []
    d = records[records["referral_eligible_r"] == 1].copy()
    if d.empty:
        return []
    rows = []
    groups = list(d["group"].drop_duplicates())
    for group_value in groups + ["ALL"]:
        g = d if group_value == "ALL" else d[d["group"] == group_value]
        if g.empty:
            continue
        fnr = float(g["cf_missed_referral_m"].mean())
        constraint = float(g["cf_geographic_constraint_c"].mean())
        joint = float(g["cf_joint_burden_j"].mean())
        expected = fnr * constraint
        c1 = g[g["cf_geographic_constraint_c"] == 1]
        c0 = g[g["cf_geographic_constraint_c"] == 0]
        dependence = (
            float(c1["cf_missed_referral_m"].mean()) - float(c0["cf_missed_referral_m"].mean())
        ) if len(c1) and len(c0) else np.nan
        rows.append({
            "scenario": scenario,
            "group": str(group_value),
            "n_referral_eligible": int(len(g)),
            "fnr_missed_referral": fnr,
            "geographic_constraint": constraint,
            "joint_burden_j": joint,
            "independence_j_ind": expected,
            "excess_e": joint - expected,
            "dependence_d": dependence,
        })
    return rows


def _run_article1_s0_s3_interventions(
        records: pd.DataFrame,
        travel_threshold_min: float,
        ai_gap_closure: float = 1.0,
        geography_time_reduction: float = 0.25,
) -> tuple[list[dict], pd.DataFrame]:
    """Run S0-S3 counterfactual interventions on fixed patient-level pathways.

    S0: baseline predictions + baseline geography.
    S1: algorithmic counterfactual only. Close ``ai_gap_closure`` fraction of the
        baseline FNR gap by converting the highest-probability missed positives in
        the higher-FNR subgroup into detected positives. This is an auditable
        counterfactual error-repair policy, not a claim about a deployed mitigation.
    S2: geography counterfactual only. Reduce all referral travel times by the
        prespecified proportional amount while holding predictions fixed.
    S3: combine S1 and S2.

    All four scenarios operate on the same patients, labels, baseline predictions,
    facility matches, and baseline travel-time estimates.
    """
    if records is None or records.empty:
        return [], pd.DataFrame()
    if not 0 <= float(ai_gap_closure) <= 1:
        raise ValueError("ai_gap_closure must be between 0 and 1.")
    if not 0 <= float(geography_time_reduction) < 1:
        raise ValueError("geography_time_reduction must be in [0,1).")
    if float(travel_threshold_min) <= 0:
        raise ValueError("travel_threshold_min must be > 0.")

    base = records.copy().reset_index(drop=True)
    required = {
        "referral_eligible_r", "missed_referral_opportunity_m", "travel_time_min",
        "prediction_probability", "group"
    }
    missing = sorted(required.difference(base.columns))
    if missing:
        raise ValueError(f"S0-S3 intervention suite missing required columns: {', '.join(missing)}")

    eligible = base[base["referral_eligible_r"] == 1].copy()
    fnr_by_group = eligible.groupby("group")["missed_referral_opportunity_m"].mean()
    target_group = None
    reference_fnr = np.nan
    target_fnr = np.nan
    n_repairs = 0
    repair_indices = []
    if len(fnr_by_group) >= 2:
        target_group = fnr_by_group.idxmax()
        target_fnr = float(fnr_by_group.max())
        reference_fnr = float(fnr_by_group.min())
        target_n = int((eligible["group"] == target_group).sum())
        desired_fnr = target_fnr - float(ai_gap_closure) * (target_fnr - reference_fnr)
        current_misses = int(((eligible["group"] == target_group) & (eligible["missed_referral_opportunity_m"] == 1)).sum())
        desired_misses = int(round(desired_fnr * target_n))
        n_repairs = max(0, current_misses - desired_misses)
        candidates = eligible[
            (eligible["group"] == target_group)
            & (eligible["missed_referral_opportunity_m"] == 1)
        ].sort_values(["prediction_probability"], ascending=False)
        repair_indices = list(candidates.head(n_repairs).index)

    scenarios = []
    patient_frames = []
    for scenario, use_ai, use_geo in [
        ("S0 Baseline", False, False),
        ("S1 AI improvement", True, False),
        ("S2 Geography improvement", False, True),
        ("S3 Combined improvement", True, True),
    ]:
        d = base.copy()
        d["cf_missed_referral_m"] = d["missed_referral_opportunity_m"].astype(int)
        d["cf_travel_time_min"] = pd.to_numeric(d["travel_time_min"], errors="coerce")
        if use_ai and repair_indices:
            d.loc[repair_indices, "cf_missed_referral_m"] = 0
        if use_geo:
            d["cf_travel_time_min"] = d["cf_travel_time_min"] * (1.0 - float(geography_time_reduction))
        d["cf_geographic_constraint_c"] = (d["cf_travel_time_min"] > float(travel_threshold_min)).astype(int)
        d["cf_joint_burden_j"] = (
            (d["cf_missed_referral_m"] == 1) & (d["cf_geographic_constraint_c"] == 1)
        ).astype(int)
        d["scenario"] = scenario
        d["ai_intervention_applied"] = bool(use_ai)
        d["geography_intervention_applied"] = bool(use_geo)
        patient_frames.append(d)
        scenarios.extend(_article1_summarize_counterfactual(d, scenario))

    meta = {
        "scenario": "INTERVENTION_METADATA",
        "group": "ALL",
        "n_referral_eligible": int(len(eligible)),
        "fnr_missed_referral": np.nan,
        "geographic_constraint": np.nan,
        "joint_burden_j": np.nan,
        "independence_j_ind": np.nan,
        "excess_e": np.nan,
        "dependence_d": np.nan,
        "ai_target_group": str(target_group) if target_group is not None else None,
        "baseline_target_group_fnr": target_fnr,
        "baseline_reference_fnr": reference_fnr,
        "ai_gap_closure": float(ai_gap_closure),
        "ai_repaired_missed_referrals": int(n_repairs),
        "geography_time_reduction": float(geography_time_reduction),
        "travel_threshold_min": float(travel_threshold_min),
        "intervention_scope": "simulation counterfactual; not a causal estimate or implemented policy",
    }
    scenarios.append(meta)
    return scenarios, pd.concat(patient_frames, ignore_index=True)



def _article1_tidy_from_result(result: dict, model_variant: str) -> list[dict]:
    """Convert one repeated-seed result into tidy Article 1 estimand rows."""
    summary = result.get("article1_summary") or {}
    rows = []
    balanced_accuracy = 0.5 * (float(result.get("sensitivity", np.nan)) + float(result.get("specificity", np.nan)))
    subgroups = summary.get("subgroups") or {}
    for group, values in subgroups.items():
        rows.append({
            "run_id": int(result.get("run_id", 0)),
            "seed": int(result.get("seed", 0)),
            "model_variant": str(model_variant),
            "group": str(group),
            "n_referral_eligible": int(values.get("n_referral_eligible", 0)),
            "fnr_missed_referral": float(values.get("fnr", np.nan)),
            "geographic_constraint": float(values.get("constraint_rate", np.nan)),
            "joint_burden_j": float(values.get("joint_burden", np.nan)),
            "independence_j_ind": float(values.get("independence_expected", np.nan)),
            "excess_e": float(values.get("excess_beyond_independence", np.nan)),
            "dependence_d": float(values.get("dependence_gap", np.nan)) if pd.notna(values.get("dependence_gap", np.nan)) else np.nan,
            "accuracy": float(result.get("accuracy", np.nan)),
            "auc": float(result.get("auc", np.nan)),
            "balanced_accuracy": float(balanced_accuracy),
            "sensitivity": float(result.get("sensitivity", np.nan)),
            "specificity": float(result.get("specificity", np.nan)),
            "travel_threshold_min": float(summary.get("travel_threshold_min", np.nan)),
            "routing_method": ",".join(map(str, summary.get("routing_method_observed", []))),
            "group_location_coupling": float(summary.get("group_location_coupling", np.nan)),
        })
    return rows


def _article1_run_level_bootstrap_ci(tidy: pd.DataFrame, n_boot: int = 1000, seed: int = 20260912) -> pd.DataFrame:
    """Bootstrap repeated-seed estimands at the run level.

    Resampling whole run IDs preserves within-run patient dependence and reflects
    retraining/test-split variability. CIs are descriptive simulation uncertainty,
    not population-representative confidence intervals for Nigeria.
    """
    if tidy is None or tidy.empty:
        return pd.DataFrame()
    metrics = ["fnr_missed_referral", "geographic_constraint", "joint_burden_j", "independence_j_ind", "excess_e", "dependence_d"]
    out = []
    rng = np.random.default_rng(int(seed))
    for (variant, group), d in tidy.groupby(["model_variant", "group"], dropna=False):
        run_ids = np.array(sorted(d["run_id"].dropna().unique()))
        if len(run_ids) == 0:
            continue
        per_run = d.groupby("run_id", as_index=False)[metrics].mean(numeric_only=True).set_index("run_id")
        for metric in metrics:
            vals = per_run[metric].dropna()
            if vals.empty:
                continue
            point = float(vals.mean())
            if len(vals) < 2 or int(n_boot) < 20:
                lo = hi = np.nan
            else:
                arr = vals.to_numpy(dtype=float)
                boot = np.empty(int(n_boot), dtype=float)
                for b in range(int(n_boot)):
                    boot[b] = float(rng.choice(arr, size=len(arr), replace=True).mean())
                lo, hi = [float(x) for x in np.quantile(boot, [0.025, 0.975])]
            out.append({
                "model_variant": str(variant), "group": str(group), "metric": metric,
                "estimate": point, "ci95_low": lo, "ci95_high": hi,
                "n_runs": int(len(vals)), "bootstrap_reps": int(n_boot),
                "bootstrap_unit": "run/seed",
            })
    return pd.DataFrame(out)


def _article1_performance_match_table(model_rows: pd.DataFrame, tolerance: float = 0.03) -> pd.DataFrame:
    """Flag model variants whose aggregate predictive performance matches baseline.

    Matching requires absolute mean differences from baseline within ``tolerance``
    simultaneously for accuracy, AUROC, and balanced accuracy.
    """
    if model_rows is None or model_rows.empty:
        return pd.DataFrame()
    perf_cols = ["accuracy", "auc", "balanced_accuracy", "sensitivity", "specificity"]
    perf = model_rows.groupby("model_variant", as_index=False)[perf_cols].mean(numeric_only=True)
    baseline = perf[perf["model_variant"] == "none"]
    if baseline.empty:
        baseline = perf.iloc[[0]]
    b = baseline.iloc[0]
    rows = []
    for _, r in perf.iterrows():
        da = abs(float(r["accuracy"]) - float(b["accuracy"]))
        du = abs(float(r["auc"]) - float(b["auc"]))
        db = abs(float(r["balanced_accuracy"]) - float(b["balanced_accuracy"]))
        rows.append({
            **{c: r[c] for c in perf.columns},
            "delta_accuracy_vs_baseline": da,
            "delta_auc_vs_baseline": du,
            "delta_balanced_accuracy_vs_baseline": db,
            "performance_tolerance": float(tolerance),
            "performance_matched": bool(da <= tolerance and du <= tolerance and db <= tolerance),
        })
    return pd.DataFrame(rows)


def init_health_session_state():
    """Initializes persistent sidebar configuration keys to prevent state loss on re-runs."""
    _hc_valid = list(simulation_config.BIAS_TYPES) + [b for b in ["gender", "linguistic"] if
                                                      b not in simulation_config.BIAS_TYPES]
    _hc_defaults = [b for b in ["demographic", "socioeconomic", "gender"] if b in _hc_valid]
    defaults = {
        # Config Keys - Aligned with cfg_health_* Widget Keys
        "cfg_health_target_country": "All",
        "cfg_health_data_source": "Synthetic Only",
        "cfg_health_setting": "Primary Care",
        "cfg_health_prediction_task": "Disease Risk Screening",
        "cfg_health_region": "Nigeria",
        "cfg_health_low_income_ratio": 0.40,
        "cfg_health_uninsured_ratio": 0.35,
        "cfg_health_mitigation_strategy": "None (Baseline)",
        "cfg_health_active_threshold": 0.50,
        "cfg_health_selected_biases": _hc_defaults,
        "cfg_health_bias_intensity": 0.25,
        "cfg_health_access_inequality": 0.30,
        "cfg_health_poison_rate": 0.05,
        "cfg_health_sample_size": 5000,
        "cfg_health_n_runs": 3,
        "cfg_health_article1_intervention_enabled": True,
        "cfg_health_article1_ai_gap_closure": 1.0,
        "cfg_health_article1_geo_time_reduction": 0.25,
        "cfg_health_season_profile": "Dry Season",
        "cfg_health_terrain_type": "Rural Unpaved",
        "cfg_health_llm_provider": "Groq (Llama 3.3 / 3.1)",
        "cfg_health_api_key": "",
        "cfg_health_enable_gender_audit": True,
        "cfg_health_enable_governance": True,
        "cfg_health_governance_policy": "majority_vote",
        "cfg_health_enable_redteam": False,
        "cfg_health_enable_ai_safety": False,
        "cfg_health_enable_lifecycle": False,
        "cfg_health_enable_eco": False,
        "cfg_health_enable_dynamic": False,
        "cfg_health_enable_arena": False,
        "cfg_health_enable_agent_economy": False,
        # Output State
        "health_run_history": [],
        "latest_impact_matrix": None,
        "health_feature_outputs": {},
        "health_xai_results": {},
        "health_longitudinal": None,
        "health_federated": None,
        "health_snapshot_history": [],
        "health_dynamic_report": {},
         "health_spatial_report": {},
         "health_redteam_report": {},
         "health_experiment_config": {},
         "health_article1_patient_records": [],
         "health_article1_summary": {},
         "health_article1_threshold_sensitivity": [],
        "health_real_models": {},
        "health_safety_report": {},
        "health_lifecycle_report": {},
        "active_llm_out": None,
        "active_nearest_facs": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_health_session_state()
try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, page_header, plotly_theme as _ptheme

    inject_css("health")
    ACCENT = DOMAIN_ACCENTS["health"]
except ImportError:
    ACCENT = "#0891b2"


# ── Step 2 & 3: Helper Engines ─────────────────────────────────────────────────

def generate_clinical_sankey(impact_matrix: dict) -> go.Figure:
    """Generates a Plotly Sankey diagram mapping patient cohorts to clinical diagnostic outcomes."""
    if not impact_matrix:
        return go.Figure()

    disadv = impact_matrix.get("disadvantaged", {})
    priv = impact_matrix.get("privileged", {})

    dis_count = max(int(disadv.get("Count", 0)), 0)
    priv_count = max(int(priv.get("Count", 0)), 0)
    gc = impact_matrix.get("group_confusion", {})
    dis_cm = gc.get("0", {})
    priv_cm = gc.get("1", {})
    dis_tp = int(dis_cm.get("tp", round(dis_count * disadv.get("TPR", 0.0))))
    dis_fn = int(dis_cm.get("fn", round(dis_count * disadv.get("FNR", 0.0))))
    priv_tp = int(priv_cm.get("tp", round(priv_count * priv.get("TPR", 0.0))))
    priv_fn = int(priv_cm.get("fn", round(priv_count * priv.get("FNR", 0.0))))

    labels = [
        "Total Patient Cohort",  # 0
        "Underserved Cohort",  # 1
        "Privileged Cohort",  # 2
        "Correct Diagnosis (TP)",  # 3
        "Under-Diagnosed (FN)",  # 4
    ]

    source = [0, 0, 1, 1, 2, 2]
    target = [1, 2, 3, 4, 3, 4]
    value = [dis_count, priv_count, max(dis_tp, 0), max(dis_fn, 0), max(priv_tp, 0), max(priv_fn, 0)]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=18,
            line=dict(color="#0F172A", width=0.5),
            label=labels,
            color=["#0284C7", "#E11D48", "#2563EB", "#16A34A", "#DC2626"]
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=["#CBD5E1", "#CBD5E1", "#86EFAC", "#FECACA", "#86EFAC", "#FECACA"]
        )
    )])
    fig.update_layout(title_text="🏥 Clinical Diagnostic Pathway Disparity Flow", font_size=12, height=380,
                      margin=dict(t=40, b=10, l=10, r=10))
    return fig



def _train_and_score(X, y, demo, feature_names=None, random_state=42, mitigation_strategy="none", threshold=0.5):
    """Train once, retain probabilities/artifacts, and apply mitigation without hiding failures."""
    if X is None or len(X) < 20:
        raise ValueError("Insufficient model-ready observations.")
    feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
    X_tr, X_te, y_tr, y_te, demo_tr, demo_te = train_test_split(
        X, y, demo, test_size=0.30, random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=6)

    sample_weight = None
    if mitigation_strategy == "reweighing":
        # Inverse-frequency weighting across (sensitive group, outcome) strata.
        strata = pd.Series([f"{g}_{yy}" for g, yy in zip(demo_tr, y_tr)])
        counts = strata.value_counts()
        sample_weight = np.array([1.0 / counts[z] for z in strata], dtype=float)
        sample_weight *= len(sample_weight) / sample_weight.sum()
    clf.fit(X_tr_s, y_tr, sample_weight=sample_weight)

    y_prob = clf.predict_proba(X_te_s)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_te_s).astype(float)
    baseline_pred = (y_prob >= 0.50).astype(int)
    y_pred = (y_prob >= float(threshold)).astype(int)

    # For the selected post-processing strategy, tune group thresholds on a
    # validation split carved from training data, then apply to untouched test data.
    if mitigation_strategy == "post_processing" and len(np.unique(demo_tr)) >= 2:
        X_fit, X_val, y_fit, y_val, d_fit, d_val = train_test_split(
            X_tr_s, y_tr, demo_tr, test_size=0.25, random_state=random_state + 991,
            stratify=y_tr if len(np.unique(y_tr)) > 1 else None
        )
        clf.fit(X_fit, y_fit, sample_weight=None)
        p_val = clf.predict_proba(X_val)[:, 1]
        groups = np.unique(d_val)
        grid = np.linspace(max(0.05, threshold - 0.25), min(0.95, threshold + 0.25), 19)
        group_thresholds = {}
        for g in groups:
            best_t, best_score = threshold, float("inf")
            for t in grid:
                preds = (p_val[d_val == g] >= t).astype(int)
                yy = y_val[d_val == g]
                pos = yy == 1
                neg = yy == 0
                tpr = np.mean(preds[pos] == 1) if pos.any() else 0.0
                fpr = np.mean(preds[neg] == 1) if neg.any() else 0.0
                ref_tprs = []
                ref_fprs = []
                for h in groups:
                    if h == g: continue
                    ph = (p_val[d_val == h] >= threshold).astype(int)
                    yh = y_val[d_val == h]
                    ph_pos = yh == 1; ph_neg = yh == 0
                    ref_tprs.append(np.mean(ph[ph_pos] == 1) if ph_pos.any() else 0.0)
                    ref_fprs.append(np.mean(ph[ph_neg] == 1) if ph_neg.any() else 0.0)
                score = abs(tpr - np.mean(ref_tprs or [tpr])) + abs(fpr - np.mean(ref_fprs or [fpr]))
                if score < best_score:
                    best_score, best_t = score, float(t)
            group_thresholds[int(g)] = best_t
        y_pred = np.array([int(p >= group_thresholds.get(int(g), threshold)) for p, g in zip(y_prob, demo_te)])
    else:
        group_thresholds = {}

    acc = float(accuracy_score(y_te, y_pred))
    rec = float(recall_score(y_te, y_pred, zero_division=0))
    prec = float(precision_score(y_te, y_pred, zero_division=0))
    f1 = float(f1_score(y_te, y_pred, zero_division=0))
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    metrics = {"accuracy": acc, "recall": rec, "sensitivity": rec, "precision": prec, "f1": f1, "fpr": fpr,
               "auc": float(roc_auc_score(y_te, y_prob)) if len(np.unique(y_te)) > 1 else 0.5,
               "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
               "threshold": float(threshold), "group_thresholds": group_thresholds,
               "baseline_accuracy": float(accuracy_score(y_te, baseline_pred)),
               "baseline_recall": float(recall_score(y_te, baseline_pred, zero_division=0))}
    artifact = ModelArtifact(clf, scaler, list(feature_names), X_tr_s, X_te_s, y_tr, y_te, demo_tr, demo_te, y_prob, y_pred, baseline_pred, float(threshold))
    return metrics, artifact


def _run_health_model_redteam(artifact: ModelArtifact) -> dict:
    """Run controlled robustness probes against the actual trained healthcare model."""
    X = np.asarray(artifact.X_test)
    y = np.asarray(artifact.y_test)
    d = np.asarray(artifact.demo_test)
    base_pred = artifact.baseline_prediction
    base_acc = float(accuracy_score(y, base_pred))
    base_fair = calculate_fairness_metrics(y, base_pred, d)
    base_equity = float(base_fair.get("fairness_score", 0.0))
    probes = {}
    rng = np.random.default_rng(2026)
    scale = np.nanstd(X, axis=0)
    scale[scale == 0] = 1.0
    for name, strength in [("feature_noise", 0.10), ("feature_masking", 0.20), ("combined_shift", 0.15)]:
        X_adv = X.copy()
        if name in ("feature_noise", "combined_shift"):
            X_adv += rng.normal(0, strength, X_adv.shape) * scale
        if name in ("feature_masking", "combined_shift"):
            n_mask = max(1, int(X_adv.shape[1] * strength))
            cols = rng.choice(X_adv.shape[1], size=n_mask, replace=False)
            X_adv[:, cols] = 0.0
        p_adv = artifact.model.predict_proba(X_adv)[:, 1] if hasattr(artifact.model, "predict_proba") else artifact.model.predict(X_adv)
        pred_adv = (p_adv >= artifact.threshold).astype(int)
        adv_acc = float(accuracy_score(y, pred_adv))
        fair_adv = calculate_fairness_metrics(y, pred_adv, d)
        probes[name] = {
            "accuracy": adv_acc,
            "accuracy_degradation": base_acc - adv_acc,
            "equity_score": float(fair_adv.get("fairness_score", 0.0)),
            "equity_degradation": base_equity - float(fair_adv.get("fairness_score", 0.0)),
        }
    probes["baseline_accuracy"] = base_acc
    return {"model_linked": True, "probes": probes, "note": "Controlled numeric perturbation probes; multimodal attacks remain in the separate feature module."}



def _summarize_article1_coupling_scenario(summary: dict, records: pd.DataFrame, coupling: float) -> list[dict]:
    """Return tidy subgroup/overall estimands for one coupling scenario.

    The underlying model predictions are held fixed. Only the structural
    patient-location assignment parameter lambda changes between scenarios.
    """
    if records is None or records.empty:
        return []
    d = records[records["referral_eligible_r"] == 1].copy()
    if d.empty:
        return []
    rows = []
    group_values = list(d["group"].drop_duplicates())
    for group_value in group_values + ["ALL"]:
        g = d if group_value == "ALL" else d[d["group"] == group_value]
        if g.empty:
            continue
        fnr = float(g["missed_referral_opportunity_m"].mean())
        constraint = float(g["geographic_constraint_c"].mean())
        joint = float(g["joint_burden_j"].mean())
        independence = fnr * constraint
        constrained = g[g["geographic_constraint_c"] == 1]
        accessible = g[g["geographic_constraint_c"] == 0]
        dependence = (
            float(constrained["missed_referral_opportunity_m"].mean())
            - float(accessible["missed_referral_opportunity_m"].mean())
        ) if len(constrained) and len(accessible) else np.nan
        rows.append({
            "coupling_lambda": float(coupling),
            "group": str(group_value),
            "n_referral_eligible": int(len(g)),
            "fnr_missed_referral": fnr,
            "geographic_constraint": constraint,
            "joint_burden_j": joint,
            "independence_j_ind": independence,
            "excess_e": joint - independence,
            "dependence_d": dependence,
            "travel_threshold_min": float(summary.get("travel_threshold_min", np.nan)),
            "routing_method_requested": summary.get("routing_method_requested"),
            "routing_method_observed": summary.get("routing_method_observed"),
        })
    return rows

def _run_one(
        data_source, n_samples, selected_biases, bias_intensity,
        poison_rate, access_inequality, run_idx,
        enable_redteam=False, enable_governance=True, governance_policy="majority_vote",
        enable_arena=False, enable_agent_economy=False, enable_gender_audit=True,
        selected_state="Nigeria (National Average)", low_income_ratio=0.40, uninsured_ratio=0.35,
        mitigation_strategy="none", threshold=0.5, experiment_seed=42,
        article1_config=None, article1_facilities=None):
    """Single reproducible healthcare experiment. No silent synthetic fallback."""
    seed = int(experiment_seed) + int(run_idx)
    rng = np.random.default_rng(seed)
    if data_source not in _pipeline.available_datasets:
        raise ValueError(f"Unknown healthcare dataset: {data_source}")

    loader = _pipeline.available_datasets[data_source]
    try:
        # Existing loaders remain compatible; synthetic generation receives the explicit seed.
        if data_source == "Synthetic Only":
            df_ds = loader(n_samples, seed=seed, low_income_ratio=low_income_ratio, uninsured_ratio=uninsured_ratio)
        else:
            df_ds = loader(n_samples)
    except Exception as exc:
        raise RuntimeError(f"Dataset '{data_source}' failed to load: {exc}") from exc
    if df_ds is None or df_ds.empty:
        raise RuntimeError(f"Dataset '{data_source}' returned no records. Synthetic fallback is disabled.")

    # Make population controls explicit for external/real datasets. These are
    # simulated equity attributes, not claims about the source population.
    if data_source != "Synthetic Only":
        df_ds = df_ds.copy()
        df_ds["income_level"] = np.where(
            rng.random(len(df_ds)) < low_income_ratio,
            rng.uniform(0.05, 0.49, len(df_ds)),
            rng.uniform(0.50, 1.00, len(df_ds)),
        )
        df_ds["insurance"] = rng.binomial(1, 1 - uninsured_ratio, len(df_ds))
        df_ds["access_score"] = rng.uniform(0.3, 1.0, len(df_ds))

    if "target" not in df_ds.columns:
        raise ValueError(f"Dataset '{data_source}' has no target column.")
    y = df_ds["target"].astype(int).to_numpy()
    if "demographic_group" in df_ds.columns:
        demo = df_ds["demographic_group"].to_numpy()
    elif "sex" in df_ds.columns:
        demo = df_ds["sex"].astype(int).to_numpy()
    elif "income_level" in df_ds.columns:
        demo = (df_ds["income_level"].to_numpy() < 0.5).astype(int)
    else:
        demo = rng.integers(0, 2, size=len(df_ds))

    excluded = {"target", "demographic_group", "_africa_centric", "_description"}
    feat_cols = [c for c in df_ds.columns if c not in excluded]
    numeric_cols = list(df_ds[feat_cols].select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        raise ValueError(f"Dataset '{data_source}' has no numeric model features.")
    X = df_ds[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(df_ds[numeric_cols].median(numeric_only=True)).to_numpy(dtype=float)

    # Access inequality is an explicit simulation perturbation, recorded as such.
    if access_inequality > 0:
        mask = demo == 0
        if mask.any():
            access_noise = rng.normal(0, access_inequality * 0.3, (mask.sum(), X.shape[1]))
            X[mask] += access_noise

    _vb = list(simulation_config.BIAS_TYPES) + [b for b in ["gender", "linguistic"] if b not in simulation_config.BIAS_TYPES]
    applied_biases = []
    for bt in [b for b in selected_biases if b in _vb]:
        X, y, demo = apply_bias(X, y, bt, bias_intensity, demographic_info=demo)
        applied_biases.append(bt)

    if poison_rate > 0:
        X, y, demo = simulate_data_poisoning(X, y, poison_rate, attack_type="label_flipping", demographic_info=demo, targeted=False)

    metrics, artifact = _train_and_score(
        X, y, demo, feature_names=numeric_cols, random_state=seed,
        mitigation_strategy=mitigation_strategy, threshold=threshold
    )
    y_te, y_pred, demo_te = artifact.y_test, artifact.y_prediction, artifact.demo_test

    fair = calculate_fairness_metrics(y_te, y_pred, demo_te)
    gender_audit = None
    if enable_gender_audit:
        try:
            gender_audit = run_gender_equity_audit(y_te, y_pred, demo_te, 0.34)
        except Exception as exc:
            gender_audit = {"error": str(exc)}

    # Actual confusion matrices by group are retained for clinical impact/Sankey.
    group_confusion = {}
    for g in np.unique(demo_te):
        m = demo_te == g
        cm = confusion_matrix(y_te[m], y_pred[m], labels=[0, 1]).ravel()
        tn, fp, fn, tp = [int(v) for v in cm]
        group_confusion[str(g)] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "count": int(m.sum())}

    article1_summary = {}
    article1_patient_records = None
    if article1_config is not None and getattr(article1_config, "enabled", False):
        if article1_facilities is None or article1_facilities.empty:
            raise ValueError("Article 1 is enabled but no GRID3/facility dataset was supplied.")
        article1_summary, article1_patient_records = _run_article1_joint_burden(
            artifact=artifact, facilities_df=article1_facilities, cfg=article1_config, seed=seed
        )

    impact_mat = compute_clinical_impact_matrix(y_te, y_pred, demo_te)
    impact_mat["group_confusion"] = group_confusion
    impact_mat["fnr_gap_threshold"] = 0.05
    impact_mat["threshold_type"] = "GAGS proposed research threshold; not a universal regulatory requirement"
    st.session_state["latest_impact_matrix"] = impact_mat

    # XAI receives the real feature names from the selected dataset.
    if run_idx == 0:
        try:
            from components.governance_logic import ExplainableModel, generate_compliance_report
            xm = ExplainableModel(domain="health")
            xm.model = artifact.model
            xm._X_train = artifact.X_train
            xm._is_fitted = True
            xm.feature_names = list(artifact.feature_names)
            fi = xm.feature_importance(artifact.X_test, artifact.y_test, n_repeats=6)
            _xai = {"feature_importance": fi.__dict__ if hasattr(fi, "__dict__") else fi,
                    "feature_names": artifact.feature_names}
            denied = np.where((artifact.y_test == 1) & (artifact.y_prediction == 0))[0]
            if len(denied):
                expl = xm.explain_instance(artifact.X_test[denied[0]])
                cf = xm.counterfactual(artifact.X_test[denied[0]])
                _xai["instance_explanation"] = expl.__dict__ if hasattr(expl, "__dict__") else expl
                _xai["counterfactual"] = cf.__dict__ if hasattr(cf, "__dict__") else cf
            st.session_state.health_xai_results = _xai
        except Exception as exc:
            st.session_state.health_xai_results = {"error": str(exc)}

    dataset_kind = "synthetic" if data_source in {"Synthetic Only", "Hybrid (Synthetic + Real)"} else "real_or_external"
    provenance = {
        "dataset": data_source, "dataset_kind": dataset_kind,
        "feature_names": numeric_cols,
        "synthetic_equity_attributes": [c for c in ["income_level", "insurance", "access_score", "education_level"] if c in df_ds.columns],
        "simulation_attributes": ["access_inequality", "bias_intensity", "poison_rate"],
        "note": "Real clinical data may have simulated socioeconomic/access attributes; fairness findings are conditional on those attributes."
    }

    return {
        "run_id": run_idx + 1, "seed": seed, "data_source": data_source,
        "n_samples": int(len(df_ds)), "n_features": int(X.shape[1]), "feature_names": numeric_cols,
        "provenance": provenance,
        "accuracy": metrics["accuracy"], "recall": metrics["recall"], "sensitivity": metrics["sensitivity"],
        "precision": metrics["precision"], "f1": metrics["f1"], "fpr": metrics["fpr"],
        "specificity": float(metrics["tn"] / (metrics["tn"] + metrics["fp"])) if (metrics["tn"] + metrics["fp"]) else 0.0,
        "auc": metrics["auc"], "baseline_accuracy": metrics["baseline_accuracy"], "baseline_recall": metrics["baseline_recall"],
        "tn": metrics["tn"], "fp": metrics["fp"], "fn": metrics["fn"], "tp": metrics["tp"],
        "acc_hi": float(accuracy_score(y_te[demo_te == 1], y_pred[demo_te == 1])) if (demo_te == 1).any() else metrics["accuracy"],
        "acc_lo": float(accuracy_score(y_te[demo_te == 0], y_pred[demo_te == 0])) if (demo_te == 0).any() else metrics["accuracy"],
        "sens_hi": float(recall_score(y_te[demo_te == 1], y_pred[demo_te == 1], zero_division=0)) if (demo_te == 1).any() else metrics["recall"],
        "sens_lo": float(recall_score(y_te[demo_te == 0], y_pred[demo_te == 0], zero_division=0)) if (demo_te == 0).any() else metrics["recall"],
        "adv_hi": float(accuracy_score(y_te[demo_te == 1], y_pred[demo_te == 1])) if (demo_te == 1).any() else metrics["accuracy"],
        "adv_lo": float(accuracy_score(y_te[demo_te == 0], y_pred[demo_te == 0])) if (demo_te == 0).any() else metrics["accuracy"],
        "equity_score": fair.get("fairness_score", 0.5), "fairness_score": fair.get("fairness_score", 0.5),
        "demographic_parity": fair.get("demographic_parity_difference", 0),
        "equalized_odds": fair.get("equalized_odds_difference", 0),
        "bias_intensity": bias_intensity, "poison_rate": poison_rate,
        "access_inequality": access_inequality, "low_income_ratio": low_income_ratio, "uninsured_ratio": uninsured_ratio,
        "biases": ", ".join(applied_biases) or "None",
        "gender_gap": getattr(gender_audit, "overall_gender_gap", 0.0) if gender_audit and not isinstance(gender_audit, dict) else 0.0,
        "warnings": [], "subgroup_tpr": {str(g): float(recall_score(y_te[demo_te == g], y_pred[demo_te == g], zero_division=0)) for g in np.unique(demo_te)},
        "group_confusion": group_confusion, "impact_matrix": impact_mat,
        "mitigation_strategy": mitigation_strategy, "threshold": threshold,
        "model_artifact": artifact,
        "article1_summary": article1_summary,
        "article1_patient_records": article1_patient_records,
    }


# ── Sidebar Setup ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.sidebar.header("🌍 Africa-Centric Context")
    target_country = st.sidebar.selectbox(
        "Select Target Region",
        ["All", "NGA", "KEN", "ZAF", "GHA"],
        key="cfg_health_target_country"
    )
    iso3_filter = None if target_country == "All" else target_country

    df_afrimed = load_afrimed_qa()
    df_facilities = load_hdx_facilities(country_iso3=iso3_filter)

    language_switcher(location="sidebar")
    st.divider()

    selected_state = state_selector(key="_state_1healthcareequity", location="sidebar")
    state_info_card(selected_state)

    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #0284C7 0%, #0F172A 100%); 
                    border-radius: 8px; padding: 10px 14px; margin: 10px 0px; color: white;'>
            <div style='font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.85;'>
                Clinical AI Engine
            </div>
            <div style='font-size: 1.05rem; font-weight: 700; margin-top: 2px;'>
                Healthcare Equity
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    role_switcher("health")
    progress_tracker(location="sidebar")
    role_algo_banner("health")

    _role_algo, _role_algo_label, _ = get_role_algo("health")
    st.divider()

    st.markdown(
        "<div style='font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;'>👁️ Perspective View</div>",
        unsafe_allow_html=True,
    )
    view_mode = st.radio(
        t("perspective"),
        ["Industry", "Research", "Policy Brief"],
        horizontal=True,
        key="_vm_health",
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("🏥 Clinical Context & Data")

    data_source = st.selectbox(
        "Dataset Selection",
        list(_pipeline.available_datasets.keys()),
        format_func=lambda k: {
            "uci_heart": "📈 UCI Heart Disease",
            "pima_diabetes": "🩸 PIMA Diabetes (Women)",
            "breast_cancer": "🔬 Breast Cancer Wisconsin",
            "africa_centric": "🌍 Nigeria Health Cohort",
            "abuja_maternal": "🏥 Nigeria Maternal Health",
            "abuja_multilingual": "⚡ Nigeria Multilingual ECG",
            "abuja_insurance": "💳 Nigeria Insurance Access",
        }.get(k, k),
        key="cfg_health_data_source",
    )

    with st.expander("🩺 Care Setting & Regional Scope", expanded=False):
        healthcare_setting = st.selectbox(
            "Healthcare Setting",
            ["Primary Care", "Secondary Hospital", "Tertiary/Teaching Hospital", "Community Clinic", "Telemedicine"],
            key="cfg_health_setting",
        )
        prediction_task = st.selectbox(
            "Prediction Task",
            ["Disease Risk Screening", "Readmission Risk", "Diagnosis Support", "Treatment Recommendation", "Triage Priority"],
            key="cfg_health_prediction_task",
        )
        region = st.selectbox(
            "Target Region",
            ["Nigeria", "Lagos (Nigeria)", "Sub-Saharan Africa", "South Asia", "Global (Generic)"],
            key="cfg_health_region",
        )

    with st.expander("👥 Patient Population Demographics", expanded=False):
        low_income_ratio = st.slider("Low-Income Ratio", 0.0, 1.0, key="cfg_health_low_income_ratio", step=0.05)
        uninsured_ratio = st.slider("Uninsured Ratio", 0.0, 1.0, key="cfg_health_uninsured_ratio", step=0.05)

    st.divider()
    st.subheader(f"🎭 {t('bias_config')}")

    mitigation_strategy, active_threshold = render_mitigation_workbench()

    with st.expander("⚙️ Bias Vector Controls", expanded=False):
        _hc_valid = list(simulation_config.BIAS_TYPES) + [b for b in ["gender", "linguistic"] if b not in simulation_config.BIAS_TYPES]
        selected_biases = st.multiselect(
            "Active Bias Types",
            options=_hc_valid,
            key="cfg_health_selected_biases",
        )
        bias_intensity = st.slider("Bias Intensity", 0.0, float(simulation_config.MAX_BIAS_FACTOR), key="cfg_health_bias_intensity", step=0.05)
        access_inequality = st.slider("Access Inequality", 0.0, 1.0, key="cfg_health_access_inequality", step=0.05)

    with st.expander("⚔️ Attack & Sim Engine", expanded=False):
        poison_rate = st.slider("Data Poisoning Rate", 0.0, 0.5, key="cfg_health_poison_rate", step=0.01, format="%.2f")
        st.markdown("---")
        sample_size = st.number_input("Cohort Sample Size (N)", 500, 50000, key="cfg_health_sample_size", step=500)
        n_runs = st.slider("Simulation Runs", 1, 8, key="cfg_health_n_runs")

    st.divider()
    st.subheader("🔬 Article 1 Research Layer")
    with st.expander("Referral Geography & Joint Burden", expanded=False):
        article1_enabled = st.toggle(
            "Enable Article 1 referral-access experiment", value=False,
            key="cfg_health_article1_enabled"
        )
        article1_facility_upload = st.file_uploader(
            "GRID3 / health-facility workbook (XLSX/CSV)",
            type=["xlsx", "xls", "csv"], key="cfg_health_article1_facilities"
        )
        article1_routing_method = st.selectbox(
            "Routing method", ["geodesic_proxy", "osrm"], index=0,
            help="geodesic_proxy is for development/sensitivity only. Use OSRM for the planned road-network primary analysis.",
            key="cfg_health_article1_routing_method"
        )
        article1_travel_threshold = st.number_input(
            "Geographic-constraint threshold (minutes)", min_value=5.0, max_value=240.0,
            value=60.0, step=5.0, key="cfg_health_article1_travel_threshold"
        )
        article1_group_coupling = st.slider(
            "Simulated group–geography coupling", 0.0, 1.0, 0.0, 0.05,
            help="0 = independent geography. Values >0 intentionally simulate structural concentration of disadvantaged patients in poorer-access origins.",
            key="cfg_health_article1_group_coupling"
        )
        article1_speed = st.number_input(
            "Proxy speed (km/h; ignored by OSRM)", min_value=5.0, max_value=100.0,
            value=40.0, step=5.0, key="cfg_health_article1_speed"
        )
        article1_threshold_sensitivity_enabled = st.toggle(
            "Run travel-threshold sensitivity suite", value=True,
            key="cfg_health_article1_threshold_sensitivity_enabled",
            help="Reuses the same routed patient pathways and recomputes geographic constraint and joint burden at prespecified thresholds; it does not retrain or reroute."
        )
        article1_sensitivity_thresholds = st.multiselect(
            "Sensitivity thresholds (minutes)",
            options=[15, 30, 45, 60, 75, 90, 120],
            default=[30, 45, 60, 90],
            key="cfg_health_article1_sensitivity_thresholds"
        )
        article1_coupling_sensitivity_enabled = st.toggle(
            "Run group–geography coupling suite", value=True,
            key="cfg_health_article1_coupling_sensitivity_enabled",
            help="Holds the trained model and predictions fixed while rerunning only the simulated patient-location assignment across prespecified coupling values."
        )
        article1_coupling_values = st.multiselect(
            "Coupling values (λ)",
            options=[0.0, 0.25, 0.50, 0.75, 1.0],
            default=[0.0, 0.25, 0.50, 0.75, 1.0],
            key="cfg_health_article1_coupling_values"
        )
        st.markdown("**S0–S3 intervention counterfactuals**")
        article1_intervention_enabled = st.toggle(
            "Run S0–S3 intervention suite", value=True,
            key="cfg_health_article1_intervention_enabled",
            help="Compares baseline, algorithm-only, geography-only, and combined counterfactual interventions on the same patient pathways."
        )
        article1_ai_gap_closure = st.slider(
            "S1 AI intervention: fraction of subgroup FNR gap closed",
            min_value=0.0, max_value=1.0, value=1.0, step=0.10,
            key="cfg_health_article1_ai_gap_closure",
            help="Counterfactually repairs the highest-probability missed positives in the higher-FNR subgroup. This is not a deployed mitigation algorithm."
        )
        article1_geo_time_reduction = st.slider(
            "S2 geography intervention: referral travel-time reduction",
            min_value=0.0, max_value=0.75, value=0.25, step=0.05, format="%.0f%%",
            key="cfg_health_article1_geo_time_reduction",
            help="Applies a proportional system-level reduction to the same referral travel-time estimates. It is a scenario parameter, not an observed policy effect."
        )
        st.markdown("**Publication-grade repeated-seed runner**")
        article1_publication_mode = st.toggle(
            "Enable publication runner", value=False,
            key="cfg_health_article1_publication_mode",
            help="Uses repeated seeds, run-level bootstrap intervals, and performance-matched model comparisons. This can be computationally expensive."
        )
        article1_publication_runs = st.number_input(
            "Repeated seeds", min_value=5, max_value=100, value=30, step=5,
            key="cfg_health_article1_publication_runs"
        )
        article1_bootstrap_reps = st.number_input(
            "Run-level bootstrap replicates", min_value=100, max_value=10000, value=1000, step=100,
            key="cfg_health_article1_bootstrap_reps"
        )
        article1_model_variants = st.multiselect(
            "Model variants for performance matching",
            options=["none", "reweighing", "post_processing"],
            default=["none", "reweighing", "post_processing"],
            key="cfg_health_article1_model_variants",
            help="All variants use the same seed sequence and Article 1 geography assumptions."
        )
        article1_performance_tolerance = st.slider(
            "Performance-match tolerance (absolute)", min_value=0.005, max_value=0.10,
            value=0.03, step=0.005, format="%.3f",
            key="cfg_health_article1_performance_tolerance",
            help="A variant is performance-matched only if mean accuracy, AUROC and balanced accuracy are each within this tolerance of baseline."
        )
        st.caption("Article 1 uses a prespecified screen-to-referral policy: benchmark-positive patients are eligible for higher-level diagnostic assessment. This is not an emergency-triage rule or individualized clinical recommendation.")

    article1_facilities_df = pd.DataFrame()
    if article1_facility_upload is not None:
        try:
            if str(article1_facility_upload.name).lower().endswith((".xlsx", ".xls")):
                article1_facilities_df = pd.read_excel(article1_facility_upload)
            else:
                article1_facilities_df = pd.read_csv(article1_facility_upload)
        except Exception as exc:
            st.error(f"Article 1 facility dataset could not be loaded: {exc}")

    article1_config = Article1ReferralConfig(
        enabled=bool(article1_enabled),
        travel_threshold_min=float(article1_travel_threshold),
        routing_method=str(article1_routing_method),
        assumed_speed_kmh=float(article1_speed),
        group_location_coupling=float(article1_group_coupling),
    )

    st.divider()
    st.subheader("🧩 Analytical Modules")

    with st.expander("🛡️ Safety, Audit & Governance Toggles", expanded=False):
        enable_gender_audit = st.toggle("⚖️ Gender Equity Audit", key="cfg_health_enable_gender_audit")
        enable_governance = st.toggle("🏛️ Governance Layer", key="cfg_health_enable_governance")
        governance_policy = st.selectbox(
            "Governance Policy",
            ["majority_vote", "supermajority", "consensus", "weighted_expert"],
            disabled=not st.session_state.get("cfg_health_enable_governance"),
            key="cfg_health_governance_policy",)
        st.markdown("---")
        enable_redteam = st.toggle("🎯 Multimodal Red Team", key="cfg_health_enable_redteam")
        enable_ai_safety = st.toggle("🛡️ AI Safety Analysis", key="cfg_health_enable_ai_safety")
        enable_lifecycle = st.toggle("🔄 Lifecycle Management", key="cfg_health_enable_lifecycle")
        enable_eco = st.toggle("🌱 Eco Analysis", key="cfg_health_enable_eco")
        enable_dynamic = st.toggle("🔮 Dynamic Systems", key="cfg_health_enable_dynamic")
        enable_arena = st.toggle("⚔️ Strategic Arena", key="cfg_health_enable_arena")
        enable_agent_economy = st.toggle("💰 Agent Economy", key="cfg_health_enable_agent_economy")

    st.divider()
    _real_model_cfg = real_model_sidebar_controls()

    st.divider()
    col_r, col_x = st.columns([3, 2])
    run_button = col_r.button("Run Simulation", type="primary", use_container_width=True)

    if col_x.button(t("reset"), use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("health_") or k.startswith("cfg_health_"):
                del st.session_state[k]
        st.rerun()

    st.caption("🟢 Engine State: Active | Standard: NDPR / FMOH AI Governance Framework")
# ── Page Header ────────────────────────────────────────────────────────────────
st.markdown(
    """<div class="page-header" style="--ac:#0891b2;"><p style="font-family:'DM Mono',monospace;font-size:.69rem;letter-spacing:.16em;text-transform:uppercase;opacity:.5;margin:0 0 .55rem;display:flex;align-items:center;gap:.45rem;"><span style="width:16px;height:1px;background:#0891b2;opacity:.55;display:inline-block;"></span>HEALTHCARE · GAGS v1.0 · Nigeria</p><h1 style="font-family:'Syne',sans-serif!important;font-size:2.5rem!important;font-weight:800!important;line-height:1.08!important;letter-spacing:-.03em!important;margin:0 0 .6rem!important;">Healthcare Equity Simulation</h1><p style="margin:0;opacity:.72;font-size:.96rem;max-width:660px;line-height:1.65;">Test AI diagnostic bias across income, gender, and insurance status — calibrated to regional demographics and spatial health infrastructure.</p></div>""",
    unsafe_allow_html=True
)

# ── Main Simulation Execution Loop ─────────────────────────────────────────────
if run_button:
    effective_n_runs = int(article1_publication_runs) if (article1_config.enabled and article1_publication_mode) else int(n_runs)
    experiment_config = ExperimentConfig(
        data_source=data_source, n_samples=int(sample_size), n_runs=int(effective_n_runs),
        selected_biases=tuple(selected_biases), bias_intensity=float(bias_intensity),
        poison_rate=float(poison_rate), access_inequality=float(access_inequality),
        low_income_ratio=float(low_income_ratio), uninsured_ratio=float(uninsured_ratio),
        mitigation_strategy=str(mitigation_strategy), threshold=float(active_threshold),
        selected_state=str(selected_state), healthcare_setting=str(healthcare_setting),
        prediction_task=str(prediction_task), region=str(region), seed=42,
    )
    experiment_config.validate()
    st.session_state["health_experiment_config"] = asdict(experiment_config)
    st.session_state["health_experiment_config"]["article1_referral"] = asdict(article1_config)
    st.session_state["health_experiment_config"]["article1_threshold_sensitivity"] = {
        "enabled": bool(article1_threshold_sensitivity_enabled),
        "thresholds_min": [float(x) for x in article1_sensitivity_thresholds],
        "method": "post-routing threshold reclassification",
    }
    st.session_state["health_experiment_config"]["article1_coupling_sensitivity"] = {
        "enabled": bool(article1_coupling_sensitivity_enabled),
        "coupling_values": [float(x) for x in article1_coupling_values],
        "method": "fixed-model structural-location reassignment",
    }
    st.session_state["health_experiment_config"]["article1_interventions"] = {
        "enabled": bool(article1_intervention_enabled),
        "scenarios": ["S0 Baseline", "S1 AI improvement", "S2 Geography improvement", "S3 Combined improvement"],
        "ai_gap_closure": float(article1_ai_gap_closure),
        "geography_time_reduction": float(article1_geo_time_reduction),
        "method": "fixed-patient counterfactual intervention decomposition",
    }
    st.session_state["health_experiment_config"]["article1_publication_runner"] = {
        "enabled": bool(article1_publication_mode),
        "repeated_seeds": int(effective_n_runs),
        "bootstrap_reps": int(article1_bootstrap_reps),
        "bootstrap_unit": "run/seed",
        "model_variants": list(article1_model_variants),
        "performance_match_tolerance": float(article1_performance_tolerance),
        "matching_metrics": ["accuracy", "auc", "balanced_accuracy"],
    }
    article1_config.validate()
    if article1_config.enabled and article1_facilities_df.empty:
        raise ValueError("Article 1 is enabled. Upload the GRID3/facility dataset before running the simulation.")
    st.session_state["health_article1_patient_records"] = []
    st.session_state["health_article1_summary"] = {}
    st.session_state["health_article1_threshold_sensitivity"] = []
    st.session_state["health_article1_coupling_sensitivity"] = []
    st.session_state["health_article1_interventions"] = []
    st.session_state["health_article1_intervention_patients"] = []
    st.session_state["health_article1_publication_master"] = []
    st.session_state["health_article1_publication_ci"] = pd.DataFrame()
    st.session_state["health_article1_performance_match"] = pd.DataFrame()
    st.session_state.health_run_history = []
    st.session_state["health_feature_outputs"] = {}
    prog = st.progress(0, text="Executing Healthcare Simulation...")

    for i in range(effective_n_runs):
        prog.progress((i) / effective_n_runs, text=f"Run {i + 1} of {effective_n_runs}…")
        with st.spinner(f"Simulating Iteration {i + 1}/{effective_n_runs}"):
            result = _run_one(
                data_source=data_source,
                n_samples=sample_size,
                selected_biases=selected_biases,
                bias_intensity=bias_intensity,
                poison_rate=poison_rate,
                access_inequality=access_inequality,
                run_idx=i,
                enable_redteam=enable_redteam,
                enable_governance=enable_governance,
                governance_policy=governance_policy,
                enable_arena=enable_arena,
                enable_agent_economy=enable_agent_economy,
                enable_gender_audit=enable_gender_audit,
                selected_state=selected_state,
                low_income_ratio=low_income_ratio,
                uninsured_ratio=uninsured_ratio,
                mitigation_strategy=mitigation_strategy,
                threshold=active_threshold,
                experiment_seed=42,
                article1_config=article1_config,
                article1_facilities=article1_facilities_df,
            )
            st.session_state.health_run_history.append(result)
            if article1_publication_mode and article1_config.enabled:
                _primary_variant = str(mitigation_strategy)
                st.session_state["health_article1_publication_master"].extend(
                    _article1_tidy_from_result(result, _primary_variant)
                )
                # Fit additional mitigation variants with the same seed sequence.
                for _variant in [v for v in article1_model_variants if str(v) != _primary_variant]:
                    _variant_result = _run_one(
                        data_source=data_source, n_samples=sample_size,
                        selected_biases=selected_biases, bias_intensity=bias_intensity,
                        poison_rate=poison_rate, access_inequality=access_inequality,
                        run_idx=i, enable_redteam=False, enable_governance=False,
                        enable_arena=False, enable_agent_economy=False, enable_gender_audit=False,
                        selected_state=selected_state, low_income_ratio=low_income_ratio,
                        uninsured_ratio=uninsured_ratio, mitigation_strategy=str(_variant),
                        threshold=active_threshold, experiment_seed=42,
                        article1_config=article1_config, article1_facilities=article1_facilities_df,
                    )
                    st.session_state["health_article1_publication_master"].extend(
                        _article1_tidy_from_result(_variant_result, str(_variant))
                    )
            if result.get("article1_patient_records") is not None:
                _a1 = result["article1_patient_records"].copy()
                _a1["run_id"] = int(result.get("run_id", i + 1))
                _a1["seed"] = int(result.get("seed", 42 + i))
                st.session_state["health_article1_patient_records"].append(_a1)
                st.session_state["health_article1_summary"] = result.get("article1_summary", {})
                if article1_threshold_sensitivity_enabled:
                    for _thr in sorted(set(float(x) for x in article1_sensitivity_thresholds)):
                        _sens_rows = _summarize_article1_records_at_threshold(_a1, _thr)
                        for _sr in _sens_rows:
                            _sr["run_id"] = int(result.get("run_id", i + 1))
                            _sr["seed"] = int(result.get("seed", 42 + i))
                            _sr["routing_method"] = result.get("article1_summary", {}).get("routing_method_observed")
                        st.session_state["health_article1_threshold_sensitivity"].extend(_sens_rows)
                if article1_coupling_sensitivity_enabled and result.get("model_artifact") is not None:
                    for _lam in sorted(set(float(x) for x in article1_coupling_values)):
                        _coupling_cfg = replace(article1_config, group_location_coupling=float(_lam))
                        _c_summary, _c_records = _run_article1_joint_burden(
                            result["model_artifact"], article1_facilities_df,
                            _coupling_cfg, int(result.get("seed", 42 + i))
                        )
                        _c_rows = _summarize_article1_coupling_scenario(_c_summary, _c_records, _lam)
                        for _cr in _c_rows:
                            _cr["run_id"] = int(result.get("run_id", i + 1))
                            _cr["seed"] = int(result.get("seed", 42 + i))
                            _cr["accuracy"] = float(result.get("accuracy", np.nan))
                            _cr["auc"] = float(result.get("auc", np.nan)) if result.get("auc") is not None else np.nan
                        st.session_state["health_article1_coupling_sensitivity"].extend(_c_rows)
                if article1_intervention_enabled:
                    _i_rows, _i_patients = _run_article1_s0_s3_interventions(
                        _a1,
                        travel_threshold_min=float(article1_config.travel_threshold_min),
                        ai_gap_closure=float(article1_ai_gap_closure),
                        geography_time_reduction=float(article1_geo_time_reduction),
                    )
                    for _ir in _i_rows:
                        _ir["run_id"] = int(result.get("run_id", i + 1))
                        _ir["seed"] = int(result.get("seed", 42 + i))
                    _i_patients["run_id"] = int(result.get("run_id", i + 1))
                    _i_patients["seed"] = int(result.get("seed", 42 + i))
                    st.session_state["health_article1_interventions"].extend(_i_rows)
                    st.session_state["health_article1_intervention_patients"].append(_i_patients)
            result["experiment_id"] = hashlib.sha256(json.dumps({"data_source": data_source, "seed": result.get("seed"), "biases": selected_biases, "mitigation": mitigation_strategy, "threshold": active_threshold}, sort_keys=True, default=str).encode()).hexdigest()[:16]

            save_to_history(
                "health_snapshot_history",
                label=f"Run {i + 1} | bias={bias_intensity:.2f} | {data_source[:12]}",
                metrics={
                    "accuracy": result.get("accuracy", 0),
                    "equity_score": result.get("equity_score", 0),
                    "demographic_parity": result.get("demographic_parity", 0),
                    "sensitivity": result.get("sensitivity", 0),
                },
                config={
                    "data_source": data_source,
                    "bias_intensity": bias_intensity,
                    "poison_rate": poison_rate,
                    "selected_biases": selected_biases,
                },
            )

    if article1_publication_mode and article1_config.enabled and st.session_state.get("health_article1_publication_master"):
        _pub_master = pd.DataFrame(st.session_state["health_article1_publication_master"])
        st.session_state["health_article1_publication_ci"] = _article1_run_level_bootstrap_ci(
            _pub_master, n_boot=int(article1_bootstrap_reps), seed=20260912
        )
        _perf_source = _pub_master.drop_duplicates(subset=["seed", "model_variant"])[
            ["seed", "model_variant", "accuracy", "auc", "balanced_accuracy", "sensitivity", "specificity"]
        ]
        st.session_state["health_article1_performance_match"] = _article1_performance_match_table(
            _perf_source, tolerance=float(article1_performance_tolerance)
        )

    if any([_real_model_cfg["enable_datasets"], _real_model_cfg["enable_huggingface"], _real_model_cfg["enable_apis"]]):
        with st.spinner("Running real model comparison..."):
            st.session_state["health_real_models"] = run_real_model_comparison(
                n_samples=sample_size,
                enable_datasets=_real_model_cfg["enable_datasets"],
                enable_huggingface=_real_model_cfg["enable_huggingface"],
                enable_apis=_real_model_cfg["enable_apis"],
                dataset_sources=_real_model_cfg["dataset_sources"],
                hf_sources=_real_model_cfg["hf_sources"],
                api_sources=_real_model_cfg["api_sources"],
                model_type=_real_model_cfg["model_type"],
            )

    _fout = st.session_state["health_feature_outputs"]
    _last_run = st.session_state.health_run_history[-1]

    if enable_redteam:
        try:
            model_redteam = _run_health_model_redteam(_last_run["model_artifact"])
            _fout["model_redteam"] = model_redteam
        except Exception as exc:
            _fout["model_redteam"] = {"model_linked": False, "error": str(exc)}

    if enable_governance:
        equity = float(_last_run.get("equity_score", 0.0))
        fnr_gap = abs(float((_last_run.get("impact_matrix") or {}).get("fnr_disparity", 0.0)))
        accuracy = float(_last_run.get("accuracy", 0.0))
        # GAGS governance policy is an experimental decision rule, not a regulatory finding.
        rt = _fout.get("model_redteam", {})
        evidence = {"accuracy": accuracy, "equity_score": equity, "fnr_gap": fnr_gap,
                    "redteam_enabled": bool(enable_redteam),
                    "redteam_model_linked": bool(rt.get("model_linked", False)),
                    "mitigation": mitigation_strategy}
        if governance_policy == "consensus":
            outcome = "approved" if equity >= 0.80 and fnr_gap < 0.05 and accuracy >= 0.70 else "review_required"
        elif governance_policy == "supermajority":
            outcome = "approved" if equity >= 0.80 and fnr_gap < 0.05 and accuracy >= 0.70 else "review_required"
        elif governance_policy == "weighted_expert":
            outcome = "approved" if (0.5 * accuracy + 0.5 * equity) >= 0.80 and fnr_gap < 0.05 else "review_required"
        else:
            outcome = "approved" if equity >= 0.80 and fnr_gap < 0.05 and accuracy >= 0.70 else "review_required"
        vote_for = int(round(100 * (0.5 * accuracy + 0.5 * equity)))
        vote_for = max(0, min(100, vote_for))
        vote_against = 100 - vote_for
        ledger_payload = {"policy": governance_policy, "outcome": outcome, "evidence": evidence, "seed": _last_run.get("seed")}
        ledger_hash = hashlib.sha256(json.dumps(ledger_payload, sort_keys=True).encode()).hexdigest()[:16]
        _fout["governance"] = {
            "policy": governance_policy, "outcome": outcome,
            "tally": {"for": vote_for, "against": vote_against, "abstain": 0},
            "evidence": evidence, "ledger_hash": ledger_hash, "ledger_entries": len(st.session_state.health_run_history),
        }
    if enable_redteam:
        try:
            _fout["multimodal_redteam"] = run_redteam_simulation(domain="health")
        except Exception as exc:
            _fout["multimodal_redteam"] = {"error": str(exc)}
    if enable_agent_economy:
        _fout["agent_economy"] = run_agent_economy_simulation(domain="health")
    if enable_arena:
        _fout["arena"] = run_arena_simulation(domain="health")

    if enable_ai_safety:
        artifact = _last_run.get("model_artifact")
        if artifact is not None:
            st.session_state["health_safety_report"] = run_ai_safety_suite(
                X_train=artifact.X_train, y_train=artifact.y_train,
                X_test=artifact.X_test, y_test=artifact.y_test,
                model=artifact.model, domain="health",
                enable_robustness=True, enable_ood=True, enable_uncertainty=True, enable_checklists=True,
                simulation_metrics={"fairness_score": _last_run.get("fairness_score", 0.5),
                                    "robustness_score": _last_run.get("accuracy", 0.0)}
            )
        else:
            st.session_state["health_safety_report"] = {"error": "No trained healthcare model artifact available."}

    if enable_lifecycle or enable_eco:
        st.session_state["health_lifecycle_report"] = run_lifecycle_suite(
            domain="health", algo_key="rf", algo_label="Random Forest",
            n_samples=sample_size, n_runs=effective_n_runs, n_features=int(_last_run.get("n_features", 0)),
            metrics=_last_run, enable_registry=enable_lifecycle, enable_eco=enable_eco
        )

    if enable_dynamic:
        _ds_params = derive_ds_params(domain="health", run_history=st.session_state.health_run_history)
        st.session_state["health_dynamic_report"] = run_dynamic_systems_suite(
            domain="health", y_true=_ds_params["y_true"], y_pred=_ds_params["y_pred"],
            sensitive=_ds_params["sensitive"], bias_intensity=bias_intensity,
            governance_strength=0.8, regulatory_pressure=0.7, market_pressure=0.5, n_agents=100
        )

    prog.progress(1.0, text="Complete!")
    prog.empty()

# ── Results Dashboard ──────────────────────────────────────────────────────────
if st.session_state.health_run_history:
    df = pd.DataFrame(st.session_state.health_run_history)
    feats = st.session_state.get("health_feature_outputs", {})
    info = st.session_state.get("health_dataset_info", {})

    st.markdown("## 📊 Healthcare Equity Dashboard")
    role_banner("health")
    role_brief_banner("health")
    national_live_banner()

    avg_acc = df["accuracy"].mean()
    avg_equity = df["equity_score"].mean()
    avg_sens = df["sensitivity"].mean()
    avg_gap = (df["adv_hi"] - df["adv_lo"]).abs().mean()
    acc_std = float(df["accuracy"].std(ddof=1)) if len(df) > 1 else 0.0
    eq_std = float(df["equity_score"].std(ddof=1)) if len(df) > 1 else 0.0
    ci_mult = 1.96 / np.sqrt(len(df)) if len(df) > 1 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="med-card"><p style="margin:0;font-size:.8rem;color:#64748B;">🎯 Prediction Accuracy</p><p style="margin:0;font-size:1.8rem;font-weight:700;color:#0284C7;">{avg_acc:.1%}</p></div>',
            unsafe_allow_html=True)
    with k2:
        colour = "#16a34a" if avg_equity >= 0.80 else "#e67e22" if avg_equity >= 0.5 else "#ef4444"
        st.markdown(
            f'<div class="med-card"><p style="margin:0;font-size:.8rem;color:#64748B;">⚖️ Algorithmic Equity</p><p style="margin:0;font-size:1.8rem;font-weight:700;color:{colour};">{avg_equity:.2f}<span style="font-size:.9rem">/1.0</span></p></div>',
            unsafe_allow_html=True)
    with k3:
        st.markdown(
            f'<div class="med-card"><p style="margin:0;font-size:.8rem;color:#64748B;">🩺 Avg Sensitivity</p><p style="margin:0;font-size:1.8rem;font-weight:700;color:#7C3AED;">{avg_sens:.1%}</p></div>',
            unsafe_allow_html=True)
    with k4:
        gap_colour = "#ef4444" if avg_gap > 0.15 else "#e67e22" if avg_gap > 0.05 else "#16a34a"
        st.markdown(
            f'<div class="med-card"><p style="margin:0;font-size:.8rem;color:#64748B;">📈 Adverse Outcome Gap</p><p style="margin:0;font-size:1.8rem;font-weight:700;color:{gap_colour};">{avg_gap:.1%}</p></div>',
            unsafe_allow_html=True)

    if "governance" in feats:
        gov = feats["governance"]
        tally = gov.get("tally", {})
        st.info(
            f"🏛️ **Governance Consensus Vote:** {gov.get('policy')} — **{str(gov.get('outcome')).upper()}** (For: {tally.get('for', 0)} | Against: {tally.get('against', 0)}) | Hash: `{gov.get('ledger_hash')}`")

    st.caption(f"Reproducibility: {len(df)} run(s) · Accuracy mean ± approx. 95% CI: {avg_acc:.1%} ± {ci_mult*acc_std:.1%} · Equity mean ± approx. 95% CI: {avg_equity:.2f} ± {ci_mult*eq_std:.2f}")

    # ── Tab Navigation Setup ───────────────────────────────────────────────────
    _tab_labels = [
        "Performance", "Equity", "Clinical Impact","Dynamic Systems", "Feature Modules", "Explainable AI", "Compliance", "Longitudinal",
"Data Analysis","🔬 Article 1 — Referral Equity","Raw Results", "Case Study", "🔬 Real Models",
    ]

    _tabs_obj = st.tabs(_tab_labels)
    T = {name: tab_ref for name, tab_ref in zip(_tab_labels, _tabs_obj)}

    # ── 1. Performance Tab ─────────────────────────────────────────────────────
    with T["Performance"]:
        # ── 1. Execution State & Data Context ─────────────────────────────────────
        run_history = st.session_state.get("health_run_history", [])
        active_dataset = st.session_state.get("cfg_health_data_source", "Selected Dataset")
        active_task = st.session_state.get("cfg_health_prediction_task", "Clinical Risk")
        mitigation_mode = st.session_state.get("cfg_health_mitigation_strategy", "Baseline")

        st.subheader("🎯 Clinical Accuracy vs. Equity Trade-Off")
        st.caption(
            f"Evaluating Model Performance for **{active_task}** on **{active_dataset.upper()}** "
            f"| Active Strategy: **{mitigation_mode}**"
        )

        # ── 2. Build DataFrame from run history (with safe fallbacks) ──────────────
        if run_history:
            records = []
            for i, r in enumerate(run_history):
                record = {
                    "run_id": f"Run {i + 1}",
                    "accuracy": float(r.get("accuracy", 0.0)),
                    "sensitivity": float(r.get("sensitivity", r.get("recall", 0.0))),
                    "specificity": float(r.get("specificity", 0.0)),
                    "equity_score": float(r.get("equity_score", r.get("fairness_score", 0.0))),
                    "bias_intensity": float(r.get("bias_intensity", 0.0)),
                }
                # Capture any subgroup TPR / sensitivity dictionaries
                subgroup_tpr = r.get("subgroup_tpr") or r.get("group_tpr") or {}
                record["subgroup_tpr"] = subgroup_tpr
                records.append(record)
            df = pd.DataFrame(records)
        else:
            # Clearly labelled demo data
            st.info("ℹ️ Showing demo data — run a simulation to see live results.")
            df = pd.DataFrame({
                "run_id": ["Run 1", "Run 2", "Run 3"],
                "accuracy": [0.88, 0.84, 0.82],
                "sensitivity": [0.85, 0.81, 0.79],
                "specificity": [0.90, 0.87, 0.85],
                "equity_score": [0.65, 0.78, 0.88],
                "bias_intensity": [0.40, 0.25, 0.10],
                "subgroup_tpr": [
                    {"Female": 0.72, "Male": 0.92, "Rural": 0.68, "Urban": 0.89},
                    {"Female": 0.78, "Male": 0.88, "Rural": 0.74, "Urban": 0.86},
                    {"Female": 0.82, "Male": 0.85, "Rural": 0.79, "Urban": 0.84},
                ],
            })

        latest = df.iloc[-1]
        avg_acc = df["accuracy"].mean()
        avg_sens = df["sensitivity"].mean()
        avg_spec = df["specificity"].mean()
        avg_eq = df["equity_score"].mean()

        # ── 3. Top-Level Metric Summary ───────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Overall Accuracy",
            f"{latest['accuracy']:.1%}",
            delta=f"{(latest['accuracy'] - avg_acc):+.1%} vs Avg",
            help="Overall proportion of correct clinical classifications."
        )
        m2.metric(
            "Sensitivity (Recall)",
            f"{latest['sensitivity']:.1%}",
            delta=f"{(latest['sensitivity'] - avg_sens):+.1%} vs Avg",
            help="Proportion of actual positive cases correctly identified. Critical for avoiding missed diagnoses."
        )
        m3.metric(
            "Specificity",
            f"{latest['specificity']:.1%}",
            delta=f"{(latest['specificity'] - avg_spec):+.1%} vs Avg",
            help="Proportion of negative cases correctly ruled out."
        )
        m4.metric(
            "Equity Score",
            f"{latest['equity_score']:.1%}",
            delta=f"{(latest['equity_score'] - avg_eq):+.1%} vs Avg",
            delta_color="normal" if latest["equity_score"] >= 0.80 else "inverse",
            help="Fairness parity across demographic cohorts. The 80% value is a GAGS research target, not a universal regulatory threshold."
        )

        st.markdown("---")

        # ── 4. Pareto Chart + Dynamic Subgroup Breakdown ──────────────────────────
        col_chart, col_details = st.columns([3, 2])

        with col_chart:
            st.markdown("**Pareto Frontier: Accuracy vs. Equity Score**")
            fig_scatter = px.scatter(
                df,
                x="equity_score",
                y="accuracy",
                color="bias_intensity",
                size=[12] * len(df),
                hover_name="run_id",
                hover_data={
                    "sensitivity": ":.1%",
                    "specificity": ":.1%",
                    "bias_intensity": ":.2f",
                },
                color_continuous_scale="RdYlGn_r",
                labels={
                    "equity_score": "Equity Score (Fairness Parity)",
                    "accuracy": "Clinical Accuracy",
                    "bias_intensity": "Bias Intensity",
                },
            )
            fig_scatter.add_vline(
                x=0.80,
                line_dash="dash",
                line_color="#10B981",
                annotation_text="80% Parity Goal",
            )
            fig_scatter.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis=dict(tickformat=".0%", range=[0.40, 1.02]),
                yaxis=dict(tickformat=".0%", range=[0.50, 1.02]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_details:
            st.markdown("**Disaggregated Sensitivity by Subgroup**")
            st.caption("True Positive Rate (Sensitivity) across demographic cohorts")

            subgroup_tpr = latest.get("subgroup_tpr") or {}
            if not isinstance(subgroup_tpr, dict):
                subgroup_tpr = {}

            if not subgroup_tpr:
                st.info("No subgroup sensitivity data available for the latest run.")
            else:
                # Sort by sensitivity ascending so the worst-performing group appears first
                sorted_groups = sorted(subgroup_tpr.items(), key=lambda x: float(x[1]))

                values = [float(v) for _, v in sorted_groups]
                max_tpr = max(values) if values else 0.0
                min_tpr = min(values) if values else 0.0
                gap = max_tpr - min_tpr

                for group_name, tpr in sorted_groups:
                    tpr = float(tpr)
                    # Colour the progress bar text via caption
                    label = f"{group_name}: {tpr:.1%}"
                    st.progress(min(max(tpr, 0.0), 1.0), text=label)

                # Gap interpretation
                if gap > 0.10:
                    st.error(
                        f"⚠️ **High Clinical Disparity**: {gap:.1%} sensitivity gap between "
                        f"best and worst performing groups. Highest under-diagnosis risk in "
                        f"**{sorted_groups[0][0]}**."
                    )
                elif gap > 0.05:
                    st.warning(
                        f"⚡ **Moderate Disparity**: {gap:.1%} sensitivity gap observed "
                        f"(lowest in **{sorted_groups[0][0]}**)."
                    )
                else:
                    st.success(
                        f"✅ **Equitable Clinical Sensitivity**: "
                        f"Maximum gap is only {gap:.1%} across cohorts."
                    )

        # ── 5. Clinical Takeaway ──────────────────────────────────────────────────
        with st.expander("💡 Understanding Clinical Performance & Equity Trade-Offs", expanded=False):
            st.markdown(
                """
                * **Why Equity Score Matters**: High overall accuracy can mask dangerous 
                  under-performance in under-represented cohorts (e.g., rural, low-wealth, or female patients).
                * **The Pareto Goal**: Optimal clinical AI sits in the **top-right quadrant** 
                  (Accuracy ≥ 85% **and** Equity ≥ 80%).
                * **Mitigation Guidance**: If equity drops below 80%, change the **Mitigation Strategy** 
                  in the sidebar (Reweighing or post-processing threshold adjustment).
                * **Sensitivity Gaps**: A gap > 10% between groups usually indicates clinically 
                  meaningful under-diagnosis risk for the lower-performing cohort.
                """
            )
    # ── 2. Equity Tab (Guarded Safe Defaults + Extracted USSD Engine) ──────────
    with T["Equity"]:
        st.markdown(
            """
            <style>
            .eq-hero {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 1.25rem;
                color: #f8fafc;
            }
            .eq-badge {
                background: #0891b2;
                color: #ffffff;
                font-size: 0.72rem;
                font-weight: 700;
                padding: 3px 10px;
                border-radius: 12px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="eq-hero">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span class="eq-badge">Spatial & Access Intelligence Engine</span>
                    <span style="font-size: 0.8rem; color: #94a3b8;">WHO / OSRM Integrated Framework</span>
                </div>
                <h2 style="margin: 0; font-size: 1.6rem; font-weight: 800; color: #ffffff;">Healthcare Equity & Spatial Infrastructure Audit</h2>
                <p style="margin: 0.4rem 0 0 0; font-size: 0.88rem; color: #cbd5e1; max-width: 850px;">
                    Simulate real-world transfer barriers, severe weather penalties, and multi-agent clinical triage feasibility across Sub-Saharan infrastructure grids.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Top Control Bar: Global Audit Setup (Guarded defaults) ──
        with st.container(border=True):
            st.markdown("**🌐 Global Audit & Transport Calibration**")
            cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns([1.1, 1, 1, 1.1])

            with cfg_c1:
                location_mode = st.radio(
                    "Patient Origin Selector:",
                    ["Default Coordinates", "Select from Uploaded Data"],
                    horizontal=True,
                )
                p_lat, p_lon = 9.0820, 8.6753

            with cfg_c2:
                season_profile = st.selectbox(
                    "🌧️ Weather Profile:",
                    ["Dry Season", "Peak Rainy Season"],
                    help="Rainy season adds dynamic travel time penalties (+30% to +120%) due to unpaved road degradation.",
                    key="global_season_profile"
                )

            with cfg_c3:
                terrain_type = st.selectbox(
                    "🏞️ Terrain Profile:",
                    ["Urban Paved", "Rural Unpaved", "Off-Road Terrain"],
                    key="global_terrain_type"
                )
                avg_speed_kmh = (
                    50.0 if terrain_type == "Urban Paved"
                    else 30.0 if terrain_type == "Rural Unpaved"
                    else 15.0
                )

            with cfg_c4:
                llm_provider = st.selectbox(
                    "LLM Inference Engine:",
                    [
                        "Groq (Llama 3.3 / 3.1)",
                        "Hugging Face (Mistral)",
                        "OpenAI (GPT-4o-mini)",
                    ],
                    key="global_llm_provider"
                )
                api_key_input = st.text_input(
                    "API Key (or leave empty for Demo):",
                    value="",
                    type="password",
                    key="global_api_key_input"
                )

        # ── Studio Workspace Tabs ──
        equity_studio_tab, dataset_spatial_tab = st.tabs(
            ["🔬 Interactive Audit Workbench", "🗺️ Infrastructure Data & Spatial Map"]
        )

        # Safe Default Initialization for Active Data
        active_df = pd.DataFrame()

        # ── TAB A: Infrastructure & Spatial Map Data Studio ──
        with dataset_spatial_tab:
            st.markdown("**📂 Step 1: Upload & Schema Alignment**")
            with st.container(border=True):
                up_col1, up_col2 = st.columns(2)
                with up_col1:
                    facility_file = st.file_uploader(
                        "Upload Health Facilities (.xlsx, .csv)",
                        type=["xlsx", "xls", "csv"],
                        key="custom_fac_upload",
                    )
                    raw_fac_df = load_user_dataset(facility_file)
                with up_col2:
                    scenario_file = st.file_uploader(
                        "Upload Scenarios/AfriMed-QA (.csv, .xlsx)",
                        type=["csv", "xlsx", "xls"],
                        key="custom_scenario_upload",
                    )
                    raw_scenario_df = load_user_dataset(scenario_file)

                fac_mapping = {}
                if not raw_fac_df.empty:
                    fac_mapping = detect_or_select_columns(raw_fac_df, key_prefix="fac_schema")
                    active_df = raw_fac_df.copy()
                    active_df["lat"] = pd.to_numeric(active_df[fac_mapping["lat"]], errors="coerce")
                    active_df["lon"] = pd.to_numeric(active_df[fac_mapping["lon"]], errors="coerce")
                    active_df["facility_name"] = active_df[fac_mapping["name"]].astype(str)
                    active_df["facility_tier"] = (
                        active_df[fac_mapping["tier"]].astype(str)
                        if fac_mapping.get("tier")
                        else "Unclassified"
                    )
                    active_df["state"] = (
                        active_df[fac_mapping["state"]].astype(str)
                        if fac_mapping.get("state")
                        else "Unknown"
                    )
                    active_df["lga"] = (
                        active_df[fac_mapping["lga"]].astype(str)
                        if fac_mapping.get("lga")
                        else "Unknown"
                    )
                    active_df = active_df.dropna(subset=["lat", "lon"])

            if location_mode == "Select from Uploaded Data" and not active_df.empty:
                sel_fac = st.selectbox(
                    "Select Origin Facility Anchor:",
                    sorted(active_df["facility_name"].unique()),
                )
                match_row = active_df[active_df["facility_name"] == sel_fac].iloc[0]
                p_lat, p_lon = float(match_row["lat"]), float(match_row["lon"])

            st.markdown("**🗺️ Step 2: Geospatial Map & Infrastructure Filters**")
            if not active_df.empty:
                with st.container(border=True):
                    filter_c1, filter_c2, filter_c3 = st.columns(3)
                    filtered_df = active_df.copy()
                    with filter_c1:
                        states_list = sorted(active_df["state"].unique()) if "state" in active_df.columns else []
                        selected_states = st.multiselect("Filter by State/Region:", states_list,
                                                         key="filter_state_select")
                        if selected_states:
                            filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]
                    with filter_c2:
                        tiers_list = sorted(
                            filtered_df["facility_tier"].unique()) if "facility_tier" in filtered_df.columns else []
                        selected_tiers = st.multiselect("Filter by Facility Tier:", tiers_list,
                                                        key="filter_tier_select")
                        if selected_tiers:
                            filtered_df = filtered_df[filtered_df["facility_tier"].isin(selected_tiers)]
                    with filter_c3:
                        if "lga" in filtered_df.columns:
                            selected_lgas = st.multiselect("Filter by LGA:", sorted(filtered_df["lga"].unique()),
                                                           key="filter_lga_select")
                            if selected_lgas:
                                filtered_df = filtered_df[filtered_df["lga"].isin(selected_lgas)]

                    fig_map = px.scatter_mapbox(
                        filtered_df, lat="lat", lon="lon",
                        hover_name="facility_name", color="facility_tier",
                        zoom=5, mapbox_style="carto-positron", opacity=0.8, height=420,
                    )
                    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
                    st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info(
                    "💡 Upload health facility spatial datasets above to unlock interactive geospatial filtering and GIS mapping.")

        # ── TAB B: Interactive Audit Workbench ──
        with equity_studio_tab:
            tab1, tab2, tab3 = st.tabs([
                "👤 Single-Patient Live Audit",
                "📊 Batch Benchmark Audit",
                "📡 Infrastructure Mitigation & USSD",
            ])

            with tab1:
                col1, col2 = st.columns([1.1, 1], gap="medium")

                with col1:
                    with st.container(border=True):
                        st.markdown("**📋 Vignette & Clinical Profile Configuration**")

                        scenario_mode = st.radio(
                            "Select Scenario Source:",
                            ["⚡ Preset Vignettes", "🤖 AI Synthesizer", "✏️ Custom Manual Override"],
                            horizontal=True,
                            key="health_scenario_source_radio",
                        )

                        if scenario_mode == "⚡ Preset Vignettes":
                            sel_preset = st.selectbox("Select Preset Template:",
                                                      list(CLINICAL_VIGNETTES_PRESETS.keys()))
                            selected_scenario = sel_preset

                            if st.session_state.get(
                                    "last_preset") != sel_preset or "active_vignette_text" not in st.session_state:
                                st.session_state["active_vignette_text"] = CLINICAL_VIGNETTES_PRESETS[sel_preset][
                                    "text"]
                                st.session_state["last_preset"] = sel_preset

                        elif scenario_mode == "🤖 AI Synthesizer":
                            p_col1, p_col2, p_col3 = st.columns(3)
                            with p_col1:
                                g_spec = st.selectbox("Target Specialty:", CLINICAL_SPECIALTIES, key="syn_spec_select")
                            with p_col2:
                                g_cohort = st.selectbox(
                                    "Patient Cohort & Gender:",
                                    [
                                        "Maternal / Pregnant Female",
                                        "Pediatric (Under 5)",
                                        "Neonatal Emergency",
                                        "Adult Female (Reproductive Age)",
                                        "Adult Male (Trauma/Acute)",
                                        "Geriatric Patient (>65 yrs)",
                                    ],
                                    key="syn_cohort_select"
                                )
                            with p_col3:
                                g_complexity = st.selectbox("Clinical Urgency:", ["Standard Care", "High-Risk / Acute",
                                                                                  "Critical / Resuscitative"],
                                                            key="syn_comp_select")

                            p_col4, p_col5, p_col6 = st.columns(3)
                            with p_col4:
                                g_setting = st.selectbox("Geospatial Context:",
                                                         ["Rural Remote LGA", "Peri-Urban Informal Settlement",
                                                          "Riverine / Island Territory",
                                                          "Border / Conflict-Affected Zone"], key="syn_setting_select")
                            with p_col5:
                                g_care_point = st.selectbox("Initial Point of Care:",
                                                            ["Primary Health Centre (PHC)", "Community Health Post",
                                                             "Faith-Based / Mission Clinic",
                                                             "Traditional Birth Attendant (TBA) Home",
                                                             "Private Patent Medicine Vendor (PPMV)"],
                                                            key="syn_care_point_select")
                            with p_col6:
                                g_time_window = st.selectbox("Presentation Window:",
                                                             ["Daytime Regular Hours (08:00 - 16:00)",
                                                              "Evening Shift (16:00 - 00:00)",
                                                              "Nighttime Emergency (00:00 - 06:00)",
                                                              "Peak Weekend / Holiday"], key="syn_time_window_select")

                            with st.expander("Facility Deficits & Modifiers", expanded=False):
                                adv_col1, adv_col2 = st.columns(2)
                                with adv_col1:
                                    g_constraints = st.multiselect("Facility Deficits:", ["No Functional Blood Bank",
                                                                                          "No Emergency Surgical Theater",
                                                                                          "No On-Site Oxygen Concentrators",
                                                                                          "Lack of Specialist Medical Staff",
                                                                                          "Intermittent Solar Power Only",
                                                                                          "Out-of-Stock Essential Medications"],
                                                                   default=["No Functional Blood Bank"],
                                                                   key="syn_deficits_select")
                                    g_transport = st.multiselect("Transport Barriers:",
                                                                 ["Unpaved Dirt Tracks / Mud Degradation",
                                                                  "Seasonal River Flooding / Boat Required",
                                                                  "No Local Ambulance Service Available",
                                                                  "Curfew / Nighttime Travel Insecurity"],
                                                                 default=["Unpaved Dirt Tracks / Mud Degradation"],
                                                                 key="syn_transport_select")
                                with adv_col2:
                                    g_vulnerabilities = st.multiselect("Patient Vulnerabilities:",
                                                                       ["Low Household Income / Out-of-Pocket Payment",
                                                                        "Uninsured / Non-NHIA Enrollee",
                                                                        "Severe Pre-existing Anemia / Malnutrition",
                                                                        "Language / Ethnic Minority Barrier",
                                                                        "No Male Escort (Sociocultural Constraint)"],
                                                                       default=[
                                                                           "Low Household Income / Out-of-Pocket Payment"],
                                                                       key="syn_vuln_select")

                            selected_scenario = f"AI Vignette: {g_spec} ({g_cohort} at {g_setting})"

                            if st.button("✨ Synthesize Dynamic Vignette", use_container_width=True):
                                with st.spinner(f"Synthesizing clinical case via {llm_provider}..."):
                                    generated_text = generate_llm_vignette(
                                        specialty=g_spec,
                                        provider=llm_provider,
                                        api_key=api_key_input,
                                        complexity=g_complexity,
                                        geospatial_setting=g_setting,
                                        patient_cohort=g_cohort,
                                        care_point=g_care_point,
                                        time_window=g_time_window,
                                        facility_constraints=g_constraints,
                                        patient_vulnerabilities=g_vulnerabilities,
                                        transport_barriers=g_transport,
                                    )
                                    st.session_state["active_vignette_text"] = generated_text
                                    st.success("Case presentation synthesized!")

                        else:
                            selected_scenario = st.text_input("Custom Scenario Name:",
                                                              value="Custom Healthcare Scenario",
                                                              key="custom_scenario_name_input")

                        st.session_state["selected_scenario"] = selected_scenario

                        active_vignette = st.text_area(
                            "Active Vignette Narrative:",
                            value=st.session_state.get("active_vignette_text", ""),
                            height=130,
                        )
                        st.session_state["active_vignette_text"] = active_vignette

                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            run_agent_audit = st.button("🤖 Multi-Agent Audit", type="primary", use_container_width=True)
                        with btn_col2:
                            run_spatial_audit = st.button("⚡ Spatial & Clinical Audit", use_container_width=True)

                        if run_agent_audit:
                            if not active_vignette.strip():
                                st.warning("Please enter or synthesize a vignette first.")
                            else:
                                with st.spinner(f"Running multi-agent pipeline ({season_profile} / {terrain_type})..."):
                                    pipeline_results = run_multi_agent_triage_pipeline(
                                        vignette_text=active_vignette,
                                        facilities_df=active_df,
                                        patient_lat=p_lat,
                                        patient_lon=p_lon,
                                        api_key=api_key_input,
                                        provider=llm_provider,
                                        season_profile=season_profile,
                                        terrain_type=terrain_type,
                                    )
                                    st.session_state["active_llm_out"] = pipeline_results["clinical_assessment"]
                                    st.session_state["active_nearest_facs"] = pipeline_results[
                                        "matched_facilities_df"].head(5)
                                    st.session_state["ussd_dispatch"] = pipeline_results["ussd_dispatch_payload"]
                                    st.session_state["health_spatial_report"] = pipeline_results
                                st.success("Audit complete!")

                        if run_spatial_audit:
                            if not active_vignette.strip():
                                st.warning("Please enter or synthesize a vignette first.")
                            else:
                                with st.spinner("Analyzing spatial routing & clinical pathway..."):
                                    llm_out = generate_clinical_recommendation(
                                        vignette_text=active_vignette,
                                        api_key=api_key_input,
                                        provider=llm_provider,
                                    )
                                    st.session_state["active_llm_out"] = llm_out

                                    if not active_df.empty:
                                        # Consistently use calculate_facility_access or OSRM route helper
                                        access_results = calculate_facility_access(
                                            df_facilities=active_df,
                                            patient_lat=p_lat,
                                            patient_lon=p_lon,
                                            season_profile=season_profile,
                                            terrain_type=terrain_type,
                                        )
                                        st.session_state["active_nearest_facs"] = access_results.head(5)

                                    st.session_state["health_spatial_report"] = {
                                        "scenario": selected_scenario,
                                        "vignette": active_vignette,
                                        "llm_recommendation": llm_out,
                                    }
                                st.success("Spatial road-network audit completed!")

                # Live Results Column
                with col2:
                    st.markdown("**🩺 Audit Intelligence & Diagnostic Feasibility**")
                    llm_out_data = st.session_state.get("active_llm_out")

                    if llm_out_data:
                        with st.container(border=True):
                            if isinstance(llm_out_data, dict):
                                m_c1, m_c2 = st.columns(2)
                                with m_c1:
                                    urgency_val = llm_out_data.get('urgency', 'CRITICAL')
                                    st.metric("Clinical Urgency", urgency_val)
                                with m_c2:
                                    st.metric("Target Facility Tier", llm_out_data.get('required_tier', 'Tertiary'))

                                st.markdown("---")
                                st.markdown(f"**Summary:** {llm_out_data.get('summary', '')}")

                                if llm_out_data.get("risk_factors"):
                                    st.warning("**Risk Factors:** " + ", ".join(llm_out_data.get("risk_factors", [])))
                                if llm_out_data.get("action_plan"):
                                    st.success(f"**Action Plan:** {llm_out_data.get('action_plan')}")
                            else:
                                st.info(f"**Recommendation:**\n\n{llm_out_data}")

                    nearest_df = st.session_state.get("active_nearest_facs")
                    if isinstance(nearest_df, pd.DataFrame) and not nearest_df.empty:
                        with st.container(border=True):
                            st.markdown("**🏥 Matched Referral Facilities & Road Access Metrics**")
                            target_cols = ["facility_name", "facility_tier", "distance_km", "travel_time_min"]
                            available_cols = [c for c in target_cols if c in nearest_df.columns]
                            st.dataframe(
                                nearest_df[available_cols] if available_cols else nearest_df,
                                use_container_width=True,
                                height=200,
                            )
                    elif nearest_df is not None:
                        st.info("💡 Load health facilities to calculate live road network travel times.")

                    if st.session_state.get("ussd_dispatch"):
                        with st.container(border=True):
                            st.markdown("**📱 USSD / SMS Low-Bandwidth Dispatch Payload**")
                            st.code(st.session_state["ussd_dispatch"], language="text")

            with tab2:
                st.markdown("#### 📊 Batch Benchmark Analysis")
                st.caption("Batch evaluates datasets for spatial and demographic under-diagnosis risks.")
                if "df_afrimed" in locals() and not df_afrimed.empty:
                    st.dataframe(df_afrimed.head(10), use_container_width=True)

            with tab3:
                # Render Extracted USSD & Low-Bandwidth Component
                render_ussd_triage_section(
                    active_df=active_df,
                    p_lat=p_lat,
                    p_lon=p_lon,
                    season_profile=season_profile,
                    terrain_type=terrain_type,
                    llm_provider=llm_provider,
                    api_key_input=api_key_input,
                )

        # ── Downstream Components & Report Generators ──
        st.divider()
        lifecycle_report = st.session_state.get("health_lifecycle_report", {})
        if isinstance(lifecycle_report, str):
            try:
                lifecycle_report = json.loads(lifecycle_report)
            except Exception:
                lifecycle_report = {"error": lifecycle_report}

        latest_results = st.session_state.get("health_dynamic_report", {})
        if isinstance(latest_results, str):
            try:
                latest_results = json.loads(latest_results)
            except Exception:
                latest_results = {"summary": latest_results}

    # ── 3. Clinical Impact Tab ─────────────────────────────────────────────────
    with T["Clinical Impact"]:
        st.subheader("Clinical Impact & Outcome Safety")
        st.caption(
            "Under-diagnosis risk, diagnostic error distribution, and NDPR / FMOH compliance."
        )


        # ── 1. Resolve the real impact matrix (never a raw run dict) ─────────────
        def _is_impact_matrix(obj) -> bool:
            return (
                    isinstance(obj, dict)
                    and ("disadvantaged" in obj or "fnr_disparity" in obj)
            )


        impact_matrix = st.session_state.get("latest_impact_matrix")
        if not _is_impact_matrix(impact_matrix):
            history = st.session_state.get("health_run_history") or []
            last = history[-1] if history and isinstance(history[-1], dict) else {}
            impact_matrix = last.get("impact_matrix") if _is_impact_matrix(last.get("impact_matrix")) else None

        if not _is_impact_matrix(impact_matrix):
            st.info(
                "No clinical impact data yet.\n\n"
                "Configure the cohort in the sidebar and click **Run Simulation** "
                "to populate under-diagnosis analytics."
            )
        else:
            disadv = impact_matrix.get("disadvantaged") or {}
            priv = impact_matrix.get("privileged") or {}


            def _f(d, k, default=0.0):
                v = d.get(k, default)
                try:
                    v = float(v)
                    return default if (v != v) else v  # NaN guard
                except (TypeError, ValueError):
                    return default


            def _i(d, k, default=0):
                try:
                    return int(d.get(k, default) or default)
                except (TypeError, ValueError):
                    return default


            fnr_dis = _f(disadv, "FNR")
            fnr_priv = _f(priv, "FNR")
            tpr_dis = _f(disadv, "TPR")
            tpr_priv = _f(priv, "TPR")
            fpr_dis = _f(disadv, "FPR")
            fpr_priv = _f(priv, "FPR")
            n_dis = _i(disadv, "Count")
            n_priv = _i(priv, "Count")

            fnr_gap = _f(impact_matrix, "fnr_disparity", fnr_dis - fnr_priv)
            eq_opp = _f(impact_matrix, "equal_opportunity_ratio")
            if eq_opp == 0.0 and tpr_priv > 0:
                eq_opp = tpr_dis / tpr_priv

            # Absolute missed cases (more honest than rates alone)
            fn_dis = int(round(fnr_dis * n_dis))
            fn_priv = int(round(fnr_priv * n_priv))

            is_compliant = bool(
                impact_matrix.get("is_ndpr_compliant",
                                  impact_matrix.get("is_fmoh_compliant",
                                                    impact_matrix.get("fnr_gap_within_threshold",
                                                                      impact_matrix.get("is_fda_compliant",
                                                                                        abs(fnr_gap) < 0.05))))
            )

            # ── 2. Headline clinical KPIs (derived, no invented deltas) ──────────
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Under-Diagnosis Gap (FNR)",
                f"{fnr_gap:+.1%}",
                delta="Within 5pp" if abs(fnr_gap) < 0.05 else "Exceeds 5pp limit",
                delta_color="normal" if abs(fnr_gap) < 0.05 else "inverse",
                help="Disadvantaged FNR − Privileged FNR. Positive = more missed disease in underserved patients.",
            )
            k2.metric(
                "Missed Cases (Underserved)",
                f"{fn_dis}",
                delta=f"{fnr_dis:.1%} FNR  ·  n={n_dis}",
                delta_color="inverse" if fnr_dis > fnr_priv else "off",
                help="Estimated false negatives in the underserved cohort (rate × sample size).",
            )
            k3.metric(
                "Equal Opportunity Ratio",
                f"{eq_opp:.2f}",
                delta="Fair (0.80–1.25)" if 0.80 <= eq_opp <= 1.25 else "Outside fairness band",
                delta_color="normal" if 0.80 <= eq_opp <= 1.25 else "inverse",
                help="TPR_underserved / TPR_privileged. 1.0 = equal sensitivity.",
            )
            k4.metric(
                "NDPR / FMOH Audit",
                "COMPLIANT" if is_compliant else "ACTION REQUIRED",
                help="Passes when |FNR gap| < 5 percentage points.",
            )

            if abs(fnr_gap) >= 0.10:
                st.error(
                    f"High under-diagnosis risk: underserved FNR is {fnr_dis:.1%} vs "
                    f"{fnr_priv:.1%} in the privileged cohort ({fnr_gap:+.1%} gap)."
                )
            elif abs(fnr_gap) >= 0.05:
                st.warning(
                    f"Elevated FNR gap of {fnr_gap:+.1%}. Review reweighing or group-specific thresholds."
                )
            else:
                st.success("Under-diagnosis disparity is within the 5pp NDPR / FMOH band.")
            with st.container(border=True):
                render_medical_compliance_panel(impact_matrix, show_metrics=False)
            st.markdown("---")

            # ── 3. Side-by-side cohort rates ─────────────────────────────────────
            left, right = st.columns(2)
            with left:
                st.markdown("**Underserved cohort**")
                st.caption(f"n = {n_dis:,}")
                st.progress(min(max(fnr_dis, 0.0), 1.0), text=f"FNR (missed disease)  {fnr_dis:.1%}")
                st.progress(min(max(tpr_dis, 0.0), 1.0), text=f"TPR (sensitivity)     {tpr_dis:.1%}")
                st.progress(min(max(fpr_dis, 0.0), 1.0), text=f"FPR (over-diagnosis)  {fpr_dis:.1%}")
            with right:
                st.markdown("**Privileged cohort**")
                st.caption(f"n = {n_priv:,}")
                st.progress(min(max(fnr_priv, 0.0), 1.0), text=f"FNR (missed disease)  {fnr_priv:.1%}")
                st.progress(min(max(tpr_priv, 0.0), 1.0), text=f"TPR (sensitivity)     {tpr_priv:.1%}")
                st.progress(min(max(fpr_priv, 0.0), 1.0), text=f"FPR (over-diagnosis)  {fpr_priv:.1%}")

            # ── 5. Referral / diagnosis flow ─────────────────────────────────────
            if CHARTS_OK:
                with st.expander("Patient triage & under-diagnosis flow", expanded=True):
                    try:
                        sankey_fig = generate_clinical_sankey(impact_matrix)
                        if sankey_fig and getattr(sankey_fig, "data", None):
                            st.plotly_chart(sankey_fig, use_container_width=True)
                        else:
                            st.warning("Insufficient flow data to construct the Sankey diagram.")
                    except Exception as e:
                        st.error(f"Error rendering Sankey flow chart: {e}")

            # ── 6. Optional: spatial audit coupling ──────────────────────────────
            spatial = st.session_state.get("health_spatial_report") or {}
            if isinstance(spatial, dict) and spatial.get("matched_facilities_df") is not None:
                with st.expander("Infrastructure feasibility (latest spatial audit)", expanded=False):
                    facs = spatial.get("matched_facilities_df")
                    if isinstance(facs, pd.DataFrame) and not facs.empty:
                        cols = [c for c in ["facility_name", "facility_tier", "distance_km", "travel_time_min"] if
                                c in facs.columns]
                        st.dataframe(facs[cols].head(5) if cols else facs.head(5), use_container_width=True)
                    rec = spatial.get("clinical_assessment") or spatial.get("llm_recommendation")
                    if rec:
                        st.caption(str(rec)[:500])

            with st.expander("How to read these numbers", expanded=False):
                st.markdown(
                    """
                    - **FNR (False Negative Rate)** is the share of true disease that the model missed.
                      In screening, this is the clinical harm metric that matters most.
                    - **FNR gap > 0** means underserved patients are missed more often than privileged patients.
                    - **Rates without counts are misleading** — a 20% FNR on 8 patients is not the same
                      as 20% on 800. Absolute missed-case counts are shown above.
                    - **Equal opportunity ratio** near 1.0 means similar sensitivity across groups.
                    - Mitigation: raise recall for the underserved group via reweighing or a
                      group-specific decision threshold (sidebar).
                    """
                )

    # ── 4. Feature Modules Tab ────────────────────────────────────────────────
    with T["Feature Modules"]:
        st.header("🧩 Feature & Analytical Module Diagnostics")
        st.caption("Real-time results and evaluations based on active sidebar modules.")

        history = st.session_state.get("health_run_history", [])
        any_active = False

        # 1. ⚖️ Gender Equity Audit Results
        if st.session_state.get("cfg_health_enable_gender_audit", False):
            any_active = True
            with st.expander(" Dynamic Gender Equity Audit Diagnostics", expanded=True):
                history = st.session_state.get("health_run_history", [])
                latest_run = history[-1] if history else {}
                render_gender_equity_audit(latest_run)

        # 2. 🏛️ Governance Layer Results
        if st.session_state.get("cfg_health_enable_governance", False):
            any_active = True
            with st.expander("🏛️ AI Governance & Decision Arbitration", expanded=True):
                policy = st.session_state.get("cfg_health_governance_policy", "majority_vote")
                st.success(f"🟢 **Enforced Governance Policy:** `{policy.upper()}`")
                st.json({
                    "policy_mode": policy,
                    "framework_standard": "NDPR / FMOH AI Governance Framework",
                    "consensus_threshold": "66%" if policy == "supermajority" else "50%",
                    "status": "Active & Monitored"
                })

        # 3. 🛡️ AI Safety Analysis (Nested Component)
        if st.session_state.get("cfg_health_enable_ai_safety", False):
            any_active = True
            with st.expander("🛡️ AI Safety & Robustness Report", expanded=True):
                # Calls your existing renderer directly inside the expander
                render_safety_tab(st.session_state.get("health_safety_report", {}))

        # 4. 🌱 Eco Analysis Results
        if st.session_state.get("cfg_health_enable_eco", False):
            any_active = True
            with st.expander("🌱 Eco Analysis & Compute Efficiency", expanded=True):
                eco_data = st.session_state.get("health_lifecycle_report", {}) or {}
                if isinstance(eco_data, str):
                    try:
                        eco_data = json.loads(eco_data)
                    except Exception:
                       eco_data = {}

                last_run = (st.session_state.get("health_run_history") or [{}])[-1]
                render_eco_tab(
                    eco_data,
                    algo_key=last_run.get("algorithm", "random_forest"),
                    n_samples=int(last_run.get("n_samples", 2000)),
                    n_runs=int(last_run.get("n_runs", len(st.session_state.get("health_run_history", [])) or 1)),
                )

        # 5. 🔄 Lifecycle Management Results
        if st.session_state.get("cfg_health_enable_lifecycle", False):
            any_active = True
            with st.expander("🔄 Lifecycle & Drift Monitoring", expanded=True):
                lifecycle_data = st.session_state.get("health_lifecycle_report", {})
                if isinstance(lifecycle_data, str):
                    try:
                        lifecycle_data = json.loads(lifecycle_data)
                    except Exception:
                        lifecycle_data = {"error": lifecycle_data}

                render_lifecycle_tab("healthcare", lifecycle_data)

        # Fallback when all toggles are disabled in the sidebar
        if not any_active:
            st.info("💡 Enable toggles under **'Analytical Modules'** in the sidebar to view detailed diagnostics here.")
    # ── 5. Data Analysis Tab ──────────────────────────────────────────────────
    with T["Data Analysis"]:
        st.markdown("### Dataset & Simulation Run History")
        st.dataframe(df, use_container_width=True)

    # ── 6. Explainable AI Tab ─────────────────────────────────────────────────
    with T["Explainable AI"]:
        st.markdown("### Model Explainability (SHAP & Counterfactuals)")
        xai_res = st.session_state.get("health_xai_results", {})
        if "feature_importance" in xai_res:
            st.json(xai_res["feature_importance"])
        else:
            st.info("Run a simulation pass to generate feature importance and explanations.")

    # ── 7. Compliance Tab ─────────────────────────────────────────────────────
    with T["Compliance"]:
        impact_matrix = st.session_state.get("latest_impact_matrix")
        render_medical_compliance_panel(impact_matrix, show_metrics=True)
        st.divider()
        nigeria_compliance_panel(st.session_state.get("health_feature_outputs", {}))

    # ── 8. Longitudinal Tab ───────────────────────────────────────────────────
    with T["Longitudinal"]:
        st.markdown("### Longitudinal Bias Drift Over Time")
        if st.session_state.get("health_longitudinal"):
            st.json(st.session_state["health_longitudinal"])

    # ── 9. Federated Tab ──────────────────────────────────────────────────────
    #with T["Federated"]:
     #   st.markdown("### Federated Learning & Heterogeneity")
      #  if st.session_state.get("health_federated"):
       #     st.json(st.session_state["health_federated"])

    # ── Article 1 Referral Equity Research Dashboard ─────────────────────────────
    with T["🔬 Article 1 — Referral Equity"]:
        st.markdown("### 🔬 Article 1 — Referral Equity")
        st.caption("Patient-level coupling of AI missed referral opportunities with geographic referral constraints. Research simulation only; not an emergency-triage or individualized clinical recommendation.")

        _a1_frames = st.session_state.get("health_article1_patient_records", [])
        _a1_summary = st.session_state.get("health_article1_summary", {})
        if not _a1_frames or not _a1_summary:
            st.info("Enable the Article 1 referral-access experiment in the sidebar, upload the GRID3 facility workbook, and run the simulation to populate this dashboard.")
        else:
            _a1_all = pd.concat(_a1_frames, ignore_index=True)
            _eligible = _a1_all[_a1_all["referral_eligible_r"] == 1].copy()

            # Aggregate directly from all patient-level records across repeated runs.
            _rows = []
            for _g, _d in _eligible.groupby("group", dropna=False):
                _fnr = float(_d["missed_referral_opportunity_m"].mean()) if len(_d) else np.nan
                _cr = float(_d["geographic_constraint_c"].mean()) if len(_d) else np.nan
                _jb = float(_d["joint_burden_j"].mean()) if len(_d) else np.nan
                _ind = _fnr * _cr
                _c1 = _d[_d["geographic_constraint_c"] == 1]
                _c0 = _d[_d["geographic_constraint_c"] == 0]
                _dep = (float(_c1["missed_referral_opportunity_m"].mean()) - float(_c0["missed_referral_opportunity_m"].mean())) if (len(_c1) and len(_c0)) else np.nan
                _label = "Disadvantaged" if str(_g) in {"0", "0.0"} else ("Advantaged" if str(_g) in {"1", "1.0"} else str(_g))
                _rows.append({"Group": _label, "Raw group": _g, "N referral-eligible": int(len(_d)), "FNR / missed referral": _fnr, "Geographic constraint": _cr, "Joint burden J": _jb, "Independence J_ind": _ind, "Excess E": _jb-_ind, "Dependence D": _dep})
            _metric_df = pd.DataFrame(_rows)

            _dis = _metric_df[_metric_df["Group"] == "Disadvantaged"]
            _adv = _metric_df[_metric_df["Group"] == "Advantaged"]
            def _gap(col):
                if len(_dis) and len(_adv):
                    return float(_dis.iloc[0][col] - _adv.iloc[0][col])
                return np.nan

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Referral-eligible records", f"{len(_eligible):,}")
            k2.metric("ΔFNR (Disadv. − Adv.)", "N/A" if np.isnan(_gap("FNR / missed referral")) else f"{_gap('FNR / missed referral'):+.1%}")
            k3.metric("Δ Geographic Constraint", "N/A" if np.isnan(_gap("Geographic constraint")) else f"{_gap('Geographic constraint'):+.1%}")
            k4.metric("Δ Joint Burden", "N/A" if np.isnan(_gap("Joint burden J")) else f"{_gap('Joint burden J'):+.1%}")

            st.markdown("#### Primary Article 1 estimands")
            _display_metrics = _metric_df.copy()
            for _c in ["FNR / missed referral", "Geographic constraint", "Joint burden J", "Independence J_ind", "Excess E", "Dependence D"]:
                _display_metrics[_c] = _display_metrics[_c].map(lambda x: "N/A" if pd.isna(x) else f"{x:.3f}")
            st.dataframe(_display_metrics.drop(columns=["Raw group"], errors="ignore"), use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                _plot = _metric_df.melt(id_vars=["Group"], value_vars=["FNR / missed referral", "Geographic constraint", "Joint burden J"], var_name="Measure", value_name="Rate")
                fig = px.bar(_plot, x="Group", y="Rate", barmode="group", facet_col="Measure", title="Algorithmic Miss, Geographic Constraint, and Joint Burden")
                fig.update_yaxes(tickformat=".0%", range=[0, max(1.0, float(_plot["Rate"].max()) * 1.15 if len(_plot) else 1.0)])
                fig.update_layout(showlegend=False, height=430)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                _decomp = _metric_df.melt(id_vars=["Group"], value_vars=["Joint burden J", "Independence J_ind", "Excess E"], var_name="Component", value_name="Value")
                fig2 = px.bar(_decomp, x="Group", y="Value", color="Component", barmode="group", title="Joint-Burden Decomposition")
                fig2.update_yaxes(tickformat=".0%")
                fig2.update_layout(height=430)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### Travel-threshold sensitivity")
            _sens_records = st.session_state.get("health_article1_threshold_sensitivity", [])
            if _sens_records:
                _sens = pd.DataFrame(_sens_records)
                _sens["Group label"] = _sens["group"].map(lambda g: "All referral-eligible" if str(g) == "ALL" else ("Disadvantaged" if str(g) in {"0", "0.0"} else ("Advantaged" if str(g) in {"1", "1.0"} else str(g))))
                _sens_agg = _sens.groupby(["threshold_min", "Group label"], as_index=False).agg(
                    runs=("run_id", "nunique"),
                    n_referral_eligible=("n_referral_eligible", "sum"),
                    fnr_missed_referral=("fnr_missed_referral", "mean"),
                    geographic_constraint=("geographic_constraint", "mean"),
                    joint_burden_j=("joint_burden_j", "mean"),
                    independence_j_ind=("independence_j_ind", "mean"),
                    excess_e=("excess_e", "mean"),
                    dependence_d=("dependence_d", "mean"),
                )
                _sens_groups = _sens_agg[_sens_agg["Group label"] != "All referral-eligible"].copy()
                if len(_sens_groups):
                    fig_s1 = px.line(
                        _sens_groups, x="threshold_min", y="joint_burden_j", color="Group label", markers=True,
                        title="Joint Burden Across Geographic-Constraint Thresholds",
                        labels={"threshold_min":"Travel-time threshold (minutes)", "joint_burden_j":"Joint burden J"}
                    )
                    fig_s1.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig_s1, use_container_width=True)

                _overall_sens = _sens_agg[_sens_agg["Group label"] == "All referral-eligible"].copy()
                if len(_overall_sens):
                    _long = _overall_sens.melt(
                        id_vars=["threshold_min"],
                        value_vars=["geographic_constraint", "joint_burden_j", "independence_j_ind", "excess_e"],
                        var_name="Measure", value_name="Rate"
                    )
                    fig_s2 = px.line(
                        _long, x="threshold_min", y="Rate", color="Measure", markers=True,
                        title="Overall Threshold Sensitivity: Constraint, Joint Burden, and Decomposition",
                        labels={"threshold_min":"Travel-time threshold (minutes)"}
                    )
                    fig_s2.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig_s2, use_container_width=True)

                _sens_display = _sens_agg.copy()
                for _c in ["fnr_missed_referral", "geographic_constraint", "joint_burden_j", "independence_j_ind", "excess_e", "dependence_d"]:
                    _sens_display[_c] = _sens_display[_c].map(lambda x: "N/A" if pd.isna(x) else f"{x:.3f}")
                st.dataframe(_sens_display, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Article 1 Threshold-Sensitivity Results (CSV)",
                    _sens.to_csv(index=False),
                    file_name="article1_threshold_sensitivity.csv", mime="text/csv", key="a1_sensitivity_csv"
                )
                st.caption("Sensitivity rows reuse the same routed pathways and change only the travel-time threshold defining C. This isolates threshold-definition sensitivity from model retraining and routing variation.")
            else:
                st.info("Enable the travel-threshold sensitivity suite in the Article 1 sidebar controls to populate this section.")

            st.markdown("#### Group–geography coupling sensitivity")
            _coupling_records = st.session_state.get("health_article1_coupling_sensitivity", [])
            if _coupling_records:
                _coup = pd.DataFrame(_coupling_records)
                _coup["Group label"] = _coup["group"].map(lambda g: "All referral-eligible" if str(g) == "ALL" else ("Disadvantaged" if str(g) in {"0", "0.0"} else ("Advantaged" if str(g) in {"1", "1.0"} else str(g))))
                _coup_agg = _coup.groupby(["coupling_lambda", "Group label"], as_index=False).agg(
                    runs=("run_id", "nunique"),
                    n_referral_eligible=("n_referral_eligible", "sum"),
                    fnr_missed_referral=("fnr_missed_referral", "mean"),
                    geographic_constraint=("geographic_constraint", "mean"),
                    joint_burden_j=("joint_burden_j", "mean"),
                    independence_j_ind=("independence_j_ind", "mean"),
                    excess_e=("excess_e", "mean"),
                    dependence_d=("dependence_d", "mean"),
                )
                _coup_groups = _coup_agg[_coup_agg["Group label"] != "All referral-eligible"].copy()
                if len(_coup_groups):
                    fig_c1 = px.line(
                        _coup_groups, x="coupling_lambda", y="joint_burden_j", color="Group label", markers=True,
                        title="Joint Burden as Structural Group–Geography Coupling Increases",
                        labels={"coupling_lambda":"Coupling λ", "joint_burden_j":"Joint burden J"}
                    )
                    fig_c1.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig_c1, use_container_width=True)
                    fig_c2 = px.line(
                        _coup_groups, x="coupling_lambda", y="geographic_constraint", color="Group label", markers=True,
                        title="Geographic Constraint by Group Across Coupling Scenarios",
                        labels={"coupling_lambda":"Coupling λ", "geographic_constraint":"Geographic constraint C"}
                    )
                    fig_c2.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig_c2, use_container_width=True)

                _pivot_j = _coup_groups.pivot(index="coupling_lambda", columns="Group label", values="joint_burden_j") if len(_coup_groups) else pd.DataFrame()
                if {"Disadvantaged", "Advantaged"}.issubset(set(_pivot_j.columns)):
                    _gap = (_pivot_j["Disadvantaged"] - _pivot_j["Advantaged"]).reset_index(name="delta_joint_disadv_minus_adv")
                    fig_c3 = px.line(_gap, x="coupling_lambda", y="delta_joint_disadv_minus_adv", markers=True,
                                     title="Joint-Burden Disparity Across Coupling Scenarios",
                                     labels={"coupling_lambda":"Coupling λ", "delta_joint_disadv_minus_adv":"ΔJ (disadvantaged − advantaged)"})
                    fig_c3.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig_c3, use_container_width=True)

                _coup_display = _coup_agg.copy()
                for _c in ["fnr_missed_referral", "geographic_constraint", "joint_burden_j", "independence_j_ind", "excess_e", "dependence_d"]:
                    _coup_display[_c] = _coup_display[_c].map(lambda x: "N/A" if pd.isna(x) else f"{x:.3f}")
                st.dataframe(_coup_display, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Article 1 Coupling-Sensitivity Results (CSV)",
                    _coup.to_csv(index=False),
                    file_name="article1_coupling_sensitivity.csv", mime="text/csv", key="a1_coupling_sensitivity_csv"
                )
                st.caption("This suite holds the fitted model, predictions, referral threshold, and facility network fixed. λ changes only the simulated concentration of patient groups in better- versus worse-access origin pools; it is a controlled structural scenario, not an observed Nigerian inequality estimate.")
            else:
                st.info("Enable the group–geography coupling suite in the Article 1 sidebar controls to populate this section.")

            st.markdown("#### S0–S3 intervention counterfactuals")
            _int_rows = st.session_state.get("health_article1_interventions", [])
            if _int_rows:
                _int_df = pd.DataFrame(_int_rows)
                _int_df = _int_df[_int_df["scenario"] != "INTERVENTION_METADATA"].copy()
                _int_df["Group label"] = _int_df["group"].map({"0":"Disadvantaged", "0.0":"Disadvantaged", "1":"Advantaged", "1.0":"Advantaged", "ALL":"All referral-eligible"}).fillna(_int_df["group"].astype(str))
                _int_agg = _int_df.groupby(["scenario","Group label"], as_index=False).agg(
                    fnr_missed_referral=("fnr_missed_referral","mean"),
                    geographic_constraint=("geographic_constraint","mean"),
                    joint_burden_j=("joint_burden_j","mean"),
                    independence_j_ind=("independence_j_ind","mean"),
                    excess_e=("excess_e","mean"),
                    dependence_d=("dependence_d","mean"),
                )
                _scenario_order = ["S0 Baseline", "S1 AI improvement", "S2 Geography improvement", "S3 Combined improvement"]
                _int_agg["scenario"] = pd.Categorical(_int_agg["scenario"], categories=_scenario_order, ordered=True)
                _int_agg = _int_agg.sort_values(["scenario","Group label"])
                _all_int = _int_agg[_int_agg["Group label"] == "All referral-eligible"].copy()
                if len(_all_int):
                    fig_i1 = px.bar(
                        _all_int, x="scenario", y="joint_burden_j",
                        title="S0–S3: Overall Joint Burden",
                        labels={"scenario":"Scenario", "joint_burden_j":"Joint burden J"}
                    )
                    fig_i1.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig_i1, use_container_width=True)

                _group_int = _int_agg[_int_agg["Group label"] != "All referral-eligible"].copy()
                if len(_group_int):
                    fig_i2 = px.bar(
                        _group_int, x="scenario", y="joint_burden_j", color="Group label", barmode="group",
                        title="S0–S3: Joint Burden by Group",
                        labels={"scenario":"Scenario", "joint_burden_j":"Joint burden J"}
                    )
                    fig_i2.update_yaxes(tickformat=".1%")
                    st.plotly_chart(fig_i2, use_container_width=True)

                    _piv = _group_int.pivot(index="scenario", columns="Group label", values="joint_burden_j")
                    if {"Disadvantaged", "Advantaged"}.issubset(set(_piv.columns)):
                        _dgap = (_piv["Disadvantaged"] - _piv["Advantaged"]).reset_index(name="delta_joint")
                        fig_i3 = px.bar(
                            _dgap, x="scenario", y="delta_joint",
                            title="S0–S3: Joint-Burden Disparity",
                            labels={"scenario":"Scenario", "delta_joint":"ΔJ (disadvantaged − advantaged)"}
                        )
                        fig_i3.update_yaxes(tickformat=".1%")
                        st.plotly_chart(fig_i3, use_container_width=True)

                _int_show = _int_agg.copy()
                for _c in ["fnr_missed_referral","geographic_constraint","joint_burden_j","independence_j_ind","excess_e","dependence_d"]:
                    _int_show[_c] = _int_show[_c].map(lambda x: "N/A" if pd.isna(x) else f"{x:.3f}")
                st.dataframe(_int_show, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Article 1 S0–S3 Intervention Results (CSV)",
                    pd.DataFrame(_int_rows).to_csv(index=False),
                    file_name="article1_s0_s3_interventions.csv", mime="text/csv", key="a1_intervention_csv"
                )
                _int_pat = st.session_state.get("health_article1_intervention_patients", [])
                if _int_pat:
                    st.download_button(
                        "⬇️ Article 1 S0–S3 Patient Counterfactuals (CSV)",
                        pd.concat(_int_pat, ignore_index=True).to_csv(index=False),
                        file_name="article1_s0_s3_patient_counterfactuals.csv", mime="text/csv", key="a1_intervention_patients_csv"
                    )
                st.caption("S1 is a counterfactual FNR-gap repair, S2 is a proportional travel-time improvement, and S3 combines both. These scenarios estimate simulated technical leverage, not causal effects of a real Nigerian intervention.")
            else:
                st.info("Enable the S0–S3 intervention suite in the Article 1 sidebar controls to populate this section.")

            st.markdown("#### Referral pathway")
            _n_r = int(len(_eligible)); _n_m = int(_eligible["missed_referral_opportunity_m"].sum()); _n_detect = _n_r - _n_m
            _miss = _eligible[_eligible["missed_referral_opportunity_m"] == 1]
            _det = _eligible[_eligible["missed_referral_opportunity_m"] == 0]
            _m_c = int(_miss["geographic_constraint_c"].sum()); _m_a = len(_miss)-_m_c
            _d_c = int(_det["geographic_constraint_c"].sum()); _d_a = len(_det)-_d_c
            _sankey = go.Figure(go.Sankey(node=dict(label=["Referral eligible", "Detected", "Missed referral opportunity", "Detected + accessible", "Detected + constrained", "Missed + accessible", "Joint burden (missed + constrained)"]), link=dict(source=[0,0,1,1,2,2], target=[1,2,3,4,5,6], value=[_n_detect,_n_m,_d_a,_d_c,_m_a,_m_c])))
            _sankey.update_layout(height=420, title="Screen-to-Referral Pathway")
            st.plotly_chart(_sankey, use_container_width=True)

            st.markdown("#### Referral geography")
            _map_df = _a1_all.drop_duplicates(subset=["origin_facility_id", "referral_facility_id"])[["origin_lat","origin_lon","origin_facility_name","origin_state","referral_lat","referral_lon","referral_facility_name","referral_facility_type","straight_line_km"]].copy()
            _orig = _map_df[["origin_lat","origin_lon","origin_facility_name","origin_state"]].rename(columns={"origin_lat":"lat","origin_lon":"lon","origin_facility_name":"facility","origin_state":"state"}).drop_duplicates()
            _orig["role"] = "Patient-origin proxy (Primary)"
            _dest = _map_df[["referral_lat","referral_lon","referral_facility_name","referral_facility_type"]].rename(columns={"referral_lat":"lat","referral_lon":"lon","referral_facility_name":"facility","referral_facility_type":"state"}).drop_duplicates()
            _dest["role"] = "Eligible higher-level referral"
            _map_points = pd.concat([_orig, _dest], ignore_index=True)
            if len(_map_points):
                fig3 = px.scatter_map(_map_points, lat="lat", lon="lon", hover_name="facility", hover_data=["role", "state"], color="role", zoom=4.2, height=520, title="Simulated Patient-Origin Proxies and Matched Referral Facilities")
                st.plotly_chart(fig3, use_container_width=True)
            st.caption("Map origins are simulated patient-location proxies anchored to functional primary facilities. Referral destinations are eligible functional secondary/tertiary facilities; this does not establish disease-specific service capability.")

            st.markdown("#### Patient-level audit records")
            _show_cols = ["run_id","seed","patient_index","group","referral_eligible_r","predicted_positive","prediction_probability","missed_referral_opportunity_m","origin_facility_name","origin_state","referral_facility_name","referral_facility_type","straight_line_km","travel_time_min","geographic_constraint_c","joint_burden_j","routing_method"]
            st.dataframe(_a1_all[[c for c in _show_cols if c in _a1_all.columns]], use_container_width=True, hide_index=True, height=420)

            st.markdown("#### Publication-grade repeated-seed analysis")
            _pub_master = pd.DataFrame(st.session_state.get("health_article1_publication_master", []))
            _pub_ci = st.session_state.get("health_article1_publication_ci", pd.DataFrame())
            _pub_match = st.session_state.get("health_article1_performance_match", pd.DataFrame())
            if not _pub_master.empty:
                _nseeds = int(_pub_master["seed"].nunique())
                _nmodels = int(_pub_master["model_variant"].nunique())
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Repeated seeds", _nseeds)
                pc2.metric("Model variants", _nmodels)
                pc3.metric("Bootstrap replicates", int(st.session_state.get("health_experiment_config", {}).get("article1_publication_runner", {}).get("bootstrap_reps", 0)))
                st.caption("95% intervals resample whole seed-level runs, preserving within-run patient dependence. They quantify simulation/retraining uncertainty, not population-representative uncertainty for Nigeria.")
                if isinstance(_pub_match, pd.DataFrame) and not _pub_match.empty:
                    st.markdown("**Performance-matched model comparison**")
                    st.dataframe(_pub_match, use_container_width=True, hide_index=True)
                if isinstance(_pub_ci, pd.DataFrame) and not _pub_ci.empty:
                    st.markdown("**Run-level bootstrap estimates**")
                    st.dataframe(_pub_ci, use_container_width=True, hide_index=True)
                    _jci = _pub_ci[(_pub_ci["metric"] == "joint_burden_j")].copy()
                    if not _jci.empty:
                        fig_pub = px.bar(_jci, x="model_variant", y="estimate", color="group", barmode="group", title="Repeated-seed mean joint burden by model variant")
                        st.plotly_chart(fig_pub, use_container_width=True)
                st.download_button("⬇️ Publication Master Results (CSV)", _pub_master.to_csv(index=False), file_name="article1_publication_master_results.csv", mime="text/csv", key="a1_pub_master_csv")
                if isinstance(_pub_ci, pd.DataFrame) and not _pub_ci.empty:
                    st.download_button("⬇️ Run-Level Bootstrap CIs (CSV)", _pub_ci.to_csv(index=False), file_name="article1_publication_bootstrap_ci.csv", mime="text/csv", key="a1_pub_ci_csv")
                if isinstance(_pub_match, pd.DataFrame) and not _pub_match.empty:
                    st.download_button("⬇️ Performance-Match Table (CSV)", _pub_match.to_csv(index=False), file_name="article1_performance_match.csv", mime="text/csv", key="a1_pub_match_csv")
            else:
                st.info("Enable the publication runner in the Article 1 sidebar controls and run the simulation to generate repeated-seed inference and performance-matched comparisons.")

            st.markdown("#### Experiment provenance & assumptions")
            _prov = {
                "Dataset": st.session_state.get("cfg_health_data_source"),
                "Referral policy": _a1_summary.get("referral_policy"),
                "Analysis population": _a1_summary.get("analysis_population"),
                "Routing requested": _a1_summary.get("routing_method_requested"),
                "Routing observed": _a1_summary.get("routing_method_observed"),
                "Constraint threshold (min)": _a1_summary.get("travel_threshold_min"),
                "Threshold sensitivity enabled": bool(st.session_state.get("health_experiment_config", {}).get("article1_threshold_sensitivity", {}).get("enabled", False)),
                "Sensitivity thresholds (min)": st.session_state.get("health_experiment_config", {}).get("article1_threshold_sensitivity", {}).get("thresholds_min", []),
                "Group–geography coupling": _a1_summary.get("group_location_coupling"),
                "Coupling sensitivity enabled": bool(st.session_state.get("health_experiment_config", {}).get("article1_coupling_sensitivity", {}).get("enabled", False)),
                "Coupling values (λ)": st.session_state.get("health_experiment_config", {}).get("article1_coupling_sensitivity", {}).get("coupling_values", []),
                "S0–S3 interventions enabled": bool(st.session_state.get("health_experiment_config", {}).get("article1_interventions", {}).get("enabled", False)),
                "AI FNR-gap closure": st.session_state.get("health_experiment_config", {}).get("article1_interventions", {}).get("ai_gap_closure"),
                "Geography travel-time reduction": st.session_state.get("health_experiment_config", {}).get("article1_interventions", {}).get("geography_time_reduction"),
                "Primary origin facilities": _a1_summary.get("n_primary_origin_facilities"),
                "Eligible referral facilities": _a1_summary.get("n_eligible_referral_facilities"),
                "Eligible levels": _a1_summary.get("eligible_referral_types"),
                "Functional statuses": _a1_summary.get("functional_statuses"),
                "Warning": _a1_summary.get("warning"),
            }
            st.json(_prov)
            st.download_button("⬇️ Article 1 Patient-Level Referral Results (CSV)", _a1_all.to_csv(index=False), file_name="article1_patient_referral_results.csv", mime="text/csv", key="a1_tab_csv")
            st.download_button("⬇️ Article 1 Joint-Burden Summary (JSON)", json.dumps(_a1_summary, indent=2, default=str), file_name="article1_joint_burden_summary.json", mime="application/json", key="a1_tab_json")

    # ── 10. Raw Results Tab ───────────────────────────────────────────────────
    with T["Raw Results"]:
        st.markdown("### Complete Simulation Raw Data")
        serializable_history = []
        for rr in st.session_state.health_run_history:
            clean = {k: v for k, v in rr.items() if k not in {"model_artifact", "article1_patient_records"}}
            serializable_history.append(clean)
        raw_json = json.dumps(serializable_history, indent=2, default=str)
        config_json = json.dumps(st.session_state.get("health_experiment_config", {}), indent=2, default=str)
        st.download_button("⬇️ Download Results (JSON)", raw_json, file_name="healthcare_equity_results.json", mime="application/json")
        st.download_button("⬇️ Download Experiment Configuration (JSON)", config_json, file_name="healthcare_equity_experiment_config.json", mime="application/json")
        st.download_button("⬇️ Download Results (CSV)", pd.DataFrame(serializable_history).drop(columns=["provenance", "feature_names", "group_confusion", "impact_matrix", "subgroup_tpr", "article1_summary"], errors="ignore").to_csv(index=False), file_name="healthcare_equity_results.csv", mime="text/csv")
        _a1_frames = st.session_state.get("health_article1_patient_records", [])
        if _a1_frames:
            _a1_export = pd.concat(_a1_frames, ignore_index=True)
            st.download_button(
                "⬇️ Article 1 Patient-Level Referral Results (CSV)",
                _a1_export.to_csv(index=False),
                file_name="article1_patient_referral_results.csv", mime="text/csv"
            )
            st.download_button(
                "⬇️ Article 1 Joint-Burden Summary (JSON)",
                json.dumps(st.session_state.get("health_article1_summary", {}), indent=2, default=str),
                file_name="article1_joint_burden_summary.json", mime="application/json"
            )
        _a1_sens = st.session_state.get("health_article1_threshold_sensitivity", [])
        if _a1_sens:
            st.download_button(
                "⬇️ Article 1 Threshold-Sensitivity Results (CSV)",
                pd.DataFrame(_a1_sens).to_csv(index=False),
                file_name="article1_threshold_sensitivity.csv", mime="text/csv", key="a1_raw_sensitivity_csv"
            )
        _a1_intervention_rows = st.session_state.get("health_article1_interventions", [])
        if _a1_intervention_rows:
            st.download_button(
                "⬇️ Article 1 S0–S3 Interventions (CSV)",
                pd.DataFrame(_a1_intervention_rows).to_csv(index=False),
                file_name="article1_s0_s3_interventions.csv", mime="text/csv", key="raw_a1_intervention_csv"
            )
        _a1_coupling_rows = st.session_state.get("health_article1_coupling_sensitivity", [])
        if _a1_coupling_rows:
            st.download_button(
                "⬇️ Article 1 Coupling-Sensitivity Results (CSV)",
                pd.DataFrame(_a1_coupling_rows).to_csv(index=False),
                file_name="article1_coupling_sensitivity.csv", mime="text/csv", key="raw_a1_coupling_csv"
            )
        st.json(serializable_history)

    # ── 11. AI Safety Tab ─────────────────────────────────────────────────────
   # with T["AI Safety"]:
    #    render_safety_tab(st.session_state.get("health_safety_report", {}))

    # ── 12. Lifecycle Tab ─────────────────────────────────────────────────────
    #with T["Lifecycle"]:
      #  lifecycle_data = st.session_state.get("health_lifecycle_report", {})
       # if isinstance(lifecycle_data, str):
        #    try:
         #       lifecycle_data = json.loads(lifecycle_data)
          #  except Exception:
           #     lifecycle_data = {"error": lifecycle_data}

        #render_lifecycle_tab("healthcare", lifecycle_data)

    # ── 13. Case Study Tab ────────────────────────────────────────────────────
    with T["Case Study"]:
        current_scenario = st.session_state.get("selected_scenario", "Healthcare Equity Assessment")
        scenario_story_banner("health", current_scenario)

        results = st.session_state.get("health_spatial_report", st.session_state.get("health_feature_outputs", {}))
        domain_challenge_panel("health", results)

    # ── 14. Eco Score Tab ─────────────────────────────────────────────────────
    #with T["Eco Score"]:
     #   eco_data = st.session_state.get("health_lifecycle_report", {}) or {}
     #   if isinstance(eco_data, str):
      #      try:
       #         eco_data = json.loads(eco_data)
        #    except Exception:
         #       eco_data = {}

        #last_run = (st.session_state.get("health_run_history") or [{}])[-1]

        #render_eco_tab(
         #   eco_data,
          #  algo_key=last_run.get("algorithm", "random_forest"),
           # n_samples=int(last_run.get("n_samples", 2000)),
            #n_runs=int(last_run.get("n_runs", len(st.session_state.get("health_run_history", [])) or 1)),
        #)

    # ── 15. Dynamic Systems Tab ───────────────────────────────────────────────
    with T["Dynamic Systems"]:
        render_dynamic_systems_tab(
            st.session_state.get("health_dynamic_report", {}),
            domain="health",
            ds_key="health_dynamic_report"
        )

    # ── 16. Real Models Tab (Conditional) ────────────────────────────────────
    if "🔬 Real Models" in T:
        with T["🔬 Real Models"]:
            render_real_model_tab(st.session_state.get("health_real_models", {}))

    # ── History Browser & Annotations Footer ──────────────────────────────────
    history_browser(
        "health_snapshot_history",
        domain="health",
        key_metrics=["accuracy", "equity_score", "demographic_parity"],
    )

    annotation_panel(
        "health_annotations",
        context_label=f"{len(st.session_state.health_run_history)} Healthcare run(s)",
    )

    # ── Policy recommendations ────────────────────────────────────────────────
    st.divider()
    st.markdown("## Policy Recommendations")

    if info.get("is_africa"):
        st.markdown("""
        <div class="alert-info">
            <strong>Africa-Centric Findings (Feature 3)</strong><br>
            • Deploy USSD/SMS fallback interfaces to close the digital inclusion gap<br>
            • Mandate multilingual model validation across Hausa, Yoruba, Igbo, and English<br>
            • Apply gender-stratified resampling to close the diagnostic accuracy gap<br>
            • Align with UNESCO Women4EthicalAI principles for all clinical AI deployments
        </div>""", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.info("""
**For Healthcare Providers**
1. Publish validation studies with group-stratified performance metrics
2. Require clinician review of all high-stakes AI recommendations
3. Commission third-party bias audits annually
4. Ensure training data represents all patient populations
5. Implement clear AI override protocols for clinical staff

**Clinical Safety**
1. Monitor false-negative rates by demographic group continuously
2. Set group-specific decision thresholds for critical conditions
3. Integrate bias monitoring into existing quality-improvement programmes
""")
    with r2:
        st.success("""
**For AI Developers**
1. Test across diverse populations before deployment
2. Provide clinician-interpretable explanations for every prediction
3. Continuously track performance disaggregated by demographic group
4. Conduct adversarial red-teaming (text, image, deepfake) before release
5. Co-design with clinicians, patients, and ethicists

**For Regulators**
1. Mandate fairness testing (GAGS proposed equity score ≥ 0.70 (research threshold)) for clinical AI approval
2. Require disclosure of performance by demographic group
3. Establish post-market surveillance requirements
4. Adopt blockchain-ledger governance for algorithm-change tracking
""")

    st.caption(
        "Disclaimer: Simulation for educational/research purposes. "
        "Real clinical AI requires extensive validation, regulatory approval, and ethical review."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Welcome screen (no results yet)
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## Welcome to Healthcare Equity Simulation")
    st.markdown("""
    Configure your scenario in the sidebar and click **Run Simulation** to begin.
    This module integrates all five GAGS v1.0 feature upgrades alongside the
    existing hybrid data pipeline.
    """)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("Feature 1 — Agent Economy", "card-blue",
         "Autonomous agents bid for ICU beds, diagnostic compute, and specialist time via Vickrey auctions. Tracks resource permeability across patient groups."),
        ("Feature 2 — Multimodal Red Team", "card-orange",
         "Text injection, adversarial image perturbation, and deepfake attacks on clinical data — with immersive VR scenario descriptions."),
        ("Feature 3 — Africa-Centric", "card-green",
         "Nigeria scenario presets, multilingual fairness bias, and a UNESCO Women4EthicalAI gender equity audit with digital inclusion metrics."),
    ]
    for col, (title, cls, desc) in zip([c1, c2, c3], cards):
        col.markdown(
            f'<div class="{cls}"><strong>{title}</strong><p style="font-size:.87rem;margin:.5rem 0 0;">{desc}</p></div>',
            unsafe_allow_html=True)

    c4, c5, _ = st.columns(3)
    extra_cards = [
        ("Feature 4 — Governance Layer", "card-purple",
         "Citizen assembly votes on healthcare AI policies. AI detects bias drift and auto-reverses harmful decisions, logged on a chained ledger."),
        ("Feature 5 — Strategic Arena", "card-blue",
         "Agents negotiate, deceive, and form coalitions under partial observability. Logs reveal emergent social behaviour in healthcare resource allocation."),
    ]
    for col, (title, cls, desc) in zip([c4, c5], extra_cards):
        col.markdown(
            f'<div class="{cls}"><strong>{title}</strong><p style="font-size:.87rem;margin:.5rem 0 0;">{desc}</p></div>',
            unsafe_allow_html=True)

    with st.expander("How to Use", expanded=False):
        st.markdown("""
        1. **Select a data source** — synthetic, real-world (UCI/PIMA/WBC), hybrid, or Abuja Africa-centric
        2. **Enable feature modules** in the sidebar (Multimodal Red Team, Governance, Arena, Agent Economy, Gender Audit)
        3. **Configure bias types and intensity** — including new `gender` and `linguistic` types
        4. **Set patient demographics** and access inequality factors
        5. **Run** — results appear across analysis tabs
        6. **Review the governance banner** — automatic vote outcome and blockchain hash displayed after each run
        7. **Download** results (CSV) or configuration (JSON) from the Raw Results tab
        """)

# Footer
st.divider()
st.markdown("""
<div style="text-align:center;color:#7f8c8d;padding:1.5rem 0;">
    <strong>Healthcare Equity Simulation • GAGS Framework v1.0</strong><br>
    Features: AI Agent Economy · Multimodal Red Teaming · Africa-Centric/Gender Equity ·
    Hybrid Governance · Strategic Social Reasoning
</div>
""", unsafe_allow_html=True)