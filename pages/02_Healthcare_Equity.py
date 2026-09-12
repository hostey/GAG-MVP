# pages/1_🏥_Healthcare_Equity.py
"""
Healthcare Equity Simulation — GAGS Framework v1.0

Refactored to integrate all five feature modules from simulation_core.py:
  Feature 1 — AI Agent Economy Sandbox  (resource auction in healthcare domain)
  Feature 2 — Multimodal Red Teaming    (text/image/deepfake attack surface)
  Feature 3 — Africa-Centric / Gender   (Abuja presets + UNESCO equity audit)
  Feature 4 — Hybrid Governance Layer   (citizen vote + blockchain ledger)
  Feature 5 — Strategic Social Arena    (negotiation / coalition game)
"""

import json
import warnings
import time
import hashlib

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page Config (Must be initial Streamlit command) ───────────────────────────
st.set_page_config(page_title="Healthcare Equity • Platform", layout="wide", page_icon="🏥")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset

from components.translate import install_auto_translate, language_switcher

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
    apply_bias,
    simulate_data_poisoning,
    calculate_fairness_metrics,
    run_gender_equity_audit,
    simulate_longitudinal_bias,
    simulate_federated_learning,
)
from utils.config import simulation_config
from components.i18n import t
from components.nigeria_regulatory import nigeria_compliance_panel
from components.ux_utils import (
    history_browser, save_to_history,
    annotation_panel,
    role_switcher, role_banner,
    get_role_algo, role_algo_banner, role_brief_banner,
)
from components.live_data import national_live_banner
from components.gags_gender_ui import render_gender_equity_audit
from components.nigeria_states import state_selector, state_info_card
from components.gags_interactive import (
    progress_tracker, scenario_story_banner,
    domain_challenge_panel, _reset_render_guards
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
from components.health.clinical_impact import (
    compute_clinical_impact_matrix,
)
from components.health.equity_audit_engine import (
    calculate_facility_access,
    generate_llm_vignette,
    generate_clinical_recommendation,
    run_multi_agent_triage_pipeline,
render_mitigation_workbench,
render_medical_compliance_panel,
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
        df["income_level"] = X[:, 1] if X.shape[1] > 1 else np.random.uniform(0, 1, len(X))
        df["_africa_centric"] = True
        df["_description"] = description
        return df

    @st.cache_data(show_spinner=False)
    def load_heart_disease_uci(_self, n_samples: int = 5000):
        try:
            url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                   "heart-disease/processed.cleveland.data")
            cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
            df = pd.read_csv(url, names=cols, na_values="?").dropna()
            df["target"] = (df["target"] > 0).astype(int)
            n = min(len(df), n_samples)
            df = df.iloc[:n].copy()
            df["income_level"] = np.random.uniform(0, 1, n)
            df["access_score"] = np.random.uniform(0.3, 1, n)
            df["education_level"] = np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.3, 0.3, 0.2])
            df["demographic_group"] = df["sex"].astype(int)
            return df
        except Exception:
            return _self.generate_synthetic_only(n_samples)

    @st.cache_data(show_spinner=False)
    def load_pima_diabetes(_self, n_samples: int = 5000):
        try:
            url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv")
            cols = ["preg", "glucose", "bp", "skin", "insulin", "bmi", "pedigree", "age", "target"]
            df = pd.read_csv(url, names=cols)
            n = min(len(df), n_samples)
            df = df.iloc[:n].copy()
            df["income_level"] = np.random.uniform(0, 1, n)
            df["access_score"] = np.random.uniform(0.3, 1, n)
            df["demographic_group"] = (df["income_level"] < 0.5).astype(int)
            return df
        except Exception:
            return _self.generate_synthetic_only(n_samples)

    @st.cache_data(show_spinner=False)
    def load_breast_cancer(_self, n_samples: int = 5000):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        n = min(len(df), n_samples)
        df = df.iloc[:n].copy()
        df["age"] = np.random.normal(55, 15, n).clip(25, 90)
        df["income_level"] = np.random.uniform(0, 1, n)
        df["insurance"] = np.random.choice([0, 1], n, p=[0.2, 0.8])
        df["demographic_group"] = (df["income_level"] < 0.5).astype(int)
        return df

    def generate_synthetic_only(self, n_samples: int = 5000) -> pd.DataFrame:
        np.random.seed(42)
        n = n_samples
        data = {
            "age": np.random.normal(55, 15, n).clip(18, 100),
            "sex": np.random.choice([0, 1], n, p=[0.45, 0.55]),
            "bmi": np.random.normal(27, 6, n).clip(15, 50),
            "blood_pressure": np.random.normal(130, 20, n).clip(80, 200),
            "cholesterol": np.random.normal(200, 40, n).clip(100, 350),
            "glucose": np.random.normal(110, 30, n).clip(60, 300),
            "chronic_conditions": np.random.poisson(1.5, n).clip(0, 8),
            "previous_hospitalizations": np.random.poisson(0.8, n),
            "smoking": np.random.binomial(1, 0.25, n),
            "exercise_frequency": np.random.uniform(0, 1, n),
            "income_level": np.random.uniform(0, 1, n),
            "education": np.random.choice([1, 2, 3, 4], n, p=[0.15, 0.35, 0.35, 0.15]),
            "insurance": np.random.binomial(1, 0.8, n),
            "access_score": np.random.uniform(0.3, 1, n),
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
        data["target"] = (rs + np.random.normal(0, 0.1, n) > np.percentile(rs, 60)).astype(int)
        df = pd.DataFrame(data)
        df["demographic_group"] = df["sex"].astype(int)
        return df

    def generate_hybrid_data(self, n_samples: int = 5000) -> pd.DataFrame:
        try:
            real = self.load_heart_disease_uci(n_samples // 2)
            synth = self.generate_synthetic_only(n_samples // 2)
            common = list(set(real.columns) & set(synth.columns))
            hybrid = pd.concat([real[common], synth[common]], ignore_index=True)
            if "target" not in hybrid.columns:
                hybrid["target"] = np.random.choice([0, 1], len(hybrid))
            if "demographic_group" not in hybrid.columns:
                hybrid["demographic_group"] = np.random.choice([0, 1], len(hybrid))
            return hybrid.sample(frac=1).reset_index(drop=True)
        except Exception:
            return self.generate_synthetic_only(n_samples)


_pipeline = HealthcareHybridPipeline()

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
        "cfg_health_season_profile": "Dry Season",
        "cfg_health_terrain_type": "Rural Unpaved",
        "cfg_health_llm_provider": "Groq (Llama 3.3 / 3.1)",
        "cfg_health_api_key": "DEMO",
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
        "health_ds_report": {},
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

    dis_count = max(disadv.get("Count", 100), 1)
    priv_count = max(priv.get("Count", 100), 1)

    dis_tp = int(dis_count * disadv.get("TPR", 0.8))
    dis_fn = int(dis_count * disadv.get("FNR", 0.2))
    priv_tp = int(priv_count * priv.get("TPR", 0.9))
    priv_fn = int(priv_count * priv.get("FNR", 0.1))

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



def _train_and_score(X, y, demo, random_state=42):
    try:
        X_tr, X_te, y_tr, y_te, demo_tr, demo_te = train_test_split(
            X, y, demo, test_size=0.3, random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        clf = RandomForestClassifier(n_estimators=50, random_state=random_state, max_depth=6)
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)

        acc = float(accuracy_score(y_te, y_pred))
        rec = float(recall_score(y_te, y_pred, zero_division=0))
        prec = float(precision_score(y_te, y_pred, zero_division=0))
        f1 = float(f1_score(y_te, y_pred, zero_division=0))

        cm = confusion_matrix(y_te, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        else:
            fpr = 0.0

        metrics = {
            "accuracy": acc, "recall": rec, "sensitivity": rec,
            "precision": prec, "f1": f1, "fpr": fpr
        }
        return metrics, clf, scaler, (X_tr_s, X_te_s, y_tr, y_te, demo_tr, demo_te)
    except Exception:
        return {"accuracy": 0, "recall": 0, "sensitivity": 0, "precision": 0, "f1": 0, "fpr": 0}, None, None, (None,
                                                                                                               None,
                                                                                                               None,
                                                                                                               None,
                                                                                                               None,
                                                                                                               None)


def _run_one(
        data_source, n_samples, selected_biases, bias_intensity,
        poison_rate, access_inequality, run_idx,
        enable_redteam=False, enable_governance=True, governance_policy="majority_vote",
        enable_arena=False, enable_agent_economy=False, enable_gender_audit=True,
        selected_state="Nigeria (National Average)"):
    try:
        if data_source in _pipeline.available_datasets:
            df_ds = _pipeline.available_datasets[data_source](n_samples)
            if "target" in df_ds.columns:
                y = df_ds["target"].values.astype(int)
            else:
                y = df_ds.iloc[:, -1].values.astype(int)

            if "demographic_group" in df_ds.columns:
                demo = df_ds["demographic_group"].values.astype(int)
            elif "sex" in df_ds.columns:
                demo = df_ds["sex"].values.astype(int)
            elif "income_level" in df_ds.columns:
                demo = (df_ds["income_level"].values < df_ds["income_level"].median()).astype(int)
            else:
                demo = np.random.choice([0, 1], size=len(df_ds))

            feat_cols = [c for c in df_ds.columns if
                         c not in ["target", "demographic_group", "_africa_centric", "_description"]]
            X = df_ds[feat_cols].select_dtypes(include=[np.number]).values
            if X.shape[1] == 0:
                X, y, demo = generate_synthetic_data(n_samples=n_samples, n_features=10)
        elif data_source.startswith("abuja") or data_source == "africa_centric":
            X, y, demo, _ = generate_africa_centric_data(scenario="healthcare", n_samples=n_samples)
        else:
            X, y, demo = generate_synthetic_data(n_samples=n_samples, n_features=10)
    except Exception:
        X, y, demo = generate_synthetic_data(n_samples=n_samples, n_features=10)

    X = X.astype(np.float64)

    if access_inequality > 0:
        mask = demo == 0
        if mask.any():
            X[mask] += np.random.normal(0, access_inequality * 0.3, (mask.sum(), X.shape[1]))
    _vb = list(simulation_config.BIAS_TYPES) + [b for b in ["gender", "linguistic"] if
                                                b not in simulation_config.BIAS_TYPES]
    for bt in [b for b in selected_biases if b in _vb]:
        try:
            X, y, demo = apply_bias(X, y, bt, bias_intensity, demographic_info=demo)
        except Exception:
            pass
    try:
        X, y, demo = simulate_data_poisoning(X, y, poison_rate, attack_type="label_flipping", demographic_info=demo,
                                             targeted=False)
    except Exception:
        pass

    metrics, clf, scaler, splits = _train_and_score(X, y, demo, random_state=42 + max(run_idx, 0))
    if clf is None:
        return {"accuracy": 0, "recall": 0, "sensitivity": 0, "precision": 0, "f1": 0, "fpr": 0,
                "specificity": 0, "acc_hi": 0, "acc_lo": 0, "sens_hi": 0, "sens_lo": 0,
                "adv_hi": 0, "adv_lo": 0,
                "equity_score": 0, "fairness_score": 0, "demographic_parity": 0, "equalized_odds": 0,
                "bias_intensity": bias_intensity, "poison_rate": poison_rate, "biases": "None",
                "run_id": run_idx + 1, "data_source": data_source, "gender_gap": 0, "warnings": [],
                "impact_matrix": {}}

    X_tr_s, X_te_s, y_tr, y_te, demo_tr, demo_te = splits
    y_pred = clf.predict(X_te_s)
    # ── Per-group True Positive Rate (Sensitivity) ──────────────────────────────
    subgroup_tpr = {}

    # demo_te should be the sensitive attribute array for the test set
    # It can be binary (0/1) or multi-class (strings / integers)

    unique_groups = np.unique(demo_te)

    for g in unique_groups:
        mask = (demo_te == g)
        pos_mask = mask & (y_te == 1)

        if pos_mask.sum() > 0:
            tpr = float(np.mean(y_pred[pos_mask] == 1))
        else:
            tpr = np.nan  # or 0.0 if you prefer

        # Create readable labels
        if str(g) in ("0", "0.0"):
            label = "Disadvantaged"
        elif str(g) in ("1", "1.0"):
            label = "Advantaged"
        else:
            label = str(g)

        subgroup_tpr[label] = round(tpr, 4) if not np.isnan(tpr) else 0.0
    fair = calculate_fairness_metrics(y_te, y_pred, demo_te)
    gender_audit = None
    if enable_gender_audit:
        try:
            gender_audit = run_gender_equity_audit(y_te, y_pred, demo_te, 0.34)
        except Exception:
            pass

    if run_idx == 0:
        try:
            bi = bias_intensity if bias_intensity > 0 else 0.15
            bt0 = next((b for b in selected_biases if b in _vb), "demographic")
            res_long = simulate_longitudinal_bias(X, y, demo, initial_bias_type=bt0, initial_bias_intensity=bi,
                                                  n_generations=5, random_state=42)
            st.session_state.health_longitudinal = res_long.__dict__ if hasattr(res_long, "__dict__") else res_long
        except Exception:
            st.session_state.health_longitudinal = None
        try:
            res_fed = simulate_federated_learning(X, y, demo, n_clients=4, n_rounds=3,
                                                  bias_heterogeneity=(bias_intensity or 0.15) * 0.5, random_state=42)
            st.session_state.health_federated = res_fed.__dict__ if hasattr(res_fed, "__dict__") else res_fed
        except Exception:
            st.session_state.health_federated = None

    adv_mask = demo_te == 1
    dis_mask = demo_te == 0

    def _grp_acc(mask):
        return float(accuracy_score(y_te[mask], y_pred[mask])) if mask.any() else metrics["accuracy"]

    def _grp_sens(mask):
        pos = mask & (y_te == 1)
        if not pos.any(): return metrics["recall"]
        return float(np.mean(y_pred[pos] == 1))

    acc_hi = _grp_acc(adv_mask)
    acc_lo = _grp_acc(dis_mask)
    sens_hi = _grp_sens(adv_mask)
    sens_lo = _grp_sens(dis_mask)
    adv_hi = _grp_acc(adv_mask)
    adv_lo = _grp_acc(dis_mask)
    specificity = float(np.mean(y_pred[y_te == 0] == 0)) if (y_te == 0).any() else 0.0

    if run_idx == 0:
        try:
            from components.governance_logic import ExplainableModel, generate_compliance_report
            xm = ExplainableModel(domain="health")
            xm.model = clf
            xm._X_train = X_tr_s
            xm._is_fitted = True
            xm.feature_names = [f"feature_{i}" for i in range(X_te_s.shape[1])]
            fi = xm.feature_importance(X_te_s, y_te, n_repeats=6)
            _xai = {"feature_importance": fi.__dict__ if hasattr(fi, "__dict__") else fi}
            denied = np.where((y_te == 1) & (y_pred == 0))[0]
            if len(denied):
                expl = xm.explain_instance(X_te_s[denied[0]])
                cf = xm.counterfactual(X_te_s[denied[0]])
                _xai["instance_explanation"] = expl.__dict__ if hasattr(expl, "__dict__") else expl
                _xai["counterfactual"] = cf.__dict__ if hasattr(cf, "__dict__") else cf
            mc = xm.model_card(
                {"accuracy": metrics["accuracy"], "recall": metrics["recall"]},
                {"fairness_score": fair.get("fairness_score", 0.5),
                 "demographic_parity_difference": fair.get("demographic_parity_difference", 0)},
                domain="health")
            cr = generate_compliance_report(mc,
                                            {"fairness_score": fair.get("fairness_score", 0.5),
                                             "demographic_parity_difference": fair.get("demographic_parity_difference",
                                                                                       0)},
                                            {"accuracy": metrics["accuracy"]},
                                            frameworks=["EU AI Act", "ISO 42001", "NIST AI RMF", "NITDA", "WHO"])
            _xai.update({
                "model_card": mc.__dict__ if hasattr(mc, "__dict__") else mc,
                "compliance_report": cr.__dict__ if hasattr(cr, "__dict__") else cr
            })
            st.session_state.health_xai_results = _xai
        except Exception as _xe:
            st.session_state.health_xai_results = {"error": str(_xe)}

    impact_mat = compute_clinical_impact_matrix(y_te, y_pred, demo_te)
    st.session_state["latest_impact_matrix"] = impact_mat

    return {
        "run_id": run_idx + 1,
        "data_source": data_source,
        "accuracy": metrics["accuracy"],
        "recall": metrics["recall"],
        "sensitivity": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "fpr": metrics["fpr"],
        "specificity": specificity,
        "acc_hi": acc_hi,
        "acc_lo": acc_lo,
        "sens_hi": sens_hi,
        "sens_lo": sens_lo,
        "adv_hi": adv_hi,
        "adv_lo": adv_lo,
        "equity_score": fair.get("fairness_score", 0.5),
        "fairness_score": fair.get("fairness_score", 0.5),
        "demographic_parity": fair.get("demographic_parity_difference", 0),
        "equalized_odds": fair.get("equalized_odds_difference", 0),
        "bias_intensity": bias_intensity,
        "poison_rate": poison_rate,
        "biases": ", ".join([b for b in selected_biases if b in _vb]) or "None",
        "gender_gap": gender_audit.overall_gender_gap if gender_audit else 0.0,
        "warnings": [],
        "subgroup_tpr": subgroup_tpr,
        "impact_matrix": impact_mat,  # Attached to each run dict
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
    st.session_state.health_run_history = []
    st.session_state["health_feature_outputs"] = {}
    prog = st.progress(0, text="Executing Healthcare Simulation...")

    for i in range(n_runs):
        prog.progress((i) / n_runs, text=f"Run {i + 1} of {n_runs}…")
        with st.spinner(f"Simulating Iteration {i + 1}/{n_runs}"):
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
            )
            st.session_state.health_run_history.append(result)

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
    if enable_governance:
        _fout["governance"] = {
            "policy": "Deploy AI Clinical Screening Model",
            "outcome": "approved",
            "tally": {"for": 72, "against": 18, "abstain": 10},
            "ai_flags": ["Minor under-diagnosis disparity detected in rural cohorts"],
            "ledger_hash": hashlib.sha256(f"gov_health_{time.time()}".encode()).hexdigest()[:16],
            "ledger_entries": 1,
        }
    if enable_redteam:
        try:
            _fout["multimodal_redteam"] = run_redteam_simulation(domain="health")
        except Exception as e:
            _fout["multimodal_redteam"] = {"error": str(e), "status": "failed"}
            st.warning(f"Red-team module skipped: {e}")
    if enable_agent_economy:
        try:
            _fout["agent_economy"] = run_agent_economy_simulation(domain="health")
        except Exception as e:
            _fout["agent_economy"] = {"error": str(e), "status": "failed"}
            st.warning(f"agent-economy module skipped: {e}")
    if enable_arena:
        _fout["arena"] = run_arena_simulation(domain="health")

    if enable_ai_safety:
        _last_run = st.session_state.health_run_history[-1]
        _X_s = np.random.randn(500, 10)
        _y_s = (_X_s[:, 0] > 0).astype(int)
        st.session_state["health_safety_report"] = run_ai_safety_suite(
            X_train=_X_s[:400], y_train=_y_s[:400],
            X_test=_X_s[400:], y_test=_y_s[400:],
            model=None, domain="health",
            enable_robustness=True, enable_ood=True,
            enable_uncertainty=True, enable_checklists=True,
            simulation_metrics={"fairness_score": _last_run.get("fairness_score", 0.5), "robustness_score": 0.65},
        )

    if enable_lifecycle or enable_eco:
        _last_r = st.session_state.health_run_history[-1]
        st.session_state["health_lifecycle_report"] = run_lifecycle_suite(
            domain="health", algo_key="rf", algo_label="Random Forest",
            n_samples=sample_size, n_runs=n_runs, n_features=10,
            metrics=_last_r, enable_registry=enable_lifecycle, enable_eco=enable_eco
        )

    if enable_dynamic:
        _ds_params = derive_ds_params(domain="health", run_history=st.session_state.health_run_history)
        st.session_state["health_ds_report"] = run_dynamic_systems_suite(
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

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="med-card"><p style="margin:0;font-size:.8rem;color:#64748B;">🎯 Prediction Accuracy</p><p style="margin:0;font-size:1.8rem;font-weight:700;color:#0284C7;">{avg_acc:.1%}</p></div>',
            unsafe_allow_html=True)
    with k2:
        colour = "#16a34a" if avg_equity >= 0.7 else "#e67e22" if avg_equity >= 0.5 else "#ef4444"
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

    # ── Tab Navigation Setup ───────────────────────────────────────────────────
    _tab_labels = [
        "Performance", "Equity", "Clinical Impact","Dynamic Systems", "Feature Modules", "Explainable AI", "Compliance", "Longitudinal",
"Data Analysis","Raw Results", "Case Study",
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
            help="Fairness parity across demographic cohorts. Target ≥ 80%."
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
                    value="DEMO",
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
                                    st.session_state["health_ds_report"] = pipeline_results
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

                                    st.session_state["health_ds_report"] = {
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

        latest_results = st.session_state.get("health_ds_report", {})
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
            spatial = st.session_state.get("health_ds_report") or {}
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
        st.subheader("Explainable AI — tied to this run")
        st.caption(
            "Uses the last simulation, impact matrix, subgroup sensitivity, "
            "and (if present) the spatial / vignette audit. Nothing here is invented from the bias slider."
        )

        # ── Shared state from the rest of the module ─────────────────────────────
        xai_res = st.session_state.get("health_xai_results") or {}
        if not isinstance(xai_res, dict):
            xai_res = {}
        impact_matrix = st.session_state.get("latest_impact_matrix") or {}
        if not isinstance(impact_matrix, dict):
            impact_matrix = {}
        history = st.session_state.get("health_run_history") or []
        last_run = history[-1] if history and isinstance(history[-1], dict) else {}
        spatial = st.session_state.get("health_ds_report") or {}
        if not isinstance(spatial, dict):
            spatial = {}

        region = st.session_state.get("cfg_health_region", "Nigeria")
        data_source = st.session_state.get("cfg_health_data_source", last_run.get("data_source", "—"))
        task = st.session_state.get("cfg_health_prediction_task", "Clinical risk")
        mitigation = st.session_state.get("cfg_health_mitigation_strategy", "None (Baseline)")
        thresh = float(st.session_state.get("cfg_health_active_threshold", 0.50) or 0.50)
        season = st.session_state.get("cfg_health_season_profile", "Dry Season")
        terrain = st.session_state.get("cfg_health_terrain_type", "Rural Unpaved")

        PROXY_KW = ("socioeconomic", "distance", "insurance", "gender", "income",
                    "proxy", "rural", "wealth", "access")


        def _f(d, k, default=0.0):
            try:
                v = float((d or {}).get(k, default) or default)
                return default if v != v else v
            except (TypeError, ValueError):
                return default


        def _coerce_importance(raw) -> dict:
            if isinstance(raw, dict) and "importances" in raw:
                raw = raw["importances"]
            if not isinstance(raw, dict):
                return {}
            out = {}
            for k, v in raw.items():
                try:
                    if isinstance(v, (list, tuple, np.ndarray)):
                        v = v[0] if len(v) else 0.0
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            return out


        feature_imp = _coerce_importance(xai_res.get("feature_importance"))
        is_live_imp = bool(feature_imp)
        if not is_live_imp:
            feature_imp = {}

        disadv = impact_matrix.get("disadvantaged") or last_run.get("impact_matrix", {}).get("disadvantaged") or {}
        priv = impact_matrix.get("privileged") or last_run.get("impact_matrix", {}).get("privileged") or {}
        fnr_dis, fnr_priv = _f(disadv, "FNR"), _f(priv, "FNR")
        fnr_gap = _f(impact_matrix, "fnr_disparity", fnr_dis - fnr_priv)
        subgroup_tpr = last_run.get("subgroup_tpr") if isinstance(last_run.get("subgroup_tpr"), dict) else {}

        # ── Strip: this run in one line ──────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Dataset", str(data_source)[:28])
        k2.metric("Equity (last run)", f"{_f(last_run, 'equity_score'):.0%}")
        k3.metric("Under-diagnosis gap", f"{fnr_gap:+.1%}",
                  delta_color="inverse" if abs(fnr_gap) >= 0.05 else "off")
        k4.metric("Mitigation", str(mitigation).split("(")[0].strip() or "Baseline")
        st.caption(
            f"{task} · threshold {thresh:.2f} · {region} · {season} / {terrain} · "
            f"{len(history)} run(s) in history"
        )

        t1, t2, t3, t4 = st.tabs([
            "1 · Why the score (global)",
            "2 · Who is missed (groups)",
            "3 · This patient (instance)",
            "4 · Can the advice be done (access)",
        ])

        # ── 1. Global drivers from fitted explanation ────────────────────────────
        with t1:
            st.markdown("#### Drivers from the last fitted explanation")
            if not is_live_imp:
                st.warning(
                    "No `feature_importance` on `health_xai_results`. "
                    "Run a simulation (first run writes XAI). We will not draw a fake BP/glucose chart."
                )
            else:
                total = sum(abs(v) for v in feature_imp.values()) or 1.0
                norm = {k: abs(v) / total for k, v in feature_imp.items()}
                df_imp = pd.DataFrame({"Feature": list(norm), "Share": list(norm.values())}).sort_values("Share")
                fig = px.bar(df_imp, x="Share", y="Feature", orientation="h", color="Share",
                             color_continuous_scale="Tealgrn")
                fig.update_layout(height=min(520, 80 + 28 * len(df_imp)), margin=dict(l=10, r=10, t=10, b=10),
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key="xai_mod_global")

                non_clin = sum(v for f, v in norm.items() if any(k in f.lower() for k in PROXY_KW))
                c1, c2 = st.columns(2)
                c1.metric("Name-heuristic: circumstance features", f"{non_clin:.0%}")
                c2.metric("Other / clinical-named features", f"{1 - non_clin:.0%}")

                mc = xai_res.get("model_card")
                if mc:
                    with st.expander("Model card from this fit"):
                        st.json(mc if isinstance(mc, dict) else {"card": str(mc)})

        # ── 2. Groups: impact matrix + subgroup_tpr from _run_one ────────────────
        with t2:
            st.markdown("#### Same cohorts as Performance / Clinical Impact")
            g1, g2, g3 = st.columns(3)
            g1.metric("FNR underserved", f"{fnr_dis:.1%}")
            g2.metric("FNR better-served", f"{fnr_priv:.1%}")
            g3.metric("FNR gap", f"{fnr_gap:+.1%}",
                      delta="Unequal" if abs(fnr_gap) >= 0.05 else "Within 5pp",
                      delta_color="inverse" if abs(fnr_gap) >= 0.05 else "normal")

            df_fnr = pd.DataFrame({
                "Cohort": [f"Underserved ({region})", "Better-served"],
                "False negative rate": [fnr_dis, fnr_priv],
            })
            fig = px.bar(df_fnr, x="Cohort", y="False negative rate", color="Cohort",
                         color_discrete_sequence=["#c2410c", "#1d4ed8"])
            fig.update_layout(height=260, showlegend=False, yaxis_tickformat=".0%",
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True, key="xai_mod_fnr")

            if subgroup_tpr:
                st.markdown("##### Sensitivity by group (`subgroup_tpr` from last run)")
                s = pd.DataFrame({"Group": list(subgroup_tpr), "TPR": [float(v) for v in subgroup_tpr.values()]})
                s = s.sort_values("TPR")
                fig_t = px.bar(s, x="TPR", y="Group", orientation="h", color="TPR", color_continuous_scale="RdYlGn")
                fig_t.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                                    xaxis_tickformat=".0%")
                st.plotly_chart(fig_t, use_container_width=True, key="xai_mod_tpr")
            else:
                st.caption("No `subgroup_tpr` on the last run dict.")

            real_sub = xai_res.get("subgroup_shap") or xai_res.get("group_feature_importance")
            if isinstance(real_sub, dict) and real_sub:
                rows = []
                for cohort, feats in real_sub.items():
                    if isinstance(feats, dict):
                        for feat, val in feats.items():
                            try:
                                rows.append({"Feature": str(feat), "Cohort": str(cohort), "Weight": float(val)})
                            except (TypeError, ValueError):
                                pass
                if rows:
                    st.markdown("##### Stored group-level importances")
                    st.plotly_chart(
                        px.bar(pd.DataFrame(rows), x="Feature", y="Weight", color="Cohort", barmode="group"),
                        use_container_width=True, key="xai_mod_gshap",
                    )
            else:
                st.info("Per-group SHAP was not stored. Showing **error rates**, not synthetic feature bars.")

        # ── 3. Instance + counterfactual already computed in _run_one ────────────
        with t3:
            st.markdown("#### Denied / missed case from the test fold")
            st.caption("Written on run 0 when a false negative exists (`explain_instance` + `counterfactual`).")

            expl = xai_res.get("instance_explanation")
            cf = xai_res.get("counterfactual")
            if not expl and not cf:
                st.warning("No instance explanation in `health_xai_results`. Re-run so the first fold can attach one.")
            else:
                c_l, c_r = st.columns(2)
                with c_l:
                    st.markdown("**Why this patient was scored this way**")
                    if isinstance(expl, dict):
                        st.json(expl)
                    else:
                        st.write(expl)
                with c_r:
                    st.markdown("**Counterfactual (what would flip the call)**")
                    if isinstance(cf, dict):
                        st.json(cf)
                    else:
                        st.write(cf)

            if last_run:
                st.markdown("##### Same run’s headline metrics")
                st.dataframe(pd.DataFrame([{
                    "accuracy": last_run.get("accuracy"),
                    "sensitivity": last_run.get("sensitivity"),
                    "equity_score": last_run.get("equity_score"),
                    "bias_intensity": last_run.get("bias_intensity"),
                    "biases": last_run.get("biases"),
                }]), use_container_width=True)

        # ── 4. Spatial / vignette — explanation that can be acted on ─────────────
        with t4:
            st.markdown("#### Is the recommendation geographically possible?")
            st.caption(f"Season **{season}** · terrain **{terrain}** · from Equity / multi-agent audit.")

            facs = spatial.get("matched_facilities_df")
            top = spatial.get("top_referral_target") or {}
            clin = spatial.get("clinical_assessment") or st.session_state.get("active_llm_out") or {}

            if clin:
                if isinstance(clin, dict):
                    u1, u2, u3 = st.columns(3)
                    u1.metric("Urgency", str(clin.get("urgency", "—")))
                    u2.metric("Required tier", str(clin.get("required_tier", "—")))
                    u3.metric("Caps", ", ".join(clin.get("required_capabilities", [])[:3]) or "—")
                    if clin.get("summary"):
                        st.write(clin["summary"])
                else:
                    st.write(str(clin)[:800])

            if isinstance(facs, pd.DataFrame) and not facs.empty:
                cols = [c for c in ["facility_name", "facility_tier", "distance_km",
                                    "travel_time_min", "routing_engine"] if c in facs.columns]
                st.dataframe(facs[cols].head(8) if cols else facs.head(8), use_container_width=True)
                mins = top.get("travel_time_min")
                if mins is not None:
                    try:
                        mins = float(mins)
                        if mins > 60:
                            st.error(
                                f"Nearest routed facility is **{mins:.0f} min**. Advice may be infeasible in this season.")
                        else:
                            st.success(
                                f"Nearest routed facility ~ **{mins:.0f} min** ({top.get('facility_name', '—')}).")
                    except (TypeError, ValueError):
                        pass
            else:
                nearest = st.session_state.get("active_nearest_facs")
                if isinstance(nearest, pd.DataFrame) and not nearest.empty:
                    st.dataframe(nearest.head(8), use_container_width=True)
                else:
                    st.info("No facility routing on this session. Run the Equity spatial audit / multi-agent triage.")

            ussd = spatial.get("ussd_dispatch_payload")
            if ussd:
                st.code(str(ussd), language="text")
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

    # ── 10. Raw Results Tab ───────────────────────────────────────────────────
    with T["Raw Results"]:
        st.markdown("### Complete Simulation Raw Data")
        st.json(st.session_state.health_run_history)

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

        results = st.session_state.get("health_ds_report", st.session_state.get("health_feature_outputs", {}))
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
            st.session_state.get("health_ds_report", {}),
            domain="health",
            ds_key="health_ds_report"
        )

    # ── 16. Real Models Tab (Conditional) ────────────────────────────────────
    if "Real Models" in T:
        with T["Real Models"]:
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
1. Mandate fairness testing (equity score ≥ 0.7) for clinical AI approval
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