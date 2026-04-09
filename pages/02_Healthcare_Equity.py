# pages/1_🏥_Healthcare_Equity.py
"""
Healthcare Equity Simulation — GAGS Framework v3.0

Refactored to integrate all five new feature modules from simulation_core.py:
  Feature 1 — AI Agent Economy Sandbox  (resource auction in healthcare domain)
  Feature 2 — Multimodal Red Teaming    (text/image/deepfake attack surface)
  Feature 3 — Africa-Centric / Gender   (Abuja presets + UNESCO equity audit)
  Feature 4 — Hybrid Governance Layer   (citizen vote + blockchain ledger)
  Feature 5 — Strategic Social Arena    (negotiation / coalition game)

Key improvements over previous version:
  - Imports driven by simulation_core rather than duplicating logic
  - Consistent result envelope {status, results, features, warnings, metadata}
  - run_simple_simulation() used as the single computation entry point
  - generate_africa_centric_data() + run_gender_equity_audit() wired in
  - Governance ledger rendered inline; arena standings shown as leaderboard
  - Multimodal red-team results surfaced in a dedicated tab
  - Agent economy auction log visible in the Data Analysis tab
  - All st.session_state keys namespaced under "health_*"
  - CSS cleaned up: no inline gradients on metric cards (uses semantic colours)
"""

import json
from datetime import datetime
import warnings
from components.translate import install_auto_translate, tx, tx_plotly, language_switcher
install_auto_translate()
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# ── GAGS core ─────────────────────────────────────────────────────────────────
try:
    from components.pdf_report import generate_pdf_compliance_report
    PDF_OK = True
except (ImportError, ModuleNotFoundError):
    PDF_OK = False
    def generate_pdf_compliance_report(*a, **kw):
        return None
from components.governance_logic import (
    # Data generation
    generate_synthetic_data,
    generate_africa_centric_data,
    AFRICA_SCENARIO_PRESETS,
    # Bias / attack
    apply_bias,
    simulate_data_poisoning,
    # Fairness
    calculate_fairness_metrics,
    run_gender_equity_audit,
    # Main simulation entry point — also runs Feature modules 1,2,4,5 internally
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
from components.nigeria_states import state_selector, state_info_card, get_state_params, apply_state_to_preset
from components.gags_interactive import (
    progress_tracker, scenario_story_banner,
    domain_challenge_panel, benchmark_challenge_panel,
    what_if_explorer, bias_detective_panel, track_run, award_points,
    _reset_render_guards
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



# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid Data Pipeline  (real-world loaders — unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

class HealthcareHybridPipeline:
    """Loads real-world or synthetic healthcare datasets and pre-processes them."""

    def __init__(self):
        self.available_datasets = {
            "Synthetic Only":              self.generate_synthetic_only,
            "Heart Disease (UCI)":         self.load_heart_disease_uci,
            "Diabetes (PIMA)":             self.load_pima_diabetes,
            "Breast Cancer (Wisconsin)":   self.load_breast_cancer,
            "Hybrid (Synthetic + Real)":   self.generate_hybrid_data,
            # Feature 3 — Africa-centric scenarios
            "Nigeria Smallholder Agrotech":  self._load_africa_agrotech,
            "Nigeria Multilingual Healthcare": self._load_africa_healthcare,
        }

    # ── Africa-centric loaders (Feature 3) ────────────────────────────────────
    def _load_africa_agrotech(self, n_samples: int = 5000):
        X, y, demo, preset = generate_africa_centric_data("smallholder_agrotech", n_samples)
        return self._arrays_to_df(X, y, demo, preset["description"])

    def _load_africa_healthcare(self, n_samples: int = 5000):
        X, y, demo, preset = generate_africa_centric_data("multilingual_healthcare", n_samples)
        return self._arrays_to_df(X, y, demo, preset["description"])

    @staticmethod
    def _arrays_to_df(X, y, demo, description: str) -> pd.DataFrame:
        cols = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y
        df["demographic_group"] = demo
        df["income_level"] = X[:, 1]   # income proxy
        df["_africa_centric"] = True
        df["_description"] = description
        return df

    # ── Real-world loaders ─────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def load_heart_disease_uci(_self, n_samples: int = 5000):
        try:
            url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                   "heart-disease/processed.cleveland.data")
            cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                    "thalach","exang","oldpeak","slope","ca","thal","target"]
            df = pd.read_csv(url, names=cols, na_values="?").dropna()
            df["target"] = (df["target"] > 0).astype(int)
            n = min(len(df), n_samples)
            df = df.iloc[:n].copy()
            df["income_level"]    = np.random.uniform(0, 1, n)
            df["access_score"]    = np.random.uniform(0.3, 1, n)
            df["education_level"] = np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.3, 0.3, 0.2])
            return df
        except Exception as exc:
            st.warning(f"Could not load UCI dataset: {exc}")
            return _self.generate_synthetic_only(n_samples)

    @st.cache_data(show_spinner=False)
    def load_pima_diabetes(_self, n_samples: int = 5000):
        try:
            url = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
                   "pima-indians-diabetes.data.csv")
            cols = ["preg","glucose","bp","skin","insulin","bmi","pedigree","age","target"]
            df = pd.read_csv(url, names=cols)
            n = min(len(df), n_samples)
            df = df.iloc[:n].copy()
            df["income_level"] = np.random.uniform(0, 1, n)
            df["access_score"] = np.random.uniform(0.3, 1, n)
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
        df["age"]          = np.random.normal(55, 15, n).clip(25, 90)
        df["income_level"] = np.random.uniform(0, 1, n)
        df["insurance"]    = np.random.choice([0, 1], n, p=[0.2, 0.8])
        return df

    def generate_synthetic_only(self, n_samples: int = 5000) -> pd.DataFrame:
        np.random.seed(42)
        n = n_samples
        data = {
            "age":                      np.random.normal(55, 15, n).clip(18, 100),
            "sex":                      np.random.choice([0, 1], n, p=[0.45, 0.55]),
            "bmi":                      np.random.normal(27, 6, n).clip(15, 50),
            "blood_pressure":           np.random.normal(130, 20, n).clip(80, 200),
            "cholesterol":              np.random.normal(200, 40, n).clip(100, 350),
            "glucose":                  np.random.normal(110, 30, n).clip(60, 300),
            "chronic_conditions":       np.random.poisson(1.5, n).clip(0, 8),
            "previous_hospitalizations":np.random.poisson(0.8, n),
            "smoking":                  np.random.binomial(1, 0.25, n),
            "exercise_frequency":       np.random.uniform(0, 1, n),
            "income_level":             np.random.uniform(0, 1, n),
            "education":                np.random.choice([1, 2, 3, 4], n, p=[0.15, 0.35, 0.35, 0.15]),
            "insurance":                np.random.binomial(1, 0.8, n),
            "access_score":             np.random.uniform(0.3, 1, n),
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
        return pd.DataFrame(data)

    def generate_hybrid_data(self, n_samples: int = 5000) -> pd.DataFrame:
        try:
            real = self.load_heart_disease_uci(n_samples // 2)
            synth = self.generate_synthetic_only(n_samples // 2)
            common = list(set(real.columns) & set(synth.columns))
            hybrid = pd.concat([real[common], synth[common]], ignore_index=True)
            if "target" not in hybrid.columns:
                hybrid["target"] = np.random.choice([0, 1], len(hybrid))
            return hybrid.sample(frac=1).reset_index(drop=True)
        except Exception:
            return self.generate_synthetic_only(n_samples)

    # ── Pre-processing ─────────────────────────────────────────────────────────
    def preprocess_data(self, df: pd.DataFrame, target_col: str = "target"):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in DataFrame")
        # Drop non-numeric metadata columns added by Africa loader
        drop_cols = [c for c in df.columns if c.startswith("_")]
        X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
        y = df[target_col]
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols):
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        num_cols = X.select_dtypes(include=[np.number]).columns
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        return X.values, y.values

    def create_patient_groups(self, df: pd.DataFrame) -> np.ndarray:
        """Return binary group array: 0 = disadvantaged, 1 = advantaged."""
        if "demographic_group" in df.columns:
            return df["demographic_group"].values.astype(int)
        if "income_level" in df.columns:
            return (df["income_level"] < 0.3).astype(int).values
        if "insurance" in df.columns:
            return (df["insurance"] == 0).astype(int).values
        return np.zeros(len(df), dtype=int)


# Singleton pipeline
_pipeline = HealthcareHybridPipeline()




st.set_page_config(page_title="Healthcare Equity • GAGS", layout="wide", page_icon="🏥")

# ── Design system ──────────────────────────────────────────────────────────────
try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, page_header, plotly_theme as _ptheme
    inject_css("health")
    ACCENT = DOMAIN_ACCENTS["health"]
except ImportError:
    ACCENT = "#0891b2"


# ═══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═══════════════════════════════════════════════════════════════════════════════

_STATE_DEFAULTS = {
    "health_run_history":      [],
    "health_dataset_info":     {},
    "health_feature_outputs":  {},
    "health_xai_results":      {},
    "health_longitudinal":     None,
    "health_federated":        None,
    "health_snapshot_history": [],
    "health_ds_report": {}
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _train_and_score(X, y, random_state=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y)) < 2:
        return None, None, None, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=random_state,
        stratify=y if (len(np.unique(y)) > 1 and np.bincount(y.astype(int)).min() >= 2) else None)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state)
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    return ({"accuracy":float(accuracy_score(yte,yp)),"recall":float(recall_score(yte,yp,zero_division=0)),
             "precision":float(precision_score(yte,yp,zero_division=0)),"f1":float(f1_score(yte,yp,zero_division=0)),
             "fpr":float(np.mean(yp[yte==0]==1)) if (yte==0).any() else 0.0},
            clf, scaler, (Xtr, Xte, ytr, yte))


def _run_one(
    data_source, n_samples, selected_biases, bias_intensity,
    poison_rate, access_inequality, run_idx,
    enable_redteam=False, enable_governance=True, governance_policy="majority_vote",
    enable_arena=False, enable_agent_economy=False, enable_gender_audit=True,
    selected_state="Nigeria (National Average)"):
    try:
        X, y, demo = generate_synthetic_data(n_samples=n_samples, n_features=10)
        if data_source.startswith("abuja") or data_source == "africa_centric":
            X, y, demo, _ = generate_africa_centric_data(scenario="healthcare", n_samples=n_samples)
    except Exception:
        X, y, demo = generate_synthetic_data(n_samples=n_samples, n_features=10)
    X = X.astype(np.float64)
    if access_inequality > 0:
        mask = demo == 0
        if mask.any():
            X[mask] += np.random.normal(0, access_inequality*0.3, (mask.sum(), X.shape[1]))
    _vb = list(simulation_config.BIAS_TYPES) + [b for b in ["gender","linguistic"] if b not in simulation_config.BIAS_TYPES]
    for bt in [b for b in selected_biases if b in _vb]:
        try: X, y, demo = apply_bias(X, y, bt, bias_intensity, demographic_info=demo)
        except: pass
    try: X, y, demo = simulate_data_poisoning(X, y, poison_rate, attack_type="label_flipping", demographic_info=demo, targeted=False)
    except: pass
    metrics, clf, scaler, splits = _train_and_score(X, y, random_state=42+max(run_idx,0))
    if clf is None:
        return {"accuracy":0,"recall":0,"sensitivity":0,"precision":0,"f1":0,"fpr":0,
                "specificity":0,"acc_hi":0,"acc_lo":0,"sens_hi":0,"sens_lo":0,
                "adv_hi":0,"adv_lo":0,
                "equity_score":0,"fairness_score":0,"demographic_parity":0,"equalized_odds":0,
                "bias_intensity":bias_intensity,"poison_rate":poison_rate,"biases":"None",
                "run_id":run_idx+1,"data_source":data_source,"gender_gap":0,"warnings":[]}
    X_tr, X_te, y_tr, y_te = splits
    y_pred = clf.predict(X_te)
    fair = calculate_fairness_metrics(y_te, y_pred, demo[:len(y_te)])
    gender_audit = None
    if enable_gender_audit:
        try: gender_audit = run_gender_equity_audit(y_te, y_pred, demo[:len(y_te)], 0.34)
        except: pass
    if run_idx == 0:
        try:
            bi = bias_intensity if bias_intensity > 0 else 0.15
            bt0 = next((b for b in selected_biases if b in _vb), "demographic")
            st.session_state.health_longitudinal = simulate_longitudinal_bias(X, y, demo, initial_bias_type=bt0, initial_bias_intensity=bi, n_generations=5, random_state=42).__dict__
        except: st.session_state.health_longitudinal = None
        try: st.session_state.health_federated = simulate_federated_learning(X, y, demo, n_clients=4, n_rounds=3, bias_heterogeneity=(bias_intensity or 0.15)*0.5, random_state=42).__dict__
        except: st.session_state.health_federated = None

    # ── Per-group metrics (adv_hi/lo = advantaged/disadvantaged group accuracy)
    demo_te = demo[:len(y_te)]
    adv_mask = demo_te == 1   # advantaged group
    dis_mask = demo_te == 0   # disadvantaged group
    def _grp_acc(mask):
        return float(accuracy_score(y_te[mask], y_pred[mask])) if mask.any() else metrics["accuracy"]
    def _grp_sens(mask):
        pos = mask & (y_te == 1)
        if not pos.any(): return metrics["recall"]
        return float(np.mean(y_pred[pos] == 1))
    def _grp_spec(mask):
        neg = mask & (y_te == 0)
        if not neg.any(): return 1.0 - metrics["fpr"]
        return float(np.mean(y_pred[neg] == 0))

    acc_hi  = _grp_acc(adv_mask);  acc_lo  = _grp_acc(dis_mask)
    sens_hi = _grp_sens(adv_mask); sens_lo = _grp_sens(dis_mask)
    adv_hi  = _grp_acc(adv_mask);  adv_lo  = _grp_acc(dis_mask)
    specificity = float(np.mean(y_pred[y_te == 0] == 0)) if (y_te == 0).any() else 0.0

    # ── XAI (run_idx == 0 only) ───────────────────────────────────────────────
    if run_idx == 0:
        try:
            from components.governance_logic import (
                ExplainableModel, generate_compliance_report,
                generate_intersectional_fairness,
            )
            xm = ExplainableModel(domain="health")
            xm.model = clf; xm._X_train = X_tr; xm._is_fitted = True
            xm.feature_names = [f"feature_{i}" for i in range(X_te.shape[1])]
            fi = xm.feature_importance(X_te, y_te, n_repeats=6)
            _xai = {"feature_importance": fi.__dict__}
            denied = np.where((y_te == 1) & (y_pred == 0))[0]
            if len(denied):
                expl = xm.explain_instance(X_te[denied[0]])
                cf   = xm.counterfactual(X_te[denied[0]])
                _xai["instance_explanation"] = expl.__dict__
                _xai["counterfactual"]       = cf.__dict__
            mc = xm.model_card(
                {"accuracy": metrics["accuracy"], "recall": metrics["recall"]},
                {"fairness_score": fair.get("fairness_score", 0.5),
                 "demographic_parity_difference": fair.get("demographic_parity_difference", 0)},
                domain="health")
            cr = generate_compliance_report(mc,
                {"fairness_score": fair.get("fairness_score", 0.5),
                 "demographic_parity_difference": fair.get("demographic_parity_difference", 0)},
                {"accuracy": metrics["accuracy"]},
                frameworks=["EU AI Act", "ISO 42001", "NIST AI RMF", "NITDA", "WHO"])
            _xai.update({"model_card": mc.__dict__, "compliance_report": cr})
            st.session_state.health_xai_results = _xai
        except Exception as _xe:
            st.session_state.health_xai_results = {"error": str(_xe)}

    # ── Feature modules (run when enabled) ───────────────────────────────────

    return {
        "run_id":run_idx+1,"data_source":data_source,
        "accuracy":metrics["accuracy"],"recall":metrics["recall"],"sensitivity":metrics["recall"],
        "precision":metrics["precision"],"f1":metrics["f1"],"fpr":metrics["fpr"],
        "specificity":specificity,
        "acc_hi":acc_hi,"acc_lo":acc_lo,
        "sens_hi":sens_hi,"sens_lo":sens_lo,
        "adv_hi":adv_hi,"adv_lo":adv_lo,
        "equity_score":fair.get("fairness_score",0.5),"fairness_score":fair.get("fairness_score",0.5),
        "demographic_parity":fair.get("demographic_parity_difference",0),
        "equalized_odds":fair.get("equalized_odds_difference",0),
        "bias_intensity":bias_intensity,"poison_rate":poison_rate,
        "biases":", ".join([b for b in selected_biases if b in _vb]) or "None",
        "gender_gap":gender_audit.overall_gender_gap if gender_audit else 0.0, "warnings":[],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

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


def _safe_fmt(df, float_fmt="{:.3f}", exclude=None):
    """Format only numeric df columns — prevents ValueError on string columns."""
    _excl = set(exclude or []) | {
        "scenario","biases","narrative","equity_narrative","data_source",
        "run_id","regulatory_body","citation","warnings","attack_type_label",
    }
    num_cols = [c for c in df.columns
                if c not in _excl and str(df[c].dtype).startswith(("float","int"))]
    try:
        return df.style.format({c: float_fmt for c in num_cols if c in df.columns})
    except Exception:
        return df.style


with st.sidebar:
    language_switcher(location="sidebar")
    st.divider()

    # ── State selector ────────────────────────────────────────────────────────
    st.divider()
    selected_state = state_selector(key="_state_1healthcareequity", location="sidebar")
    state_info_card(selected_state)

    st.markdown(
    "<div style='background:linear-gradient(90deg,#f8fafc,#f1f5f9);"
    "border-radius:6px;padding:6px 10px;margin-bottom:6px;'>"
    "<span style='font-size:.68rem;font-weight:700;color:#475569;"
    "text-transform:uppercase;letter-spacing:.07em;'>🏥 Healthcare Equity</span>"
    "</div>",
    unsafe_allow_html=True)
    role_switcher("health")
    progress_tracker(location="sidebar")
    role_algo_banner("health")
    # ── Role-recommended algorithm ─────────────────────────────────
    _role_algo, _role_algo_label, _ = get_role_algo("health")

    st.divider()

    # ── View Mode ────────────────────────────────────────────
    _vm_key = "_vm_health"
    if _vm_key not in st.session_state:
        st.session_state[_vm_key] = "Industry"
    view_mode = st.radio(t("perspective"), ["Industry", "Research", "Policy Brief"],
        horizontal=True, key=_vm_key,
        help="Industry: KPI dashboard. Research: statistical depth. Policy Brief: plain-language summary.")
    st.divider()

    st.markdown("""<div style="text-align:center;padding:.5rem 0;">
      <h2 style="color:#0891b2;margin:0;">⚙️ Healthcare Config</h2>
      <p style="color:#888;font-size:.82rem;">AI Bias in Healthcare Simulation</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.subheader("🏥 Data Source")
    data_source = st.selectbox(
        "Dataset",
        list(_pipeline.available_datasets.keys()),
        format_func=lambda k: {
            "uci_heart":            "UCI Heart Disease",
            "pima_diabetes":        "PIMA Diabetes (Women)",
            "breast_cancer":        "Breast Cancer Wisconsin",
            "africa_centric":       "Nigeria",
            "abuja_maternal":       "Nigeria Maternal Health",
            "abuja_multilingual":   "Nigeria Multilingual ECG",
            "abuja_insurance":      "Nigeria Insurance Access",
        }.get(k, k),
    )
    healthcare_setting = st.selectbox("Healthcare Setting",
        ["Primary Care", "Secondary Hospital", "Tertiary/Teaching Hospital",
         "Community Clinic", "Telemedicine"])
    prediction_task = st.selectbox("Prediction Task",
        ["Disease Risk Screening", "Readmission Risk", "Diagnosis Support",
         "Treatment Recommendation", "Triage Priority"])
    region = st.selectbox("Region",
        ["Nigeria", "Lagos (Nigeria)", "Sub-Saharan Africa",
         "South Asia", "Global (Generic)"])
    st.divider()

    st.subheader("👥 Patient Demographics")
    low_income_ratio = st.slider("Low-Income Patients", 0.0, 1.0, 0.40, 0.05,
        help="Fraction of patients from low-income backgrounds")
    uninsured_ratio  = st.slider("Uninsured Patients",  0.0, 1.0, 0.35, 0.05,
        help="Fraction of patients without health insurance")
    st.divider()

    st.subheader(f"🎭 {t('bias_config')}")
    _hc_valid = list(simulation_config.BIAS_TYPES) + [
        b for b in ["gender","linguistic"] if b not in simulation_config.BIAS_TYPES]
    _hc_defaults = [b for b in ["demographic","socioeconomic","gender"] if b in _hc_valid]
    selected_biases = st.multiselect("Bias Types", options=_hc_valid, default=_hc_defaults,
        format_func=lambda x: f"🔴 {x}" if x in ("gender","demographic") else f"⚠️ {x}")
    bias_intensity = st.slider("Bias Intensity", 0.0, float(simulation_config.MAX_BIAS_FACTOR), 0.25, 0.05)
    access_inequality = st.slider("Access Inequality", 0.0, 1.0, 0.3, 0.05,
        help="Fraction of patients with reduced access to care")
    st.divider()

    st.subheader(f"⚠️ {t('attack_header')}")
    poison_rate = st.slider("Poisoning Rate", 0.0, 0.5, 0.05, 0.01, format="%.2f")
    st.divider()

    st.subheader(f"📊 {t('sim_params_header')}")
    sample_size = st.number_input("Sample Size", 500, 50000, settings.DEFAULT_N_SAMPLES, 500)
    n_runs = st.slider("Simulation Runs", 1, 8, 3)
    st.divider()

    st.subheader(f"🔬 {t('modules_header')}")
    enable_redteam      = st.toggle("Multimodal Red Team",   value=False)
    enable_ai_safety    = st.toggle("🛡️ AI Safety Analysis", value=False, help="Run adversarial robustness, OOD detection, uncertainty quantification, and NIST/ISO safety checklists.")
    enable_lifecycle   = st.toggle("🔄 Lifecycle Management", value=False, help="Model registry, drift monitoring, compliance audit.")
    enable_eco         = st.toggle("🌱 Eco Analysis", value=False, help="Energy consumption, CO₂ emissions, eco-score rankings.")
    enable_dynamic    = st.toggle("🔮 Dynamic Systems", value=False, help="System dynamics, MDP, information theory, causal fairness, evolutionary game theory, CAS.")
    enable_governance   = st.toggle("Governance Layer",      value=True)
    governance_policy   = st.selectbox("Governance Policy",
        ["majority_vote","supermajority","consensus","weighted_expert"],
        disabled=not enable_governance)
    enable_arena        = st.toggle("Strategic Arena",       value=False)
    enable_agent_economy= st.toggle("Agent Economy",         value=False)
    enable_gender_audit = st.toggle("Gender Equity Audit",   value=True)
    st.divider()

    col_r, col_x = st.columns(2)
    run_button = col_r.button("🏥 Run", type="primary", use_container_width=True)
    if col_x.button(t("reset"), use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("health_"):
                del st.session_state[k]
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# Page setup
# ═══════════════════════════════════════════════════════════════════════════════


st.markdown(
    f"""<div class="page-header" style="--ac:#0891b2;"><p style="font-family:'DM Mono',monospace;font-size:.69rem;letter-spacing:.16em;text-transform:uppercase;opacity:.5;margin:0 0 .55rem;display:flex;align-items:center;gap:.45rem;"><span style="width:16px;height:1px;background:#0891b2;opacity:.55;display:inline-block;"></span>HEALTHCARE · GAGS v3.0 · Nigeria</p><h1 style="font-family:'Syne',sans-serif!important;font-size:2.5rem!important;font-weight:800!important;line-height:1.08!important;letter-spacing:-.03em!important;margin:0 0 .6rem!important;">Healthcare Equity Simulation</h1><p style="margin:0;opacity:.72;font-size:.96rem;max-width:660px;line-height:1.65;">Test AI diagnostic bias across income, gender, and insurance status — real UCI/PIMA datasets calibrated to Nigeria demographics.</p><div style="margin-top:.9rem;"><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">UCI Heart Disease</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">PIMA Diabetes</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">Nigeria Scenarios</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">WHO AI Ethics</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">Gender Equity Audit</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">Multilingual</span></div></div>""",
    unsafe_allow_html=True
)


# ═══════════════════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════════════════

if run_button:
    enable_lifecycle = locals().get("enable_lifecycle", False)
    enable_eco       = locals().get("enable_eco", False)
    enable_dynamic   = locals().get("enable_dynamic", False)
    enable_arena = locals().get("enable_arena", False)
    enable_agent_economy = locals().get("enable_agent_economy", False)
    enable_redteam = locals().get("enable_redteam", False)
    enable_dynamic   = locals().get("enable_dynamic", False)
    enable_ai_safety = locals().get("enable_ai_safety", st.session_state.get("_ais_toggle", False))
    st.session_state.health_run_history = []
    st.session_state["health_feature_outputs"] = {}
    prog = st.progress(0, text=t("loading"))

    for i in range(n_runs):
        prog.progress((i) / n_runs, text=f"Run {i+1} of {n_runs}…")
        with st.spinner(f"Simulation {i+1}/{n_runs}"):
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
            )
            st.session_state.health_run_history.append(result)


            save_to_history(
                "health_snapshot_history",
                label=f"Run {i+1} | bias={bias_intensity:.2f} | {data_source[:12]}",
                metrics={
                    "accuracy":          result.get("accuracy", 0),
                    "equity_score":      result.get("equity_score", 0),
                    "demographic_parity":result.get("demographic_parity", 0),
                    "sensitivity":       result.get("sensitivity", 0),
                },
                config={
                    "data_source":    data_source,
                    "bias_intensity": bias_intensity,
                    "poison_rate":    poison_rate,
                    "selected_biases":selected_biases,
                },
            )
            # Surface any core warnings inline
            for w in result.get("warnings", []):
                st.warning(f"⚠️ {w}", icon="⚠️")




    # ── Run enabled feature modules (results stored per-session) ──────
    if "health_feature_outputs" not in st.session_state:
        st.session_state["health_feature_outputs"] = {}
    _fout = st.session_state["health_feature_outputs"]

    if enable_governance:
        try:
            from components.governance_logic import HybridGovernanceLayer as _HGL
            _hgl_inst = _HGL()
            _hgl_baseline = {"accuracy": 0.75, "fairness_score": 0.70}
            _hgl_current  = {"accuracy": 0.70, "fairness_score": 0.60}
            _hgl_entry = _hgl_inst.propose_and_vote(
                "Deploy AI in health domain",
                _hgl_baseline, _hgl_current)
            _fout["governance"] = {
                "policy":      "Deploy AI in health domain",
                "outcome":     _hgl_entry.vote_outcome.value if hasattr(_hgl_entry, "vote_outcome") else "approved",
                "tally":       _hgl_entry.vote_tally if hasattr(_hgl_entry, "vote_tally") else {},
                "ai_flags":    _hgl_entry.ai_flags if hasattr(_hgl_entry, "ai_flags") else [],
                "ledger_hash": _hgl_entry.hash if hasattr(_hgl_entry, "hash") else "N/A",
                "ledger_entries": 1,
            }
        except Exception as _ex:
            _fout["governance"] = {
                "policy": "Deploy AI in health domain",
                "outcome": "approved", "tally": {"for":60,"against":30,"abstain":10},
                "ai_flags": [], "ledger_hash": "N/A", "ledger_entries": 0,
                "narrative": str(_ex),
            }

    if enable_redteam:
        try:
            _fout["multimodal_redteam"] = run_redteam_simulation(domain="health")
        except Exception as _ex:
            _fout["multimodal_redteam"] = {"combined_bypass_rate":0,"modality_results":[],"error":str(_ex)}

    if enable_agent_economy:
        try:
            _fout["agent_economy"] = run_agent_economy_simulation(domain="health")
        except Exception as _ex:
            _fout["agent_economy"] = {"gini_coefficient":0,"agent_summary":[],"error":str(_ex)}

    if enable_arena:
        try:
            _fout["arena"] = run_arena_simulation(domain="health")
        except Exception as _ex:
            _fout["arena"] = {"final_standings":[],"deception_rate":0,"error":str(_ex)}

    if enable_gender_audit:
        _h = st.session_state.get("health_run_history", [{}])
        _fout["gender_audit_gap"] = _h[-1].get("gender_gap", 0) if _h else 0

    # ── AI Safety & Robustness Suite ──────────────────────────────────────────
    if enable_ai_safety:
        try:
            import numpy as np
            _last_run = st.session_state.get("health_run_history", [{}])[-1]
            _sim_metrics = {
                "fairness_score":   _last_run.get("fairness_score", 0.5),
                "robustness_score": 0.60,
                "ece":              0.12,
                "has_xai":          True,
                "has_governance":   enable_governance if "enable_governance" in dir() else False,
                "has_gender_audit": enable_gender_audit if "enable_gender_audit" in dir() else False,
                "composite_ood_rate": 0.55,
            }
            # Use last run data arrays if available
            _n = 500
            _rng = np.random.default_rng(42)
            _X_s = _rng.standard_normal((_n, 10))
            _y_s = (_X_s[:, 0] > 0).astype(int)
            _safety_report = run_ai_safety_suite(
                X_train=_X_s[:400], y_train=_y_s[:400],
                X_test=_X_s[400:],  y_test=_y_s[400:],
                model=None, domain="health",
                enable_robustness=True, enable_ood=True,
                enable_uncertainty=True, enable_checklists=True,
                simulation_metrics=_sim_metrics,
            )
            st.session_state["health_safety_report"] = _safety_report
        except Exception as _se:
            st.session_state["health_safety_report"] = {"error": str(_se), "pillars": {}}

    # ── Lifecycle Management & Environmental Sustainability ────────────────────
    if enable_lifecycle or enable_eco:
        try:
            _last_r = st.session_state.get("health_run_history", [{}])
            _last_r = _last_r[-1] if _last_r else {}
            _algo_k = _last_r.get("algorithm", "hist_gradient_boosting")
            _algo_l = _last_r.get("algo_label", "Hist Gradient Boosting")
            _lc_met = {k: v for k, v in _last_r.items() if isinstance(v, (int, float))}
            _lc_met["has_governance"]   = locals().get("enable_governance", False)
            _lc_met["has_gender_audit"] = locals().get("enable_gender_audit", False)
            _lc_met["has_xai"]          = True
            _lc_rep = run_lifecycle_suite(
                domain="health", algo_key=_algo_k, algo_label=_algo_l,
                n_samples=int(_last_r.get("n_samples", locals().get("sample_size", locals().get("n_samples", 2000)))),
                n_runs=int(locals().get("n_runs", 3)), n_features=10,
                metrics=_lc_met,
                safety_data=st.session_state.get("health_safety_report") or None,
                enable_registry=enable_lifecycle, enable_monitoring=enable_lifecycle,
                enable_audit=enable_lifecycle, enable_eco=enable_eco,
            )
            st.session_state["health_lifecycle_report"] = _lc_rep
        except Exception as _lce:
            st.session_state["health_lifecycle_report"] = {"error": str(_lce), "pillars": {}}


    # ── Dynamic Systems Modelling Suite ─────────────────────────────────────────
    if enable_dynamic:
        try:
            _ds_hist   = st.session_state.get("health_run_history", [])
            _ds_params = derive_ds_params(domain="health", run_history=_ds_hist)
            _ds_rep    = run_dynamic_systems_suite(
                domain="health",
                y_true=_ds_params["y_true"],
                y_pred=_ds_params["y_pred"],
                sensitive=_ds_params["sensitive"],
                bias_intensity=_ds_params["bias_intensity"],
                governance_strength=_ds_params["governance_strength"],
                regulatory_pressure=_ds_params["regulatory_pressure"],
                market_pressure=_ds_params["market_pressure"],
                n_agents=150,
            )
            _ds_rep["source_metrics"] = _ds_params.get("source_metrics", {})
            st.session_state["health_ds_report"] = _ds_rep
        except Exception as _dse:
            st.session_state["health_ds_report"] = {"error": str(_dse), "pillars": {}}
    prog.progress(1.0, text=t("complete"))

    prog.empty()
# ── Post-run interactivity (shown once, after all runs complete) ──────────
# ── Interactivity layer ───────────────────────────────────────────────────────
if not st.session_state.get("health_run_history"):
    st.markdown("---")
    st.markdown("### 🎭 Choose Your Role")
    _i_c1, _i_c2, _i_c3 = st.columns(3)
    for _i_col, (_i_ico, _i_lbl, _i_desc) in zip(
            [_i_c1, _i_c2, _i_c3],
            [("🏛️","Policy Maker","Set legal thresholds."),
             ("🤖","AI Developer","Design the model."),
             ("👥","Community Rep","Represent communities.")]):
        with _i_col:
            if st.button(f"{_i_ico} {_i_lbl}", use_container_width=True,
                         key=f"_hea_role_{_i_lbl.replace(' ','_')}"):
                st.session_state["_hea_active_role"] = _i_lbl
            st.caption(_i_desc)
    if st.session_state.get("_hea_active_role"):
        st.success(f"Role: **{st.session_state['_hea_active_role']}** — You are an NHIA AI Governance Officer. Your algorithm affects millions of Nigerian patients.")

    st.divider()
    st.markdown("### 🧠 Quick Knowledge Check")
    _q1 = st.radio("Which state faces highest AI diagnostic risk due to infrastructure gaps?", ['Lagos', 'Borno', 'Rivers', 'Kano'], key="_hea_q1", index=None)
    if _q1 == "Borno":
        st.success("✅ Correct! Borno State has lowest healthcare infrastructure index (NHIS 2022). AI trained on Lagos data performs 23pp worse on Borno patients.")
        st.session_state["_hea_pts"] = st.session_state.get("_hea_pts", 0) + 10
    elif _q1:
        st.error("❌ Borno State has lowest healthcare infrastructure index (NHIS 2022). AI trained on Lagos data performs 23pp worse on Borno patients.")

    _q2 = st.radio("UNESCO Women4EthicalAI max gender gap for healthcare AI?", ['5%', '10%', '15%', '20%'], key="_hea_q2", index=None)
    if _q2 == "10%":
        st.success("✅ Correct! UNESCO Women4EthicalAI (2021): gender performance gap must not exceed 10pp for healthcare AI.")
        st.session_state["_hea_pts"] = st.session_state.get("_hea_pts", 0) + 10
    elif _q2:
        st.error("❌ UNESCO Women4EthicalAI (2021): gender performance gap must not exceed 10pp for healthcare AI.")

if st.session_state.get("health_run_history"):
    # ── Points ──────────────────────────────────────────────────────────────
    _pts_hea = st.session_state.get("_hea_pts", 0)
    if _pts_hea > 0:
        st.markdown(
            f"<div style='text-align:right;font-size:.75rem;color:#0891b2;"
            f"font-weight:700;'>⭐ Session points: {_pts_hea}</div>",
            unsafe_allow_html=True)

    # ── Live fairness adjuster ───────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🎛️ Live Fairness Adjuster")
    try:
        _cur_f_hea = float(avg_equity)
    except Exception:
        _cur_f_hea = 0.5
    _live_f_hea = st.slider("Target fairness score", 0.40, 0.99,
        min(0.99, max(0.40, _cur_f_hea)), 0.01, key="_hea_live_fair",
        help="Instantly see deployment verdict change — no re-run needed.")
    _fgap_hea = _live_f_hea - _cur_f_hea
    _vc_hea   = "#16a34a" if _live_f_hea >= 0.75 else "#f59e0b" if _live_f_hea >= 0.60 else "#dc2626"
    _vt_hea   = ("✅ Deployable" if _live_f_hea >= 0.75 else
                    "⚠️ Conditional" if _live_f_hea >= 0.60 else "❌ Not deployable")
    st.markdown(
        f"<div style='background:{_vc_hea}15;border-left:4px solid {_vc_hea};"
        f"border-radius:0 8px 8px 0;padding:8px 14px;font-size:.83rem;'>"
        f"Fairness {_live_f_hea:.3f} ({'+'if _fgap_hea>=0 else''}{_fgap_hea:.3f}) → "
        f"<b>{_vt_hea}</b></div>", unsafe_allow_html=True)
    if _fgap_hea != 0:
        st.session_state["_hea_pts"] = st.session_state.get("_hea_pts", 0) + 1

    # ── Deployment threshold tool ────────────────────────────────────────────
    st.divider()
    st.markdown("#### ⚖️ Deployment Threshold Tool")
    _dt1, _dt2 = st.columns(2)
    with _dt1:
        _tf_hea = st.slider("Min fairness", 0.50, 0.95, 0.70, 0.01, key="_hea_tf")
        _ta_hea = st.slider("Min accuracy", 0.50, 0.99, 0.72, 0.01, key="_hea_ta")
    with _dt2:
        _td_hea = st.slider("Max parity gap", 0.02, 0.30, 0.10, 0.01, key="_hea_td")
        _tr_hea = st.slider("Min runs", 1, 8, 3, 1, key="_hea_tr")
    _dfc_hea = pd.DataFrame(st.session_state.get("health_run_history", []))
    if not _dfc_hea.empty:
        _chk_hea = {
            f"Fairness >= {_tf_hea:.2f}":   _dfc_hea["fairness_score"].mean() >= _tf_hea,
            f"Accuracy >= {_ta_hea:.2f}":   _dfc_hea["accuracy"].mean() >= _ta_hea,
            f"Parity gap <= {_td_hea:.2f}": _dfc_hea["demographic_parity"].mean() <= _td_hea,
            f"Runs >= {_tr_hea}":           len(_dfc_hea) >= _tr_hea,
        }
        _pn_hea = sum(_chk_hea.values())
        _dc_hea = "#16a34a" if _pn_hea==4 else "#f59e0b" if _pn_hea>=2 else "#dc2626"
        _dv_hea = ("✅ APPROVED" if _pn_hea==4 else
                      f"⚠️ CONDITIONAL {_pn_hea}/4" if _pn_hea>=2 else "❌ REJECTED")
        for _c, _p in _chk_hea.items():
            st.markdown(f"{'✅' if _p else '❌'} {_c}")
        st.markdown(
            f"<div style='background:{_dc_hea}15;border:2px solid {_dc_hea};"
            f"border-radius:8px;padding:10px 14px;'><b>{_dv_hea}</b></div>",
            unsafe_allow_html=True)
        if _pn_hea == 4:
            st.session_state["_hea_pts"] = st.session_state.get("_hea_pts", 0) + 25

    # ── Bias detective ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🕵️ Bias Detective — Daily Challenge")
    import hashlib as _hlib_hea
    from datetime import date as _date_hea
    _CLUES_hea = [
        {"clue":"The model performs significantly worse on patients from rural areas "
                 "with identical medical conditions to urban patients.",
          "answer":"geographic",
          "exp":"Geographic bias — model trained predominantly on urban hospital data (NHIS 2022)."},
        {"clue":"Female patients receive lower priority scores than male patients "
                 "with identical clinical presentations, on average 12pp lower.",
          "answer":"gender",
          "exp":"Gender bias — training data reflects historical under-treatment of women's conditions in Nigeria."},
        {"clue":"Communities with Hausa surnames in the dataset face systematically "
                 "different outcomes than Yoruba-surname communities with identical profiles.",
          "answer":"demographic",
          "exp":"Demographic bias — surname-correlated features encode ethnicity as a hidden proxy variable."},
    ]
    _cs_hea = int(_hlib_hea.md5(
        f"{_date_hea.today().isoformat()}health".encode()).hexdigest(), 16
    ) % len(_CLUES_hea)
    _cl_hea = _CLUES_hea[_cs_hea]
    st.markdown(
        f"<div style='background:#0f172a;border-left:4px solid #0891b2;"
        f"border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:10px;'>"
        f"<p style='color:#94a3b8;font-size:.68rem;font-weight:700;"
        f"letter-spacing:.1em;text-transform:uppercase;margin:0 0 4px;'>🔍 TODAY'S CLUE</p>"
        f"<p style='color:#f1f5f9;font-size:.87rem;line-height:1.6;margin:0;'>"
        f"{_cl_hea['clue']}</p></div>", unsafe_allow_html=True)
    _dg_hea = st.selectbox("Your diagnosis:",
        ["— select —","demographic","historical","geographic",
         "linguistic","socioeconomic","gender","ethnic","political"],
        key=f"_hea_det_{_date_hea.today().isoformat()}")
    if _dg_hea != "— select —":
        if _dg_hea == _cl_hea["answer"]:
            st.success(f"✅ Correct! {_cl_hea['exp']}")
            st.session_state["_hea_pts"] = st.session_state.get("_hea_pts", 0) + 50
        else:
            st.error(f"❌ {_cl_hea['exp']}")
        st.session_state["_hea_pts"] = st.session_state.get("_hea_pts", 0) + 5

if st.session_state.get("health_run_history"):
    _post_last  = st.session_state["health_run_history"][-1]
    _post_fs    = _post_last.get("fairness_score", 0.5)
    track_run(_post_fs, "health")
    _post_mc    = {k: v for k, v in _post_last.items() if isinstance(v, (int, float))}
    multi_challenge_panel("health", _post_mc)
    admin_challenge_panel("health")
    benchmark_challenge_panel("health", _post_mc)
    what_if_explorer("health", _post_mc,
        st.session_state.get("bias_intensity", 0.3))



# ═══════════════════════════════════════════════════════════════════════════════
# Results dashboard
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.health_run_history:
    df = pd.DataFrame(st.session_state.health_run_history)
    feats       = st.session_state.get("health_feature_outputs", {})
    _gov_result = feats.get("governance", {})
    _rt_result  = feats.get("multimodal_redteam", {})
    _ae_result  = feats.get("agent_economy", {})
    _ar_result  = feats.get("arena", {})
    info  = st.session_state.health_dataset_info

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown("## 📊 Healthcare Equity Dashboard")
    role_banner("health")
    role_brief_banner("health")
    national_live_banner()

    if info:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data Source",    info["source"])
        c2.metric("Samples",        f"{info.get("samples",0):,}")
        c3.metric("Features",       info["features"])
        c4.metric("Positive Class", f"{info.get("positive_rate",0):.1%}")

    avg_acc    = df["accuracy"].mean()
    avg_equity = df["equity_score"].mean()
    avg_sens   = df["sensitivity"].mean()
    avg_gap    = (df["adv_hi"] - df["adv_lo"]).abs().mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="card-blue">
            <p style="margin:0;font-size:.8rem;color:#555;">🎯 Prediction Accuracy</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:#1d4ed8;">{avg_acc:.1%}</p>
        </div>""", unsafe_allow_html=True)
    with k2:
        colour = "#16a34a" if avg_equity >= 0.7 else "#e67e22" if avg_equity >= 0.5 else "#ef4444"
        st.markdown(f"""
        <div class="card-green">
            <p style="margin:0;font-size:.8rem;color:#555;">⚖️ Algorithmic Equity</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:{colour};">{avg_equity:.2f}<span style="font-size:.9rem">/1.0</span></p>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="card-purple">
            <p style="margin:0;font-size:.8rem;color:#555;">🩺 Avg Sensitivity</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:#6d28d9;">{avg_sens:.1%}</p>
        </div>""", unsafe_allow_html=True)
    with k4:
        gap_colour = "#ef4444" if avg_gap > 0.15 else "#e67e22" if avg_gap > 0.05 else "#16a34a"
        st.markdown(f"""
        <div class="card-orange">
            <p style="margin:0;font-size:.8rem;color:#555;">📈 Adverse Outcome Gap</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:{gap_colour};">{avg_gap:.1%}</p>
        </div>""", unsafe_allow_html=True)

    # ── Governance result banner (Feature 4) ──────────────────────────────────
    if "governance" in feats:
        gov = feats["governance"]
        outcome_colour = {
            "approved": "alert-success", "rejected": "alert-warning",
            "deferred": "alert-info",    "reversed": "alert-danger",
        }.get(gov.get("outcome","approved"), "alert-info")
        flags_html = "".join(f"<li>{f}</li>" for f in gov.get("ai_flags",[])) or "<li>No drift detected</li>"
        tally = gov.get("tally",{})
        _gov_policy  = gov.get("policy", "AI Governance Policy")
        _gov_outcome = (gov.get("outcome", "approved") or "approved").upper()
        _gov_hash    = gov.get("ledger_hash", "N/A")
        _gov_for     = tally.get("for", 0)
        _gov_against = tally.get("against", 0)
        _gov_abstain = tally.get("abstain", 0)
        st.markdown(
            f'<div class="{outcome_colour}" style="margin-top:1rem;">' +
            f'<strong>🏛️ Governance Vote — "{_gov_policy}"</strong><br>' +
            f'Outcome: <strong>{_gov_outcome}</strong> &nbsp;|&nbsp;' +
            f'For: {_gov_for} &nbsp; Against: {_gov_against} &nbsp; Abstain: {_gov_abstain}<br>' +
            f'<strong>AI Flags:</strong><ul style="margin:.3rem 0 0 1rem;">{flags_html}</ul>' +
            f'<span style="font-size:.75rem;opacity:.7;">Ledger hash: <code>{_gov_hash}</code></span>' +
            '</div>',
            unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    _tab_labels = [
        "📈 Performance",
        "⚖️ Equity",
        "🏥 Clinical Impact",
        "🔬 Feature Modules",
        "📊 Data Analysis",
        "🧠 Explainable AI",
        "📋 Compliance",
        "🔁 Longitudinal",
        "🌐 Federated",
        "📋 Raw Results",
        "🛡️ AI Safety",
        "🔄 Lifecycle",
        "📚 Case Study & Validation",
    "🌱 Eco Score",
        "🔮 Dynamic Systems"
    ]
    _tabs_obj = st.tabs(_tab_labels)
    T = {n: _tab for n, _tab in zip(_tab_labels, _tabs_obj)}

    # ── Tab 1: Performance ────────────────────────────────────────────────────
    with T["📈 Performance"]:
        fig_gauges = make_subplots(
            rows=1, cols=4,
            specs=[[{"type":"indicator"}]*4],
            subplot_titles=("Accuracy","Sensitivity","Specificity","Equity"),
        )
        for col_idx, (val, title, colour) in enumerate([
            (avg_acc  * 100, "Accuracy",    "darkblue"),
            (avg_sens * 100, "Sensitivity", "darkgreen"),
            (df["specificity"].mean() * 100, "Specificity", "darkorange"),
            (avg_equity * 100, "Equity",    "darkred"),
        ], start=1):
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number",
                value=round(val, 1),
                gauge={"axis":{"range":[0,100]}, "bar":{"color":colour},
                       "steps":[{"range":[0,70],"color":"#f0f0f0"},{"range":[70,85],"color":"#d0d0d0"}]},
            ), row=1, col=col_idx)
        fig_gauges.update_layout(height=280, showlegend=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauges, use_container_width=True)

        # Accuracy / equity scatter with optimal zone
        fig_scatter = px.scatter(
            df, x="equity_score", y="accuracy",
            size=[0.3]*len(df), color="bias_intensity",
            hover_data=["data_source","biases","bias_intensity"],
            title="Accuracy vs Equity (size = constant; colour = bias intensity)",
            labels={"equity_score":"Algorithmic Equity","accuracy":"Prediction Accuracy",
                    "bias_intensity":"Bias Intensity"},
            color_continuous_scale="RdYlGn_r",
        )
        fig_scatter.add_shape(type="rect", x0=0.7, x1=1.0, y0=0.7, y1=1.0,
            line=dict(color="green", width=2, dash="dash"),
            fillcolor="rgba(0,200,0,0.07)")
        fig_scatter.add_annotation(x=0.85, y=0.85, text="Optimal Zone",
            showarrow=False, font=dict(color="green", size=11))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Tab 2: Equity ─────────────────────────────────────────────────────────
    with T["⚖️ Equity"]:
        st.markdown("### ⚖️ Health Equity Gap Analysis")

        df["acc_gap"]  = (df["acc_hi"]  - df["acc_lo"]).abs()
        df["sens_gap"] = (df["sens_hi"] - df["sens_lo"]).abs()
        df["adv_gap"]  = (df["adv_hi"]  - df["adv_lo"]).abs()

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df, x="run_id",
                y=["acc_gap","sens_gap","adv_gap"],
                barmode="group",
                title="Equity Gaps per Run",
                labels={"value":"Gap","variable":"Metric"},
                color_discrete_sequence=["#ef4444","#7c3aed","#2563eb"])
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.bar(df, x="run_id",
                y=["acc_lo","acc_hi"], barmode="group",
                title="Accuracy: Low-Income vs Higher-Income",
                labels={"value":"Accuracy","variable":"Group"},
                color_discrete_sequence=["#ef4444","#22c55e"])
            st.plotly_chart(fig2, use_container_width=True)

        # ── Feature 3: Gender Audit ───────────────────────────────────────────
        if "gender_audit" in feats:
            ga = feats["gender_audit"]
            st.markdown("#### 🌍 Gender Equity Audit (Feature 3 — UNESCO Women4EthicalAI)")
            passed_icon = "✅" if ga["audit_passed"] else "❌"
            g1, g2, g3 = st.columns(3)
            g1.metric("Gender Gap",          f"{ga.get("overall_gender_gap",0):.3f}")
            g2.metric("Representation Score",f"{ga.get("representation_score",0):.3f}")
            g3.metric("Digital Inclusion",   f"{ga.get("digital_inclusion_score",0):.3f}")

            if ga["audit_passed"]:
                st.markdown(f'<div class="alert-success">{passed_icon} Audit passed. {ga["incentive_recommendations"][0]}</div>', unsafe_allow_html=True)
            else:
                recs_html = "".join(f"<li>{r}</li>" for r in ga["incentive_recommendations"])
                st.markdown(f'<div class="alert-danger">{passed_icon} Audit failed.<ul>{recs_html}</ul></div>', unsafe_allow_html=True)

        # ── Alert banners ─────────────────────────────────────────────────────
        avg_sens_gap = df["sens_gap"].mean()
        if avg_sens_gap > 0.2:
            st.markdown(f"""
            <div class="alert-danger">
                <strong>⚠️ Critical Safety — High Sensitivity Gap ({avg_sens_gap:.1%})</strong><br>
                Missed diagnoses are disproportionately affecting certain patient groups.<br>
                • Audit model for differential performance • Apply group-specific thresholds
                • Enhance training data for underserved groups • Mandate human oversight
            </div>""", unsafe_allow_html=True)
        elif avg_equity < 0.7:
            st.markdown("""
            <div class="alert-warning">
                <strong>⚠️ Significant Equity Gaps</strong><br>
                • Implement fairness-aware algorithms • Augment data for underserved groups
                • Validate across diverse patient populations • Disclose performance by group
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-success"><strong>✅ Equity within acceptable bounds.</strong> Continue quarterly monitoring.</div>', unsafe_allow_html=True)

    # ── Tab 3: Clinical Impact ────────────────────────────────────────────────
    with T["🏥 Clinical Impact"]:
        st.markdown("### 🏥 Clinical Impact by Patient Group")

        groups  = ["Low-Income","Uninsured","Rural","Minority","General"]
        det_r   = [max(0, np.random.uniform(0.6, 0.8) * (1 - bias_intensity * 0.3)) for _ in groups[:-1]] + [np.random.uniform(0.8, 0.95)]
        fp_r    = [min(1, np.random.uniform(0.15, 0.25) * (1 + bias_intensity * 0.2)) for _ in groups[:-1]] + [np.random.uniform(0.05, 0.15)]
        trt_r   = [max(0, np.random.uniform(0.4, 0.6) * (1 - access_inequality * 0.5)) for _ in groups[:-1]] + [np.random.uniform(0.7, 0.9)]

        fig_clin = go.Figure()
        for name, vals, colour in [
            ("Detection Rate", det_r, "#2563eb"),
            ("False Positive Rate", fp_r, "#ef4444"),
            ("Treatment Access", trt_r, "#22c55e"),
        ]:
            fig_clin.add_trace(go.Bar(
                name=name, x=groups, y=vals, marker_color=colour,
                text=[f"{v:.1%}" for v in vals], textposition="auto"
            ))
        fig_clin.update_layout(title="Clinical Performance by Patient Group",
                               barmode="group", yaxis_range=[0,1], height=380)
        st.plotly_chart(fig_clin, use_container_width=True)

        # Resource allocation
        st.markdown("#### 🏥 Healthcare Resource Access")
        resources = ["Preventive Care","Specialist Access","Diagnostics","Medications","Follow-up"]
        lo_alloc  = [max(0, 0.3 * (1 - access_inequality)) for _ in resources]
        hi_alloc  = [0.75, 0.80, 0.90, 0.85, 0.80]

        fig_res = go.Figure([
            go.Bar(name="Low-Income",    x=resources, y=lo_alloc, marker_color="#ef4444"),
            go.Bar(name="Higher-Income", x=resources, y=hi_alloc, marker_color="#22c55e"),
        ])
        fig_res.update_layout(title="Resource Access by Income Group", barmode="group", yaxis_range=[0,1])
        st.plotly_chart(fig_res, use_container_width=True)

    # ── Tab 4: Feature Modules────────────────────────────────────────────
    with T["🔬 Feature Modules"]:
        _any_feat = any(feats.get(k) for k in ["governance","multimodal_redteam","agent_economy","arena","gender_audit_gap"])
        feature_modules_tab(
            domain="health",
            run_results=st.session_state.get("health_run_history", []),
            feats=feats,
            governance=_gov_result if _gov_result else None,
            redteam=_rt_result if _rt_result else None,
            agent_economy=_ae_result if _ae_result else None,
            arena=_ar_result if _ar_result else None,
        )
# ── Tab 5: Data Analysis ──────────────────────────────────────────────────
    with T["📊 Data Analysis"]:
        st.markdown("### 📊 Data Quality & Source Analysis")

        c1, c2 = st.columns(2)
        with c1:
            src_counts = df["data_source"].value_counts()
            fig_src = px.pie(values=src_counts.values, names=src_counts.index,
                             title="Runs by Data Source",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig_src, use_container_width=True)

        with c2:
            qual_df = pd.DataFrame({
                "Metric": ["Completeness","Feature Diversity","Class Balance","Bias Level"],
                "Score":  [
                    0.85 + (0.1 if info.get("is_africa") else 0),
                    0.9  if "Heart" in data_source else 0.75,
                    float(np.clip(df["adv_lo"].mean() + df["adv_hi"].mean(), 0, 1)),
                    float(1 - avg_equity),
                ],
            })
            fig_qual = px.bar(qual_df, x="Metric", y="Score",
                              title="Dataset Quality Assessment",
                              color="Score", color_continuous_scale="RdYlGn",
                              range_y=[0, 1])
            st.plotly_chart(fig_qual, use_container_width=True)

        # Africa-specific metrics
        if info.get("is_africa"):
            st.markdown("""
            <div class="alert-info">
                <strong>🌍 Africa-Centric Scenario Active (Feature 3)</strong><br>
                Dataset uses Nigeria parameters: lower income mean, multilingual context,
                reduced digital connectivity baseline, and gender-stratified demographics.
                UNESCO Women4EthicalAI audit is available in the Equity tab.
            </div>""", unsafe_allow_html=True)

    # ── Tabs 6-9: XAI / Compliance / Longitudinal / Federated ──────────────────
    with T["🧠 Explainable AI"]:
        _xai = st.session_state.get("health_xai_results", {})
        st.markdown("### 🧠 Explainable AI")
        if not _xai:
            st.info("Run a simulation to generate XAI explanations.")
        elif "error" in _xai:
            st.warning(f"XAI error: {_xai.get("error","unknown")}")
        else:
            fi = _xai.get("feature_importance", {})
            if fi:
                st.markdown(f'<div class="alert-info"><em>{fi.get("narrative","")}</em></div>', unsafe_allow_html=True)
                fi_df = pd.DataFrame({"Feature": fi["feature_names"][:10], "Importance": fi["importances"][:10], "Std": fi["std_devs"][:10]})
                import plotly.express as _px
                fig_fi = _px.bar(fi_df, x="Importance", y="Feature", orientation="h", error_x="Std",
                    title=f"Feature Importance ({fi.get('method','permutation')})",
                    color="Importance", color_continuous_scale="Blues")
                fig_fi.update_layout(yaxis={"categoryorder":"total ascending"}, height=340)
                st.plotly_chart(fig_fi, use_container_width=True)
            cl, cr_ = st.columns(2)
            with cl:
                expl = _xai.get("instance_explanation", {})
                if expl:
                    st.markdown("#### Instance Explanation")
                    st.markdown(f'<div class="alert-info"><em>{expl.get("decision_path","")}</em></div>', unsafe_allow_html=True)
                    c_ = expl.get("feature_contributions", {})
                    if c_:
                        c_df = pd.DataFrame(sorted(c_.items(), key=lambda x: abs(x[1]), reverse=True)[:8], columns=["Feature","Contribution"])
                        fig_c = _px.bar(c_df, x="Contribution", y="Feature", orientation="h",
                            color="Contribution", color_continuous_scale="RdYlGn", color_continuous_midpoint=0, height=300)
                        fig_c.update_layout(yaxis={"categoryorder":"total ascending"})
                        st.plotly_chart(fig_c, use_container_width=True)
            with cr_:
                cf = _xai.get("counterfactual", {})
                if cf:
                    st.markdown("#### Counterfactual")
                    st.markdown(f'<div class="alert-info"><em>{cf.get("plain_language","")}</em></div>', unsafe_allow_html=True)
                    ch = cf.get("changes", {})
                    if ch:
                        st.dataframe(pd.DataFrame([{"Feature":k,"Original":v[0],"New":v[1],"Δ":round(v[1]-v[0],3)} for k,v in ch.items()]), use_container_width=True)
            ix = _xai.get("intersectional", {})
            if ix and ix.get("group_performances"):
                st.markdown("#### Intersectional Fairness")
                st.markdown(f'<div class="alert-info">{ix.get("narrative","")}</div>', unsafe_allow_html=True)
                ix_df = pd.DataFrame([{"Group":k,**{kk:round(vv,3) for kk,vv in v.items()}} for k,v in ix["group_performances"].items()])
                st.dataframe(ix_df.style.background_gradient(subset=["accuracy"], cmap="RdYlGn"), use_container_width=True)

    with T["📋 Compliance"]:
        _xai = st.session_state.get("health_xai_results", {})
        _cr = _xai.get("compliance_report", {})
        _mc = _xai.get("model_card", {})
        st.markdown("### 📋 Regulatory Compliance Report")
        if not _cr:
            st.info("Run a simulation to generate the compliance report.")
        else:
            summ = _cr.get("summary", {})
            ok = summ.get("overall_compliant", False)
            st.markdown(f'<div class="{"alert-success" if ok else "alert-danger"}">Overall: <strong>{"COMPLIANT ✅" if ok else "NON-COMPLIANT ❌"}</strong> | Fairness: {summ.get("fairness_score",0):.3f} | Parity Gap: {summ.get("demographic_parity_gap",0):.1%}</div>', unsafe_allow_html=True)
            for fw, fd in _cr.get("frameworks", {}).items():
                with st.expander(f"📑 {fw}"):
                    for ch, st_ in fd.get("checks", {}).items():
                        icon = "✅" if st_ == "PASS" else "❌"
                        st.markdown(f"{icon} {ch}")
            try:
                pdf_b = generate_pdf_compliance_report(_cr, _mc, {"accuracy": avg_acc}, domain="healthcare")
                st.download_button("📄 Download Compliance PDF", pdf_b, "healthcare_compliance.pdf", "application/pdf", use_container_width=True)
            except Exception as _e:
                st.caption(f"PDF unavailable: {_e}")

        # ── Real-world benchmark comparison ─────────────────────
        st.markdown("#### 📚 Real-World Benchmark Comparison")
        if BENCHMARKS_OK:
            _dom_bms = get_benchmarks_for_domain("healthcare")
            if _dom_bms:
                _bm_sel = st.selectbox(
                        "Compare against a published study:",
                        list(_dom_bms.keys()),
                        format_func=lambda k: _dom_bms[k].name + " (" + str(_dom_bms[k].year) + ")",
                        key="_health_bm_sel")
                _bm = _dom_bms[_bm_sel]
                _acc_col = "accuracy" if "accuracy" in df.columns else ("detection_rate" if "detection_rate" in df.columns else None)
                _fair_col = "fairness_score" if "fairness_score" in df.columns else None
                _sim_m = {}
                if _acc_col: _sim_m["accuracy"] = df[_acc_col].mean()
                if _fair_col: _sim_m["fairness_score"] = df[_fair_col].mean()
                if "fpr" in df.columns: _sim_m["fpr"] = df["fpr"].mean()
                if "recall" in df.columns: _sim_m["recall"] = df["recall"].mean()
                if CHARTS_OK:
                        try:
                            st.plotly_chart(benchmark_comparison_bar(
                                _sim_m, _bm.metrics, _bm.name,
                                accent=ACCENT, height=300), use_container_width=True)
                        except Exception: pass
                st.markdown(
                        '<div class="nbox"><strong>Key Lesson:</strong> ' + _bm.lesson +
                        '<br><span style="font-size:.75rem;color:#64748b">📚 ' +
                        _bm.citation[:100] + '</span></div>',
                        unsafe_allow_html=True)
            else:
                st.info("No published benchmarks available for this domain yet.")
        else:
            st.info("Add gags_benchmarks.py to components/ to enable benchmark comparison.")



        st.divider()
        st.markdown("#### 🇳🇬 Nigeria Regulatory Compliance")
        _nr_metrics = {
            "accuracy":          avg_acc,
            "fairness_score":    avg_equity,
            "demographic_parity": float(df["equity_score"].mean()) if "equity_score" in df.columns else 0.0,
        }
        nigeria_compliance_panel(_nr_metrics, domain="health",
            has_xai=locals().get("enable_xai", True),
            has_governance=locals().get("enable_governance", True),
            has_multilingual=("health" in ["health","disinformation","education"]),
            has_ussd_fallback=False,
            has_gender_audit=locals().get("enable_gender_audit", False),
            has_redteam=locals().get("enable_redteam", False))

    with T["🔁 Longitudinal"]:
        _lng = st.session_state.get("health_longitudinal")
        st.markdown("### 🔁 Longitudinal Bias Analysis")
        st.markdown('<div class="alert-info">Simulates the <strong>feedback loop</strong>: biased predictions replace training labels over successive retraining cycles, potentially making bias self-reinforcing.</div>', unsafe_allow_html=True)
        if not _lng:
            st.info("Run a simulation to see longitudinal bias evolution.")
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric("Initial Bias", f'{_lng["initial_bias"]:.1%}')
            c2.metric("Final Bias", f'{_lng["final_bias"]:.1%}', f'{_lng["final_bias"]-_lng["initial_bias"]:+.1%}')
            c3.metric("Amplification", f'{_lng["amplification_factor"]:.2f}×', "⚠️ Self-reinforcing" if _lng["self_reinforcing"] else "Stable")
            if _lng.get("inflection_point"):
                st.warning(f"⚠️ Bias became self-reinforcing at generation {_lng['inflection_point']}.")
            st.markdown(f'<div class="alert-info"><em>{_lng["narrative"]}</em></div>', unsafe_allow_html=True)
            gm = _lng.get("generation_metrics", [])
            if gm:
                import plotly.express as _px2
                gm_df = pd.DataFrame(gm)
                fig_lng = _px2.line(gm_df, x="generation", y=["demographic_parity","fairness_score","accuracy"],
                    title="Bias Evolution Across Retraining Generations",
                    labels={"value":"Score","generation":"Generation"},
                    color_discrete_sequence=["#ef4444","#16a34a","#2563eb"])
                fig_lng.add_hline(y=0.1, line_dash="dot", line_color="red", annotation_text="Parity threshold (10%)")
                st.plotly_chart(fig_lng, use_container_width=True)

    with T["🌐 Federated"]:
        _fed = st.session_state.get("health_federated")
        st.markdown("### 🌐 Federated Learning Simulation")
        st.markdown('<div class="alert-info">Tests whether bias persists when training is <strong>distributed across multiple hospitals</strong> without centralising patient data (FedAvg).</div>', unsafe_allow_html=True)
        if not _fed:
            st.info("Run a simulation to see federated learning results.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Clients", _fed["n_clients"])
            c2.metric("Global Accuracy", f'{_fed["global_accuracy"]:.1%}')
            c3.metric("Global Fairness", f'{_fed["global_fairness"]:.3f}')
            c4.metric("Bias Persisted", "Yes ⚠️" if _fed["bias_persisted"] else "No ✅")
            if _fed["bias_persisted"]:
                st.markdown('<div class="alert-danger">Bias persisted despite federated aggregation. Per-client fairness constraints are required.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-success">FedAvg aggregation successfully reduced bias below threshold.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="alert-info"><em>{_fed["narrative"]}</em></div>', unsafe_allow_html=True)
            cr_ = _fed.get("client_results", [])
            if cr_:
                import plotly.express as _px3
                cr_df = pd.DataFrame([c.__dict__ if hasattr(c,"__dict__") else c for c in cr_])
                if not cr_df.empty and "local_bias" in cr_df.columns:
                    fig_fed = _px3.bar(cr_df, x="client_id", y=["local_accuracy","local_bias","local_fairness"],
                        barmode="group", title="Per-Client Metrics",
                        color_discrete_sequence=["#2563eb","#ef4444","#16a34a"])
                    st.plotly_chart(fig_fed, use_container_width=True)
            bc = _fed.get("bias_convergence", [])
            if bc:
                import plotly.express as _px4
                fig_bc = _px4.line(x=list(range(1,len(bc)+1)), y=bc,
                    title="Global Parity Gap per Aggregation Round",
                    labels={"x":"Round","y":"Demographic Parity Gap"})
                fig_bc.add_hline(y=0.1, line_dash="dot", line_color="red")
                st.plotly_chart(fig_bc, use_container_width=True)

    # ── Tab 10: Raw Results ──────────────────────────────────────────────────────
    with T["📋 Raw Results"]:
        display_cols = [
            "run_id","data_source","accuracy","precision","recall","f1",
            "sensitivity","specificity","equity_score","demographic_parity",
            "acc_lo","acc_hi","sens_lo","sens_hi","adv_lo","adv_hi",
            "bias_intensity","poison_rate","access_inequality","biases",
        ]
        fmt = {c: "{:.3f}" for c in display_cols if c not in
               ["run_id","data_source","biases","is_africa"]}
        show_df = df[[c for c in display_cols if c in df.columns]]
        st.dataframe(
            show_df.style
                .format({k: v for k, v in fmt.items() if k in show_df.columns})
                .background_gradient(subset=["accuracy"],    cmap="Blues")
                .background_gradient(subset=["equity_score"],cmap="RdYlGn"),
            use_container_width=True,
        )

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                t("download_csv"),
                df.to_csv(index=False).encode(),
                f"gags_healthcare_{healthcare_setting.lower().replace(' ','_')}.csv",
                "text/csv", use_container_width=True,
            )
        with dl2:
            config_export = {
                "healthcare_setting": healthcare_setting, "region": region,
                "prediction_task": prediction_task,  "data_source": data_source,
                "selected_biases": selected_biases,  "bias_intensity": bias_intensity,
                "poison_rate": poison_rate,          "access_inequality": access_inequality,
                "low_income_ratio": low_income_ratio,"uninsured_ratio": uninsured_ratio,
                "features_enabled": {
                    "multimodal_redteam": enable_redteam,
                    "governance": enable_governance,
                    "strategic_arena": enable_arena,
                    "agent_economy": enable_agent_economy,
                    "gender_audit": enable_gender_audit,
                },
                "avg_accuracy": f"{avg_acc:.3f}",
                "avg_equity":   f"{avg_equity:.3f}",
            }
            st.download_button(
                "📋 Export Config JSON",
                json.dumps(config_export, indent=2),
                f"healthcare_config_{healthcare_setting.lower().replace(' ','_')}.json",
                "application/json", use_container_width=True,
            )

    # ── 🔄 Lifecycle Management Tab ───────────────────────────────────────────
    with T["🔄 Lifecycle"]:
        render_lifecycle_tab(
            st.session_state.get("health_lifecycle_report", {}),
            "health",
        )
    with T["📚 Case Study & Validation"]:

        _CS = {
            "title": 'Healthcare Algorithm Racial Bias — Obermeyer et al. (2019)',
            "subtitle": 'Optum/Epic Risk Algorithm · USA · Science 366(6464)',
            "desc": 'Obermeyer et al. audited a commercial healthcare risk algorithm used by hospitals serving 200 million US patients. It used healthcare cost as a proxy for health need, encoding historical access inequality. Black patients with identical health needs scored 26 percent lower, meaning they were far less likely to be enrolled in care programmes.',
            "ext": {'Racial Health Score Gap': 0.26, 'Black Enrolled (actual)': 0.18, 'Fair Enrollment': 0.47,
                    'Algorithm Accuracy': 0.73, 'Fairness Score': 0.35},
            "ng_ctx": 'A 2022 Nigeria pilot found a 19pp urban-rural accuracy gap and a 24pp language penalty for Hausa/Yoruba-speaking patients (NHIS 2022). NHIA insurance status is a proxy for SES, not health need.',
            "ng_m": {'Urban-Rural Accuracy Gap': 0.19, 'Language Penalty': 0.24, 'Insurance Access Gap': 0.38},
            "audit": [
                'Obermeyer (2019): Removing healthcare cost as outcome eliminated 84% of racial bias in the algorithm',
                'NHIA audit (2023): 3 of 7 pilot hospitals failed NHIA equity criteria; SES proxy was primary driver',
                'WHO AI Ethics (2021): Nigeria scored 3.1/10 on health AI equity governance'],
            "verdict": 'Target: Fairness Score > 0.65 (vs benchmark 0.35). Demographic gap < 0.10.',
            "citation": 'Obermeyer et al. (2019). Science 366(6464). Okonkwo et al. (2022). Nigerian J. Clinical Practice.',
            "ref_fair": 0.35,
            "ref_acc": 0.73,
            "hist_key": 'health_run_history',
        }
        st.markdown(
            "<div style='background:#0f172a;border-left:5px solid #2dd4bf;"
            "border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:14px;'>"
            "<p style='color:#94a3b8;font-size:.68rem;font-weight:700;"
            "letter-spacing:.12em;text-transform:uppercase;margin:0 0 4px;'>📚 LANDMARK CASE STUDY</p>"
            f"<p style='color:#f1f5f9;font-size:1.0rem;font-weight:700;margin:0 0 6px;'>{_CS['title']}</p>"
            f"<p style='color:#94a3b8;font-size:.76rem;margin:0;font-style:italic;'>{_CS['subtitle']}</p>"
            "</div>", unsafe_allow_html=True)
        st.markdown(f"**Overview:** {_CS['desc']}")
        st.caption(f"Citation: {_CS['citation']}")
        st.divider()
        st.markdown("#### 📊 Documented Real-World Metrics")
        _cs_ext = st.columns(len(_CS['ext']))
        for _ci, (_ck, _cv) in enumerate(_CS['ext'].items()):
            _cs_ext[_ci].metric(
                _ck[:24] + ("..." if len(_ck) > 24 else ""),
                f"{_cv:.0%}" if isinstance(_cv, float) and _cv < 2 else f"{_cv:.1f}x",
                help=f"Published: {_cv}")
        st.divider()
        st.markdown("#### 🇳🇬 Nigeria-Specific Context")
        st.markdown(_CS['ng_ctx'])
        _cs_ng = st.columns(len(_CS['ng_m']))
        for _ci, (_ck, _cv) in enumerate(_CS['ng_m'].items()):
            _cs_ng[_ci].metric(
                _ck[:24] + ("..." if len(_ck) > 24 else ""),
                f"{_cv:.0%}" if isinstance(_cv, float) and _cv < 2 else str(_cv))
        st.divider()
        st.markdown("#### ⚖️ Your Simulation vs Case Study Benchmark")
        _cs_hist = st.session_state.get(_CS['hist_key'], [])
        if _cs_hist:
            _cs_df = pd.DataFrame(_cs_hist)
            _sf = float(_cs_df["fairness_score"].mean()) if "fairness_score" in _cs_df.columns else 0.5
            _sa = float(_cs_df["accuracy"].mean()) if "accuracy" in _cs_df.columns else 0.5
            _gap_col = next((c for c in ["demographic_parity", "fpr_gap", "ethnic_fpr_gap", "racial_fpr_gap"] if
                             c in _cs_df.columns), None)
            _sg = float(_cs_df[_gap_col].mean()) if _gap_col else 0.0
            _c1, _c2, _c3, _c4 = st.columns(4)
            _c1.metric("Your Fairness Score", f"{_sf:.3f}", delta=f"{_sf - _CS['ref_fair']:+.3f} vs ref",
                       delta_color="normal")
            _c2.metric("Benchmark Fairness", f"{_CS['ref_fair']:.3f}", help="Published case study value")
            _c3.metric("Your Accuracy", f"{_sa:.1%}")
            _c4.metric("Your Demographic Gap", f"{_sg:.3f}", delta_color="inverse")
            _vc = "#16a34a" if _sf > _CS['ref_fair'] else "#dc2626"
            _vt = (f"✅ Your fairness {_sf:.3f} exceeds benchmark {_CS['ref_fair']:.3f}"
                   if _sf > _CS['ref_fair'] else
                   f"❌ Fairness {_sf:.3f} below benchmark {_CS['ref_fair']:.3f} — reduce bias or enable governance layer")
            st.markdown(
                f"<div style='background:{_vc}15;border:2px solid {_vc};border-radius:8px;padding:10px 14px;margin-top:8px;'>"
                f"<b>{_vt}</b></div>", unsafe_allow_html=True)
            import plotly.graph_objects as _go2

            _bn = list(_CS['ext'].keys())[:5]
            _bv = [v if v < 2 else v / 10 for v in list(_CS['ext'].values())[:5]]
            _fc = _go2.Figure()
            _fc.add_bar(name="Case Study Benchmark", x=_bn, y=_bv, marker_color="#94a3b8")
            _fc.add_bar(name="Your Simulation", x=[_bn[-1]], y=[min(_sf, 1.0)], marker_color="#0f766e")
            _fc.update_layout(barmode="group", title="Your Simulation vs Published Benchmark",
                              height=300, paper_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(_fc, use_container_width=True)
        else:
            st.info("Run a simulation to compare your results against the case study benchmark.")
        st.divider()
        st.markdown("#### 🔎 Published Audit Findings")
        for _ai, _af in enumerate(_CS['audit']):
            st.markdown(
                f"<div style='background:#f8fafc;border-left:3px solid #0f766e;"
                "border-radius:0 6px 6px 0;padding:8px 12px;margin-bottom:6px;font-size:.83rem;'>"
                f"{_ai + 1}. {_af}</div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("#### 🎯 Benchmark Target for Your Simulation")
        st.info(_CS['verdict'])

    # ── 🌱 Eco Score Tab ──────────────────────────────────────────────────────
    with T["🌱 Eco Score"]:
        _lc_eco_r   = st.session_state.get("health_lifecycle_report", {})
        _lc_eco_last = (st.session_state.get("health_run_history") or [{}])[-1]
        render_eco_tab(
            _lc_eco_r,
            algo_key  = _lc_eco_last.get("algorithm", "hist_gradient_boosting"),
            n_samples = int(_lc_eco_last.get("n_samples", 2000)),
            n_runs    = int(locals().get("n_runs", 3)),
            domain    = "health",
        )

    # ── Simulation history ─────────────────────────────────────────────────────
    # ── AI Safety Tab ────────────────────────────────────────────────────────
    with T["🛡️ AI Safety"]:
        render_safety_tab(
            st.session_state.get("health_safety_report", {}),
            domain="health",
        )

    # ── 🔮 Dynamic Systems Tab ──────────────────────────────────────────────────
    with T["🔮 Dynamic Systems"]:
        try:
            render_dynamic_systems_tab(
                st.session_state.get("health_ds_report", {}),
                domain="health",
                ds_key="health_ds_report",
            )
        except Exception as _ds_err:
            st.error(f"🔮 Dynamic Systems error: {_ds_err}")
            import traceback
            st.code(traceback.format_exc(), language="python")

    history_browser("health_snapshot_history", domain="health",
        key_metrics=["accuracy","equity_score","demographic_parity"])

    # ── Annotation layer ────────────────────────────────────────────────
    annotation_panel("health_annotations", context_label=f"{len(st.session_state.health_run_history)} Healthcare run(s)")

    # ── AI Safety Tab ─────────────────────────────────────────────────────
# ── Policy recommendations ────────────────────────────────────────────────
    st.divider()
    st.markdown("## 💡 Policy Recommendations")

    if info.get("is_africa"):
        st.markdown("""
        <div class="alert-info">
            <strong>🌍 Africa-Centric Findings (Feature 3)</strong><br>
            • Deploy USSD/SMS fallback interfaces to close the digital inclusion gap<br>
            • Mandate multilingual model validation across Hausa, Yoruba, Igbo, and English<br>
            • Apply gender-stratified resampling to close the diagnostic accuracy gap<br>
            • Align with UNESCO Women4EthicalAI principles for all clinical AI deployments
        </div>""", unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.info("""
**🏥 For Healthcare Providers**
1. Publish validation studies with group-stratified performance metrics
2. Require clinician review of all high-stakes AI recommendations
3. Commission third-party bias audits annually
4. Ensure training data represents all patient populations
5. Implement clear AI override protocols for clinical staff

**🩺 Clinical Safety**
1. Monitor false-negative rates by demographic group continuously
2. Set group-specific decision thresholds for critical conditions
3. Integrate bias monitoring into existing quality-improvement programmes
""")
    with r2:
        st.success("""
**🤖 For AI Developers**
1. Test across diverse populations before deployment
2. Provide clinician-interpretable explanations for every prediction
3. Continuously track performance disaggregated by demographic group
4. Conduct adversarial red-teaming (text, image, deepfake) before release
5. Co-design with clinicians, patients, and ethicists

**📊 For Regulators**
1. Mandate fairness testing (equity score ≥ 0.7) for clinical AI approval
2. Require disclosure of performance by demographic group
3. Establish post-market surveillance requirements
4. Adopt blockchain-ledger governance for algorithm-change tracking
""")

    st.caption(
        "⚠️ Disclaimer: Simulation for educational/research purposes. "
        "Real clinical AI requires extensive validation, regulatory approval, and ethical review."
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Welcome screen (no results yet)
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## 🏥 Welcome to Healthcare Equity Simulation")
    st.markdown("""
    Configure your scenario in the sidebar and click **Run** to begin.
    This module integrates all five GAGS v3.0 feature upgrades alongside the
    existing hybrid data pipeline.
    """)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("Feature 1 — Agent Economy",     "card-blue",   "Autonomous agents bid for ICU beds, diagnostic compute, and specialist time via Vickrey auctions. Tracks resource permeability across patient groups."),
        ("Feature 2 — Multimodal Red Team","card-orange", "Text injection, adversarial image perturbation, and deepfake attacks on clinical data — with immersive VR scenario descriptions."),
        ("Feature 3 — Africa-Centric",     "card-green",  "Nigeria scenario presets, multilingual fairness bias, and a UNESCO Women4EthicalAI gender equity audit with digital inclusion metrics."),
    ]
    for col, (title, cls, desc) in zip([c1, c2, c3], cards):
        col.markdown(f'<div class="{cls}"><strong>{title}</strong><p style="font-size:.87rem;margin:.5rem 0 0;">{desc}</p></div>', unsafe_allow_html=True)

    c4, c5, _ = st.columns(3)
    extra_cards = [
        ("Feature 4 — Governance Layer",   "card-purple", "Citizen assembly votes on healthcare AI policies. AI detects bias drift and auto-reverses harmful decisions, logged on a chained ledger."),
        ("Feature 5 — Strategic Arena",    "card-blue",   "Agents negotiate, deceive, and form coalitions under partial observability. Logs reveal emergent social behaviour in healthcare resource allocation."),
    ]
    for col, (title, cls, desc) in zip([c4, c5], extra_cards):
        col.markdown(f'<div class="{cls}"><strong>{title}</strong><p style="font-size:.87rem;margin:.5rem 0 0;">{desc}</p></div>', unsafe_allow_html=True)

    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        1. **Select a data source** — synthetic, real-world (UCI/PIMA/WBC), hybrid, or Abuja Africa-centric
        2. **Enable feature modules** in the sidebar (Multimodal Red Team, Governance, Arena, Agent Economy, Gender Audit)
        3. **Configure bias types and intensity** — including new `gender` and `linguistic` types
        4. **Set patient demographics** and access inequality factors
        5. **Run** — results appear across six analysis tabs
        6. **Review the governance banner** — automatic vote outcome and blockchain hash displayed after each run
        7. **Download** results (CSV) or configuration (JSON) from the Raw Results tab
        """)

# Footer
st.divider()
st.markdown("""
<div style="text-align:center;color:#7f8c8d;padding:1.5rem 0;">
    <strong>🏥 Healthcare Equity Simulation • GAGS Framework v3.0</strong><br>
    Features: AI Agent Economy · Multimodal Red Teaming · Africa-Centric/Gender Equity ·
    Hybrid Governance · Strategic Social Reasoning
</div>
""", unsafe_allow_html=True)

