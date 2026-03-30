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

# ── Auto-translation: translates ALL output to active language ───────────────
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
from components.pdf_report import generate_pdf_compliance_report
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
    board_member_summary, ROLE_TAB_VISIBILITY,
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
            "Abuja Smallholder Agrotech":  self._load_africa_agrotech,
            "Abuja Multilingual Healthcare": self._load_africa_healthcare,
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
        stratify=y if len(np.unique(y)) > 1 else None)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state)
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    return ({"accuracy":float(accuracy_score(yte,yp)),"recall":float(recall_score(yte,yp,zero_division=0)),
             "precision":float(precision_score(yte,yp,zero_division=0)),"f1":float(f1_score(yte,yp,zero_division=0)),
             "fpr":float(np.mean(yp[yte==0]==1)) if (yte==0).any() else 0.0},
            clf, scaler, (Xtr, Xte, ytr, yte))


@st.cache_data(show_spinner=False)
def _run_one(
    data_source, n_samples, selected_biases, bias_intensity,
    poison_rate, access_inequality, run_idx,
    enable_redteam=False, enable_governance=True, governance_policy="majority_vote",
    enable_arena=False, enable_agent_economy=False, enable_gender_audit=True,
):
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
    role_switcher("health")
    st.divider()

    # ── View Mode ────────────────────────────────────────────
    _vm_key = "_vm_health"
    if _vm_key not in st.session_state:
        st.session_state[_vm_key] = "Industry"
    view_mode = st.radio(t("perspective"), ["Industry", "Research"],
        horizontal=True, key=_vm_key,
        help="Industry: KPI-first. Research: full statistical depth.")
    st.divider()

    st.markdown("""<div style="text-align:center;padding:.5rem 0;">
      <h2 style="color:#0891b2;margin:0;">⚙️ Healthcare Config</h2>
      <p style="color:#888;font-size:.82rem;">AI Bias in Healthcare Simulation</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.subheader("🏥 Data Source")
    data_source = st.selectbox(t("dataset"),
        list(_pipeline.available_datasets.keys()),
        format_func=lambda k: {
            "uci_heart":            "UCI Heart Disease",
            "pima_diabetes":        "PIMA Diabetes (Women)",
            "breast_cancer":        "Breast Cancer Wisconsin",
            "africa_centric":       "Nigeria (Africa-centric)",
            "abuja_maternal":       "Abuja Maternal Health",
            "abuja_multilingual":   "Abuja Multilingual ECG",
            "abuja_insurance":      "Abuja Insurance Access",
        }.get(k, k),
    )
    healthcare_setting = st.selectbox(t("healthcare_setting"),
        ["Primary Care", "Secondary Hospital", "Tertiary/Teaching Hospital",
         "Community Clinic", "Telemedicine"])
    prediction_task = st.selectbox(t("prediction_task"),
        ["Disease Risk Screening", "Readmission Risk", "Diagnosis Support",
         "Treatment Recommendation", "Triage Priority"])
    region = st.selectbox(t("region"),
        ["Abuja FCT (Nigeria)", "Lagos (Nigeria)","Kano (Nigeria)","Anambra (Nigeria)" "Sub-Saharan Africa",
         "South Asia", "Global (Generic)"])
    st.divider()

    st.subheader("👥 Patient Demographics")
    low_income_ratio = st.slider(t("low_income_patients"), 0.0, 1.0, 0.40, 0.05,
        help="Fraction of patients from low-income backgrounds")
    uninsured_ratio  = st.slider(t("uninsured_patients"),  0.0, 1.0, 0.35, 0.05,
        help="Fraction of patients without health insurance")
    st.divider()

    st.subheader(f"🎭 {t('bias_config')}")
    _hc_valid = list(simulation_config.BIAS_TYPES) + [
        b for b in ["gender","linguistic"] if b not in simulation_config.BIAS_TYPES]
    _hc_defaults = [b for b in ["demographic","socioeconomic","gender"] if b in _hc_valid]
    selected_biases = st.multiselect(t("bias_types"), options=_hc_valid, default=_hc_defaults,
        format_func=lambda x: f"🔴 {x}" if x in ("gender","demographic") else f"⚠️ {x}")
    bias_intensity = st.slider(t("bias_intensity"), 0.0, float(simulation_config.MAX_BIAS_FACTOR), 0.25, 0.05)
    access_inequality = st.slider(t("access_inequality"), 0.0, 1.0, 0.3, 0.05,
        help="Fraction of patients with reduced access to care")
    st.divider()

    st.subheader(f"⚠️ {t('attack_header')}")
    poison_rate = st.slider(t("poisoning_rate"), 0.0, 0.5, 0.05, 0.01, format="%.2f")
    st.divider()

    st.subheader(f"📊 {t('sim_params_header')}")
    sample_size = st.number_input(t("sample_size"), 500, 50000, settings.DEFAULT_N_SAMPLES, 500)
    n_runs = st.slider(t("simulation_runs"), 1, 8, 3)
    st.divider()

    st.subheader(f"🔬 {t('modules_header')}")
    enable_redteam      = st.toggle(t("multimodal_red_team"),   value=False)
    enable_governance   = st.toggle(t("governance_layer"),      value=True)
    governance_policy   = st.selectbox(t("governance_policy"),
        ["majority_vote","supermajority","consensus","weighted_expert"],
        disabled=not enable_governance)
    enable_arena        = st.toggle(t("strategic_arena"),       value=False)
    enable_agent_economy= st.toggle(t("agent_economy"),         value=False)
    enable_gender_audit = st.toggle(t("gender_equity_audit"),   value=True)
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
    f"""<div class="page-header" style="--ac:#0891b2;"><p style="font-family:'DM Mono',monospace;font-size:.69rem;letter-spacing:.16em;text-transform:uppercase;opacity:.5;margin:0 0 .55rem;display:flex;align-items:center;gap:.45rem;"><span style="width:16px;height:1px;background:#0891b2;opacity:.55;display:inline-block;"></span>HEALTHCARE · GAGS v3.0 ·</p><h1 style="font-family:'Syne',sans-serif!important;font-size:2.5rem!important;font-weight:800!important;line-height:1.08!important;letter-spacing:-.03em!important;margin:0 0 .6rem!important;">Healthcare Equity Simulation</h1><p style="margin:0;opacity:.72;font-size:.96rem;max-width:660px;line-height:1.65;">Test AI diagnostic bias across income, gender, and insurance status — real UCI/PIMA datasets calibrated to Nigeria centric demographics.</p><div style="margin-top:.9rem;"><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">UCI Heart Disease</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">PIMA Diabetes</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">Nigeria Scenarios</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">WHO AI Ethics</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">Gender Equity Audit</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#0891b2;">Multilingual</span></div></div>""",
    unsafe_allow_html=True
)


# ═══════════════════════════════════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════════════════════════════════

if run_button:
    st.session_state.health_run_history = []
    st.session_state.health_feature_outputs = {}
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

    prog.progress(1.0, text=t("complete"))
    prog.empty()


# ═══════════════════════════════════════════════════════════════════════════════
# Results dashboard
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.health_run_history:
    df = pd.DataFrame(st.session_state.health_run_history)
    feats = st.session_state.health_feature_outputs
    info  = st.session_state.health_dataset_info

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown(t("healthcare_equity_dashboard"))
    role_banner("health")

    if info:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data Source",    info["source"])
        c2.metric("Samples",        f"{info['samples']:,}")
        c3.metric("Features",       info["features"])
        c4.metric("Positive Class", f"{info['positive_rate']:.1%}")

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
        }.get(gov["outcome"], "alert-info")
        flags_html = "".join(f"<li>{f}</li>" for f in gov["ai_flags"]) or "<li>No drift detected</li>"
        tally = gov["tally"]
        st.markdown(f"""
        <div class="{outcome_colour}" style="margin-top:1rem;">
            <strong>🏛️ Governance Vote — "{gov['policy']}"</strong><br>
            Outcome: <strong>{gov['outcome'].upper()}</strong> &nbsp;|&nbsp;
            For: {tally.get('for',0)} &nbsp; Against: {tally.get('against',0)} &nbsp; Abstain: {tally.get('abstain',0)}<br>
            <strong>AI Flags:</strong><ul style="margin:.3rem 0 0 1rem;">{flags_html}</ul>
            <span style="font-size:.75rem;opacity:.7;">Ledger hash: <code>{gov['ledger_hash']}</code></span>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_labels = [
        "📈 Performance", "⚖️ Equity", "🏥 Clinical Impact",
        "🔬 Feature Modules", "📊 Data Analysis",
        "🧠 Explainable AI", "📋 Compliance", "🔁 Longitudinal", "🌐 Federated",
        "📋 Raw Results"
    ]
    # ── Share URL panel ───────────────────────────────────────────────
    _share_cfg = {"domain":"health","data_source":data_source,"selected_biases":selected_biases,"bias_intensity":bias_intensity,"poison_rate":poison_rate,"n_runs":n_runs}
    share_url_panel("health", config=_share_cfg)

    # ── Board Member view (role-specific executive summary) ───────────
    _role_now = get_active_role("health")
    if _role_now == "Board Member":
        _fair_val = avg_equity if "avg_equity" in dir() else 0.5
        _acc_val  = avg_acc if "avg_acc" in dir() else 0.5
        _ok = _fair_val >= 0.7
        _finding = ("Fairness score is within acceptable range. No critical disparities detected."
                    if _ok else "Fairness score below 0.70 — demographic disparities detected.")
        _rec = ("Continue quarterly monitoring and maintain current governance oversight."
                if _ok else "Bias mitigation required before deployment. Consult Data Science team.")
        board_member_summary("health", _acc_val, _fair_val, _ok, _finding, _rec)
    else:
        metric_glossary_expander(["accuracy", "sensitivity", "specificity", "fairness score", "demographic parity", "equalized odds", "false positive rate", "false negative rate", "gender gap", "digital inclusion score"])
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(tab_labels)

    # ── Tab 1: Performance ────────────────────────────────────────────────────
    with tab1:
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
    with tab2:
        st.markdown(t("health_equity_gap_analysis"))

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
            st.markdown(t("gender_equity_audit_feature_3_unesco_women4ethical"))
            passed_icon = "✅" if ga["audit_passed"] else "❌"
            g1, g2, g3 = st.columns(3)
            g1.metric("Gender Gap",          f"{ga['overall_gender_gap']:.3f}")
            g2.metric("Representation Score",f"{ga['representation_score']:.3f}")
            g3.metric("Digital Inclusion",   f"{ga['digital_inclusion_score']:.3f}")

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
    with tab3:
        st.markdown(t("clinical_impact_by_patient_group"))

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
        st.markdown(t("healthcare_resource_access"))
        resources = ["Preventive Care","Specialist Access","Diagnostics","Medications","Follow-up"]
        lo_alloc  = [max(0, 0.3 * (1 - access_inequality)) for _ in resources]
        hi_alloc  = [0.75, 0.80, 0.90, 0.85, 0.80]

        fig_res = go.Figure([
            go.Bar(name="Low-Income",    x=resources, y=lo_alloc, marker_color="#ef4444"),
            go.Bar(name="Higher-Income", x=resources, y=hi_alloc, marker_color="#22c55e"),
        ])
        fig_res.update_layout(title="Resource Access by Income Group", barmode="group", yaxis_range=[0,1])
        st.plotly_chart(fig_res, use_container_width=True)

    # ── Tab 4: Feature Modules ────────────────────────────────────────────────
    with tab4:
        st.markdown(t("advanced_feature_module_results"))

        # ── Feature 2: Multimodal Red Team ────────────────────────────────────
        if "multimodal_redteam" in feats:
            st.markdown(t("feature_2_multimodal_red_teaming"))
            rt = feats["multimodal_redteam"]
            rt_rows = []
            for r in rt.get("modality_results", []):
                rt_rows.append({
                    "Modality":              r["modality"],
                    "Attack Vector":         r["attack_vector"],
                    "Severity":              r["severity"],
                    "Affected Samples":      r["affected_samples"],
                    "Bypass Rate":           f"{r['bypass_rate']:.1%}",
                    "Sociotechnical Risk":   f"{r['sociotechnical_risk']:.2f}",
                })
            if rt_rows:
                st.dataframe(pd.DataFrame(rt_rows), use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("Combined Bypass Rate",       f"{rt.get('combined_bypass_rate',0):.1%}")
            c2.metric("Combined Sociotechnical Risk",f"{rt.get('combined_sociotechnical_risk',0):.2f}")

            # Show VR scenario if deepfake attack has one
            for r in rt.get("modality_results", []):
                if r.get("vr_scenario"):
                    st.markdown(f"""
                    <div class="alert-info">
                        <strong>🥽 Immersive VR/AR Scenario (Deepfake Attack)</strong><br>
                        {r['vr_scenario']}
                    </div>""", unsafe_allow_html=True)
                    break

        # ── Feature 1: Agent Economy ──────────────────────────────────────────
        if "agent_economy" in feats:
            st.markdown(t("feature_1_ai_agent_economy_healthcare_resources"))
            ae = feats["agent_economy"]
            st.metric("Economy Stability",  ae.get("economy_stability","—"))
            st.metric("Permeability Score", f"{ae.get('permeability_score',0):.4f}")
            agents_df = pd.DataFrame(ae.get("agent_summary", []))
            if not agents_df.empty:
                st.dataframe(agents_df.style.background_gradient(
                    subset=["reputation","total_spent"], cmap="Blues"
                ), use_container_width=True)

        # ── Feature 4: Governance Ledger ──────────────────────────────────────
        if "governance" in feats:
            st.markdown(t("feature_4_hybrid_governance_ledger_blockchain_styl"))
            gov = feats["governance"]
            st.markdown(f"""
            <div class="ledger-row">
                ▶ <strong>Policy:</strong> {gov['policy']}<br>
                ▶ <strong>Outcome:</strong> {gov['outcome'].upper()}<br>
                ▶ <strong>Tally:</strong> For={gov['tally'].get('for',0)}, Against={gov['tally'].get('against',0)}, Abstain={gov['tally'].get('abstain',0)}<br>
                ▶ <strong>Ledger Hash:</strong> <code>{gov['ledger_hash']}</code>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature 5: Strategic Arena ────────────────────────────────────────
        if "strategic_arena" in feats:
            st.markdown(t("feature_5_strategic_social_reasoning_arena"))
            arena = feats["strategic_arena"]
            standings = pd.DataFrame(arena.get("final_standings", []))
            if not standings.empty:
                st.dataframe(standings.style.background_gradient(
                    subset=["score"], cmap="YlGn"
                ), use_container_width=True)
            coal = arena.get("coalition_scores", {})
            if coal:
                fig_coal = px.bar(
                    x=list(coal.keys()), y=list(coal.values()),
                    title="Coalition Scores", labels={"x":"Coalition","y":"Score"},
                    color=list(coal.values()), color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_coal, use_container_width=True)

        if not any(k in feats for k in ["multimodal_redteam","agent_economy","governance","strategic_arena","gender_audit"]):
            st.info("Enable feature modules in the sidebar to see results here.")

    # ── Tab 5: Data Analysis ──────────────────────────────────────────────────
    with tab5:
        st.markdown(t("data_quality_source_analysis"))

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
                Dataset Nigeria centric parameters: lower income mean, multilingual context,
                reduced digital connectivity baseline, and gender-stratified demographics.
                UNESCO Women4EthicalAI audit is available in the Equity tab.
            </div>""", unsafe_allow_html=True)

    # ── Tabs 6-9: XAI / Compliance / Longitudinal / Federated ──────────────────
    with tab6:
        _xai = st.session_state.get("health_xai_results", {})
        st.markdown("### 🧠 Explainable AI")
        if not _xai:
            st.info("Run a simulation to generate XAI explanations.")
        elif "error" in _xai:
            st.warning(f"XAI error: {_xai["error"]}")
        else:
            fi = _xai.get("feature_importance", {})
            if fi:
                st.markdown(f'<div class="alert-info"><em>{fi.get("narrative","")}</em></div>', unsafe_allow_html=True)
                fi_df = pd.DataFrame({"Feature": fi["feature_names"][:10], "Importance": fi["importances"][:10], "Std": fi["std_devs"][:10]})
                import plotly.express as _px
                fig_fi = _px.bar(fi_df, x="Importance", y="Feature", orientation="h", error_x="Std",
                    title=f"Feature Importance ({fi.get("method","permutation")})",
                    color="Importance", color_continuous_scale="Blues")
                fig_fi.update_layout(yaxis={"categoryorder":"total ascending"}, height=340)
                st.plotly_chart(fig_fi, use_container_width=True)
            cl, cr_ = st.columns(2)
            with cl:
                expl = _xai.get("instance_explanation", {})
                if expl:
                    st.markdown(t("instance_explanation"))
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
                st.markdown(t("intersectional_fairness"))
                st.markdown(f'<div class="alert-info">{ix.get("narrative","")}</div>', unsafe_allow_html=True)
                ix_df = pd.DataFrame([{"Group":k,**{kk:round(vv,3) for kk,vv in v.items()}} for k,v in ix["group_performances"].items()])
                st.dataframe(ix_df.style.background_gradient(subset=["accuracy"], cmap="RdYlGn"), use_container_width=True)

    with tab7:
        _xai = st.session_state.get("health_xai_results", {})
        _cr = _xai.get("compliance_report", {})
        _mc = _xai.get("model_card", {})
        st.markdown(t("regulatory_compliance_report"))
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
        st.markdown(t("real_world_benchmark_comparison"))
        if BENCHMARKS_OK:
            _dom_bms = get_benchmarks_for_domain("healthcare")
            if _dom_bms:
                _bm_sel = st.selectbox(t("compare_against_a_published_study"),
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


    with tab8:
        _lng = st.session_state.get("health_longitudinal")
        st.markdown(t("longitudinal_bias_analysis"))
        st.markdown('<div class="alert-info">Simulates the <strong>feedback loop</strong>: biased predictions replace training labels over successive retraining cycles, potentially making bias self-reinforcing.</div>', unsafe_allow_html=True)
        if not _lng:
            st.info("Run a simulation to see longitudinal bias evolution.")
        else:
            c1,c2,c3 = st.columns(3)
            c1.metric("Initial Bias", f'{_lng["initial_bias"]:.1%}')
            c2.metric("Final Bias", f'{_lng["final_bias"]:.1%}', f'{_lng["final_bias"]-_lng["initial_bias"]:+.1%}')
            c3.metric("Amplification", f'{_lng["amplification_factor"]:.2f}×', "⚠️ Self-reinforcing" if _lng["self_reinforcing"] else "Stable")
            if _lng.get("inflection_point"):
                st.warning(f"⚠️ Bias became self-reinforcing at generation {_lng["inflection_point"]}.")
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

    with tab9:
        _fed = st.session_state.get("health_federated")
        st.markdown(t("federated_learning_simulation"))
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
    with tab10:
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

    # ── Simulation history ─────────────────────────────────────────────────────
    history_browser("health_snapshot_history", domain="health",
        key_metrics=["accuracy","equity_score","demographic_parity"])

    # ── Annotation layer ────────────────────────────────────────────────
    annotation_panel("health_annotations", context_label=f"{len(st.session_state.health_run_history)} Healthcare run(s)")

    # ── Policy recommendations ────────────────────────────────────────────────
    st.divider()
    st.markdown(t("policy_recommendations"))

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
    st.markdown(t("welcome_to_healthcare_equity_simulation"))
    st.markdown("""
    Configure your scenario in the sidebar and click **Run** to begin.
    This module integrates all five GAGS v3.0 feature upgrades alongside the
    existing hybrid data pipeline.
    """)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("Feature 1 — Agent Economy",     "card-blue",   "Autonomous agents bid for ICU beds, diagnostic compute, and specialist time via Vickrey auctions. Tracks resource permeability across patient groups."),
        ("Feature 2 — Multimodal Red Team","card-orange", "Text injection, adversarial image perturbation, and deepfake attacks on clinical data — with immersive VR scenario descriptions."),
        ("Feature 3 — Africa-Centric",     "card-green",  "Abuja FCT scenario presets, multilingual fairness bias, and a UNESCO Women4EthicalAI gender equity audit with digital inclusion metrics."),
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

    with st.expander(t("how_to_use"), expanded=False):
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
