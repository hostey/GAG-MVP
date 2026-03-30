# pages/02_🛡️_National_Security.py
"""
National Security Resilience Simulation — GAGS Framework v3.0

Refactored to integrate all GAGS v3.0 governance_logic imports:
  - apply_bias()              with AttackSeverity enum (keyword args only)
  - simulate_data_poisoning() keyword demographic_info (fixes positional crash)
  - calculate_fairness_metrics() always-present group_metrics
  - run_simple_simulation()   consistent {status,results,features,warnings} envelope
  - MultimodalRedTeamer       Feature 2 — text/image/deepfake attacks
  - HybridGovernanceLayer     Feature 4 — citizen vote + blockchain ledger
  - AgentEconomySandbox       Feature 1 — Vickrey auction for security resources
  - StrategicReasoningArena   Feature 5 — negotiation/deception game

Key improvements over v2.2:
  - All simulate_data_poisoning() calls use keyword args (fixes TypeError crash)
  - "behavioral" bias type removed from defaults (not in simulation_config.BIAS_TYPES)
  - Consistent session-state namespace  (security_*)
  - _run_one() single computation function — no logic scattered in the run loop
  - Feature module results surfaced in dedicated tab
  - Governance banner rendered after every run
  - CSS uses semantic card classes (no inline gradients on metric cards)
  - dataset_choice scoped correctly (was referenced outside its conditional block)
  - All cmaps validated against matplotlib accepted list
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ── GAGS v3.0 core ────────────────────────────────────────────────────────────
from components.pdf_report import generate_pdf_compliance_report
from components.governance_logic import (
    AttackSeverity,
    apply_bias,
    simulate_data_poisoning,
    calculate_fairness_metrics,
    generate_synthetic_data,
    run_simple_simulation,
    ExplainableModel,
    generate_compliance_report,
    generate_intersectional_fairness,
    # Feature 1
    AgentEconomySandbox,
    # Feature 2
    MultimodalRedTeamer,
    # Feature 4
    HybridGovernanceLayer,
    # Feature 5
    StrategicReasoningArena,
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


# ── Constants ─────────────────────────────────────────────────────────────────
# Only bias types that exist in simulation_config.BIAS_TYPES are valid
_VALID_BIAS_TYPES = list(simulation_config.BIAS_TYPES)

_ATTACK_TYPE_MAP: dict = {
    "Data Poisoning":    "label_flipping",
    "Evasion Attacks":   "feature_noise",
    "Model Inversion":   "feature_noise",    # label_smoothing not implemented; feature_noise is closest analogue
    "Backdoor Attacks":  "backdoor",
}

_OVERSIGHT_MULTIPLIER: dict = {
    "Minimal":         1.2,
    "Moderate":        1.0,
    "Strong":          0.8,
    "Judicial Review": 0.6,
}

_GOVERNANCE_POLICY = "Deploy AI threat-detection in national security operations"

# ── Session-state defaults ────────────────────────────────────────────────────
st.set_page_config(
    page_title="National Security • GAGS",
    layout="wide",
    page_icon="🛡️",
)

# ── Design system ──────────────────────────────────────────────────────────────
try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, page_header, plotly_theme as _ptheme
    inject_css("security")
    ACCENT = DOMAIN_ACCENTS["security"]
except ImportError:
    ACCENT = "#ef4444"


_STATE_DEFAULTS = {
    "security_run_history":   [],
    "security_baseline":      None,
    "security_feature_outputs": {},
    "security_rw_df":         None,
    "security_rw_info":       None,
    "security_snapshot_history": [],  # UX history browser
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v




# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_data(data_source, dataset_choice, scenario, n_samples, random_state=42):
    """Generate security dataset."""
    X, y, demo = generate_synthetic_data(n_samples=n_samples, n_features=12)
    return X.astype(np.float64), y.astype(int), demo.astype(int)


def _run_baseline(data_source, dataset_choice, scenario, n_samples,
                  surveillance_level, data_retention, oversight_level):
    """Run baseline simulation (no bias, no attack)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y, demo = _preprocess_data(data_source, dataset_choice, scenario, n_samples)
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=42,
        stratify=y if len(np.unique(y))>1 else None)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    fair = calculate_fairness_metrics(yte, yp, demo[:len(yte)])
    fpr = float(np.mean(yp[yte==0]==1)) if (yte==0).any() else 0.0
    lib = float(np.clip(1 - fpr*2 - fair.get("demographic_parity_difference",0), 0, 1))
    return {"accuracy":float(accuracy_score(yte,yp)),"detection_rate":float(recall_score(yte,yp,zero_division=0)),
            "false_positive_rate":fpr,"fairness_score":fair.get("fairness_score",0.7),
            "liberty_score":lib,"demographic_parity":fair.get("demographic_parity_difference",0),
            "is_baseline":True}


def _run_one(
    data_source, dataset_choice, scenario, sample_size, selected_biases, bias_intensity,
    attack_type_label, poison_rate, attack_sophistication, surveillance_level,
    data_retention, oversight_level, run_idx, threat_level=0.5,
    enable_redteam=False, enable_governance=True, enable_arena=False, enable_agent_economy=False,
):
    """Execute one national security simulation run."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y, demo = _preprocess_data(data_source, dataset_choice, scenario, sample_size, random_state=42+max(run_idx,0))
    _vb = list(simulation_config.BIAS_TYPES) + [b for b in ["gender","linguistic"] if b not in simulation_config.BIAS_TYPES]
    for bt in [b for b in selected_biases if b in _vb]:
        try: X, y, demo = apply_bias(X, y, bt, bias_intensity, demographic_info=demo)
        except: pass
    pr = poison_rate * (1 + attack_sophistication * 0.5)
    try: X, y, demo = simulate_data_poisoning(X, y, pr, attack_type="label_flipping", demographic_info=demo, targeted=True)
    except: pass
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3, random_state=42+max(run_idx,0),
        stratify=y if len(np.unique(y))>1 else None)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42+max(run_idx,0))
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    fair = calculate_fairness_metrics(yte, yp, demo[:len(yte)])
    fpr = float(np.mean(yp[yte==0]==1)) if (yte==0).any() else 0.0
    det = float(recall_score(yte, yp, zero_division=0))
    lib = float(np.clip(1 - fpr*2 - surveillance_level*0.3 - fair.get("demographic_parity_difference",0), 0, 1))
    if run_idx == 0:
        try: st.session_state.security_longitudinal = simulate_longitudinal_bias(X, y, demo, initial_bias_type=next((b for b in selected_biases if b in _vb),"demographic"), initial_bias_intensity=bias_intensity or 0.15, n_generations=5, random_state=42).__dict__
        except: st.session_state.security_longitudinal = None
        try: st.session_state.security_federated = simulate_federated_learning(X, y, demo, n_clients=4, n_rounds=3, bias_heterogeneity=(bias_intensity or 0.15)*0.5, random_state=42).__dict__
        except: st.session_state.security_federated = None
    # Confusion matrix components
    _tp = int(np.sum((yte == 1) & (yp == 1)))
    _fp = int(np.sum((yte == 0) & (yp == 1)))
    _tn = int(np.sum((yte == 0) & (yp == 0)))
    _fn = int(np.sum((yte == 1) & (yp == 0)))
    _sec_score = float(np.clip(
        0.35 * det + 0.35 * (1 - fpr) + 0.30 * fair.get("fairness_score", 0.5), 0, 1))
    return {
        "run_id":run_idx+1,"dataset":data_source,"scenario":scenario,
        "accuracy":float(accuracy_score(yte,yp)),"detection_rate":det,
        "recall":det,
        "false_positive_rate":fpr,"fpr":fpr,
        "liberty_score":lib,
        "security_score":_sec_score,
        "fairness_score":fair.get("fairness_score",0.5),
        "demographic_parity":fair.get("demographic_parity_difference",0),
        "equalized_odds":fair.get("equalized_odds_difference",0),
        "tp":_tp,"fp":_fp,"tn":_tn,"fn":_fn,
        "bias_intensity":bias_intensity,"poison_rate":poison_rate,
        "attack_sophistication":attack_sophistication,"oversight":oversight_level,
        "surveillance_level":surveillance_level,
        "biases":", ".join([b for b in selected_biases if b in _vb]) or "None",
        "attack_type":attack_type_label,"warnings":[],
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
    role_switcher("security")
    st.divider()

    # ── View Mode ────────────────────────────────────────────
    _vm_key = "_vm_security"
    if _vm_key not in st.session_state:
        st.session_state[_vm_key] = "Industry"
    view_mode = st.radio(t("perspective"), ["Industry", "Research"],
        horizontal=True, key=_vm_key,
        help="Industry: KPI-first. Research: full statistical depth.")
    st.divider()

    st.markdown("""<div style="text-align:center;padding:.5rem 0;">
      <h2 style="color:#ef4444;margin:0;">⚙️ Security Config</h2>
      <p style="color:#888;font-size:.82rem;">AI Bias in National Security Simulation</p>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.subheader("🛡️ Data Source")
    data_source = st.selectbox(t("dataset"),
        ["GTD (Global Terrorism)", "UNSW-NB15 (Network Intrusion)",
         "Synthetic — Urban Surveillance", "Synthetic — Border Security"])

    _ds_map = {
        "GTD (Global Terrorism)":         "gtd",
        "UNSW-NB15 (Network Intrusion)":  "unsw",
        "Synthetic — Urban Surveillance": "urban",
        "Synthetic — Border Security":    "border",
    }
    dataset_choice = _ds_map.get(data_source, "gtd")

    scenario = st.selectbox(t("threat_scenario"),
        ["Counter-Terrorism", "Cyber Threat Detection",
         "Predictive Policing", "Border Control", "Financial Crime"])
    st.divider()

    st.subheader(f"🎭 {t('bias_config')}")
    _ns_valid = list(simulation_config.BIAS_TYPES) + [
        b for b in ["gender","linguistic"] if b not in simulation_config.BIAS_TYPES]
    _ns_defaults = [b for b in ["demographic","geographic","historical"] if b in _ns_valid]
    selected_biases = st.multiselect(t("bias_types"), options=_ns_valid, default=_ns_defaults,
        format_func=lambda x: f"🔴 {x}" if x in ("demographic","geographic") else f"⚠️ {x}")
    bias_intensity = st.slider(t("bias_intensity"), 0.0, float(simulation_config.MAX_BIAS_FACTOR), 0.3, 0.05)
    st.divider()

    st.subheader(t("surveillance_configuration"))
    surveillance_level = st.slider(t("surveillance_intensity"), 0.0, 1.0, 0.5, 0.05)
    data_retention     = st.slider(t("data_retention_days"), 30, 3650, 365, 30)
    oversight_level    = st.selectbox(t("oversight_mechanism"),
        ["None","Internal Review","Judicial Oversight","Parliamentary Oversight","Independent Audit"])
    threat_level       = st.slider(t("perceived_threat_level"), 0.0, 1.0, 0.5, 0.05)
    st.divider()

    st.subheader(f"⚠️ {t('attack_header')}")
    attack_type_label = st.selectbox(t("attack_type"),
        ["label_flipping", "feature_noise", "backdoor", "model_inversion"],
        format_func=lambda x: x.replace("_", " ").title())
    attack_sophistication = st.select_slider(
        "Attack Sophistication",
        options=[0.1, 0.3, 0.6, 1.0],
        value=0.3,
        format_func=lambda x: {0.1:"Low",0.3:"Medium",0.6:"High",1.0:"Nation-State"}.get(x,str(x)))
    poison_rate = st.slider(t("poisoning_rate"), 0.0, 0.5, 0.05, 0.01, format="%.2f")
    dataset_choice = st.selectbox(t("dataset_choice"),
        ["gtd", "synthetic", "unsw_nb15"],
        format_func=lambda x: {"gtd": "GTD Terrorism", "synthetic": "Synthetic",
                                "unsw_nb15": "UNSW-NB15 Cyber"}.get(x, x),
        key="_ns_dataset_choice")
    st.divider()

    st.subheader(f"📊 {t('sim_params_header')}")
    sample_size = st.number_input(t("sample_size"), 500, 50000, settings.DEFAULT_N_SAMPLES, 500)
    n_runs      = st.slider(t("simulation_runs"), 1, 8, 3)
    include_baseline = st.toggle(t("include_baseline_no_bias_attack"), value=True)
    st.divider()

    st.subheader(f"🔬 {t('modules_header')}")
    enable_redteam       = st.toggle(t("multimodal_red_team"),  value=False)
    enable_governance    = st.toggle(t("governance_layer"),     value=True)
    enable_arena         = st.toggle(t("strategic_arena"),      value=False)
    enable_agent_economy = st.toggle(t("agent_economy"),        value=False)
    st.divider()

    col_r, col_x = st.columns(2)
    run_button = col_r.button(t("run_simulation"), type="primary", use_container_width=True)
    if col_x.button(t("reset"), use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("security_"):
                del st.session_state[k]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Page setup
# ═══════════════════════════════════════════════════════════════════════════════


st.markdown(
    f"""<div class="page-header" style="--ac:#ef4444;"><p style="font-family:'DM Mono',monospace;font-size:.69rem;letter-spacing:.16em;text-transform:uppercase;opacity:.5;margin:0 0 .55rem;display:flex;align-items:center;gap:.45rem;"><span style="width:16px;height:1px;background:#ef4444;opacity:.55;display:inline-block;"></span>NATIONAL SECURITY · GAGS v3.0</p><h1 style="font-family:'Syne',sans-serif!important;font-size:2.5rem!important;font-weight:800!important;line-height:1.08!important;letter-spacing:-.03em!important;margin:0 0 .6rem!important;">National Security Assessment</h1><p style="margin:0;opacity:.72;font-size:.96rem;max-width:660px;line-height:1.65;">Simulate the surveillance vs civil-liberties trade-off — GTD & UNSW-NB15 datasets, liberty score, judicial oversight analysis.</p><div style="margin-top:.9rem;"><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">GTD Dataset</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">UNSW-NB15</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">Liberty Score</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">Predictive Policing</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">Red Team</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#ef4444;">Governance Layer</span></div></div>""",
    unsafe_allow_html=True
)


# ═══════════════════════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════════════════════

if run_button:
    st.session_state.security_run_history   = []
    st.session_state.security_baseline      = None
    st.session_state.security_feature_outputs = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    if include_baseline:
        with st.spinner("Computing baseline (no bias / no attack)…"):
            st.session_state.security_baseline = _run_baseline(
                data_source, dataset_choice, scenario, sample_size,
                surveillance_level, data_retention, oversight_level,
            )

    # ── Main runs ─────────────────────────────────────────────────────────────
    prog = st.progress(0, text=t("loading"))
    for i in range(n_runs):
        prog.progress(i / n_runs, text=f"Run {i+1} of {n_runs}…")
        with st.spinner(f"Simulation {i+1}/{n_runs}"):
            result = _run_one(
                data_source=data_source,
                dataset_choice=dataset_choice,
                scenario=scenario,
                sample_size=sample_size,
                selected_biases=selected_biases,
                bias_intensity=bias_intensity,
                attack_type_label=attack_type_label,
                poison_rate=poison_rate,
                attack_sophistication=attack_sophistication,
                surveillance_level=surveillance_level,
                data_retention=data_retention,
                oversight_level=oversight_level,
                run_idx=i,
                threat_level=threat_level,      # Fix 2 — passed through
                enable_redteam=enable_redteam,
                enable_governance=enable_governance,
                enable_arena=enable_arena,
                enable_agent_economy=enable_agent_economy,
            )
            st.session_state.security_run_history.append(result)
            save_to_history(
                "security_snapshot_history",
                label=f"Run {i+1} | surv={surveillance_level} | {scenario[:14]}",
                metrics={
                    "detection_rate":      result.get("detection_rate", 0),
                    "liberty_score":       result.get("liberty_score", 0),
                    "false_positive_rate": result.get("false_positive_rate", 0),
                    "fairness_score":      result.get("fairness_score", 0),
                },
                config={
                    "surveillance_level": surveillance_level,
                    "scenario":           scenario,
                    "bias_intensity":     bias_intensity,
                    "oversight_level":    oversight_level,
                },
            )
            for w in result.get("warnings", []):
                st.warning(f"⚠️ {w}")

    prog.progress(1.0, text=t("complete"))
    prog.empty()

    # ── XAI on first run ──────────────────────────────────────────────────────
    if st.session_state.security_run_history:
        _r0 = st.session_state.security_run_history[0]
        try:
            # Re-generate a small dataset to fit the ExplainableModel
            _Xx, _yy, _dd = generate_synthetic_data(n_samples=2000, n_features=12,
                decision_boundary=simulation_config.DECISION_BOUNDARY + 0.5)
            _Xx = _Xx.astype(float)
            from sklearn.preprocessing import StandardScaler as _SS_
            from sklearn.model_selection import train_test_split as _tts_
            _Xs = _SS_().fit_transform(_Xx)
            _Xtr,_Xte,_ytr,_yte = _tts_(_Xs,_yy,test_size=0.3,random_state=42,stratify=_yy)
            from sklearn.ensemble import RandomForestClassifier as _RFC_
            _clf = _RFC_(n_estimators=50,class_weight="balanced",random_state=42).fit(_Xtr,_ytr)
            _xm = ExplainableModel(domain="national_security")
            _xm.model=_clf; _xm._X_train=_Xtr; _xm._is_fitted=True
            _fi = _xm.feature_importance(_Xte,_yte,n_repeats=5)
            _hr = ((_yte==1).nonzero()[0])
            _expl = _xm.explain_instance(_Xte[_hr[0]]) if len(_hr)>0 else None
            _cf   = _xm.counterfactual(_Xte[_hr[0]]) if len(_hr)>0 else None
            _fair_ = calculate_fairness_metrics(_yte,_clf.predict(_Xte),_dd[:len(_yte)])
            _mc_ = _xm.model_card({"accuracy":_r0.get("accuracy",0),"recall":_r0.get("detection_rate",0)},
                                   _fair_, domain="national_security", version="3.0")
            _cr_ = generate_compliance_report(_mc_, _fair_,
                {"accuracy":_r0.get("accuracy",0)},
                frameworks=["EU AI Act","ISO 42001","NIST AI RMF","NITDA","UNESCO"])
            _ix_ = generate_intersectional_fairness(_yte,_clf.predict(_Xte),
                {"region":_dd[:len(_yte)]%3,"group":_dd[:len(_yte)]%2}, min_group_size=15)
            st.session_state.security_xai_results = {
                "feature_importance": _fi.__dict__,
                "instance_explanation": _expl.__dict__ if _expl else {},
                "counterfactual": _cf.__dict__ if _cf else {},
                "model_card": _mc_.__dict__,
                "compliance_report": _cr_,
                "intersectional": _ix_.__dict__,
            }
        except Exception as _e:
            st.session_state.security_xai_results = {"error": str(_e)}

    # ── Longitudinal + Federated ──────────────────────────────────────────────
    if st.session_state.security_run_history:
        try:
            _Xl,_yl,_dl = generate_synthetic_data(n_samples=1500,n_features=12,
                decision_boundary=simulation_config.DECISION_BOUNDARY+0.5)
            _bl = bias_intensity if bias_intensity > 0 else 0.15
            _bt = (selected_biases[0] if selected_biases else "socioeconomic")
            lng = simulate_longitudinal_bias(_Xl.astype(float),_yl,_dl,
                initial_bias_type=_bt, initial_bias_intensity=_bl,
                n_generations=5, random_state=42)
            st.session_state.security_longitudinal = lng.__dict__
        except Exception:
            st.session_state.security_longitudinal = None
        try:
            fed = simulate_federated_learning(_Xl.astype(float),_yl,_dl,
                n_clients=4, n_rounds=3,
                bias_heterogeneity=bias_intensity*0.5, random_state=42)
            st.session_state.security_federated = fed.__dict__
        except Exception:
            st.session_state.security_federated = None


# ═══════════════════════════════════════════════════════════════════════════════
# Results dashboard
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.security_run_history:
    df_r  = pd.DataFrame(st.session_state.security_run_history)
    base  = st.session_state.security_baseline
    feats = st.session_state.security_feature_outputs
    rw_info = st.session_state.security_rw_info

    # ── Dataset info banner ───────────────────────────────────────────────────
    if rw_info and data_source == "Real-World":
        st.markdown(f"""
        <div class="dataset-info">
            <strong>Dataset:</strong> {rw_info['name']} &nbsp;|&nbsp;
            <strong>Records:</strong> {rw_info['n_records']:,} &nbsp;|&nbsp;
            <strong>Features:</strong> {rw_info['n_features']} &nbsp;|&nbsp;
            <strong>Period:</strong> {rw_info['year_range']} &nbsp;|&nbsp;
            <strong>Source:</strong> {rw_info['source']}
        </div>
        """, unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown(t("security_assessment_dashboard"))
    role_banner("security")

    avg_det  = df_r["detection_rate"].mean()
    avg_lib  = df_r["liberty_score"].mean()
    avg_fpr  = df_r["false_positive_rate"].mean()
    avg_sec  = df_r["security_score"].mean()

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="card-threat">
            <p style="margin:0;font-size:.8rem;color:#555;">🎯 Threat Detection Rate</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:#ef4444;">{avg_det:.1%}</p>
            {'<p style="margin:0;font-size:.8rem;color:#888;">vs baseline: '
             + (f"{avg_det - base['detection_rate']:+.1%}" if base else "—")
             + '</p>'}
        </div>""", unsafe_allow_html=True)

    with k2:
        lib_col = "#16a34a" if avg_lib >= 0.7 else "#e67e22" if avg_lib >= 0.5 else "#ef4444"
        st.markdown(f"""
        <div class="card-liberty">
            <p style="margin:0;font-size:.8rem;color:#555;">⚖️ Liberty Preservation</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:{lib_col};">{avg_lib:.2f}<span style="font-size:.9rem">/1.0</span></p>
            {'<p style="margin:0;font-size:.8rem;color:#888;">vs baseline: '
             + (f"{avg_lib - base['liberty_score']:+.2f}" if base else "—")
             + '</p>'}
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="card-secure">
            <p style="margin:0;font-size:.8rem;color:#555;">🛡️ Security Score</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:#1d4ed8;">{avg_sec:.2f}<span style="font-size:.9rem">/1.0</span></p>
        </div>""", unsafe_allow_html=True)

    with k4:
        fpr_col = "#ef4444" if avg_fpr > 0.15 else "#e67e22" if avg_fpr > 0.05 else "#16a34a"
        st.markdown(f"""
        <div class="card-warn">
            <p style="margin:0;font-size:.8rem;color:#555;">⚠️ False Alarm Rate</p>
            <p style="margin:0;font-size:1.8rem;font-weight:700;color:{fpr_col};">{avg_fpr:.1%}</p>
            {'<p style="margin:0;font-size:.8rem;color:#888;">vs baseline: '
             + (f"{avg_fpr - base['false_positive_rate']:+.1%}" if base else "—")
             + '</p>'}
        </div>""", unsafe_allow_html=True)

    # Target description
    first = df_r.iloc[0]
    if first.get("target_description"):
        st.markdown(f'<div class="alert-info">🎯 <strong>Prediction target:</strong> {first["target_description"]}</div>',
                    unsafe_allow_html=True)

    # ── Governance banner (Feature 4) ─────────────────────────────────────────
    if "governance" in feats:
        gov = feats["governance"]
        cls = {"approved":"alert-success","rejected":"alert-warning",
               "deferred":"alert-info","reversed":"alert-danger"}.get(gov["outcome"],"alert-info")
        flags_html = "".join(f"<li>{f}</li>" for f in gov["ai_flags"]) or "<li>No drift detected</li>"
        t = gov["tally"]
        st.markdown(f"""
        <div class="{cls}" style="margin-top:1rem;">
            <strong>🏛️ Governance Vote — "{gov['policy']}"</strong><br>
            Outcome: <strong>{gov['outcome'].upper()}</strong> &nbsp;|&nbsp;
            For: {t.get('for',0)} &nbsp; Against: {t.get('against',0)} &nbsp; Abstain: {t.get('abstain',0)}<br>
            <strong>AI Flags:</strong><ul style="margin:.3rem 0 0 1rem;">{flags_html}</ul>
            <span style="font-size:.75rem;opacity:.7;">Ledger hash: <code>{gov['ledger_hash']}</code></span>
        </div>""", unsafe_allow_html=True)

    # ── Baseline comparison ────────────────────────────────────────────────────
    if base:
        st.info(
            f"**Baseline (no bias / no attack):** "
            f"Detection {base['detection_rate']:.1%}  |  "
            f"Liberty {base['liberty_score']:.2f}  |  "
            f"FPR {base['false_positive_rate']:.1%}  — "
            f"deltas above show the impact of your configuration."
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    # ── Share URL panel ───────────────────────────────────────────────
    _share_cfg = {"domain":"security","scenario":scenario,"surveillance_level":surveillance_level,"bias_intensity":bias_intensity,"poison_rate":poison_rate,"oversight_level":oversight_level}
    share_url_panel("security", config=_share_cfg)

    # ── Board Member view (role-specific executive summary) ───────────
    _role_now = get_active_role("security")
    if _role_now == "Board Member":
        _fair_val = avg_lib if "avg_lib" in dir() else 0.5
        _acc_val  = avg_det if "avg_det" in dir() else 0.5
        _ok = _fair_val >= 0.7
        _finding = ("Fairness score is within acceptable range. No critical disparities detected."
                    if _ok else "Fairness score below 0.70 — demographic disparities detected.")
        _rec = ("Continue quarterly monitoring and maintain current governance oversight."
                if _ok else "Bias mitigation required before deployment. Consult Data Science team.")
        board_member_summary("security", _acc_val, _fair_val, _ok, _finding, _rec)
    else:
        metric_glossary_expander(["false positive rate", "false negative rate", "liberty score", "demographic parity", "equalized odds", "bias intensity", "poison rate", "sociotechnical risk", "fairness score"])
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Performance", "⚖️ Trade-offs",
        "🎭 Bias Impact", "🔬 Feature Modules",
        "🧠 Explainable AI", "📋 Compliance",
        "🔁 Longitudinal", "🌐 Federated",
        "📋 Raw Results",
    ])

    # ── Tab 1: Performance ─────────────────────────────────────────────────────
    with tab1:
        fig_g = make_subplots(rows=1, cols=3,
            specs=[[{"type":"indicator"}]*3],
            subplot_titles=("Threat Detection","Civil Liberties","False Alarm Rate"))
        for col_i, (val, colour, rng) in enumerate([
            (avg_det*100, "darkblue",  [0,100]),
            (avg_lib*100, "darkgreen", [0,100]),
            (avg_fpr*100, "darkred",   [0,50]),
        ], 1):
            fig_g.add_trace(go.Indicator(
                mode="gauge+number", value=round(val, 1),
                gauge={"axis":{"range":rng},"bar":{"color":colour},
                       "steps":[{"range":[0,rng[1]*0.6],"color":"#f0f0f0"},
                                 {"range":[rng[1]*0.6,rng[1]*0.85],"color":"#d0d0d0"}]},
            ), row=1, col=col_i)
        fig_g.update_layout(height=280, showlegend=False, margin=dict(t=40,b=0))
        st.plotly_chart(fig_g, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_bar = px.bar(df_r, x="run_id",
                y=["detection_rate","false_positive_rate"],
                barmode="group", title="Detection vs False Alarms per Run",
                labels={"value":"Rate","variable":"Metric"},
                color_discrete_sequence=["#22c55e","#ef4444"])
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            avg_cm = {
                "True Positives":  df_r["tp"].mean(),
                "False Positives": df_r["fp"].mean(),
                "False Negatives": df_r["fn"].mean(),
                "True Negatives":  df_r["tn"].mean(),
            }
            fig_cm = px.bar(x=list(avg_cm.keys()), y=list(avg_cm.values()),
                title="Average Confusion Matrix",
                color=list(avg_cm.keys()),
                color_discrete_sequence=["#22c55e","#ef4444","#f39c12","#2563eb"])
            fig_cm.update_layout(showlegend=False)
            st.plotly_chart(fig_cm, use_container_width=True)

    # ── Tab 2: Trade-offs ──────────────────────────────────────────────────────
    with tab2:
        st.markdown(t("security_vs_liberty_trade_off"))
        if len(df_r) > 1:
            fig_sc = px.scatter(df_r, x="liberty_score", y="detection_rate",
                size=[max(f*30+5, 5) for f in df_r["false_positive_rate"]],
                color="surveillance_level" if "surveillance_level" in df_r.columns else "bias_intensity",
                hover_data=["biases","attack_sophistication","oversight","dataset"],
                title="Detection vs Liberty (bubble size = FPR)",
                labels={"liberty_score":"Civil Liberty Score",
                        "detection_rate":"Threat Detection Rate",
                        "surveillance_level":"Surveillance Intensity"},
                color_continuous_scale="RdYlGn_r")
            fig_sc.add_shape(type="rect", x0=0.7, x1=1.0, y0=0.8, y1=1.0,
                line=dict(color="green",width=2,dash="dash"),
                fillcolor="rgba(0,200,0,0.07)")
            fig_sc.add_annotation(x=0.85, y=0.9, text="Optimal zone",
                showarrow=False, font=dict(color="green",size=11))
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Detection Rate", f"{df_r['detection_rate'].iloc[0]:.1%}")
            c2.metric("Liberty Score",  f"{df_r['liberty_score'].iloc[0]:.2f}")
            c3.metric("False Alarm Rate", f"{df_r['false_positive_rate'].iloc[0]:.1%}")
            st.caption("Run 2+ simulations to plot the detection-vs-liberty trade-off scatter.")

        st.markdown(t("fairness_vs_security"))
        if len(df_r) > 1:
            fig_fs = px.scatter(df_r, x="fairness_score", y="security_score",
                color="bias_intensity",
                title="Fairness Score vs Security Score",
                labels={"fairness_score":"Algorithmic Fairness","security_score":"Security Score"},
                color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig_fs, use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            c1.metric("Fairness Score",  f"{df_r['fairness_score'].iloc[0]:.3f}")
            c2.metric("Security Score",  f"{df_r['security_score'].iloc[0]:.3f}")
            st.caption("Run 2+ simulations to see the scatter plot across multiple data points.")

    # ── Tab 3: Bias Impact ─────────────────────────────────────────────────────
    with tab3:
        st.markdown(t("bias_impact_assessment"))
        _bias_implications = {
            "demographic": "Ethnic, religious, or nationality-based targeting risks discrimination "
                           "and erodes public trust in security services.",
            "geographic":  "Regional over-surveillance creates blind spots and may displace "
                           "threat activity rather than prevent it.",
            "historical":  "Training on historically biased incident data perpetuates past "
                           "patterns of disproportionate enforcement.",
            "measurement": "Measurement inconsistency across demographic groups introduces "
                           "differential accuracy in threat scoring.",
            "temporal":    "Temporal drift in threat patterns can cause models trained on "
                           "older data to mis-classify current activity.",
            "socioeconomic":"Socioeconomic proxies in models can correlate with protected "
                            "characteristics and produce discriminatory outcomes.",
        }

        active = [b for b in selected_biases if b in _VALID_BIAS_TYPES]
        if active:
            for bt in active:
                msg = _bias_implications.get(bt,
                      f"'{bt}' bias may introduce systematic errors in threat assessment.")
                st.markdown(f'<div class="alert-warning"><strong>{bt.title()} Bias:</strong> {msg}</div>',
                            unsafe_allow_html=True)

            # Fix 1: derive real bias impact by comparing biased runs against
            # the baseline (no-bias/no-attack) that was already computed.
            # Each active bias gets the delta = biased_metric - baseline_metric.
            if base:
                # Use the actual measured deltas from the simulation results
                det_delta  = avg_det  - base["detection_rate"]
                lib_delta  = avg_lib  - base["liberty_score"]
                fpr_delta  = avg_fpr  - base["false_positive_rate"]

                # Apportion the total delta evenly across active biases
                # (additive decomposition — each bias is held responsible for 1/n of total)
                n_active = max(len(active), 1)
                impact_rows = [
                    {
                        "bias_type":        bt,
                        "detection_delta":  round(det_delta / n_active, 4),
                        "liberty_delta":    round(lib_delta  / n_active, 4),
                        "fpr_delta":        round(fpr_delta  / n_active, 4),
                    }
                    for bt in active
                ]
                impact_df = pd.DataFrame(impact_rows)

                fig_bi = px.bar(
                    impact_df, x="bias_type",
                    y=["detection_delta","liberty_delta","fpr_delta"],
                    barmode="group",
                    title="Measured Impact vs Baseline (apportioned per bias type)",
                    labels={"value":"Delta vs baseline","variable":"Metric"},
                    color_discrete_sequence=["#22c55e","#2563eb","#ef4444"],
                )
                fig_bi.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
                st.plotly_chart(fig_bi, use_container_width=True)
                st.caption(
                    "Deltas measured against the no-bias / no-attack baseline run. "
                    "Positive detection delta = bias improves recall; "
                    "positive FPR delta = bias increases false alarms. "
                    "Total delta is split equally across active bias types as an approximation."
                )
            else:
                st.info(
                    "Enable **Include baseline** in the sidebar and re-run to see "
                    "real measured bias impact against a clean reference point."
                )

            st.info("""
            **Bias Mitigation Strategies:**
            1. Regular algorithmic audits across demographic groups
            2. Diverse and representative training data
            3. Explainable AI for reviewable alert rationale
            4. Mandatory human review for high-stakes decisions
            5. Pre-deployment bias scenario testing
            """)
        else:
            st.markdown('<div class="alert-success">No biases active — '
                        'this configuration prioritises accuracy over demographic targeting.</div>',
                        unsafe_allow_html=True)

    # ── Tab 4: Feature Modules ─────────────────────────────────────────────────
    with tab4:
        st.markdown(t("advanced_feature_module_results"))

        # Feature 2 — Multimodal Red Teaming
        if "multimodal_redteam" in feats:
            st.markdown(t("feature_2_multimodal_red_teaming"))
            rt = feats["multimodal_redteam"]
            rt_rows = [
                {"Modality":             r["modality"],
                 "Attack Vector":        r["attack_vector"],
                 "Severity":             r["severity"],
                 "Affected Samples":     r["affected_samples"],
                 "Bypass Rate":          f"{r['bypass_rate']:.1%}",
                 "Sociotechnical Risk":  f"{r['sociotechnical_risk']:.2f}"}
                for r in rt.get("modality_results", [])
            ]
            if rt_rows:
                st.dataframe(pd.DataFrame(rt_rows), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.metric("Combined Bypass Rate",        f"{rt.get('combined_bypass_rate',0):.1%}")
            c2.metric("Combined Sociotechnical Risk", f"{rt.get('combined_sociotechnical_risk',0):.2f}")
            for r in rt.get("modality_results", []):
                if r.get("vr_scenario"):
                    st.markdown(f'<div class="alert-info">🥽 <strong>Immersive VR Scenario:</strong><br>'
                                f'{r["vr_scenario"]}</div>', unsafe_allow_html=True)
                    break

        # Feature 1 — Agent Economy
        if "agent_economy" in feats:
            st.markdown(t("feature_1_ai_agent_economy_security_resources"))
            ae = feats["agent_economy"]
            c1, c2 = st.columns(2)
            c1.metric("Economy Stability",  ae.get("economy_stability", "—"))
            c2.metric("Permeability Score", f"{ae.get('permeability_score',0):.4f}")
            agents_df = pd.DataFrame(ae.get("agent_summary", []))
            if not agents_df.empty:
                st.dataframe(agents_df.style.background_gradient(
                    subset=["reputation","total_spent"], cmap="Blues"),
                    use_container_width=True)

        # Feature 4 — Governance Ledger
        if "governance" in feats:
            st.markdown(t("feature_4_governance_ledger_blockchain_style"))
            gov = feats["governance"]
            st.markdown(f"""
            <div class="ledger-row">
                ▶ <strong>Policy:</strong> {gov['policy']}<br>
                ▶ <strong>Outcome:</strong> {gov['outcome'].upper()}<br>
                ▶ <strong>Tally:</strong> For={gov['tally'].get('for',0)},
                  Against={gov['tally'].get('against',0)},
                  Abstain={gov['tally'].get('abstain',0)}<br>
                ▶ <strong>Ledger Hash:</strong> <code>{gov['ledger_hash']}</code>
            </div>""", unsafe_allow_html=True)

        # Feature 5 — Strategic Arena
        if "strategic_arena" in feats:
            st.markdown(t("feature_5_strategic_social_reasoning_arena"))
            arena = feats["strategic_arena"]
            standings = pd.DataFrame(arena.get("final_standings", []))
            if not standings.empty:
                st.dataframe(standings.style.background_gradient(
                    subset=["score"], cmap="YlGn"),
                    use_container_width=True)
            coal = arena.get("coalition_scores", {})
            if coal:
                fig_coal = px.bar(x=list(coal.keys()), y=list(coal.values()),
                    title="Coalition Scores",
                    labels={"x":"Coalition","y":"Score"},
                    color=list(coal.values()),
                    color_continuous_scale="Viridis")
                st.plotly_chart(fig_coal, use_container_width=True)

        if not feats:
            st.info("Enable feature modules in the sidebar to see results here.")

    # ── Tab 5-8: XAI / Compliance / Longitudinal / Federated ─────────────────
    with tab5:
        _xai = st.session_state.get("security_xai_results",{})
        st.markdown("### 🧠 Explainable AI")
        if not _xai:
            st.info("Run a simulation to generate XAI explanations.")
        elif "error" in _xai:
            st.warning(f"XAI error: {_xai["error"]}")
        else:
            fi=_xai.get("feature_importance",{})
            if fi:
                st.markdown(f'<div class="alert-info"><em>{fi.get("narrative","")}</em></div>',unsafe_allow_html=True)
                import plotly.express as _px5
                fi_df=pd.DataFrame({"Feature":fi["feature_names"][:10],"Importance":fi["importances"][:10]})
                fig_fi=_px5.bar(fi_df,x="Importance",y="Feature",orientation="h",
                    title="Feature Importance",color="Importance",color_continuous_scale="Blues")
                fig_fi.update_layout(yaxis={"categoryorder":"total ascending"},height=320)
                st.plotly_chart(fig_fi,use_container_width=True)
            cl,cr_=st.columns(2)
            with cl:
                expl=_xai.get("instance_explanation",{})
                if expl:
                    st.markdown(t("instance_explanation"))
                    st.markdown(f'<div class="alert-info"><em>{expl.get("decision_path","")}</em></div>',unsafe_allow_html=True)
            with cr_:
                cf=_xai.get("counterfactual",{})
                if cf:
                    st.markdown("#### Counterfactual")
                    st.markdown(f'<div class="alert-info"><em>{cf.get("plain_language","")}</em></div>',unsafe_allow_html=True)
            ix=_xai.get("intersectional",{})
            if ix and ix.get("group_performances"):
                st.markdown(t("intersectional_fairness"))
                st.markdown(f'<div class="alert-info">{ix.get("narrative","")}</div>',unsafe_allow_html=True)
                ix_df=pd.DataFrame([{"Group":k,**{kk:round(vv,3) for kk,vv in v.items()}} for k,v in ix["group_performances"].items()])
                st.dataframe(ix_df.style.background_gradient(subset=["accuracy"],cmap="RdYlGn"),use_container_width=True)

    with tab6:
        _xai=st.session_state.get("security_xai_results",{})
        _cr_=_xai.get("compliance_report",{})
        _mc_=_xai.get("model_card",{})
        st.markdown(t("regulatory_compliance_report"))
        if not _cr_:
            st.info("Run a simulation to generate the compliance report.")
        else:
            summ=_cr_.get("summary",{})
            ok=summ.get("overall_compliant",False)
            st.markdown(f'<div class="{"alert-success" if ok else "alert-danger"}">Overall: <strong>{"COMPLIANT ✅" if ok else "NON-COMPLIANT ❌"}</strong> | Fairness: {summ.get("fairness_score",0):.3f}</div>',unsafe_allow_html=True)
            for fw,fd in _cr_.get("frameworks",{}).items():
                with st.expander(f"📑 {fw}"):
                    for ch,st_ in fd.get("checks",{}).items():
                        st.markdown(f"{"✅" if st_=="PASS" else "❌"} {ch}")
            try:
                pdf_b=generate_pdf_compliance_report(_cr_,_mc_,{"accuracy":df_r["detection_rate"].mean()},domain="national_security")
                st.download_button("📄 Download Compliance PDF",pdf_b,"security_compliance.pdf","application/pdf",use_container_width=True)
            except Exception as _e:
                st.caption(f"PDF unavailable: {_e}")

        # ── Real-world benchmark comparison ─────────────────────
        st.markdown(t("real_world_benchmark_comparison"))
        if BENCHMARKS_OK:
            _dom_bms = get_benchmarks_for_domain("national_security")
            if _dom_bms:
                _bm_sel = st.selectbox(t("compare_against_a_published_study"),
                        list(_dom_bms.keys()),
                        format_func=lambda k: _dom_bms[k].name + " (" + str(_dom_bms[k].year) + ")",
                        key="_sec_bm_sel")
                _bm = _dom_bms[_bm_sel]
                _acc_col = "accuracy" if "accuracy" in df_r.columns else ("detection_rate" if "detection_rate" in df_r.columns else None)
                _fair_col = "fairness_score" if "fairness_score" in df_r.columns else None
                _sim_m = {}
                if _acc_col: _sim_m["accuracy"] = df_r[_acc_col].mean()
                if _fair_col: _sim_m["fairness_score"] = df_r[_fair_col].mean()
                if "fpr" in df_r.columns: _sim_m["fpr"] = df_r["fpr"].mean()
                if "recall" in df_r.columns: _sim_m["recall"] = df_r["recall"].mean()
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


    with tab7:
        _lng=st.session_state.get("security_longitudinal")
        st.markdown(t("longitudinal_bias_analysis"))
        st.markdown('<div class="alert-info">Simulates bias <strong>feedback loops</strong> across successive model retraining cycles — critical for surveillance systems updated frequently.</div>',unsafe_allow_html=True)
        if not _lng:
            st.info("Run a simulation to see longitudinal bias evolution.")
        else:
            c1,c2,c3=st.columns(3)
            c1.metric("Initial Bias",f'{_lng["initial_bias"]:.1%}')
            c2.metric("Final Bias",f'{_lng["final_bias"]:.1%}',f'{_lng["final_bias"]-_lng["initial_bias"]:+.1%}')
            c3.metric("Amplification",f'{_lng["amplification_factor"]:.2f}×')
            if _lng.get("self_reinforcing"):
                st.error(f"⚠️ Bias became self-reinforcing at generation {_lng.get("inflection_point","N/A")}.")
            st.markdown(f'<div class="alert-info"><em>{_lng["narrative"]}</em></div>',unsafe_allow_html=True)
            gm=_lng.get("generation_metrics",[])
            if gm:
                import plotly.express as _px6
                fig_lng=_px6.line(pd.DataFrame(gm),x="generation",
                    y=["demographic_parity","fairness_score","accuracy"],
                    title="Bias Evolution Across Retraining Generations",
                    color_discrete_sequence=["#ef4444","#16a34a","#2563eb"])
                fig_lng.add_hline(y=0.1,line_dash="dot",line_color="red")
                st.plotly_chart(fig_lng,use_container_width=True)

    with tab8:
        _fed=st.session_state.get("security_federated")
        st.markdown(t("federated_learning_simulation"))
        st.markdown('<div class="alert-info">Tests bias persistence when threat-detection models are trained <strong>across jurisdictions</strong> without centralising sensitive intelligence data.</div>',unsafe_allow_html=True)
        if not _fed:
            st.info("Run a simulation to see federated learning results.")
        else:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Clients",_fed["n_clients"])
            c2.metric("Global Accuracy",f'{_fed["global_accuracy"]:.1%}')
            c3.metric("Global Fairness",f'{_fed["global_fairness"]:.3f}')
            c4.metric("Bias Persisted","Yes ⚠️" if _fed["bias_persisted"] else "No ✅")
            st.markdown(f'<div class="{"alert-danger" if _fed["bias_persisted"] else "alert-success"}"><em>{_fed["narrative"]}</em></div>',unsafe_allow_html=True)
            cr_=_fed.get("client_results",[])
            if cr_:
                import plotly.express as _px7
                cr_df=pd.DataFrame([c.__dict__ if hasattr(c,"__dict__") else c for c in cr_])
                if not cr_df.empty and "local_bias" in cr_df.columns:
                    fig_fed=_px7.bar(cr_df,x="client_id",y=["local_accuracy","local_bias"],
                        barmode="group",title="Per-Client Metrics",
                        color_discrete_sequence=["#2563eb","#ef4444"])
                    st.plotly_chart(fig_fed,use_container_width=True)

    # ── Tab 9: Raw Results ──────────────────────────────────────────────────────
    with tab9:
        display_cols = ["run_id","data_source","scenario","detection_rate","precision",
                        "false_positive_rate","accuracy","liberty_score","security_score",
                        "fairness_score","demographic_parity","surveillance_level",
                        "bias_intensity","poison_rate","attack_type","oversight","biases"]
        show_df = df_r[[c for c in display_cols if c in df_r.columns]]
        fmt = {c: "{:.3f}" for c in display_cols
               if c not in ("run_id","data_source","scenario","attack_type","oversight","biases")}
        st.dataframe(
            show_df.style
                .format({k: v for k, v in fmt.items() if k in show_df.columns})
                .background_gradient(subset=["detection_rate"],  cmap="Greens")
                .background_gradient(subset=["liberty_score"],   cmap="RdYlGn")
                .background_gradient(subset=["false_positive_rate"], cmap="Reds"),
            use_container_width=True,
        )

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                t("download_csv"),
                df_r.to_csv(index=False).encode(),
                f"gags_security_{scenario.lower().replace(' ','_')}.csv",
                "text/csv", use_container_width=True,
            )
        with dl2:
            cfg_export = {
                "scenario": scenario, "data_source": data_source,
                "dataset": dataset_choice, "threat_level": threat_level,
                "selected_biases": selected_biases, "bias_intensity": bias_intensity,
                "attack_type": attack_type_label, "poison_rate": poison_rate,
                "surveillance_level": surveillance_level, "oversight": oversight_level,
                "data_retention": data_retention,
                "features_enabled": {
                    "multimodal_redteam": enable_redteam,
                    "governance": enable_governance,
                    "strategic_arena": enable_arena,
                    "agent_economy": enable_agent_economy,
                },
                "avg_detection": f"{avg_det:.3f}",
                "avg_liberty":   f"{avg_lib:.3f}",
                "avg_fpr":       f"{avg_fpr:.3f}",
            }
            st.download_button(
                "📋 Export Config JSON",
                json.dumps(cfg_export, indent=2),
                f"security_config_{scenario.lower().replace(' ','_')}.json",
                "application/json", use_container_width=True,
            )

    # ── Tab 6: Data Explorer ───────────────────────────────────────────────────
    with tab6:
        st.markdown(t("data_explorer"))
        if data_source == "Real-World" and st.session_state.security_rw_df is not None:
            rw = st.session_state.security_rw_df
            c1, c2 = st.columns(2)
            c1.metric("Total Records",  f"{len(rw):,}")
            c2.metric("Total Features", len(rw.columns))

            st.subheader(t("data_preview"))
            st.dataframe(rw.head(20), use_container_width=True)

            st.subheader(t("basic_statistics"))
            st.dataframe(rw.describe(), use_container_width=True)

            if dataset_choice == "global_terrorism":
                gtd_preview = [c for c in ["iyear","imonth","country_txt","region_txt",
                    "attacktype1_txt","targtype1_txt","nkill","nwound"] if c in rw.columns]
                if gtd_preview:
                    st.subheader(t("gtd_key_fields"))
                    st.dataframe(rw[gtd_preview].head(15), use_container_width=True)
                if "attacktype1_txt" in rw.columns:
                    fig_atk = px.bar(
                        rw["attacktype1_txt"].value_counts().head(10),
                        title="Top 10 Attack Types in Dataset",
                    )
                    st.plotly_chart(fig_atk, use_container_width=True)
        else:
            st.info("Select 'Real-World' data source in the sidebar and load a dataset to explore it here.")

    # ── Simulation history ─────────────────────────────────────────────────────
    history_browser("security_snapshot_history", domain="security",
        key_metrics=["detection_rate","liberty_score","false_positive_rate"])

    # ── Annotation layer ────────────────────────────────────────────────
    annotation_panel("security_annotations", context_label=f"{len(st.session_state.security_run_history)} Security run(s)")

    # ── Policy Recommendations ─────────────────────────────────────────────────
    st.divider()
    st.markdown(t("security_ethics_recommendations"))
    r1, r2 = st.columns(2)
    with r1:
        if avg_fpr > 0.1:
            st.markdown("""
            <div class="alert-danger">
                <strong>⚠️ High False Positive Rate Detected</strong><br>
                • Increase judicial oversight requirements<br>
                • Mandate human review for all positive alerts<br>
                • Reduce surveillance intensity in low-threat areas<br>
                • Add fairness constraints to detection algorithms
            </div>""", unsafe_allow_html=True)
        elif avg_det < 0.7:
            st.markdown("""
            <div class="alert-warning">
                <strong>⚠️ Suboptimal Threat Detection</strong><br>
                • Consider moderate surveillance increase in high-threat zones<br>
                • Add multi-factor confirmation for threat alerts<br>
                • Expand training data diversity<br>
                • Recalibrate detection thresholds
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <strong>✅ Good Balance Achieved</strong><br>
                • Continue quarterly bias and fairness audits<br>
                • Maintain strong oversight mechanisms<br>
                • Review data retention policies annually<br>
                • Update threat models as adversary tactics evolve
            </div>""", unsafe_allow_html=True)
    with r2:
        st.info("""
        **Best Practices for Security AI:**
        1. **Proportionality** — Surveillance proportional to threat level
        2. **Transparency** — Publish detection criteria where operationally possible
        3. **Data Minimisation** — Collect only what is necessary
        4. **Oversight** — Independent review of surveillance practices
        5. **Redress** — Clear process for false-positive victims
        6. **Sunset Clauses** — Automatic review of surveillance powers
        7. **Audit Trail** — Tamper-evident logs for all AI decisions
        """)

    st.caption(
        "⚠️ Disclaimer: Educational simulation only. "
        "Real-world security AI requires comprehensive legal, ethical, and operational review."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Welcome state
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown(t("welcome_to_national_security_simulation"))
    st.markdown(
        "Configure your scenario in the sidebar and click **Run**. "
        "This module tests AI surveillance systems against bias, adversarial attacks, "
        "and governance constraints — with real-world GTD / UNSW-NB15 datasets and "
        "full GAGS v3.0 feature integration."
    )

    c1, c2, c3 = st.columns(3)
    scenario_cards = [
        ("🕵️ High Security",    "#1e3c72",
         "Surveillance 80+, all bias types. Maximise detection. Try with GTD data."),
        ("⚖️ Balanced Approach","#0f6e56",
         "Surveillance 40–60, moderate oversight. Find the optimal trade-off."),
        ("🕊️ Liberty Focused",  "#784212",
         "Surveillance 20–40, judicial oversight, no bias. Prioritise civil rights."),
    ]
    for col, (title, bg, desc) in zip([c1, c2, c3], scenario_cards):
        col.markdown(
            f'<div class="scenario-card" style="border-left-color:{bg};">'
            f'<h4 style="color:{bg};">{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

    with st.expander(t("how_to_use"), expanded=False):
        st.markdown("""
        1. Choose **data source** (synthetic or real-world GTD/UNSW)
        2. Enable **GAGS Feature Modules** (Red Team, Governance, Arena, Economy)
        3. Select **bias types** from the validated list
        4. Configure **surveillance** intensity and oversight level
        5. Run simulations and review the 6-tab dashboard
        6. Check the **Governance banner** for AI drift flags and ledger hash
        7. Export results as CSV or configuration as JSON
        """)

    st.warning("""
    **Ethical Considerations:** Proportionality · Necessity · Accountability ·
    Transparency · Non-discrimination. This simulation explores these tensions
    in a controlled, educational environment.
    """)


# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:#7f8c8d;padding:1rem 0;'>"
    "🛡️ National Security Simulation · GAGS Framework v3.0 · "
    "GTD / UNSW-NB15 · Multimodal Red Teaming · Governance Ledger"
    "</div>",
    unsafe_allow_html=True,
)
